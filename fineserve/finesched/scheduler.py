import os
import signal
import asyncio
import aiohttp
import enum
import yaml
import time
import json
import numpy as np
import subprocess
from typing import Any, Dict, Iterable, List, Tuple, Set
import torch
from transformers import AutoConfig

from fineserve.finesched.placement.estimator import FineServeEstimator
from fineserve.finesched.placement.model import Llama
from fineserve.finesched.scheduling_algorithm.algorithm import get_scheduling_algorithm
from fineserve.utils.constant import (SM_HOLD_NAME_FMT, OUT_HOLD_NAME_FMT, ADD_REQ_NAME_FMT,
                                      RET_REQ_NAME_FMT, PREEMPT_REQ_NAME_FMT)
from fineserve.config import FineServeConfig
from fineserve.utils.workload_utils import get_workload, Workload, Request
from fineserve.utils.shm_utils import (create_shared_var, read_shared_var,
                                       write_shared_var,
                                       load_from_shared_var, close_shared_var,
                                       dump_reqs_to_shared_var, read_list_from_shared_var)

from fineserve.finesched.launcher import launch_server_process
from fineserve.logger import get_logger
from fineserve.utils.pipe_utils import pipeline_split

logger = get_logger()
IS_STANDALONE = os.environ.get("SCHED_STANDALONE", None)
DEBUG = os.environ.get("DEBUG", None)


class SchedStatus(enum.Enum):
    PREFILL_NOREQ = 0
    DECODE_NOSM = 1
    NO_MEM = 2
    NO_SM = 3
    RUNNING = 4
    NO_REQ = 5
    ON_HOLD = 6


class FineServeScheduler:

    def __init__(self, fineserve_config: FineServeConfig):
        self.fineserve_config = fineserve_config
        self.processes = {
            mps_percent: {}
            for mps_percent in [20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        self.node_rank = self.fineserve_config.node_rank
        self.ngpu_per_rank = self.fineserve_config.ngpu_per_node
        self.num_gpus = self.fineserve_config.num_gpus
        self.pipeline_parallel_size = 1

        ## workload related
        self._workload_queue: asyncio.PriorityQueue[Tuple[
            float, Request]] = asyncio.PriorityQueue()
        self.workload: Workload = None
        self.counter: Set[int] = set()
        self.is_finished = False

        ## scheduling queue related
        self.lock = asyncio.Lock()
        self.executing: Dict[str, Set[int]] = {}  # key: model-name, val: set of request ids
        self.in_prefill: Dict[str, Set[int]] = {}  # key: model-name, val: set of request ids
        self.waiting: Dict[str, List[Request]] = {}
        self.dropped: Dict[str, List[Request]] = {}
        self._preempted: Dict[str, List[Request]] = {}
        self.wait_for_cache: Dict[str, bool] = {}
        self.cannot_schedule: Dict[str, bool] = {}
        self.max_num_seqs: Dict[str, int] = {}
        self.hist_num_seqs: Dict[str, List[int]] = {}
        self._served_models = []
        self._name_to_model: Dict[str, str] = {}
        self._model_to_mps_percentage: Dict[str, int] = {}

        self.shm_size = 6
        self.sm_hold_name_to_shm = {}
        self.output_hold_name_to_shm = {}
        self.model_mps_dual = {}

        # stats record
        self.sched_dict = {}
        self.batches: Dict[str, List[int]] = {}

        self.enable_profiler = False
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=True,
            with_modules=True) if self.enable_profiler else None
        self.prof_out_name = f"log/profiler_fineserve/fineserve_schduler.json"

        self.cost_file = self.fineserve_config.cost_file_path
        memory_per_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        self.cost_estimator = FineServeEstimator(self.cost_file,
                                                 memory_per_gpu)
        self._served_llms = {}
        self._ttft_per_token = {}
        self.sched_alg = get_scheduling_algorithm(self.fineserve_config.schedule_approach)
        logger.info(f"Scheduling Algorithm: {self.sched_alg.__name__}")

        self._max_num_seqs = 0
        self._max_num_batched_tokens = 0
        self.gpu_memory_utilization_adjusted = {}
        self.adjust_gpu_memory_utilization()

        import os
        import glob
        all_fmts = [
            SM_HOLD_NAME_FMT,
            OUT_HOLD_NAME_FMT,
            ADD_REQ_NAME_FMT,
            RET_REQ_NAME_FMT,
            PREEMPT_REQ_NAME_FMT,
        ]
        fmts = [f"_{fmt.split('_')[1]}_" for fmt in all_fmts]
        files = glob.glob("/dev/shm/*")
        for f in files:
            if not isinstance(f, str):
                continue
            if f.startswith('/dev/shm/eic-'):
                continue    # AWS-specific
            for fmt in fmts:
                if fmt in f:    # for safety
                    os.remove(f)

    def adjust_gpu_memory_utilization(self):
        _MAX_UTIL = 0.9
        all_placement = {}
        gpu_memory_utilization = {}
        gpu_memory_utilization_adjusted = {}
        for job_config in self.fineserve_config.job_configs:
            # init llms
            model_name = job_config.name
            slo = job_config.slo
            if hasattr(self.fineserve_config, "_slo_overwrite"):
                slo = getattr(self.fineserve_config, "_slo_overwrite")
            self._served_llms[model_name] = Llama(job_config.name,
                                                  job_config.model,
                                                  job_config.model_size_org,
                                                  job_config.qformat,
                                                  job_config.rate,
                                                  job_config.avg_input_len,
                                                  job_config.avg_output_len,
                                                  placement=job_config.placement,
                                                  slo=slo)
            llm = self._served_llms[model_name]
            tp_size = len(llm.placement)
            mps = job_config.mps_percentage[0]
            ttft_per_token = \
                self.cost_estimator.get_avg_latency_per_token(llm, tp_size, mps, 1,
                                                              prefill=True)
            self._ttft_per_token[model_name] = ttft_per_token
            logger.info(f"ttft/token for {model_name}: {ttft_per_token}")

            for rank in llm.placement:
                all_placement. \
                    setdefault(rank, []). \
                    append(model_name)
                gpu_memory_utilization. \
                    setdefault(model_name, {}). \
                    setdefault(rank, job_config.gpu_memory_utilization)

        ## restrict gpu memory utilization (< _MAX_UTIL)
        for rank, placement in all_placement.items():
            gmus = np.array([gpu_memory_utilization[model_name][rank] for model_name in placement])
            gmus_adjusted = gmus / np.sum(gmus) * _MAX_UTIL

            for idx, model_name in enumerate(placement):
                gpu_memory_utilization_adjusted. \
                    setdefault(model_name, {}). \
                    setdefault(rank, gmus_adjusted[idx])

        for model_name, gmus_adj in gpu_memory_utilization_adjusted.items():
            self.gpu_memory_utilization_adjusted[model_name] = float(min(gmus_adj.values()))

        ## adjust gpu memory utilization
        for node_rank in range(self.fineserve_config.nnodes):
            rank_start = node_rank * self.fineserve_config.nproc_per_node
            rank_end = rank_start + self.fineserve_config.nproc_per_node
            total_util = []
            all_models = set()
            for rank in range(rank_start, rank_end):
                total_util_rank = np.sum([self.gpu_memory_utilization_adjusted[model_name]
                                          for model_name in all_placement[rank]])
                total_util.append(total_util_rank)
                all_models.update(all_placement[rank])

            max_total_util = np.max(total_util)
            if max_total_util <= 0:
                continue
            for model_name in all_models:
                self.gpu_memory_utilization_adjusted[model_name] *= float(_MAX_UTIL / max_total_util)


    def serve_models(self):
        """Serve all models with MPS processes."""
        np.random.seed(0)
        logger.info(f"FineServeScheduler begins serve_models()")
        port = self.fineserve_config.server_port
        block_size = self.fineserve_config.block_size

        cluster_id = 0
        time.sleep(5)
        self.model_to_blocks_per_token: Dict[str, int] = {}
        for model_id, job_config in enumerate(self.fineserve_config.job_configs):
            ## check the placement of the job_config, and skip if not included
            skip = True
            for gpu_rank in job_config.placement:
                if gpu_rank // self.ngpu_per_rank == self.node_rank:
                    skip=False
            if skip:
                continue
            model_name = job_config.name

            self._name_to_model[model_name] = job_config.model
            self.pipeline_parallel_size = job_config.pipeline_parallel_size
            self.max_num_seqs[model_name] = job_config.max_num_seqs
            self.hist_num_seqs[model_name] = []
            mps_list = [job_config.mps_percentage[0]]

            for i, mps_percentage in enumerate(mps_list):
                name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
                shm_var = create_shared_var(name,
                                            size=self.shm_size,
                                            create=True)
                self.sm_hold_name_to_shm[name] = shm_var
                name = OUT_HOLD_NAME_FMT.format(model_name, mps_percentage)
                shm_var = create_shared_var(name,
                                            size=self.shm_size,
                                            create=True)
                self.output_hold_name_to_shm[name] = shm_var

                self.processes[mps_percentage][model_name] = {}
                self._model_to_mps_percentage[model_name] = mps_percentage
                dp_rank = 0

                gmus_adj = self.gpu_memory_utilization_adjusted[model_name]
                proc = launch_server_process(
                    model_id,
                    block_size,
                    mps_percentage,
                    job_config,
                    self.fineserve_config,
                    gpu_memory_utilization=gmus_adj,
                )
                cluster_id += 1
                self.processes[mps_percentage][model_name][dp_rank] = (proc, port)
                port += 1
                if self.fineserve_config.restrict_gpu_memory_utilization:
                    self.wait_to_warmup(model_name)

            self.executing[model_name] = set()
            self.in_prefill[model_name] = set()
            self.waiting[model_name] = []
            self.dropped[model_name] = []
            self._preempted[model_name] = []
            self._served_models.append(model_name)
            self.wait_for_cache[model_name] = False
            self.cannot_schedule[model_name] = False

            self.batches[model_name] = []

            model_config = AutoConfig.from_pretrained(job_config.model)
            tensor_parallel_size = job_config.tensor_parallel_size
            num_heads = model_config.num_attention_heads // tensor_parallel_size
            partition = pipeline_split(model_config.num_hidden_layers,
                                       job_config.pipeline_parallel_size)
            num_hidden_layers = max(partition)
            self.model_to_blocks_per_token[
                model_name] = num_heads * num_hidden_layers
        self.batches["total"] = []

        logger.info(f"FineServeScheduler finished preparing models.")
        logger.info(f"Model Config: {self.fineserve_config.model_config_path}")
        logger.info(f"{yaml.dump(self.fineserve_config.model_config)}")

        self.prepare_workloads()
        logger.info(f"FineServeScheduler finished preparing workload.")

    def prepare_workloads(self):
        self.workload = self.get_workload()
        # get rate for each model
        self.model_rates = {}
        logger.info(f"Workload rates: {self.workload.workload_infos['rates']}")
        for (model_name, rate) in self.workload.workload_infos["rates"]:
            if model_name in self._served_models:
                self.model_rates[model_name] = rate

        input_len = np.array([r.data[1] for r in self.workload.requests])
        output_len = np.array([r.data[2] for r in self.workload.requests])
        seq_len = np.array([r.data[1] + r.data[2] for r in self.workload.requests])

        avg_input_len = np.mean(input_len)
        avg_output_len = np.mean(output_len)
        avg_seq_len = np.mean(seq_len)
        logger.info(f"Workload avg_input_len: {avg_input_len}")
        logger.info(f"Workload avg_output_len: {avg_output_len}")
        logger.info(f"Workload avg_seq_len: {avg_seq_len}")

        total_input_len = np.sum(input_len)
        total_output_len = np.sum(output_len)
        total_seq_len = np.sum(seq_len)
        last_arrival = self.workload.arrivals[-1]
        logger.info(f"Workload total_input_len: {total_input_len}")
        logger.info(f"Workload total_output_len: {total_output_len}")
        logger.info(f"Workload total_seq_len: {total_seq_len}")
        logger.info(f"Workload token rate: {total_seq_len / last_arrival:.3f} tokens/s")

    def clean_up(self):
        logger.info(f"Clean processes...")
        # Clear shared variables.
        for job_config in self.fineserve_config.job_configs:
            model_name = job_config.name
            for mps_percentage in job_config.mps_percentage:
                for fmt in [
                    SM_HOLD_NAME_FMT, ADD_REQ_NAME_FMT, RET_REQ_NAME_FMT,
                    PREEMPT_REQ_NAME_FMT
                ]:
                    name = fmt.format(model_name, mps_percentage)
                    close_shared_var(name)

        for mps_percent in self.processes:
            for model_name in self.processes[mps_percent]:
                for dp_rank in self.processes[mps_percent][model_name]:
                    proc, _ = self.processes[mps_percent][model_name][dp_rank]

                    output = subprocess.check_output(
                        ['pgrep', '-P', str(proc.pid)])
                    child_pids = [int(pid) for pid in output.decode().split()]

                    # Terminate the child processes
                    for pid in child_pids:
                        os.kill(pid, signal.SIGTERM)
                    os.kill(proc.pid, signal.SIGTERM)
                    logger.info(f"Kill parent process {proc.pid}, "
                                f"child processes: {child_pids}")

    def get_workload(self) -> Workload:
        workload_file = self.fineserve_config.workload_config.get(
            "workload_file", None)
        if workload_file:
            workload = Workload.from_workload_file(workload_file)
        else:
            models = [
                job_config.name
                for job_config in self.fineserve_config.job_configs
            ]
            model_paths = [
                job_config.model
                for job_config in self.fineserve_config.job_configs
            ]
            arrival_rates = self.fineserve_config.workload_config["arrival_rates"]
            start = self.fineserve_config.workload_config["start"]
            duration = self.fineserve_config.workload_config["duration"]
            dataset = self.fineserve_config.workload_config["dataset"]
            num_requests = self.fineserve_config.workload_config.get(
                "num_requests", None)
            workload = get_workload(models,
                                    arrival_rates,
                                    model_paths,
                                    start,
                                    duration,
                                    dataset_path=dataset,
                                    num_requests=num_requests)

        split_by_model = self.fineserve_config.workload_config.get(
            "split_by_model", None)
        if split_by_model is not None:
            workload = workload.split_by_model(split_by_model)

        # only serve requests in the served models
        workload = workload.split_by_models(self._served_models)

        total_num_requests = 0
        for i in range(len(workload)):
            arrival_time = workload.arrivals[i]
            request = workload.requests[i]
            if request.model_name not in self._served_models:
                continue
            self._workload_queue.put_nowait((arrival_time, request))
            total_num_requests += 1
        # we use a counter to track the number of requests status
        self.counter = set([i for i in range(total_num_requests)])
        return workload

    def get_tick(self) -> float:
        return time.perf_counter() - self.cur_tick

    async def submit_workload(self, workload):
        # dispatch workload
        while True:
            if self.is_finished:
                break
            if self._workload_queue.empty():
                break
            (arrival_time, request) = self._workload_queue.get_nowait()
            if arrival_time > self.get_tick():
                self._workload_queue.put_nowait((arrival_time, request))
                await asyncio.sleep(0.003)
                continue
            async with self.lock:
                logger.info(f"append to waiting queue of [{request.model_name}]: {request.idx} ")
                request.arrival_time = arrival_time
                if request.slo is None:
                    request.slo = self._served_llms[request.model_name].slo
                self.waiting[request.model_name].append(request)

    async def finish_requests(self):
        logger.info(f"finish_requests started")
        while True:
            if self.is_finished:
                break

            for (model_name, mps_percentage) in self.model_names:
                name = PREEMPT_REQ_NAME_FMT.format(model_name, mps_percentage)
                req_ids = load_from_shared_var(name)
                async with self.lock:
                    for req_id in req_ids[::-1]:
                        req = self.workload.requests[req_id]
                        req.output_tokens = None
                        self.executing[req.model_name].remove(req_id)
                        self.waiting[req.model_name].insert(0, req)
                if req_ids:
                    self.wait_for_cache[model_name] = True
                    self.cannot_schedule[model_name] = True
                    logger.info(f"Preempte requests {req_ids} during decoding "
                                f"({model_name}), add them to waiting queue.")

            if len(self.counter) == 0:
                self.log_stats()
                self.is_finished = True
                break
            await asyncio.sleep(0.003)

    async def signal_exec(self,
                            model_name: str,
                            mps_percentage: int):
        num_iters = 1
        name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
        shm_var = self.sm_hold_name_to_shm[name]
        write_shared_var(shm_var, num_iters)

        logger.debug(f"running_mps.append({name})")
        self.running_mps.add(name)

    async def add_requests(self,
                           model_name: str,
                           mps_percentage: int,
                           requests: List[Request]):

        # add requests
        shm_name = ADD_REQ_NAME_FMT.format(model_name, mps_percentage)
        tb_add_ids, tb_add_tokens = [], []
        async with self.lock:
            for req in requests:
                tb_add_ids.append(req.idx)
                if req.submit_time is None:
                    req.submit_time = self.get_tick()
        if tb_add_ids:
            dump_reqs_to_shared_var(shm_name, requests + tb_add_tokens)

        await self.signal_exec(model_name, mps_percentage)

    async def try_schedule_prompts(self, model):
        mps_percentage = self._model_to_mps_percentage[model]
        sm_hold_name = SM_HOLD_NAME_FMT.format(model, mps_percentage)
        running = len(self.executing[model]) + len(self.in_prefill[model])
        hold_shm = create_shared_var(sm_hold_name, create=False)
        read_on_hold = read_shared_var(hold_shm)
        if read_on_hold == 1:
            status = SchedStatus.ON_HOLD
            return status
        async with self.lock:
            if self.wait_for_cache[model]:
                status = SchedStatus.NO_MEM
                return status
            sched_rlt = self.sched_alg.schedule(self, model)
            # scheduled: List[Request]
            scheduled = sched_rlt.scheduled
            dropped = sched_rlt.dropped
        if not running and not scheduled and not dropped:
            status = SchedStatus.PREFILL_NOREQ
            return status
        status = None
        if scheduled:
            req_ids = [req.idx for req in scheduled]
            for r in scheduled:
                self.waiting[model].remove(r)
            # self.executing[model].update(req_ids)
            self.in_prefill[model].update(req_ids)
            self.batches[model].append(len(scheduled))
            self.batches["total"].append(len(scheduled))
            logger.info(f"Schedule {model} {len(scheduled)} requests: "
                        f"{req_ids}")
            # launch tasks
            await self.add_requests(model, mps_percentage, scheduled)
        if dropped:
            req_ids = [req.idx for req in dropped]
            for r in dropped:
                r.dropped = True
                self.waiting[model].remove(r)
                self.dropped[model].append(r)
                self.counter.remove(r.idx)
            logger.info(f"Drop {model} {len(dropped)} requests: "
                        f"{req_ids}")
        return status

    async def schedule(self, prompt_in_exec: bool,
                       last_sched_time: float, last_warn_time: float):

        status = None
        ## try to schedule prompts for all models
        for model in self._served_models:
            status = await self.try_schedule_prompts(model)
            if status is None:
                last_sched_time = self.get_tick()
            else:
                if self.warn_log(last_sched_time, last_warn_time, status,
                                 model, True):
                    last_warn_time = self.get_tick()
                    logger.info(f"Unfinished requests: {sorted(self.counter)}")
                if status == SchedStatus.NO_SM:
                    return status, prompt_in_exec, last_sched_time, last_warn_time
        return status, prompt_in_exec, last_sched_time, last_warn_time

    def warn_log(self, last_sched_tick, last_warn_tick, status, model,
                 prefill):
        if self.get_tick() - last_sched_tick > 35 and self.get_tick(
        ) - last_warn_tick > 35:
            tag = "prefill" if prefill else "decode"
            logger.info(f"Fail to schedule {tag} requests due to "
                        f"{status} for {model}: "
                        f"waiting {len(self.waiting[model])} "
                        f"executing {len(self.executing[model])} ")
            return True
        return False

    async def schedule_requests(self):
        self.holding_sm_mps = []
        self.running_mps = set()
        num_requests = len(self.workload)
        # setup scheduler
        need_adapt = False
        adapt_interval = 0
        last_adapt_time = self.get_tick()
        last_sched_time = last_warn_time = self.get_tick()
        last_log_time = self.get_tick()
        prompt_in_exec, has_prompt_to_schedule = False, False
        while True:
            if self.is_finished:
                break

            # for debugging
            if self.get_tick()- last_log_time > 10:
                last_log_time = self.get_tick()
                logger.info(f"remaining tasks: {self.counter}")
                for model_name in self._served_models:
                    logger.info(f"{model_name} # of waiting requests: {len(self.waiting[model_name])}")
                    logger.info(f"{model_name} in-prefill requests: {self.in_prefill[model_name]}")
                    logger.info(f"{model_name} executing requests: {self.executing[model_name]}")


            for (model_name, mps_percentage) in self.model_names:
                ## check whether there is output to deal with
                out_hold_name = OUT_HOLD_NAME_FMT.format(model_name, mps_percentage)
                output_signal = read_shared_var(self.output_hold_name_to_shm[out_hold_name])
                if output_signal == 0:
                    continue

                ## read data from shared memory and signal back that data has been dealt with
                ret_name = RET_REQ_NAME_FMT.format(model_name, mps_percentage)
                data = read_list_from_shared_var(ret_name)
                output_signal = 0
                write_shared_var(self.output_hold_name_to_shm[out_hold_name], output_signal)
                index = 0

                ## first check requests that have finished prefill phase
                num_of_prefill_finished_reqs = data[index]
                index = index + 1
                for i in range(num_of_prefill_finished_reqs):
                    the_req_id = data[index]
                    self.workload.requests[the_req_id].prefill_end_time = self.get_tick()
                    # req.prefill_end_time = self.get_tick()
                    self.in_prefill[model_name].remove(the_req_id)
                    self.executing[model_name].add(the_req_id)
                    index = index + 1
                    if DEBUG:
                        logger.debug(f"request {the_req_id} has finished prefill phase")
                req_ids = []
                ## next check finish
                while index < len(data):
                    req_idx = data[index]
                    req_ids.append(req_idx)
                    output_len = data[index + 1]
                    self.workload.requests[req_idx].output_tokens = data[index + 2:index + 2 + output_len]
                    ## check if this request was aborted
                    if -1 in self.workload.requests[req_idx].output_tokens:
                        logger.info(f"DETECTED ABORTED REQUEST {req_idx}")
                        self.workload.requests[req_idx].aborted = True
                    if DEBUG is not None:
                        logger.info(f"tokens of request id {req_idx}:")
                        logger.info(f"{self.workload.requests[req_idx].output_tokens}")
                    index = index + 2 + output_len
                async with self.lock:
                    num_reqs = len(req_ids)
                    logger.info(f"returned {num_reqs} finished requests from {model_name}: {req_ids}")
                    for req_id in req_ids:
                        self.executing[model_name].remove(req_id)
                        self.workload.requests[req_id].end_time = self.get_tick()
                        self.counter.remove(req_id)
                        finished = num_requests - len(self.counter)
                        if finished % 10 == 0:
                            logger.info(f"Finish {finished}/{num_requests} requests")

                if len(self.executing[model_name]) == 0:
                    name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
                    self.running_mps.discard(name)

            ret = await self.schedule(prompt_in_exec,
                                      last_sched_time, last_warn_time)
            status, prompt_in_exec, last_sched_time, last_warn_time = ret


            ## prevents experiment from running forever
            if self.get_tick() - last_sched_time > 60 * 2:
                logger.error("Scheduler Timeout!")
                self.is_finished = True
                break

            if need_adapt and self.get_tick(
            ) - last_adapt_time > adapt_interval:
                self.adapt_max_num_seqs()
                last_adapt_time = self.get_tick()
                adapt_interval = 8

            await asyncio.sleep(0 if status is None else 0.0001)

    def wait_to_warmup(self, model_name):
        logger.info(f"waiting for model {model_name} to warmup")
        mps_percentage = self._model_to_mps_percentage[model_name]
        hold_sm_name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
        hold_sm_var = create_shared_var(hold_sm_name, create=False)
        warmup_done = read_shared_var(hold_sm_var)
        while not warmup_done:
            time.sleep(1)
            warmup_done = read_shared_var(hold_sm_var)
        warmup_ack = 0
        write_shared_var(hold_sm_var, warmup_ack)
        logger.info(f"server {model_name}-{mps_percentage} finished warmup")
        
    async def schedule_loop(self):
        self.model_names = []
        for model_name in self._served_models:
            ## warmup already checked when launching servers, so no need to do it again
            if not self.fineserve_config.restrict_gpu_memory_utilization:
                self.wait_to_warmup(model_name)
            mps_percentage = self._model_to_mps_percentage[model_name]
            self.model_names.append((model_name, mps_percentage))

        logger.info(f"FineServeScheduler Begin to schedule requests, "
                    f"total {len(self.workload)} requests.")
        logger.info(f"model_names: {self.model_names}")
        self.cur_tick = time.perf_counter()
        await asyncio.gather(
            self.submit_workload(self.workload),
            self.schedule_requests(),
            self.finish_requests(),
        )
        logger.info(f"Finish all requests begin to exit...")
        self.clean_up()

    def log_stats(self):
        logger.info(f"Finish all requests in node {self.node_rank}, total {len(self.workload)}")

        for job_config in self.fineserve_config.job_configs:
            model_name = job_config.name
            self.sched_dict[model_name] = {}
            self.sched_dict[model_name]["model_name"] = job_config.model
            self.sched_dict[model_name]["mps"] = job_config.mps_percentage
            self.sched_dict[model_name]["first_token_latency"] = 0
            self.sched_dict[model_name]["output_per_token_latency"] = 0
            self.sched_dict[model_name]["request_num"] = 0
            self.sched_dict[model_name]["all_latency"] = []
        #
        self.sched_dict["first_token_latency"] = 0
        self.sched_dict["output_per_token_latency"] = 0
        self.sched_dict["request_num"] = 0
        #
        model_to_requests = {}
        for i in range(len(self.workload)):
            req = self.workload.requests[i]
            if req.model_name not in model_to_requests:
                model_to_requests[req.model_name] = []
            model_to_requests[req.model_name].append(req)
            if req.submit_time is not None:
                sched_lat = req.submit_time - self.workload.arrivals[i]
                logger.info(
                    f"Request {req.idx} model {req.model_name} "
                    f"prompt {req.data[1]} "
                    f"arrival {self.workload.arrivals[i]:.3f} "
                    f"submit {req.submit_time:.3f} "
                    f"prefill_end {req.prefill_end_time:.3f} "
                    f"end {req.end_time:.3f} "
                    f"sched_lat {sched_lat:.3f} ")
            else:
                sched_lat = None
                logger.info(
                    f"Request {req.idx} model {req.model_name} "
                    f"prompt {req.data[1]} "
                    f"arrival {self.workload.arrivals[i]:.3f} "
                    f"submit {req.submit_time} "
                    f"prefill_end {req.prefill_end_time} "
                    f"end {req.end_time} "
                    f"sched_lat {sched_lat} ")

        logger.info(f"Scheduling Algorithm: {self.sched_alg.__name__}")
        logger.info("Input Workload Statistics:")
        if self.workload.workload_infos:
            rates = self.workload.workload_infos.pop("rates")
            for (model, rate) in rates:
                logger.info(f"  Model: {model} request rate: {rate}")
            for key, value in self.workload.workload_infos.items():
                logger.info(f"{key}: {value}")
        total_time, total_token, total_req = 0, 0, 0
        avg_lat, first_token_lat, avg_per_output_token_lat = 0, 0, 0
        avg_slo_attainment = 0
        # list for total statistics
        columns = [
            "model_name",
            "model_type",
            "qformat",
            "slo",
            "req_idx",
            "input_len",
            "output_len",
            "max_tokens",
            "arrival_time",
            "submit_time",
            "prefill_end_time",
            "decode_submit_time",
            "end_time",
        ]
        data = []

        latency_list, ttft_list, tpot_list = [], [], []
        slo_attainment_list = []
        for model_name, requests in model_to_requests.items():
            scheduled_requests = [req for req in requests if scheduled_and_executed(req)]
            dropped_requests = [req for req in requests if not scheduled_and_executed(req)]

            data.extend([
                [
                    model_name,
                    self._served_llms[model_name].model_type,
                    self._served_llms[model_name].qformat,
                    self._served_llms[model_name].slo,
                    req.idx,
                    req.data[1],
                    len(req.output_tokens) if req.output_tokens else None,
                    req.data[2],
                    req.arrival_time,
                    req.submit_time,
                    req.prefill_end_time,
                    req.decode_submit_time,
                    req.end_time,
                ]
                for i, req in enumerate(requests)
            ])

            if not scheduled_requests:
                continue

            total_num_tokens = sum(
                [len(req.output_tokens) for req in scheduled_requests])

            elapsed_time = max([req.end_time for req in scheduled_requests])
            req_tpt = len(scheduled_requests) / elapsed_time
            token_tpt = total_num_tokens / elapsed_time
            total_time = max(total_time, elapsed_time)
            total_token += total_num_tokens
            total_req += len(scheduled_requests)

            latency_list_per_model = [
                (req.end_time - self.workload.arrivals[req.idx])
                for req in scheduled_requests
            ]

            p99 = percentile(latency_list_per_model, 99)
            p95 = percentile(latency_list_per_model, 95)
            p90 = percentile(latency_list_per_model, 90)
            latency_list.extend(latency_list_per_model)
            #
            weight = len(requests) / len(self.workload)
            avg_lat_per_model = np.mean(latency_list_per_model)

            tpot_list_per_model = [
                (req.end_time - req.prefill_end_time) / len(req.output_tokens)
                for req in scheduled_requests
            ]
            avg_per_output_token_lat_per_model = np.mean(tpot_list_per_model)
            p99_tpot = percentile(tpot_list_per_model, 99)
            p95_tpot = percentile(tpot_list_per_model, 95)
            p90_tpot = percentile(tpot_list_per_model, 90)
            tpot_list.extend(tpot_list_per_model)
            #
            ttft_list_per_model = [
                (req.prefill_end_time - self.workload.arrivals[req.idx])
                for req in scheduled_requests
            ]
            first_token_lat_per_model = np.mean(ttft_list_per_model)
            p99_ttft = percentile(ttft_list_per_model, 99)
            p95_ttft = percentile(ttft_list_per_model, 95)
            p90_ttft = percentile(ttft_list_per_model, 90)
            ttft_list.extend(ttft_list_per_model)

            llm = self._served_llms[model_name]
            ttft_list_dropped_per_model = [llm.slo * 2 for _ in dropped_requests]
            slo_attainment_per_model = np.array(ttft_list_per_model +
                                                ttft_list_dropped_per_model) < llm.slo
            avg_slo_attainment_per_model = np.mean(slo_attainment_per_model) * 100
            slo_attainment_list.extend(slo_attainment_per_model)

            avg_lat += avg_lat_per_model * weight
            avg_per_output_token_lat += avg_per_output_token_lat_per_model * weight
            first_token_lat += first_token_lat_per_model * weight
            avg_slo_attainment += avg_slo_attainment_per_model * weight

            logger.info(
                f"Name: {model_name} \n"
                f"Model: {self._name_to_model[model_name]} \n"
                f"Throughput {req_tpt:.2f} requests/s {token_tpt:.2f} tokens/s \n"
                f"avg req latency: {avg_lat_per_model:.3f} \n"
                f"avg latency of first token: {first_token_lat_per_model:.3f} \n"
                f"avg latency per output token: {avg_per_output_token_lat_per_model:.3f} \n"
                f"[avg latency] p99: {p99:.3f}, p95: {p95:.3f}, p90: {p90:.3f} \n"
                f"[TTFT] p99: {p99_ttft:.3f}, p95: {p95_ttft:.3f}, p90: {p90_ttft:.3f} \n"
                f"[TPOT] p99: {p99_tpot:.3f}, p95: {p95_tpot:.3f}, p90: {p90_tpot:.3f} \n"
                f"[SLO] avg slo attainment: {avg_slo_attainment_per_model:.3f}% with slo {llm.slo} \n"
                f"total requests: {len(requests)} \n"
                f"scheduled requests: {len(scheduled_requests)} \n"
                f"dropped requests: {len(dropped_requests)} \n"
            )

            self.sched_dict[model_name]["throughput"] = req_tpt
            self.sched_dict[model_name]["tokens_throughput"] = token_tpt

            self.sched_dict[model_name][
                "first_token_latency"] = first_token_lat_per_model
            self.sched_dict[model_name][
                "output_per_token_latency"] = avg_per_output_token_lat_per_model

            self.sched_dict[model_name]["p99[avg_latency]"] = p99
            self.sched_dict[model_name]["p95[avg_latency]"] = p95
            self.sched_dict[model_name]["p90[avg_latency]"] = p90

            self.sched_dict[model_name]["p99[TTFT]"] = p99_ttft
            self.sched_dict[model_name]["p95[TTFT]"] = p95_ttft
            self.sched_dict[model_name]["p90[TTFT]"] = p90_ttft

            self.sched_dict[model_name]["p99[TPOT]"] = p99_tpot
            self.sched_dict[model_name]["p95[TPOT]"] = p95_tpot
            self.sched_dict[model_name]["p90[TPOT]"] = p90_tpot

            self.sched_dict[model_name]["avg_slo_attainment"] = avg_slo_attainment_per_model

            self.sched_dict[model_name]["request_num"] = len(scheduled_requests)
            self.sched_dict["request_num"] += len(scheduled_requests)

        p99 = percentile(latency_list, 99)
        p95 = percentile(latency_list, 95)
        p90 = percentile(latency_list, 90)
        p99_ttft = percentile(ttft_list, 99)
        p95_ttft = percentile(ttft_list, 95)
        p90_ttft = percentile(ttft_list, 90)
        p99_tpot = percentile(tpot_list, 99)
        p95_tpot = percentile(tpot_list, 95)
        p90_tpot = percentile(tpot_list, 90)
        req_tpt = total_req / total_time if total_time > 0 else np.nan
        token_tpt = total_token / total_time if total_time > 0 else np.nan
        logger.info(
            f"System Statistics Summary: \n"
            f"Throughput {req_tpt:.2f} "
            f"requests/s {token_tpt:.2f} tokens/s \n"
            f"avg req latency: {avg_lat:.3f} \n"
            f"avg latency of first token: {first_token_lat:.3f} \n"
            f"avg latency per output token: {avg_per_output_token_lat:.3f} \n"
            f"[avg latency] p99: {p99:.3f}, p95: {p95:.3f}, p90: {p90:.3f} \n"
            f"[TTFT] p99: {p99_ttft:.3f}, p95: {p95_ttft:.3f}, p90: {p90_ttft:.3f} \n"
            f"[TPOT] p99: {p99_tpot:.3f}, p95: {p95_tpot:.3f}, p90: {p90_tpot:.3f} \n"
            f"[SLO] avg slo attainment: {avg_slo_attainment:.3f}% \n"
            f"avg batches: {np.mean(self.batches['total']):.3f}, all times: {len(self.batches['total'])} \n"
            f"historical max num seqs: {self._max_num_seqs} \n"
            f"historical max num batched tokens: {self._max_num_batched_tokens}"
        )
        #
        self.sched_dict["throughput"] = req_tpt
        self.sched_dict["tokens_throughput"] = token_tpt
        self.sched_dict["first_token_latency"] = first_token_lat
        self.sched_dict["output_per_token_latency"] = avg_per_output_token_lat
        self.sched_dict["avg_slo_attainment"] = avg_slo_attainment
        #
        self.sched_dict["p99[avg_latency]"] = p99
        self.sched_dict["p95[avg_latency]"] = p95
        self.sched_dict["p90[avg_latency]"] = p90
        #
        workload_file = self.fineserve_config.workload_config.get("workload_file", None)
        PREFIX = os.environ.get("FLEXSM_SHM_PREFIX", "")
        with open(workload_file[:-5] + f"_{PREFIX}_stats.json", "w") as f:
            json.dump(self.sched_dict, f, indent=2)
        #
        import pandas as pd
        df = pd.DataFrame(columns=columns, data=data)
        df["sched_latency"] = df["submit_time"] - df["arrival_time"]
        df["latency"] = df["end_time"] - df["arrival_time"]
        df["ttft"] = df["prefill_end_time"] - df["arrival_time"]
        df["ttft_server"] = df["prefill_end_time"] - df["submit_time"]
        df["ttft_server_per_token"] = df["ttft_server"] / df["input_len"]
        df["tpot"] = (df["end_time"] - df["prefill_end_time"]) / df["output_len"]
        if hasattr(self.fineserve_config, "_scheduler_result_file"):
            file_name = getattr(self.fineserve_config, "_scheduler_result_file")
            logger.info(f"Save result to file: {file_name}")
            df.to_csv(f"{file_name}")


def percentile(l, q):
    return np.percentile(l, q=q) if l else np.nan


def scheduled_and_executed(req: Request):
    return req.submit_time and not req.dropped and not req.aborted


## Leave following code for future work
#def decode_response(response: Dict[str, Any]) -> str:
#    output = response["text"][0]
#    return output


#async def async_post_http_request(prompt: str,
#                                  request_id: str,
#                                  api_url: str,
#                                  is_free_cache: bool = False,
#                                  max_tokens: int = 2048) -> Dict[str, Any]:
#    headers = {"User-Agent": "Test Client"}
#
#    pload = {
#        "prompt": prompt,
#        "request_id": request_id,
#        "n": 1,
#        "use_beam_search": False,
#        "temperature": 0.0,
#        "max_tokens": max_tokens,
#        "stream": False,
#        "is_free_cache": is_free_cache,
#    }
#    async with aiohttp.ClientSession() as session:
#        async with session.post(api_url, headers=headers,
#                                json=pload) as response:
#            data = await response.json()
#    return data
