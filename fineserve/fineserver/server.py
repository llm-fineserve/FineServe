import os
import argparse
import asyncio
import time
import numpy as np
from typing import Dict, Set

import pandas as pd
import torch

from vllm.sampling_params import SamplingParams
from vllm import EngineArgs, LLMEngine, RequestOutput, CompletionOutput
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import TokensPrompt
from typing import List
from fineserve.utils.constant import (SM_HOLD_NAME_FMT, OUT_HOLD_NAME_FMT, ADD_REQ_NAME_FMT,
                                RET_REQ_NAME_FMT, PREEMPT_REQ_NAME_FMT)
from fineserve.utils.workload_utils import Request
from fineserve.utils.shm_utils import (create_shared_var, read_shared_var,
                                write_shared_var, dump_to_shared_var,
                                load_from_shared_var, load_reqs_from_shared_var,
                                dump_reqs_to_shared_var, write_list_to_shared_var)
from fineserve.logger import get_logger

logger = get_logger()
##
IS_STANDALONE = os.environ.get("STANDALONE", None)
DEBUG = os.environ.get("DEBUG", None)

## Below is a dataclass from utils/workload_utils.py
# @dataclasses.dataclass
# class Request:
#    """A single request."""
#    model_name: str
#    slo: Optional[float]
#    idx: int
#    time_stamp: Dict  # debug only
#    data: Any [0]: prompt token ids, [1]: output token ids [2]: max number of output
#    submit_time: float = None  # This will be filled later
#    prefill_end_time: float = None  # This will be filled later
#    decode_submit_time: float = None  # This will be filled later
#    end_time: float = None  # This will be filled later
#    is_prefill: bool = True
#    output: str = None
#    output_idx: int = 0
#    output_tokens: Optional[List[int]] = None

## for standalone unit testing
def add_test_prompt(model_name,
                    mps_percentage):
    cur_prompt_len = 16
    max_token = 16
    random_prompt = np.random.randint(0, 24000, size=cur_prompt_len).tolist()
    data_item = (random_prompt, cur_prompt_len, max_token)
    requests: List[Request] = []
    req = Request(model_name=model_name, slo=1,
                  idx=0, time_stamp={}, data=data_item)
    requests.append(req)
    shm_name = ADD_REQ_NAME_FMT.format(
        model_name, mps_percentage)
    num_iters = 1

    for req in requests:
        if req.submit_time is None:
            req.submit_time = time.time()

    dump_reqs_to_shared_var(shm_name, requests)
    name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
    shm_var = create_shared_var(name, create=False)
    write_shared_var(shm_var, num_iters)

## for standalone unit testing
def create_shared_vars_standalone(model_name, mps_percentage):
    name = SM_HOLD_NAME_FMT.format(model_name, mps_percentage)
    shm_var = create_shared_var(name,
                                size=6,
                                create=True)
    return shm_var


class FineServeEngine:

    def __init__(self, llm_runtime: LLMEngine,
                 model_name: str,
                 model_id: int,
                 rank: int,
                 mps_percentage: int,
                 timout_s: int,
                 engine_args
                 ):

        self.llm_runtime = llm_runtime
        self.engine_args = engine_args

        self.mps_percentage = mps_percentage
        self.model_name = model_name
        self.model_id = model_id
        self.rank = rank

        self.lock = asyncio.Lock()
        sm_hold_shm_name = SM_HOLD_NAME_FMT.format(self.model_name, self.mps_percentage)
        output_hold_shm_name = OUT_HOLD_NAME_FMT.format(self.model_name, self.mps_percentage)
        self.sm_hold_shm = create_shared_var(sm_hold_shm_name,
                                             create=False)
        self.out_hold_shm = create_shared_var(output_hold_shm_name,
                                              create=False)
        self.add_req_shm_name = ADD_REQ_NAME_FMT.format(
            self.model_name, self.mps_percentage)
        self.ret_req_shm_name = RET_REQ_NAME_FMT.format(
            self.model_name, self.mps_percentage)
        self.preempt_req_shm_name = PREEMPT_REQ_NAME_FMT.format(
            self.model_name, self.mps_percentage)

        self.requests_running: Set[int] = set()
        self.requests_enqueued: Set[int] = set()
        self.requests_max_tokens: Dict[int, int] = {}

        self.requests: Dict[int, Request] = {}

        self.enable_profiler = False
        self.start_batch = 100
        self.stop_batch = 150
        self.cur_batch = 0
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_stack=True,
            with_modules=True) if self.enable_profiler else None
        model = self.model_name.split("/")[-1]
        self.sched_dict = {}
        self.prof_out_name = f"log/profiler_fineserve/profiler_{model}_mps{self.mps_percentage}_id{self.model_id}.json"
        self.ttft_compt_per_token = []
        self.ttft_compt_per_token_step = []
        self.step = 0

        self.TIMEOUT_SECS=timout_s
        self.aborted_reqs = []

    def add_requests(self):
        while True:
            batch_reqs = load_reqs_from_shared_var(self.add_req_shm_name)
            if batch_reqs:
                break
        num_requests = len(batch_reqs)
        batch_request_ids = []
        for req in batch_reqs[:num_requests]:
            if isinstance(req, Request):
                self.requests[req.idx] = req
                batch_request_ids.append(req.idx)
            else:
                batch_request_ids.append(req)
        logger.info(f"step[{self.step}] | received batch_request_ids: {batch_request_ids}")
        for i in range(num_requests):
            req_id = batch_request_ids[i]
            req = self.requests[req_id]
            prompt_tokens = TokensPrompt(prompt_token_ids=np.array(req.data[0]).tolist())
            max_tokens = req.data[2]
            logger.info(f"step[{self.step}] | req_id: {req_id}, # of prompt_tokens: {len(req.data[0])}, max_tokens: {max_tokens}")
            self.llm_runtime.add_request(str(req_id),
                                         prompt_tokens,
                                         SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens))
            self.requests_max_tokens[req_id] = max_tokens
            self.requests_enqueued.add(req_id)
            self.requests[req_id].output_idx = 0
            self.sched_dict[req_id] = {
                "start": None,
                "prefill_end": None,
                "decode_end": None,
                "num_output_token": None,
                "stats": None,
            }
            self.sched_dict[req_id]["start"] = time.time()
        batch_output_tokens = batch_reqs[num_requests:]
        return batch_request_ids, batch_output_tokens

    def warmup_engine(self):
        warmup_prompts=10
        WARMUP_PARAMS=SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
        request_id = 0
        logger.info(f"Started warmup with {warmup_prompts} prompts")
        while warmup_prompts or self.llm_runtime.has_unfinished_requests():
            if warmup_prompts:
                warmup_prompts = warmup_prompts-1
                ## borrowed offline example from vllm
                li = np.random.randint(0, 24000, size=16).tolist()
                prompt = TokensPrompt(prompt_token_ids=li)
                self.llm_runtime.add_request(str(request_id), prompt, WARMUP_PARAMS)
                request_id += 1

            request_outputs: list[RequestOutput] = self.llm_runtime.step()
            for output in request_outputs:
                pass
        logger.info(f"Ended warmup ")

    def log_request(self, req_id):
        ts_arrival = self.sched_dict[req_id]["start"]
        ts_prefill_end = self.sched_dict[req_id]["prefill_end"]
        ts_decode_end = self.sched_dict[req_id]["decode_end"]
        num_output_tokens = self.sched_dict[req_id]["num_output_token"]
        ## get time in sec
        prefill_time_s = float(ts_prefill_end - ts_arrival)
        decode_time_s = float(ts_decode_end - ts_prefill_end)
        tpot_s = float(decode_time_s) / num_output_tokens
        logger.info(
            f"step[{self.step}] | "
            f"req[{req_id}] | "
            f"prefill_time: {prefill_time_s * 1000:.3f} ms, "
            f"decode_time: {decode_time_s * 1000:.3f} ms, "
            f"tpot: {tpot_s * 1000:.3f} ms"
        )

        stats = self.sched_dict[req_id]["stats"]
        if stats is None:
            logger.info(
                f"step[{self.step}] | "
                f"req[{req_id}] | "
                f"no stats"
            )
        else:
            input_len = self.requests[req_id].data[1]
            ttft = stats.first_token_ts - stats.queued_ts
            ttft_compt = stats.first_token_ts - stats.scheduled_ts
            ttft_compt_per_token = ttft_compt / input_len if input_len > 0 else np.nan
            decode_t = stats.last_token_ts - stats.first_token_ts
            topt = decode_t / stats.num_generation_tokens if stats.num_generation_tokens > 0 else np.nan
            logger.info(
                f"step[{self.step}] | "
                f"req[{req_id}] | "
                f"num_generation_tokens: {stats.num_generation_tokens}, "
                f"prefill_time: {ttft * 1000:.3f} ms, "
                f"ttft_compt: {ttft_compt * 1000:.3f} ms, "
                f"ttft_compt/token: {ttft_compt_per_token * 1000:.3f} ms, "
                f"decode_time: {decode_t * 1000:.3f} ms, "
                f"topt: {topt * 1000: .3f} ms"
            )
            if ttft_compt_per_token is not np.nan:
                self.ttft_compt_per_token.append((req_id, input_len, ttft_compt_per_token))

    async def process_outputs(self,
                              finished_req_id_dict: Dict[int,List[int]],
                              finished_prefill_req_id_list: List[int]) -> None:
        step = self.step
        ids = list(finished_req_id_dict.keys())
        num_of_finished_prefill = len(finished_prefill_req_id_list)
        logger.info(f"step[{step}] | finished requests: {ids}")
        if DEBUG is not None:
            if num_of_finished_prefill >0:
                logger.info(f"step[{step}] | requests that finished prefill phase: {finished_prefill_req_id_list}")

            for _id in finished_req_id_dict:
                logger.info(f"step[{step}] | tokens of id: {_id}")
                logger.info(f"step[{step}] | {finished_req_id_dict[_id]}")
        output_signal = read_shared_var(self.out_hold_shm)
        ## wait until server has processed previous output
        while output_signal==1:
            time.sleep(0.001)
            output_signal = read_shared_var(self.out_hold_shm)
        output_data_list=[]
        output_data_list.append(num_of_finished_prefill)
        for _id in  finished_prefill_req_id_list:
            output_data_list.append(_id)

        for _id in finished_req_id_dict:
            output_data_list.append(_id)
            output_data_list.append(len(finished_req_id_dict[_id]))
            output_data_list = output_data_list + list(finished_req_id_dict[_id])
        while True:
            try:
                write_list_to_shared_var(self.ret_req_shm_name, output_data_list)
                break
            except FileExistsError:
                time.sleep(1 / 5e4)
        output_signal = 1
        write_shared_var(self.out_hold_shm, output_signal)

    async def exec_batch_loop(self):
        ## do warmup (sync)
        self.warmup_engine()
        ## signal the scheduler that warmup is finished
        warmup_done = 2
        write_shared_var(self.sm_hold_shm, warmup_done)

        read_signal = 0
        output_task = None
        while True:
            ## check signal
            completed_requests = []
            while not read_signal:
                read_signal = read_shared_var(self.sm_hold_shm)
                if read_signal == 1:
                    break
                time.sleep(0.001)
            self.add_requests()
            read_signal=0
            write_shared_var(self.sm_hold_shm, read_signal)
            while self.llm_runtime.has_unfinished_requests() or self.aborted_reqs:
                ## check timeout and abort
                # logger.info(f"requests_enqueued : {self.requests_enqueued}")
                # logger.info(f"requests_running: {self.requests_running}")
                ids_to_check = set()
                ids_to_check.update(self.requests_enqueued)
                ids_to_check.update(self.requests_running)
                for req_id in ids_to_check:
                    if time.time() - self.sched_dict[req_id]["start"] > self.TIMEOUT_SECS:
                        if req_id not in self.aborted_reqs:
                            self.aborted_reqs.append(req_id)    # 여기서 한번만
                            # logger.info(f"ABORTED {req_id} after {self.TIMEOUT_SECS} timeout seconds")
                            self.llm_runtime.abort_request([str(req_id)])


                finished_requests = {} # key: request id , val: output tokens
                finished_prefill_requests:List[int] = [] # list if request ids
                prev = time.time()
                ## execute a step
                request_outputs = []
                if (not hasattr(self.llm_runtime.engine_core, "outputs_queue") or   # For InprocClient
                    not self.llm_runtime.engine_core.outputs_queue.empty()):
                    logger.info(f"step[{self.step}] | executing a step!")
                    request_outputs = self.llm_runtime.step()
                now = time.time()
                elapsed = now - prev

                ## process aborted requests due to timeout
                for req_id in self.aborted_reqs:
                    logger.info(f"step[{self.step}] | ABORTED req[{req_id}] after timeout {self.TIMEOUT_SECS} s")
                    req = self.requests[req_id]
                    comp_output = CompletionOutput(index=req_id,
                                                   text=None,
                                                   token_ids=[-1],
                                                   cumulative_logprob=None,
                                                   logprobs=None)
                    output = RequestOutput(request_id=str(req_id),
                                           prompt=None,
                                           prompt_token_ids=req.data[0],
                                           prompt_logprobs=None,
                                           outputs=[comp_output],
                                           finished=True)
                    request_outputs.append(output)
                    self.aborted_reqs.remove(req_id)

                ## checkout output
                for output in request_outputs:
                    req_id = int(output.request_id)
                    if self.sched_dict[req_id]["prefill_end"] is None:
                        if DEBUG is not None:
                            logger.info(f"step[{self.step}] | req[{req_id}] | prefill end: {now}")
                        self.sched_dict[req_id]["prefill_end"] = now
                        if not output.finished:
                            self.sched_dict[req_id]["stats"] = \
                                self.llm_runtime.output_processor.request_states[str(req_id)].stats
                        finished_prefill_requests.append(req_id)
                    if output.finished:
                        self.sched_dict[req_id]["decode_end"] = now
                        req = self.requests[req_id]
                        finished_requests[req.idx] = output.outputs[0].token_ids
                        self.sched_dict[req.idx]["num_output_token"] = len(output.outputs[0].token_ids)

                ## process output async (if present)
                if finished_requests or finished_prefill_requests:
                    # check for previous output  processing task
                    if output_task is not None:
                        if DEBUG is not None:
                            logger.info(f"[DEBUG] step[{self.step}] | waiting output to be sent!")
                        await output_task
                    finished_requests_ids = list(finished_requests.keys())

                    self.requests_enqueued.difference_update(finished_prefill_requests)
                    self.requests_running.update(finished_prefill_requests)
                    self.requests_running.difference_update(finished_requests_ids)

                    completed_requests = completed_requests + finished_requests_ids
                    output_task = asyncio.create_task(
                        self.process_outputs(finished_requests, finished_prefill_requests))

                # self._assertions(request_outputs) # for debugging

                ## check for more requests
                in_read_signal = read_shared_var(self.sm_hold_shm)
                if in_read_signal == 1:
                    self.add_requests()
                    in_read_signal=0
                    write_shared_var(self.sm_hold_shm, in_read_signal)

                if request_outputs:
                    self._record_stats(request_outputs, elapsed)
                    self.step += 1

            if output_task is not None:
                await output_task
                output_task = None
            for _id in completed_requests:
                self.log_request(_id)
            self._print_stats()

    def _assertions(self, request_outputs):
        logger.debug(f"step[{self.step}] | =========================")
        logger.debug(f"step[{self.step}] | running: {len(self.requests_running)}")
        logger.debug(f"step[{self.step}] | enqueued: {len(self.requests_enqueued)}")
        logger.debug(f"step[{self.step}] | =========================")
        logger.debug(f"step[{self.step}] | request_outputs: {len(request_outputs)}")
        logger.debug(f"step[{self.step}] | request_states: {len(self.llm_runtime.output_processor.request_states)}")
        logger.debug(f"step[{self.step}] | =========================")

        ro = {r.request_id: r for r in request_outputs}
        rs = self.llm_runtime.output_processor.request_states
        ro_rs = {i: r for i, r in ro.items() if i not in rs}
        rs_ro = {i: r for i, r in rs.items() if i not in ro}

        for r in request_outputs:
            if len(r.outputs) == 0:
                raise RuntimeError(f"req[{r.request_id}] has no output")
            if r.finished:
                if r.request_id in rs:
                    raise RuntimeError(f"req[{r.request_id}] is finished "
                                       f"but still found in request_states")
            else:
                if r.request_id not in rs:
                    raise RuntimeError(f"req[{r.request_id}] is not finished yet "
                                       f"but could not found in request_states")

        for _, r in rs.items():
            if r.is_prefilling:
                if r.stats.num_generation_tokens > 0:
                    raise RuntimeError(f"req[{r.request_id}] is prefilling but "
                                       f"num_generation_tokens[{r.stats.num_generation_tokens}] > 0")
                if r.request_id in ro:
                    raise RuntimeError(f"req[{r.request_id}] is prefilling "
                                       f"but found in request_outputs")
            if not r.is_prefilling:
                if r.stats.num_generation_tokens == 0:
                    raise RuntimeError(f"req[{r.request_id}] is not prefilling "
                                       f"but num_generation_tokens[{r.stats.num_generation_tokens}] == 0")
                if r.request_id not in ro:
                    pass    # may be preempted requests
                    # raise RuntimeError(f"req[{r.request_id}] is not prefilling "
                    #                    f"but could not found in request_outputs")
                if int(r.request_id) not in self.requests_running:
                    raise RuntimeError(f"req[{r.request_id}] is not prefilling "
                                       f"but has not been seen before")

    def _record_stats(self, request_outputs, elapsed):
        batched_tokens = []
        for output in request_outputs:
            if output.outputs[0].token_ids and output.outputs[0].token_ids[0] == -1:
                continue
            bt = 0
            if self.sched_dict[int(output.request_id)]["prefill_end"] is None:
                bt = len(output.prompt_token_ids)
            batched_tokens.append(bt)

        num_seqs = len(request_outputs)
        num_batched_tokens = sum(batched_tokens)
        if num_batched_tokens > 0:
            ttft_per_token = elapsed / num_batched_tokens
            logger.info(f"step[{self.step}] | num_seqs: {num_seqs}")
            logger.info(f"step[{self.step}] | num_batched_tokens: {num_batched_tokens}")
            logger.info(f"step[{self.step}] | ttft_per_token: {ttft_per_token}")
            self.ttft_compt_per_token_step.append([self.step, num_seqs, num_batched_tokens, elapsed, ttft_per_token])

    def _print_stats(self):
        df = pd.DataFrame(columns=["req_id", "input_len", "ttft_per_token"],
                          data=self.ttft_compt_per_token)
        logger.info(f"ttft_per_token stats: ")
        logger.info(df.describe())

        df_step = pd.DataFrame(columns=["step", "num_seqs", "num_batched_tokens", "elapsed", "ttft_per_token"],
                               data=self.ttft_compt_per_token_step)
        logger.info(f"ttft_per_token_step stats: ")
        logger.info(df_step[["num_seqs", "num_batched_tokens", "ttft_per_token"]].describe())

    def serve(self):
        logger.info(
            f"FineServe engine started (MPS: {self.mps_percentage})! "
            f"max_num_seqs: {self.engine_args.max_num_seqs} "
            f"max_batched_token: "
            f"{self.engine_args.max_num_batched_tokens}")
        asyncio.run(self.exec_batch_loop())


def main(args: argparse.Namespace):
    model_id = args.model_id
    model_name = args.model_name
    engine_args = EngineArgs.from_cli_args(args)

    if IS_STANDALONE is not None:
        create_shared_vars_standalone(model_name, args.mps_percentage)
    ## Additional arguments for QoQ quantized model
    if args.quantization == "qoq":
        from vllm.v1.engine.core_qoq import QoQConfig
        engine_args.additional_config = QoQConfig(
                            kv_quant_granularity="fine_grained",
                            precision="w4a8kv4",
                            group_size=128,
                            ifb_mode=True,
                            sparse_decode_mode=0,
                            chunk_prefill_size=102400,
                            )
    llm_runtime = LLMEngine.from_engine_args(engine_args)

    fineserve_engine = FineServeEngine(llm_runtime,
                                     model_name,
                                     model_id,
                                     args.local_rank,
                                     args.mps_percentage,
                                     args.timeout,
                                     engine_args)
    if IS_STANDALONE is not None:
        add_test_prompt(model_name, args.mps_percentage)
    # Start the engine loop.
    logger.info("Server started serving")
    fineserve_engine.serve()



if __name__ == '__main__':
    parser = FlexibleArgumentParser(description='FineServe Runtime worker')
    parser.add_argument("--model-id",
                        type=int,
                        default=0,
                        help="The index of served model.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument('--runtime-profile', action='store_true')
    parser.add_argument('--mps-percentage', default=100, type=int)
    parser.add_argument('--model-name', type=str, help='string name of model')
    parser.add_argument("--global-rank", type=int, help="the global rank in cluster")
    parser.add_argument("--timeout", type=int, default=30, help="setup timeout time for triggering abort")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    main(args)
