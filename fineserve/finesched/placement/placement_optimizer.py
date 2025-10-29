import abc
import argparse
import itertools
import os
from typing import Dict, Optional, List
from dataclasses import dataclass

import numpy as np
import yaml

from fineserve.finesched.placement.cluster import Cluster, FineServeCluster
from fineserve.finesched.placement.constants import *
from fineserve.finesched.placement.estimator import Estimator, FineServeEstimator
from fineserve.finesched.placement.mesh_group import MeshGroup, GPU
from fineserve.finesched.placement.model import LLM, Llama
from fineserve.finesched.placement.profile import build_cost_file
from fineserve.finesched.placement.yaml_builder import YamlBuilder

from fineserve.logger import get_logger

logger = get_logger()


@dataclass
class ModelInfo:
    name: str
    tp_size: int
    dp_size: int
    tp_rank: int
    dp_rank: int
    weight_size: float
    req_rate: float
    slo: float
    gpu_rank: int
    node_rank: int

PLACEMENT_OPTS = ["prism", "slab-aware"]
FORCED_DP=0

class PlacementOptimizer(abc.ABC):

    def __init__(self,
                 workload_file: str,
                 cost_file: str,
                 rate_dict: Optional[Dict[str, float]] = None,
                 verbose: bool = False):
        self.workload_file = workload_file
        self.cost_file = cost_file

        with open(self.workload_file, "r") as f:
            self.model_group = yaml.safe_load(f)
        self.nnodes = self.model_group["cluster"]["nnodes"]
        self.ngpus_per_node = self.model_group["cluster"]["ngpus_per_node"]
        self.memory_per_gpu = self.model_group["cluster"].get("memory_per_gpu", MEMORY_PER_GPU)
        self.overload_threshold = self.model_group["cluster"].get("overload_threshold", 100)
        self.ngpus = self.nnodes * self.ngpus_per_node
        self.total_memory = self.ngpus * self.memory_per_gpu
        self.models: Dict[str, LLM] = {}
        self.cluster: Cluster = None
        self.cost_estimator: Estimator = None
        self.yaml_builder = YamlBuilder(self.nnodes,
                                        self.ngpus_per_node,
                                        self.memory_per_gpu,
                                        self.overload_threshold)

    @abc.abstractmethod
    def optimize(self,
                 dump_dir: str = None,
                 dump_to_yaml: bool = True,
                 verbose: bool = True):
        ...


class FineServePlacementOptimizer(PlacementOptimizer):

    def __init__(self,
                 workload_file: str,
                 cost_file: str,
                 rate_dict: Optional[Dict[str, float]] = None,
                 verbose: bool = False):
        super().__init__(workload_file, cost_file, rate_dict, verbose)
        avg_input_len = 903 * 0.8       # p95 seq len * 0.8
        avg_output_len = 631 * 0.8      # p95 seq len * 0.8
        for model_cfg in self.model_group["models"]:
            rate = rate_dict[model_cfg["name"]] if rate_dict else model_cfg["rate"]
            llm = Llama(model_cfg["name"],
                        model_cfg["model"],
                        model_cfg["model_size_org"],
                        model_cfg["qformat"],
                        rate,
                        avg_input_len,
                        avg_output_len)
            ## check if SLO was specified in the model config file, if so then set it the predefined value
            if 'slo' in model_cfg:
                llm.slo = model_cfg['slo']
                llm.slo_fixed = True
            self.models[llm.name] = llm


        self.cluster = FineServeCluster(self.nnodes,
                                        self.ngpus_per_node,
                                        self.memory_per_gpu,
                                        self.overload_threshold,
                                        100)
        self.cost_estimator = FineServeEstimator(cost_file,
                                                 self.memory_per_gpu)

        # set slo
        for llm in self.models.values():
            ## check if llm.slo was set up earlier
            if llm.slo is not None:
                continue

            p95_prefill_seq_len = 903
            p95_ttft = self.cost_estimator.get_avg_latency(llm,
                                                           llm.base_ngpu,
                                                           100,
                                                           1,
                                                           p95_prefill_seq_len,
                                                           True)
            llm.slo = float((p95_ttft ) * SLO_SCALE - SLO_BUFFER_S)
            print(f"slo for {llm.name} calculated as {llm.slo}")

        for llm in self.models.values():
            self.cost_estimator.set_tp_candidates(llm,
                                                  prefill_mps=100,
                                                  verbose=verbose)

    def get_kvpr_argmin_node(self,
                             gpu_list: list[GPU],
                             llm,
                             w_req_rate,
                             shared_kv
                             ):

        ## report impossible cases
        if len(gpu_list) < llm.tpt_config.tp_size:
            return None

        cand_list = []  # (expected_kvpr, gpu_rank)
        model_weight = llm.model_size / llm.tpt_config.tp_size
        for gpu in gpu_list:
            print()
            if shared_kv[gpu.rank] <= model_weight:
                continue
            expected_kvpr = (w_req_rate[gpu.rank] + (llm.rate / llm.slo * (llm.tpt_config.tp_size))) / (shared_kv[gpu.rank] - model_weight)
            cand_list.append((expected_kvpr, gpu.rank))

        if len(cand_list) < llm.tpt_config.tp_size:
            return None

        ## sort and extract
        cand_list.sort(key=lambda x: x[0])
        return cand_list[:llm.tpt_config.tp_size]

    def update_w_req_rates_and_shared_kv(self,
                                         gpu_rank_list: list[int],
                                         llm,
                                         w_req_rate,
                                         shared_kv):
        # left the following code in case someone wants to debug
        #print("llm's spec:")
        #print(f"llm.slo {llm.slo}")
        #print(f"tp_size: {llm.tpt_config.tp_size}")
        #print(f"llm.model_size: {llm.model_size}")
        #print(f"llm.rate: {llm.rate}")
        #print(f"calculated weight size: {(llm.model_size / llm.tpt_config.tp_size)}")
        for gpu_rank in gpu_rank_list:
            w_req_rate[gpu_rank] = w_req_rate[gpu_rank] + (llm.rate / (llm.slo * llm.tpt_config.tp_size))
            shared_kv[gpu_rank] = shared_kv[gpu_rank] - (llm.model_size / llm.tpt_config.tp_size)

    def prism(self):
        placement = []
        ## sort by weighted request rate
        llms = sorted(self.models.values(),
                      key=lambda llm: (llm.rate / (llm.slo*llm.tpt_config.tp_size)),
                      reverse=True)
        ## initiate values
        w_req_rate: Dict[int, float] = {}  # key: (gpu rank), value: rate/slo
        shared_kv: Dict[int, float] = {}  # key: (gpu rank), value: remaining gpu memory
        for i, node in enumerate(self.cluster.nodes):
            for gpu in node.gpus:
                key = gpu.rank
                w_req_rate[key] = 0.0
                shared_kv[key] = self.memory_per_gpu

        for llm in llms:
            ## search for appropriate GPU
            # /* find the best gpu best_idx that minimizes KVPR */
            # but first find best
            best_rs = float('inf')
            best_node_id = -1
            best_gpu_ranks = []
            for i, node in enumerate(self.cluster.nodes):
                ## fine the best by nodes
                best_list = self.get_kvpr_argmin_node(node.gpus, llm, w_req_rate, shared_kv)
                if best_list is None:
                    continue
                node_best_rs = sum([kvpr for (kvpr, gpu_idx) in best_list])
                if (best_node_id == -1) or best_rs > node_best_rs:
                    best_node_id = i
                    best_rs = node_best_rs
                    best_gpu_ranks = [gpu_idx for (kvpr, gpu_idx) in best_list]
            ## if unsuccessful, just continue and let the scheduler know this is impossible
            if best_node_id == -1:
                continue

            for gpu_rank in best_gpu_ranks:
                llm.placement.append(gpu_rank)
            placement.append(llm)
            self.update_w_req_rates_and_shared_kv(best_gpu_ranks, llm, w_req_rate, shared_kv)
            #print("## Updated States ##")
            #print(f"w_req_rate: {w_req_rate}")
            #print(f"shared_kv: {shared_kv}")

        return placement
    
    def slab_aware(self):
        # Estimate model memory requirements
        for llm in self.models.values():
            llm.tp_size = llm.tpt_config.tp_size
            llm.seq_len = llm.avg_input_len + llm.avg_output_len
            llm.batch_size = llm.tpt_config.rate_min_batch_size

            llm.kv_size = llm.kv_cache_size_per_batch(llm.batch_size, llm.seq_len)
            llm.required_memory = 1.3 * llm.model_size + llm.kv_size

        # Sort models by required memory in descending order
        llms = sorted(self.models.values(), key=lambda llm: llm.required_memory, reverse=True)

        # Model placement
        placement = []
        print("# |  model   | qformat  |  bs  | tp  |    w    | required_mem | req_rate/slo |  Placement  |")
        for llm in llms:
            tp_size = llm.tp_size

            # Print model information
            model = llm 
            if isinstance(model, tuple):
                model = model[0]
            print(f"# | {model.name:^8} | {model.qformat:^8} | {model.batch_size:^4} | {model.tp_size:^3} \
| {model.model_size:^7.1f} | {model.required_memory:^12.2f} | {model.rate/model.slo:^12.2f} |", end="")

            # Find best mesh for the model
            best_mesh = self._find_best_mesh(llm)

            if best_mesh:
                best_mesh.place_model(llm, 100)
            placement.append(best_mesh)

            # Print placement information
            print(f" {','.join(map(str, llm.placement)):^11} |")
        return placement

    def _find_best_mesh(self, llm):
        best_mesh_score = 0
        best_mesh = None
        for node in self.cluster.nodes:
            gpu_cand = [g for g in node.gpus if g.can_place(llm, 100, llm.tp_size)]
            if len(gpu_cand) < llm.tp_size:
                continue

            # Mesh loop
            for gpus in itertools.combinations(gpu_cand, llm.tp_size):
                mesh = MeshGroup(gpus, self.cluster.gpu_memory_utilization)

                # Calculate mesh score
                cur_mesh_score = self._calculate_mesh_score(mesh, llm)
                if cur_mesh_score > best_mesh_score:
                    best_mesh_score = cur_mesh_score
                    best_mesh = mesh
        return best_mesh

    '''
    Calculate mesh score based on GPU free memory and model requirements
    '''
    def _calculate_mesh_score(self, mesh, llm):
        #
        cur_mesh_score = 0
        for gpu in mesh.gpus:
            models = gpu.models.copy()
            models.append(llm)
            inv_block_size = 0
            for model in models:
                if isinstance(model, tuple):
                    model = model[0]
                if model.qformat == 'qoq':
                    block_scale = model.head_size + 2
                else:
                    block_scale = model.head_size + 1
                block_size = 2 * block_scale * model.num_key_value_heads * model.kv_cache_bytes * NUM_TOKENS_PER_BLOCK / (1024 ** 3)
                inv_block_size += model.tp_size / (model.num_hidden_layers * block_size)

            avg_block_size = len(models) / inv_block_size
            num_free_blocks_per_gpu = (gpu.free_memory - llm.required_memory / llm.tp_size) // avg_block_size
            cur_gpu_score = num_free_blocks_per_gpu / 1e5
            cur_mesh_score += cur_gpu_score
        return cur_mesh_score


    def setup_tp_dp(self, placement_opt: str, forced_dp=0, verbose=True) -> dict[str, Llama]:
        ## reconstruct
        new_llm_dict: dict[str, Llama] = {}
        ## no need to use tp > max_gpu among nodes
        MAX_GPU=0
        new_id = 0
        for node in self.cluster.nodes:
            if len(node.gpus) > MAX_GPU:
                MAX_GPU = len(node.gpus)
        for llm in self.models.values():
            tpt_config = None
            if not llm.tp_candidates:
                print(f"SKIPPING {llm.name} since there was no valid tp option!")
                continue
            if placement_opt == "prism":
                tpt_config = llm.query_candidate(llm.base_ngpu)
                # solution does not exist
                if tpt_config.tp_size == 0:
                    continue
            else:
                # 일단 candidate 중에 rate 달성 가능한 것들만
                tp_cand_satisfied = sorted([c for c in llm.tp_candidates.values()
                                            if c.satisfied and c.tp_size <= MAX_GPU],
                                           key=lambda c: c.rate_min_batch_size)
                if verbose:
                    print(tp_cand_satisfied)

                # chose the setup which minimizes memory usage
                if tp_cand_satisfied:
                    tpt_config = tp_cand_satisfied[0]
                else:  ## if llm does not have a valid candidate
                    print(f"Model {llm.name} does not have a valid candidate")
                    continue
            if tpt_config is None:
                continue

            tp_size = tpt_config.tp_size
            dp_size = tpt_config.dp_size
            if forced_dp:
                dp_size = forced_dp
            if verbose:
                print(f" {llm.name}: forming {dp_size} dp-instances (tp: {tp_size} , dp: {dp_size})")
                print(f"tpt_config chosen for model {llm.name} \n"
                      f"{tpt_config}")
            for j in range(dp_size):
                new_name = "llm-" + str(new_id)
                new_id +=1
                new_llm = Llama(name=new_name,
                                model=llm.model,
                                model_size_org=llm.model_size_org,
                                qformat=llm.qformat,
                                rate=llm.rate / dp_size,
                                avg_input_len=llm.avg_input_len,
                                avg_output_len=llm.avg_output_len)
                new_llm.tp_candidates = [tpt_config]
                new_llm.tpt_config = tpt_config
                new_llm.slo = llm.slo
                new_llm.slo_fixed = llm.slo_fixed
                new_llm_dict[new_name] = new_llm
        return new_llm_dict

    def optimize(self,
                 placement_opt: str = None,
                 dump_dir: str = None,
                 verbose: bool = True):
        best_placement = None
        forced_dp=0
        if FORCED_DP:
            forced_dp=FORCED_DP
        self.models = self.setup_tp_dp(placement_opt,forced_dp=forced_dp,verbose=verbose)
        if verbose:
            print("##AFTER TP/DP SETUP##")
            for name in self.models:
                print(f"name: {name} "
                      f"rate: {self.models[name].rate} ")
            print("-----------------")

        if placement_opt == "prism":
            best_placement = self.prism()
        elif placement_opt == "slab-aware":
            best_placement = self.slab_aware()

        best_tpt = None
        if best_placement is None:
            if verbose:
                print(f"Optimizer Done")
                print(f"Find no placement")
            return None

        ret = {
            "total_tpt": best_tpt,
            "placement": []
        }

        for llm in self.models.values():
            tp_size = len(llm.placement)
            candidate = llm.query_candidate(tp_size)
            expected_tpt = candidate.throughput
            ret["placement"].append({
                "name": llm.name,
                "model": llm.model,
                "model_type": llm.model_type,
                "qformat": llm.qformat,
                "placement": llm.placement,
                "expected_tpt": expected_tpt,
                "rate": llm.rate,
            })


        logger.info(f"============= Optimizer Done: {placement_opt} =============")
        self.dump_to_yaml(best_placement, dump_dir)

        return ret

    def dump_to_yaml(self,
                     placement,
                     dump_dir=None,
                     ):

        print(f"  Placement: ")
        for llm in self.models.values():
            tpt_config = llm.tpt_config
            ## restore slo
            if not llm.slo_fixed:
                p95_prefill_seq_len = 903 ## P95 Value of Input Token Length for LMSYS-CHAT
                p95_ttft = self.cost_estimator.get_avg_latency(llm,
                                                           llm.base_ngpu,
                                                           100,
                                                           1,
                                                           p95_prefill_seq_len,
                                                           True)
                llm.slo = float((p95_ttft) * SLO_SCALE + SLO_BUFFER_S)
            max_batch_size = int(tpt_config.slo_max_batch_size)
            min_batch_size = int(tpt_config.rate_min_batch_size)
            expected_tpt = tpt_config.throughput
            logger.info(f"    LLM: {llm.name}, "
                  f" DP-Degree: {llm.tpt_config.dp_size}, "
                  f"Model: {llm.model_type}, "
                  f"Quant-Format: {llm.qformat}, "
                  f"MPS: {tpt_config.mps_percentage}, "
                  f"rate: {llm.rate}, "
                  f"slo: {llm.slo}, "
                  f"min rate batch_size: {min_batch_size}, "
                  f"max slo batch_size: {max_batch_size}, "
                  f"expected_tpt: {expected_tpt:.3f}, "
                  f"placement: {sorted(llm.placement)}")

            self.yaml_builder.add_model(
                llm=llm,
                tensor_parallel_size=tpt_config.tp_size,
                placement=[rank for rank in llm.placement],
                mps_percentage=[100],
                max_num_seqs=max_batch_size,
                min_num_seqs=min_batch_size
            )

        # print(yaml_builder.to_yaml())
        fname_prefix = os.path.splitext(os.path.basename(self.workload_file))[0]
        dump_dir = os.path.dirname(self.workload_file) if dump_dir is None else dump_dir
        logger.info(f"dump_dir: {dump_dir}")
        fname = f"{fname_prefix}_cfg.yaml"
        self.yaml_builder.dump_to_file(os.path.join(dump_dir, fname))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-file", type=str, default="examples/basic/models.yaml")
    parser.add_argument("--cost-file", type=str, default="examples/placement/cost.csv")
    # for build cost file
    parser.add_argument("--profile-log-dir", type=str, default=None)
    parser.add_argument("--placement-option", choices=PLACEMENT_OPTS, type=str)
    parser.add_argument("--dump-dir", required=False, type=str, default=None)
    parser.add_argument('--verbose', required=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.profile_log_dir is not None:
        logger.info("build cost file called!")
        build_cost_file(args.cost_file, args.profile_log_dir)
    dump_dir = args.dump_dir
    if dump_dir is None:
        dump_dir = os.path.dirname(args.workload_file) + "/placement_yamls"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    opt = FineServePlacementOptimizer(args.workload_file, args.cost_file)
    opt.optimize(dump_dir=dump_dir,
                placement_opt=args.placement_option,
                verbose=args.verbose)
