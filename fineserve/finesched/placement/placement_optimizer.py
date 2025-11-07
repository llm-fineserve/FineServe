import abc
import argparse
import itertools
import os
from typing import Dict, Optional, List
from dataclasses import dataclass

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
                 slo_scale: int,
                 rate_dict: Optional[Dict[str, float]] = None,
                 verbose: bool = False):
        super().__init__(workload_file, cost_file, rate_dict, verbose)
        avg_input_len = 903 * TOKEN_LEN_HEADROOM       # p95 seq len * 0.8
        avg_output_len = 631 * TOKEN_LEN_HEADROOM      # p95 seq len * 0.8
        self.slo_scale = slo_scale
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
            llm.slo = self._calculate_slo(llm)
            print(f"slo for {llm.name} calculated as {llm.slo}")

        for llm in self.models.values():
            self.cost_estimator.set_tp_candidates(llm,
                                                  prefill_mps=100,
                                                  verbose=verbose)

    def _calculate_slo(self, llm: LLM):
        p95_prefill_seq_len = 903  # P95 Value of Input Token Length for LMSYS-CHAT
        p95_ttft = self.cost_estimator.get_avg_latency(
            llm, llm.base_ngpu, 100, 1, p95_prefill_seq_len, True)
        slo = float((p95_ttft) * self.slo_scale + SLO_BUFFER_S)
        return slo

    def get_kvpr_argmin_node(self,
                             gpu_list: list[GPU],
                             llm,
                             w_req_rate,
                             shared_kv
                             ):
        """
        Find the best GPUs that minimize KVPR (KV cache Pressure Ratio) for the given model.
        
        Args:
            gpu_list: List of available GPUs
            llm: Language model to place
            w_req_rate: Weighted request rate per GPU
            shared_kv: Available KV cache memory per GPU
            
        Returns:
            List of (expected_kvpr, gpu_rank) tuples or None if placement is impossible
        """
        # Report impossible cases
        if len(gpu_list) < llm.tpt_config.tp_size:
            return None

        cand_list = []  # (expected_kvpr, gpu_rank)
        model_weight = llm.model_size / llm.tpt_config.tp_size
        
        for gpu in gpu_list:
            # Skip GPUs with insufficient memory
            if shared_kv[gpu.rank] <= model_weight:
                continue
                
            expected_kvpr = (w_req_rate[gpu.rank] + (llm.rate / llm.slo * llm.tpt_config.tp_size)) / (shared_kv[gpu.rank] - model_weight)
            cand_list.append((expected_kvpr, gpu.rank))

        # Check if we have enough GPUs
        if len(cand_list) < llm.tpt_config.tp_size:
            return None

        # Sort and extract the best GPUs
        cand_list.sort(key=lambda x: x[0])
        return cand_list[:llm.tpt_config.tp_size]

    def update_w_req_rates_and_shared_kv(self,
                                         gpu_rank_list: list[int],
                                         llm,
                                         w_req_rate,
                                         shared_kv):
        """
        Update the weighted request rates and shared KV cache for the selected GPUs.
        
        Args:
            gpu_rank_list: List of GPU ranks where the model will be placed
            llm: Language model being placed
            w_req_rate: Dictionary mapping GPU ranks to weighted request rates
            shared_kv: Dictionary mapping GPU ranks to available KV cache memory
        """
        for gpu_rank in gpu_rank_list:
            # Update weighted request rate
            w_req_rate[gpu_rank] = w_req_rate[gpu_rank] + (llm.rate / (llm.slo * llm.tpt_config.tp_size))
            # Update available KV cache memory
            shared_kv[gpu_rank] = shared_kv[gpu_rank] - (llm.model_size / llm.tpt_config.tp_size)

    def prism(self):
        """
        PRISM placement algorithm that minimizes KVPR (KV cache Pressure Ratio).
        
        Returns:
            List of placed models
        """
        placement = []
        # Sort by weighted request rate in descending order
        llms = sorted(self.models.values(),
                      key=lambda llm: (llm.rate / (llm.slo*llm.tpt_config.tp_size)),
                      reverse=True)
                      
        # Initialize values
        w_req_rate: Dict[int, float] = {}  # key: (gpu rank), value: rate/slo
        shared_kv: Dict[int, float] = {}   # key: (gpu rank), value: remaining gpu memory
        
        for i, node in enumerate(self.cluster.nodes):
            for gpu in node.gpus:
                key = gpu.rank
                w_req_rate[key] = 0.0
                shared_kv[key] = self.memory_per_gpu

        for llm in llms:
            # Search for appropriate GPU
            best_rs = float('inf')
            best_node_id = -1
            best_gpu_ranks = []
            
            for i, node in enumerate(self.cluster.nodes):
                # Find the best by nodes
                best_list = self.get_kvpr_argmin_node(node.gpus, llm, w_req_rate, shared_kv)
                if best_list is None:
                    continue
                    
                node_best_rs = sum([kvpr for (kvpr, gpu_idx) in best_list])
                if (best_node_id == -1) or best_rs > node_best_rs:
                    best_node_id = i
                    best_rs = node_best_rs
                    best_gpu_ranks = [gpu_idx for (kvpr, gpu_idx) in best_list]
                    
            # If unsuccessful, continue and let the scheduler know this is impossible
            if best_node_id == -1:
                continue

            # Place the model on the selected GPUs
            for gpu_rank in best_gpu_ranks:
                llm.placement.append(gpu_rank)
            placement.append(llm)
            
            # Update resource utilization
            self.update_w_req_rates_and_shared_kv(best_gpu_ranks, llm, w_req_rate, shared_kv)

        return placement
    
    def slab_aware(self):
        """
        Slab-aware placement algorithm that considers memory requirements and slab allocation.
        
        Returns:
            List of placed meshes
        """
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
        logger.info("# |  model   | qformat  |  bs  | tp  |    w    | required_mem | req_rate/slo |  Placement  |")
        
        for llm in llms:
            tp_size = llm.tp_size

            # Log model information
            model = llm 
            if isinstance(model, tuple):
                model = model[0]
                
            logger.info(f"# | {model.name:^8} | {model.qformat:^8} | {model.batch_size:^4} | {model.tp_size:^3} \\ | {model.model_size:^7.1f} | {model.required_memory:^12.2f} | {model.rate/model.slo:^12.2f} |")

            # Find best mesh for the model
            best_mesh = self._find_best_mesh(llm)

            if best_mesh:
                best_mesh.place_model(llm, 100)
            placement.append(best_mesh)

            # Log placement information
            logger.info(f" {','.join(map(str, llm.placement)):^11} |")
            
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

    def _calculate_mesh_score(self, mesh, llm):
        """
        Calculate mesh score based on GPU free memory and model requirements.
        
        Args:
            mesh: Mesh group to evaluate
            llm: Language model to be placed
            
        Returns:
            float: Mesh score
        """
        cur_mesh_score = 0
        for gpu in mesh.gpus:
            models = gpu.models.copy()
            models.append(llm)
            inv_block_size = 0
            
            for model in models:
                if isinstance(model, tuple):
                    model = model[0]
                    
                # Determine block scale based on quantization format
                if model.qformat == 'qoq':
                    block_scale = model.head_size + 2
                else:
                    block_scale = model.head_size + 1
                    
                block_size = 2 * block_scale * model.num_key_value_heads * model.kv_cache_bytes * NUM_TOKENS_PER_BLOCK / (1024 ** 3)
                inv_block_size += model.tp_size / (model.num_hidden_layers * block_size)

            # Calculate average block size and free blocks
            avg_block_size = len(models) / inv_block_size
            num_free_blocks_per_gpu = (gpu.free_memory - llm.required_memory / llm.tp_size) // avg_block_size
            cur_gpu_score = num_free_blocks_per_gpu / 1e5
            cur_mesh_score += cur_gpu_score
            
        return cur_mesh_score


    def setup_tp_dp(self, placement_opt: str, forced_dp=0, verbose=True) -> dict[str, Llama]:
        """
        Setup tensor parallelism (TP) and data parallelism (DP) configurations for models.
        
        Args:
            placement_opt: Placement optimization algorithm ("prism" or "slab-aware")
            forced_dp: Forced data parallelism degree (0 means use model's suggested value)
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary mapping new model names to Llama instances
        """
        # Reconstruct model dictionary
        new_llm_dict: dict[str, Llama] = {}
        
        # Find maximum number of GPUs among nodes
        max_gpu = 0
        for node in self.cluster.nodes:
            if len(node.gpus) > max_gpu:
                max_gpu = len(node.gpus)
                
        new_id = 0
        for llm in self.models.values():
            tpt_config = None
            
            # Skip models with no valid TP candidates
            if not llm.tp_candidates:
                logger.warning(f"SKIPPING {llm.name} since there was no valid tp option!")
                continue
                
            if placement_opt == "prism":
                tpt_config = llm.query_candidate(llm.base_ngpu)
                # Skip if solution does not exist
                if tpt_config.tp_size == 0:
                    continue
            else:
                # Filter candidates that can satisfy the rate and fit within GPU limits
                # Translated comment: Select only candidates that can achieve the target rate
                tp_cand_satisfied = sorted([c for c in llm.tp_candidates.values()
                                            if c.satisfied and c.tp_size <= max_gpu],
                                           key=lambda c: c.rate_min_batch_size)
                                           
                if verbose:
                    logger.info(f"Valid TP candidates for {llm.name}: {tp_cand_satisfied}")

                # Choose the setup which minimizes memory usage
                if tp_cand_satisfied:
                    tpt_config = tp_cand_satisfied[0]
                else:
                    logger.warning(f"Model {llm.name} does not have a valid candidate")
                    continue
                    
            if tpt_config is None:
                continue

            # Determine TP and DP sizes
            tp_size = tpt_config.tp_size
            dp_size = tpt_config.dp_size
            
            if forced_dp:
                dp_size = forced_dp
                
            if verbose:
                logger.info(f"{llm.name}: forming {dp_size} dp-instances (tp: {tp_size}, dp: {dp_size})")
                logger.info(f"tpt_config chosen for model {llm.name}: {tpt_config}")
                
            # Create DP instances
            for _ in range(dp_size):
                new_name = f"llm-{new_id}"
                new_id += 1
                
                new_llm = Llama(
                    name=new_name,
                    model=llm.model,
                    model_size_org=llm.model_size_org,
                    qformat=llm.qformat,
                    rate=llm.rate / dp_size,
                    avg_input_len=llm.avg_input_len,
                    avg_output_len=llm.avg_output_len
                )
                
                new_llm.tp_candidates = [tpt_config]
                new_llm.tpt_config = tpt_config
                new_llm.slo = llm.slo
                new_llm.slo_fixed = llm.slo_fixed
                new_llm_dict[new_name] = new_llm
                
        return new_llm_dict

    def optimize(self,
                 placement_opt: str = None,
                 dump_dir: str = None,
                 forced_dp: int = 0,
                 verbose: bool = True):
        """
        Optimize model placement using the specified algorithm.
        
        Args:
            placement_opt: Placement optimization algorithm ("prism" or "slab-aware")
            dump_dir: Directory to dump YAML configuration files
            forced_dp: (experimental) Force degree of data parallelism, pass 0 if you want it to be adjusted
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary with optimization results or None if no placement found
        """
        best_placement = None

        self.models = self.setup_tp_dp(placement_opt, forced_dp=forced_dp, verbose=verbose)
        
        if verbose:
            logger.info("##AFTER TP/DP SETUP##")
            for name in self.models:
                logger.info(f"name: {name} rate: {self.models[name].rate}")
            logger.info("-----------------")

        if placement_opt == "prism":
            best_placement = self.prism()
        elif placement_opt == "slab-aware":
            best_placement = self.slab_aware()

        if best_placement is None:
            if verbose:
                logger.info("Optimizer Done")
                logger.info("Find no placement")
            return None

        ret = {
            "total_tpt": None,  # Placeholder for future use
            "placement": []
        }

        for llm in self.models.values():
            tp_size = len(llm.placement)
            candidate = llm.query_candidate(tp_size)
            expected_tpt = candidate.throughput if candidate else 0
            
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
        self.dump_to_yaml(dump_dir)

        return ret



    def dump_to_yaml(self,
                     dump_dir=None):
        """
        Dump the placement configuration to a YAML file.
        
        Args:
            placement: The placement result from the optimization algorithm
            dump_dir: Directory to dump the YAML file (uses default if None)
        """
        logger.info("  Placement: ")
        
        for llm in self.models.values():
            tpt_config = llm.tpt_config
            
            # Calculate SLO if it wasn't fixed
            if not llm.slo_fixed:
                llm.slo = self._calculate_slo(llm)
                
            max_batch_size = int(tpt_config.slo_max_batch_size)
            min_batch_size = int(tpt_config.rate_min_batch_size)
            expected_tpt = tpt_config.throughput
            
            logger.info(f"    LLM: {llm.name}, "
                        f"DP-Degree: {llm.tpt_config.dp_size}, "
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

        # Generate filename and dump to file
        fname_prefix = os.path.splitext(os.path.basename(self.workload_file))[0]
        dump_dir = os.path.dirname(self.workload_file) if dump_dir is None else dump_dir
        logger.info(f"dump_dir: {dump_dir}")
        fname = f"{fname_prefix}_cfg.yaml"
        self.yaml_builder.dump_to_file(os.path.join(dump_dir, fname))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FineServe Placement Optimizer")
    parser.add_argument("--workload-file", type=str, default="examples/basic/models.yaml",
                        help="Path to the workload configuration file")
    parser.add_argument("--cost-file", type=str, default="examples/placement/cost.csv",
                        help="Path to the cost data file")
    parser.add_argument("--profile-log-dir", type=str, default=None,
                        help="Directory containing profile logs for building cost file")
    parser.add_argument("--placement-option", choices=PLACEMENT_OPTS, type=str,
                        help="Placement optimization algorithm to use")
    parser.add_argument("--force-dp", type=int, default=0, required=False,
                        help="The amount of data parallelism(=replica) to be forced")
    parser.add_argument("--slo-scale", type=int, default=SLO_SCALE, required=False,
                        help="The scale of SLO that will be multiplied")
    parser.add_argument("--dump-dir", required=False, type=str, default=None,
                        help="Directory to dump placement configuration files")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help="Enable verbose logging")
    args = parser.parse_args()
    return args


def main():
    """Main entry point for the placement optimizer."""
    args = parse_args()
    
    # Build cost file if profile logs are provided
    if args.profile_log_dir is not None:
        logger.info("Building cost file from profile logs...")
        build_cost_file(args.cost_file, args.profile_log_dir)
    
    # Set up dump directory
    dump_dir = args.dump_dir
    if dump_dir is None:
        dump_dir = os.path.join(os.path.dirname(args.workload_file), "placement_yamls")
    
    # Create dump directory if it doesn't exist
    os.makedirs(dump_dir, exist_ok=True)
    
    # Run the placement optimizer
    opt = FineServePlacementOptimizer(args.workload_file,
                                      args.cost_file,
                                      args.slo_scale
    )
    opt.optimize(
        dump_dir=dump_dir,
        placement_opt=args.placement_option,
        forced_dp=args.force_dp,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()