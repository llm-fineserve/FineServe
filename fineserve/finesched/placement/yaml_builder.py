import sys

import ruamel.yaml

from fineserve.finesched.placement.constants import MEMORY_PER_GPU
from fineserve.finesched.placement.model import LLM


def flist(x):
    """Create a flow-style YAML list."""
    retval = ruamel.yaml.comments.CommentedSeq(x)
    retval.fa.set_flow_style()
    return retval


yaml = ruamel.yaml.YAML()


class YamlBuilder:
    """Builder for YAML configuration files."""

    def __init__(self,
                 nnodes: int,
                 ngpus_per_node: int,
                 memory_per_gpu: int = MEMORY_PER_GPU,
                 overload_threshold: int = 100):
        """
        Initialize YAML builder.
        
        Args:
            nnodes: Number of nodes
            ngpus_per_node: Number of GPUs per node
            memory_per_gpu: Memory per GPU in GB
            overload_threshold: Overload threshold
        """
        self.nnodes = nnodes
        self.ngpus_per_node = ngpus_per_node
        self.memory_per_gpu = memory_per_gpu
        self.ngpus = self.nnodes * self.ngpus_per_node
        self.total_memory = self.ngpus * self.memory_per_gpu
        self.overload_threshold = overload_threshold
        
        self.data = {
            "cluster": {
                "nnodes": self.nnodes,
                "ngpus_per_node": self.ngpus_per_node,
                "memory_per_gpu": self.memory_per_gpu,
                "num_gpus": self.ngpus,
                "total_memory": self.total_memory,
                "max_num_seqs": 256,    # default
                "overload_threshold": self.overload_threshold,
                "gpu_memory_utilization": None,
            },
            "models": [],
        }
        self.placement_memories = []

    def add_model(self,
                  llm: LLM,
                  tensor_parallel_size: int,
                  placement: list,
                  mps_percentage: list,
                  max_num_seqs: int,
                  min_num_seqs: int):
        """
        Add a model to the configuration.
        
        Args:
            llm: Language model to add
            tensor_parallel_size: Tensor parallelism degree
            placement: List of GPU ranks for placement
            mps_percentage: List of MPS percentages
            max_num_seqs: Maximum number of sequences
            min_num_seqs: Minimum number of sequences
        """
        # Calculate memory requirements
        bs = llm.tpt_config.rate_min_batch_size
        kv_cache = llm.kv_cache_size_per_batch(bs)
        memory = llm.placement_memory + kv_cache
        total_memory = len(placement) * self.memory_per_gpu
        
        # Create model data dictionary
        model_data = {
            "name": str(llm.name),
            "model": str(llm.model),
            "qformat": str(llm.qformat),
            "tensor_parallel_size": int(tensor_parallel_size),
            "pipeline_parallel_size": int(1),
            "placement": flist([int(rank) for rank in placement]),
            "mps_percentage": flist([int(mps) for mps in mps_percentage]),
            "gpu_memory_utilization": float(memory / total_memory) if total_memory > 0 else None,
            "kv_size_gb": float(kv_cache),
            "max_num_seqs": int(max_num_seqs),
            "min_num_seqs": int(min_num_seqs),
            "model_dtype": str(llm.quant.weights.name),
            "model_size_org": float(llm.model_size_org),
            "num_hidden_layers": int(llm.num_hidden_layers),
            "rate": float(llm.rate),
            "slo": float(llm.slo),
            "avg_input_len": float(llm.avg_input_len),
            "avg_output_len": float(llm.avg_output_len),
        }
        
        self.data["models"].append(model_data)
        self.placement_memories.append(memory)
        self.update_utilization()

    def update_utilization(self):
        """Update the cluster's GPU memory utilization."""
        if self.total_memory > 0:
            self.data["cluster"]["gpu_memory_utilization"] = sum(self.placement_memories) / self.total_memory
        else:
            self.data["cluster"]["gpu_memory_utilization"] = 0.0

    def build(self):
        """
        Build the configuration data.
        
        Returns:
            Configuration data dictionary
        """
        return self.data

    def to_yaml(self, f=sys.stdout):
        """
        Convert the configuration to YAML format.
        
        Args:
            f: File-like object to write to (default: stdout)
            
        Returns:
            YAML representation
        """
        return yaml.dump(self.data, f)

    def dump_to_file(self, file_path: str):
        """
        Dump the configuration to a YAML file.
        
        Args:
            file_path: Path to the output file
        """
        if not self.data["models"]:
            return
            
        self.data["workloads"] = {}
        with open(file_path, 'w') as file:
            self.to_yaml(file)

        print(f"YAML saved to {file_path}")