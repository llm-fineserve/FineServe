from typing import List, Dict, Any

SLAB_SIZE = 2 * 1024 * 1024
NUM_LAYERS = 64


class FineServeJobConfig:
    """Configuration for one job.

    Args:
        model: Name or path of the huggingface model to use.
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
    """

    def __init__(self,
                 name: str,
                 model: str,
                 qformat: str,
                 pipeline_parallel_size: int,
                 tensor_parallel_size: int,
                 placement: List[int],
                 mps_percentage: List[int],
                 gpu_memory_utilization: float,
                 kv_size_gb: float,
                 min_num_seqs: int,
                 max_num_seqs: int,
                 max_model_len: int,
                 model_dtype: str,
                 model_size_org: int,
                 num_hidden_layers: int,
                 rate: float,
                 slo: float,
                 avg_input_len: float,
                 avg_output_len: float):
        self.name = name
        self.model = model
        self.qformat = qformat
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.placement = placement
        self.mps_percentage = mps_percentage
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_size_gb = kv_size_gb
        self.min_num_seqs = min_num_seqs
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.model_dtype = model_dtype
        self.model_size_org = model_size_org
        self.num_hidden_layers = num_hidden_layers
        self.rate = rate
        self.slo = slo
        self.avg_input_len = avg_input_len
        self.avg_output_len = avg_output_len


class FineServeConfig:
    """Configuration for FineServe.

    Args:
        job_configs: List of JobConfig.
        num_gpus: Number of GPUs to use.
        block_size: Token block size.
        gpu_memory_utilization: The percentage of GPU memory to be used for the
            flexstore.
    """

    def __init__(self,
                 job_configs: List[FineServeJobConfig],
                 num_gpus: int,
                 ray_node_address: str,
                 base_ray_port: int,
                 num_ray_cluster: int,
                 mps_dir: str,
                 block_size: int,
                 overload_threshold: int,
                 gpu_memory_utilization: float,
                 restrict_gpu_memory_utilization: bool,
                 max_num_batched_tokens: int,
                 max_num_partial_prefills: int,
                 max_num_seqs: int,
                 manager_host: str,
                 manager_port: int,
                 server_port: int,
                 workload_config: Dict[str, Any],
                 model_config: Dict[Any, Any],
                 model_config_path: str,
                 cost_file_path: str,
                 schedule_approach: str,
                 nnodes: int,
                 nproc_per_node: int,
                 ngpu_per_node: int,
                 node_rank: int,
                 master_addr: str,
                 master_port: int):
        self.job_configs = job_configs
        self.num_gpus = num_gpus
        self.ray_node_address = ray_node_address
        self.base_ray_port = base_ray_port
        self.num_ray_cluster = num_ray_cluster
        self.mps_dir = mps_dir
        self.block_size = block_size
        self.overload_threshold = overload_threshold
        self.gpu_memory_utilization = gpu_memory_utilization
        self.restrict_gpu_memory_utilization = restrict_gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens      # 높이면 TTFT 단축, 낮추면 ITL 단축
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_num_seqs = max_num_seqs
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.server_port = server_port
        self.workload_config = workload_config
        self.model_config = model_config
        self.model_config_path = model_config_path
        self.cost_file_path = cost_file_path
        self.schedule_approach = schedule_approach
        self.nnodes = nnodes
        self.nproc_per_node = nproc_per_node
        self.ngpu_per_node = ngpu_per_node
        self.node_rank = node_rank
        self.master_addr = master_addr
        self.master_port = master_port

        self.head_size = 128

        self.num_runtime_processes = 0
        for job_config in self.job_configs:
            self.num_runtime_processes += len(job_config.mps_percentage)
