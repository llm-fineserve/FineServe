import torch
from typing import Dict, List, Union
from fineserve.config import FineServeJobConfig
from fineserve.utils.quant_format import get_dtype

def print_gpu_mem_usage():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        used_in_total = (used / total) * 100 if total > 0 else 0
        print(f"\t- Mem usage: {used_in_total:.4f}% ({used / 1024**3:.4f}GB / {total / 1024**3:.4f}GB)")

def get_slab_id(block_id: int, num_blocks: int):
    slab_id = block_id // num_blocks
    return slab_id

def gb_to_bytes(gb: int):
    return gb * (1024 ** 3)

def bytes_to_gb(bytes: int):
    return bytes / (1024 ** 3)

def cal_total_weight_size(job_configs: List[FineServeJobConfig]):
    total_weights_gb = 0
    for job_config in job_configs:
        total_weights_gb += job_config.model_size_org * (get_dtype(job_config.model_dtype).bit / 8) / job_config.tensor_parallel_size
        
    total_weights_size = gb_to_bytes(total_weights_gb)
    return total_weights_size