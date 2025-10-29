import math
import torch
import threading
from typing import Dict, List, Union
from fineserve.config import FineServeJobConfig
from fineserve.kvslab.kv_slab_table import KVSlabTable
from fineserve.kvslab.utils import print_gpu_mem_usage, get_slab_id, cal_total_weight_size, bytes_to_gb
from fineserve.logger import get_logger
from fineserve.utils.quant_format import get_dtype

logger = get_logger()

class KVSlabManager:
    def __init__(self, num_placed_models, max_num_batched_tokens):
        # shared kv tensor
        self.shared_kv: torch.Tensor
        
        # slab info
        self.total_num_slabs: int
        self.slab_size: int

        self.kv_slab_tables: Dict[int, KVSlabTable] = {} # key: num_layers

        # engine info
        self.num_placed_models = num_placed_models
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_conn_engines = 0
        self.engine_info: Dict[int, Dict[str, Union[str, int, List[int]]]] = {}
        self.lock = threading.Lock()
    
    def init(self, device, job_configs: List[FineServeJobConfig]):
        total_weight_size = cal_total_weight_size(job_configs)
        cur_available, total = torch.cuda.mem_get_info(device=device)
        available = int(total - total_weight_size * 1.8)
        
        logger.info(f"Available w/o activation buffer: {bytes_to_gb(cur_available):.2f}GB, "
                    f"Cal. total_weights_gb: {bytes_to_gb(total_weight_size)}GB, "
                    f"Cal. available KV size: {bytes_to_gb(available):.2f}GB")
        
        self.slab_size = self._compute_slab_size() * 2 # TODO: parameterize this
        self.total_num_slabs = available // 2 // self.slab_size

        shared_kv_shape = (2 * self.total_num_slabs * self.slab_size)
        self.shared_kv = torch.empty(shared_kv_shape, device=device, dtype=torch.int8)
        self.shared_kv.share_memory_()

        logger.info(f"Shared KV tensor is created: ({self.total_num_slabs} slabs x {self.slab_size/1024:.2f}KB)")
        print_gpu_mem_usage()

        # kv slab table init
        self._init_kv_slab_table()

        # slab pool assignment
        weighted_demands, total_weighted_demand = self._calculate_engine_slab_demand()
        num_assigned_slabs = self._assign_slab_pool(weighted_demands, total_weighted_demand)
        self.num_assigned_slabs = num_assigned_slabs
        logger.info(f"Assigned slabs: {self.num_assigned_slabs} / {self.total_num_slabs}")

    def _init_kv_slab_table(self):
        for engine_info in self.engine_info.values():
            num_layers = engine_info['num_layers']
            if num_layers not in self.kv_slab_tables:
                self.kv_slab_tables[num_layers] = KVSlabTable()
            
            self.kv_slab_tables[num_layers].register_format(
                engine_info['tokens_per_block'],
                engine_info['token_size'],
                engine_info['scale_size']
            )
    
    def _compute_slab_size(self):
        slab_size = self.engine_info[0]['block_size']
        
        for engine_info in self.engine_info.values():
            block_size = engine_info['block_size']
            slab_size = math.lcm(slab_size, block_size)
        
        return slab_size
    
    def _calculate_engine_slab_demand(self):
        weighted_demands = {}
        total_weighted_demand = 0.0
        
        for num_layers in self.kv_slab_tables.keys():
            rate_for_group = 0
            engine_infos = self._get_engine_info_by_num_layers(num_layers)
            for engine_info in engine_infos:
                rate_for_group += engine_info['rate']
            
            demand = float(rate_for_group * num_layers)
            weighted_demands[num_layers] = demand
            total_weighted_demand += demand

        return weighted_demands, total_weighted_demand

    def _assign_slab_pool(self, weighted_demands: Dict[int, int], total_weighted_demand: float):
        total_slabs = self.total_num_slabs
        shared_kv_slab_offset = 0
        num_assigned_slabs = 0 
        min_slab_pool_sizes = {}
        
        sorted_tables = sorted(self.kv_slab_tables.items())

        for num_layers, kv_slab_table in sorted_tables:
            minimum_num_blocks = 0
            max_num_blocks_per_slab = 0
            minimum_num_slabs = 0

            engine_infos = self._get_engine_info_by_num_layers(num_layers)
            for engine_info in engine_infos:
                minimum_num_blocks = self.max_num_batched_tokens // engine_info['tokens_per_block']
                max_num_blocks_per_slab = self.slab_size // engine_info['block_size']
                minimum_num_slabs += (math.ceil(minimum_num_blocks / max_num_blocks_per_slab)) * num_layers
            
            min_slab_pool_sizes[num_layers] = minimum_num_slabs
            total_slabs -= minimum_num_slabs

            assert total_slabs >= 0, "Available KV cache size cannot satisfy minimum required num blocks processing 1 sequence of max_num_batched_tokens (4096)"
        
        for num_layers, kv_slab_table in sorted_tables:
            demand = weighted_demands[num_layers]

            share_ratio = demand / total_weighted_demand if total_weighted_demand > 0 else (1.0 / len(self.kv_slab_tables))
            slab_pool_size = int(total_slabs * share_ratio)
            slab_pool_size = (slab_pool_size - (slab_pool_size % num_layers)) + min_slab_pool_sizes[num_layers]
            num_slabs_per_layer = slab_pool_size // num_layers

            engine_infos = self._get_engine_info_by_num_layers(num_layers)
            for engine_info in engine_infos:
                engine_info['slab_pool_size'] = slab_pool_size
                engine_info['shared_kv_slab_offset'] = shared_kv_slab_offset

            kv_slab_table.init(num_slabs_per_layer, self.slab_size)
            
            # increase offset by num_slabs_per_layer
            shared_kv_slab_offset += slab_pool_size
            num_assigned_slabs += slab_pool_size
            
            assert num_assigned_slabs <= self.total_num_slabs, "Assigned slabs > Total available slabs. \
                Available KV cache memory size cannot satisfy minimum required number of blocks of the placed models."
        
        return num_assigned_slabs


    def _generate_engine_id(self):
        with self.lock:
            engine_id = self.num_conn_engines
            self.num_conn_engines += 1
            return engine_id
    
    def _get_engine_info_by_num_layers(self, num_layers):
        engines = []
        for engine_info in self.engine_info.values():
            if num_layers == engine_info['num_layers']:
                engines.append(engine_info)
            
        return engines

    def register_engine(self, model_name: str, num_layers: int, 
                        tokens_per_block: int, token_size: int, scale_size: int,
                        job_configs: list[FineServeJobConfig], local_rank: int):
        
        engine_id = self._generate_engine_id()
        # initialize engine_info
        rate = None
        for job_config in job_configs:
            if job_config.model == model_name:
                rate = job_config.rate

        assert rate, f"{model_name} does not exist in FineServeJobConfig"
        self.engine_info[engine_id] = {
            "model_name": model_name,
            "num_layers": num_layers, 
            "rate": rate,
            "tokens_per_block": tokens_per_block,
            "token_size": token_size,
            "scale_size": scale_size,
            "block_size": tokens_per_block * token_size + scale_size,
        }

        logger.info(f"Local rank {local_rank} Engine {engine_id} is registered. ({self.num_conn_engines}/{self.num_placed_models}) \
                    \n\tmodel_name: {self.engine_info[engine_id]['model_name']} \
                    \n\tnum_layers: {self.engine_info[engine_id]['num_layers']} \
                    \n\trequest_rate: {self.engine_info[engine_id]['rate']} \
                    \n\ttokens_per_block: {self.engine_info[engine_id]['tokens_per_block']} \
                    \n\ttoken_size: {self.engine_info[engine_id]['token_size']} \
                    \n\tscale_size: {self.engine_info[engine_id]['scale_size']} \
                    \n\tblock_size: {self.engine_info[engine_id]['block_size']}")
        
        return engine_id

    def get_num_conn_engines(self):
        with self.lock:
            num_conn_engines = self.num_conn_engines
        return num_conn_engines

    def get_engine_info(self, engine_id: int):
        if engine_id in self.engine_info:
            return self.engine_info[engine_id]
        else:
            raise ValueError(f"Engine id: {engine_id} is not registered.")
    
    def get_num_free_blocks(self, engine_id: int, block_size: int):
        num_layers = self.engine_info[engine_id]['num_layers']
        return self.kv_slab_tables[num_layers].get_num_free_blocks(block_size)

    def get_shared_kv(self) -> torch.Tensor:
        shared_kv_storage = self.shared_kv.untyped_storage()
    
        (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
        ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = shared_kv_storage._share_cuda_()
        
        shared_kv_info = {
            "dtype": self.shared_kv.dtype,
            "tensor_size": self.shared_kv.shape,
            "tensor_stride": self.shared_kv.stride(),
            "tensor_offset": self.shared_kv.storage_offset(),
            "storage_cls": type(shared_kv_storage),
            "storage_device": storage_device,
            "storage_handle": storage_handle,
            "storage_size_bytes": storage_size_bytes,
            "storage_offset_bytes": storage_offset_bytes,
            "requires_grad": False,
            "ref_counter_handle": ref_counter_handle,
            "ref_counter_offset": ref_counter_offset,
            "event_handle": event_handle,
            "event_sync_required": event_sync_required,
        }
        return shared_kv_info

    def get_format_info(self, engine_id: int, block_size: int) -> Dict[str, int]:
        num_layers = self.engine_info[engine_id]['num_layers']
        return self.kv_slab_tables[num_layers].format_info[block_size]

    def alloc(self, engine_id: int, block_size: int, num_blocks: int) -> List[int]:
        num_layers = self.engine_info[engine_id]['num_layers']
        slab_table = self.kv_slab_tables[num_layers]
        block_ids = slab_table.alloc_blocks(block_size, num_blocks)
        return block_ids

    def free(self, engine_id: int, block_size: int, block_ids: List[int]):
        num_layers = self.engine_info[engine_id]['num_layers']
        slab_table = self.kv_slab_tables[num_layers]
        slab_table.free_blocks(block_size, block_ids)
        

class AsyncKVSlabManager:
    def __init__(self, num_placed_models, max_num_batched_tokens):
        self.manager = KVSlabManager(num_placed_models, max_num_batched_tokens)
    
    async def init(self, *args, **kwargs):
        self.manager.init(*args, **kwargs)

    async def register_engine(self, *args, **kwargs):
        return self.manager.register_engine(*args, **kwargs)

    async def get_shared_kv(self, *args, **kwargs):
        return self.manager.get_shared_kv(*args, **kwargs)
    
    async def alloc(self, *args, **kwargs):
        try:
            return self.manager.alloc(*args, **kwargs)
        except MemoryError:
            return None
        
    async def free(self, *args, **kwargs):
        self.manager.free(*args, **kwargs)
    
    async def get_slab_size(self) -> int:
        return self.manager.slab_size
    
    async def get_num_conn_engines(self):
        return self.manager.get_num_conn_engines()
    
    async def get_engine_info(self, *args, **kwargs):
        return self.manager.get_engine_info(*args, **kwargs)

    async def get_num_free_blocks(self, *args, **kwargs):
        return self.manager.get_num_free_blocks(*args, **kwargs)
    
    async def get_format_info(self, *args, **kwargs):
        return self.manager.get_format_info(*args, **kwargs)