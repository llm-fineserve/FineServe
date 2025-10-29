import torch
import threading
from collections import deque
from typing import Dict, List, Set, Deque
from fineserve.kvslab.kv_slab import KVSlab
from fineserve.logger import get_logger

logger = get_logger()

class KVSlabTable:
    """        
        KV block memory layout:
        
        block_size = token_size * tokens_per_block (N) + scale_size

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ^^^^^^^^^^^                          ^^^^^^^^
        token_size                          scale_size
        ----------------------------------------------
        | Token 1 | Token 2 | ... | Token N | Scales | 
        ----------------------------------------------
    """
    def __init__(self):
        self.slab_size: int
        self.num_slabs: int
        self.slabs: Dict[int, KVSlab]
        self.free_slabs: Deque[int]
        
        self.partial_slabs: Dict[int, Set[int]] = {}
        self.full_slabs: Dict[int, Set[int]] = {}
        self.format_info: Dict[int, Dict[str, int]] = {} # key: block_size, value: format_meata_data
        
        self.free_slabs_lock = threading.Lock()
        self.format_locks: Dict[int, threading.Lock] = {}
    
    def init(self, num_slabs: int, slab_size: int):
        self.slab_size = slab_size
        self.num_slabs = num_slabs
        
        self.slabs = dict(
            [(slab_id, KVSlab(slab_id, self.slab_size)) for slab_id in range(num_slabs)]
        )
        self.free_slabs = deque(
            [ slab_id for slab_id in range(num_slabs) ]
        )

        # num_blocks calculation
        for block_size, block_info in self.format_info.items():
            block_info['num_blocks'] = self.slab_size // block_size

    def register_format(self, tokens_per_block: int, token_size: int, scale_size: int):
        block_size = token_size * tokens_per_block + scale_size
        
        if block_size in self.format_info:
            return
        self.format_info[block_size] = {}
        self.format_info[block_size]['tokens_per_block'] = tokens_per_block
        self.format_info[block_size]['token_size'] = token_size
        self.format_info[block_size]['scale_size'] = scale_size
        self.format_info[block_size]['block_size'] = block_size

        self.partial_slabs[block_size] = set()
        self.full_slabs[block_size] = set()
        self.format_locks[block_size] = threading.Lock()
        
        logger.info(f"Block size: {block_size/1024:.2f}KB is registered. \
                    \n\t- tokens_per_block: {self.format_info[block_size]['tokens_per_block']} \
                    \n\t- token_size: {self.format_info[block_size]['token_size']} \
                    \n\t- scale_size: {self.format_info[block_size]['scale_size']}")
        
    def get_num_free_slabs(self):
        with self.free_slabs_lock:
            return len(self.free_slabs)
    
    def get_num_free_blocks(self, block_size: int):
        num_free_blocks = 0
        num_free_blocks += self.get_num_free_slabs() * self.format_info[block_size]['num_blocks']
        
        with self.format_locks[block_size]:
            if self.partial_slabs[block_size]:
                for slab_id in self.partial_slabs[block_size]:
                    target_slab = self.slabs[slab_id]
                    num_free_blocks += target_slab.get_num_free_blocks()
    
        return num_free_blocks

    def alloc_blocks(self, block_size: int, num_blocks: int):
        block_ids = []
        remaining_blocks = num_blocks

        while remaining_blocks > 0:
            with self.format_locks[block_size]:
                if self.partial_slabs[block_size]:
                    slab_id = self.partial_slabs[block_size].pop()
                else:
                    slab_id = None
            
            if not slab_id:
                with self.free_slabs_lock:
                    if not self.free_slabs:
                        raise MemoryError("Cannot allocate a new slab")
                    slab_id = self.free_slabs.pop()
                
                    target_slab = self.slabs[slab_id]
                    target_slab.init(block_size)
            else:
                target_slab = self.slabs[slab_id]    
                
            while not target_slab.is_full() and remaining_blocks > 0:
                block_id = target_slab.alloc_block()
                block_ids.append(block_id)
                remaining_blocks -= 1
            
            with self.format_locks[block_size]:
                if not target_slab.is_full():
                    self.partial_slabs[block_size].add(slab_id)
                else:
                    self.full_slabs[block_size].add(slab_id)

        return block_ids

    def free_blocks(self, block_size:int, block_ids: List[int]):
        slabs_to_update: Dict[int, List[int]] = {}
        num_blocks_per_slab = self.format_info[block_size]['num_blocks']
            
        for block_id in block_ids:
            slab_id = block_id // num_blocks_per_slab
            if slab_id not in slabs_to_update:
                slabs_to_update[slab_id] = []
            local_block_id = block_id % num_blocks_per_slab
            slabs_to_update[slab_id].append(local_block_id)

        for slab_id, local_block_ids in slabs_to_update.items():
            target_slab = self.slabs[slab_id]

            with self.format_locks[block_size]:
                is_full_before = target_slab.is_full()

                for local_block_id in local_block_ids:
                    target_slab.free_block(local_block_id)
                
                is_empty_after = target_slab.is_empty()
                
                if is_full_before and not is_empty_after:
                    self.full_slabs[block_size].remove(slab_id)
                    self.partial_slabs[block_size].add(slab_id)
                
                if is_empty_after:
                    self.partial_slabs[block_size].discard(slab_id)

            if is_empty_after:
                with self.free_slabs_lock:
                    target_slab.reset()
                    self.free_slabs.append(slab_id)