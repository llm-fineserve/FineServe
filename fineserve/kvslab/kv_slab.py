from typing import List
from fineserve.config import SLAB_SIZE

class KVSlab:
    def __init__(self, slab_id: int, slab_size: int = SLAB_SIZE):
        self.slab_id = slab_id
        self.slab_size = slab_size # typically 2MB
        self.block_size = 0 # in bytes
        self.num_blocks = 0
        self.free_block_list: List[int] = []

    def init(self, block_size: int):
        """Divide a single KVSlab into multiple blocks"""
        self.block_size = block_size # in bytes
        self.num_blocks = self.slab_size // self.block_size
        self.pad_size = self.slab_size - self.block_size * self.num_blocks
        self.free_block_list = [i for i in range(self.num_blocks)]
        
    def reset(self):
        self.block_size = 0
        self.num_blocks = 0
        self.free_block_list.clear()
    
    def is_full(self) -> bool:
        return not self.free_block_list
    
    def is_empty(self) -> bool:
        return len(self.free_block_list) == self.num_blocks
    
    def get_num_free_blocks(self) -> int:
        return len(self.free_block_list)
    
    def get_block_usage(self) -> int:
        return self.num_blocks - len(self.free_block_list)
    
    def get_mem_usage(self) -> int:
        return self.get_block_usage() * self.block_size
    
    def alloc_block(self) -> int:
        if self.is_full():
            raise MemoryError(f"Slab {self.slab_id} is full.")
        local_block_id = self.free_block_list.pop()
        global_block_id = self.slab_id * self.num_blocks + local_block_id
        return global_block_id
    
    def free_block(self, block_id: int):
        assert block_id < self.num_blocks, f"KVSlab: block_id is not in valid range. slab_id: {self.slab_id} {block_id} >= {self.num_blocks}"
        self.free_block_list.append(block_id)
