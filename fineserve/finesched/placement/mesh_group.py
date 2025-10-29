import numpy as np

from fineserve.finesched.placement.constants import MEMORY_PER_GPU
from fineserve.finesched.placement.model import LLM


class GPU:

    def __init__(self,
                 rank: int = 0,
                 total_memory: int = MEMORY_PER_GPU,
                 total_mps: int = 100,
                 overload_threshold: int = 20):
        self.rank = rank
        self.total_memory = total_memory
        self.free_memory = total_memory
        self.total_mps = total_mps
        self.free_mps = total_mps
        self.overload_threshold = overload_threshold
        self.models: list[tuple[LLM, int]] = []

    @property
    def memory_utilization(self):
        return self.free_memory / self.total_memory

    @property
    def overloaded(self):
        return self.free_mps < 0

    def can_place(self, llm: LLM, mps: int, tp_size: int):
        if self.free_memory < llm.required_memory / tp_size:
            return False
        return True


    def place_model(self, llm: LLM, mps: int, tp_size: int = 1):
        self.models.append((llm, mps))
        self.free_memory -= llm.required_memory / tp_size
        self.free_mps -= mps
        llm.place(self.rank)

    def __repr__(self) -> str:
        return (f"GPU("
                f"rank={repr(self.rank)}, "
                f"total_memory={repr(self.total_memory)}, "
                f"free_memory={repr(self.free_memory)}, "
                f"gpu_memory_utilization={repr(self.memory_utilization)}, "
                f"total_mps={repr(self.total_mps)}, "
                f"free_mps={repr(self.free_mps)}, "
                f"overload_threshold={repr(self.overload_threshold)}, "
                f"overloaded={repr(self.overloaded)}, "
                f"models={repr(self.models)}"
                f")")


class MeshGroup:

    def __init__(self,
                 gpus: list[GPU],
                 gpu_memory_utilization: float = 1.0):
        self.gpus = gpus
        self.gpu_memory_utilization = gpu_memory_utilization

    @property
    def ngpus(self):
        return len(self.gpus)

    @property
    def total_memory(self):
        return sum([g.total_memory for g in self.gpus])

    @property
    def free_memory(self):
        return sum([g.free_memory for g in self.gpus])

    @property
    def total_mps(self):
        return sum([g.total_mps for g in self.gpus])

    @property
    def free_mps(self):
        return sum([g.free_mps for g in self.gpus])

    @property
    def models(self):
        return set(sum([g.models for g in self.gpus], []))

    def can_place(self, llm: LLM, mps: int):
        return np.all([gpu.can_place(llm, mps, self.ngpus) for gpu in self.gpus])

    def place_model(self, llm: LLM, mps: int):
        for gpu in self.gpus:
            gpu.place_model(llm, mps, self.ngpus)
        # self.models.append((llm, mps))

    def __repr__(self) -> str:
        return (f"MeshGroup("
                f"ngpus={repr(self.ngpus)}, "
                f"gpus={repr(sorted([g.rank for g in self.gpus]))}, "
                # f"gpu_memory_utilization={repr(self.gpu_memory_utilization)}, "
                f"gpu_memory={repr(self.total_memory)}, "
                f"free_gpu_memory={repr(self.free_memory)}, "
                f"models={repr(self.models)}"
                f")")
