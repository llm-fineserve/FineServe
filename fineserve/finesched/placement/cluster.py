import abc
import itertools

from fineserve.finesched.placement.mesh_group import MeshGroup, GPU


class Cluster(abc.ABC):

    def __init__(self,
                 nnodes: int,
                 ngpus_per_node: int,
                 memory_per_gpu: int,
                 overload_threshold: int,
                 gpu_memory_utilization: float):
        self.nnodes = nnodes
        self.ngpus_per_node = ngpus_per_node
        self.memory_per_gpu = memory_per_gpu
        self.gpu_memory_utilization = gpu_memory_utilization
        self.overload_threshold = overload_threshold

        self.ngpus = self.nnodes * self.ngpus_per_node
        self.total_memory = self.ngpus * self.memory_per_gpu
        self.gpus = [
            GPU(rank, self.memory_per_gpu, 100, self.overload_threshold)
            for rank in range(self.nnodes * self.ngpus_per_node)
        ]
        self.nodes = [
            MeshGroup(self.gpus[ngpus_per_node * node_idx:
                                ngpus_per_node * (node_idx + 1)],
                      self.gpu_memory_utilization)
            for node_idx in range(nnodes)
        ]


class FineServeCluster(Cluster):
    def __init__(self,
                 nnodes: int,
                 ngpus_per_node: int,
                 memory_per_gpu: int,
                 overload_threshold: int,
                 gpu_memory_utilization: float):
        super().__init__(nnodes,
                         ngpus_per_node,
                         memory_per_gpu,
                         overload_threshold,
                         gpu_memory_utilization)