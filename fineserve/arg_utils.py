import argparse
import dataclasses
import yaml
import torch
from torch.distributed.argparse_util import env
from dataclasses import dataclass
from typing import Optional

from fineserve.config import FineServeJobConfig, FineServeConfig

DTYPE_MAP = {"fp16": torch.float16}


@dataclass
class FineServeArgs:
    """Arguments for FineServe"""
    model_config: str
    mps_dir: Optional[str] = None
    ray_node_address: str = "127.0.0.1"
    base_ray_port: int = 6379
    num_ray_cluster: int = 4
    block_size: int = 32
    gpu_memory_utilization: float = 0.90
    restrict_gpu_memory_utilization: bool = False
    max_num_batched_tokens: int = 2048
    max_num_partial_prefills: int = 1
    max_num_seqs: int = 256
    manager_host: str = "127.0.0.1"
    manager_port: int = 5555
    server_port: int = 50060
    cost_file: str = None
    workload_file: str = None
    split_by_model: str = None
    schedule_approach: str = "saab"
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--model-config',
                            type=str,
                            help='path of the serving job config file')
        parser.add_argument('--mps-dir',
                            type=str,
                            default=FineServeArgs.mps_dir,
                            help='path of the mps directory')
        parser.add_argument('--ray-node-address',
                            type=str,
                            default=FineServeArgs.ray_node_address,
                            help='ray node address')
        parser.add_argument('--base-ray-port',
                            type=int,
                            default=FineServeArgs.base_ray_port,
                            help='the base port of ray cluster')
        parser.add_argument('--num-ray-cluster',
                            type=int,
                            default=FineServeArgs.num_ray_cluster,
                            help='the number of ray cluster')
        # FlexStore arguments.
        parser.add_argument('--block-size',
                            type=int,
                            default=FineServeArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=FineServeArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                                 'the flexstore')
        parser.add_argument('--restrict-gpu-memory-utilization',
                            action="store_true",
                            default=FineServeArgs.restrict_gpu_memory_utilization,
                            help='restrict GPU memory utilization or not')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=FineServeArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                                 'iteration')
        parser.add_argument('--max-num-partial-prefills',
                            type=int,
                            default=FineServeArgs.max_num_partial_prefills,
                            help='maximum number of partial prefills per iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=FineServeArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--manager-host',
                            type=str,
                            default=FineServeArgs.manager_host,
                            help='the host address of resource manager')
        parser.add_argument('--manager-port',
                            type=int,
                            default=FineServeArgs.manager_port,
                            help='the port of resource manager')
        parser.add_argument('--server-port',
                            type=int,
                            default=FineServeArgs.server_port,
                            help='the port of vllm server')
        parser.add_argument('--cost-file',
                            type=str,
                            default=FineServeArgs.cost_file,
                            help='the path of cost file')
        parser.add_argument('--workload-file',
                            type=str,
                            default=FineServeArgs.workload_file,
                            help='the path of workload file')
        parser.add_argument('--split-by-model',
                            type=str,
                            default=FineServeArgs.split_by_model,
                            help='split the workload by model')
        parser.add_argument('--schedule-approach',
                            type=str,
                            default=FineServeArgs.schedule_approach,
                            choices=["fcfs", "sjf", "lsf", "saab", "prism"],
                            help='schedule approach')
        # launch configs
        parser.add_argument("--nnodes",
                            type=int,
                            default=FineServeArgs.nnodes,
                            help="Number of nodes")
        parser.add_argument("--nproc-per-node",
                            type=int,
                            default=FineServeArgs.nproc_per_node,
                            help="Number of workers per node.")
        parser.add_argument("--node-rank",
                            type=int,
                            action=env,
                            default=FineServeArgs.node_rank,
                            help="Rank of the node for multi-node distributed training.")
        parser.add_argument("--master-addr",
                            default=FineServeArgs.master_addr,
                            type=str,
                            action=env,
                            help="Address of the master node (rank 0) that only used for static rendezvous. "
                                 "It should be either the IP address or the hostname of rank 0. "
                                 "For single node multi-proc training the --master-addr can simply be 127.0.0.1; "
                                 "IPv6 should have the pattern `[0:0:0:0:0:0:0:1]`.")
        parser.add_argument("--master-port",
                            default=FineServeArgs.master_port,
                            type=int,
                            action=env,
                            help="Port on the master node (rank 0) to be used for communication during "
                                 "distributed training. It is only used for static rendezvous.")
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'FineServeArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        fineserve_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return fineserve_args

    def create_config(self) -> FineServeConfig:
        assert self.model_config is not None, "model_config is not specified"
        with open(self.model_config, "r") as f:
            model_config = yaml.safe_load(f)
        # overwrite
        cluster = model_config["cluster"]
        self.nnodes = cluster["nnodes"]
        self.nproc_per_node = cluster["ngpus_per_node"]
        self.max_num_seqs = cluster["max_num_seqs"]
        self.overload_threshold = cluster["overload_threshold"]
        self.gpu_memory_utilization = cluster["gpu_memory_utilization"]
        num_gpus = cluster["num_gpus"]

        if self.split_by_model is not None:
            print(f"{'='*30} Split By Model ({self.split_by_model}) {'='*30}")
            model_config["workloads"]["split_by_model"] = self.split_by_model
        else:
            assert model_config["workloads"].get("split_by_model", None) is \
                None, "split_by_model shouldn't be specified in config"

        job_configs = []
        for model in model_config["models"]:
            if self.split_by_model is not None and \
                    model["name"] != self.split_by_model:
                continue

            job_cfg = FineServeJobConfig(
                name=model["name"],
                model=model["model"],
                qformat=model["qformat"],
                pipeline_parallel_size=model["pipeline_parallel_size"],
                tensor_parallel_size=model["tensor_parallel_size"],
                placement=model["placement"],
                mps_percentage=model["mps_percentage"],
                gpu_memory_utilization=model.get("gpu_memory_utilization", cluster["gpu_memory_utilization"]),
                kv_size_gb=model.get("kv_size_gb", 0.0),
                min_num_seqs=model.get("min_num_seqs", 0),
                max_num_seqs=model.get("max_num_seqs", cluster["max_num_seqs"]),
                max_model_len=model.get("max_model_len"),
                model_dtype=model["model_dtype"],
                model_size_org=model["model_size_org"],
                num_hidden_layers=model["num_hidden_layers"],
                rate=model["rate"],
                slo=model["slo"],
                avg_input_len=model["avg_input_len"],
                avg_output_len=model["avg_output_len"],
            )
            if self.split_by_model is not None:
                num_gpus = len(model["placement"])
            job_configs.append(job_cfg)
            assert model["pipeline_parallel_size"] * model[
                "tensor_parallel_size"] <= num_gpus, f"Exceeds {num_gpus} GPUs"
        assert len(job_configs) > 0, "No job is specified"

        if self.workload_file is not None:
            assert model_config["workloads"].get(
                "workload_file") is None, "workload_file is specified twice"
            model_config["workloads"]["workload_file"] = self.workload_file
        # else:
        #     assert model_config["workloads"].get(
        #         "workload_file") is not None, "workload_file is not specified"
        #     self.workload_file = model_config["workloads"]["workload_file"]

        config = FineServeConfig(
            job_configs=job_configs,
            num_gpus=num_gpus,
            ray_node_address=self.ray_node_address,
            base_ray_port=self.base_ray_port,
            num_ray_cluster=self.num_ray_cluster,
            mps_dir=self.mps_dir,
            block_size=self.block_size,
            overload_threshold=self.overload_threshold,
            gpu_memory_utilization=self.gpu_memory_utilization,
            restrict_gpu_memory_utilization=self.restrict_gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_num_seqs=self.max_num_seqs,
            manager_host=self.manager_host,
            manager_port=self.manager_port,
            server_port=self.server_port,
            workload_config=model_config["workloads"],
            model_config=model_config,
            model_config_path=self.model_config,
            cost_file_path=self.cost_file,
            schedule_approach=self.schedule_approach,
            nnodes=self.nnodes,
            nproc_per_node=self.nproc_per_node,
            ngpu_per_node=cluster["ngpus_per_node"],
            node_rank=self.node_rank,
            master_addr=self.master_addr,
            master_port=self.master_port)
        return config
