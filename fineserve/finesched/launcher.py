import copy
import sys
import os
import subprocess
from pathlib import Path

from fineserve.config import FineServeJobConfig, FineServeConfig, SLAB_SIZE
from fineserve.logger import get_logger
from fineserve.utils.port_utils import get_manager_port
from fineserve.utils.quant_format import get_quant

logger = get_logger()


def launch_server_process(model_id,
                          block_size,
                          mps_percentage,
                          job_config: FineServeJobConfig,
                          cluster_config: FineServeConfig,
                          gpu_memory_utilization=None,
                          logfile=None):
    # we have left the options we have tried out in case someone needs it for debugging or further work in the future
    # prefill_option = "--is-prefill" if is_prefill else ""
    # split_option = f"--split-by-model {split_by_model}" if split_by_model else ""
    print(f"job_config.placement: {job_config.placement}")
    print(f"job_config.nproc_per_node: {cluster_config.ngpu_per_node}")
    logger.info(f"model_id: {model_id}")
    local_ranks = [
        rank % cluster_config.ngpu_per_node
        for rank in job_config.placement
    ]
    kv_slab_ports = [
        get_manager_port(cluster_config.manager_port, local_rank)
        for local_rank in local_ranks
    ]

    exc = sys.executable
    fineserve_dir = Path(__file__).parents[1]
    script = os.path.join(fineserve_dir, "fineserver/server.py")
    ## vllms default max_num_batched_tokens is 2048, max_num_batched_tokens >= max_num_seqs
    ## hence, we check amd cap max_num_seqs (512), increase this if you must
    ## also the max size for cuda graphs are 512
    max_num_seqs=min(job_config.max_num_seqs, 256)

    cmd = [
        exc,
        script,
        "--model-id", str(model_id),
        "--model-name", str(job_config.name),
        "--model", str(job_config.model),
        "--tensor-parallel-size", str(job_config.tensor_parallel_size),
        "--pipeline-parallel-size", str(job_config.pipeline_parallel_size),
        "--block-size", str(block_size),
        "--swap-space", str(1),
        "--max-num-seqs", str(max_num_seqs),
        # "--enable-chunked-prefill",
        # "--max-num-partial-prefills", str(cluster_config.max_num_partial_prefills),
        "--mps-percentage", str(mps_percentage),
        # "--enforce-eager",
        # "--no-enable-prefix-caching",
        "--kv-slab-size", str(SLAB_SIZE),
        "--kv-slab-host", str(cluster_config.manager_host),
    ]
    cmd.extend(["--kv-slab-ports"]+[str(port) for port in kv_slab_ports])

    if job_config.max_model_len:
        cmd.extend(["--max-model-len", str(job_config.max_model_len)])

    VLLM_USE_KV_SLAB = 1
    if cluster_config.restrict_gpu_memory_utilization:
        VLLM_USE_KV_SLAB = 0
        gmu = gpu_memory_utilization if gpu_memory_utilization else job_config.gpu_memory_utilization
        cmd.extend(["--gpu-memory-utilization", str(gmu)])
        cmd.extend(["--max-model-len", str(4096)])

    quant = get_quant(job_config.qformat)
    logger.info(f" quant: {quant}")
    if quant.vllm_option is not None:
        strs = quant.vllm_option.split()
        cmd.extend(strs)

    proc_env = copy.deepcopy(os.environ)
    proc_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(rank) for rank in local_ranks])
    proc_env["VLLM_USE_KV_SLAB"] = str(VLLM_USE_KV_SLAB)
    proc_env[f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_percentage)
    proc_env["VLLM_USE_FLASHINFER_SAMPLER"] = '0'
    # proc_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = '0'      # use InprocClient

    if quant.name == "qoq":
        proc_env["QSERVE_VIA_VLLM"] = '1'
        proc_env['NUM_RETRIEVAL_GPU_PAGE_BLOCKS']='3000'
        proc_env['NUM_STREAMING_GPU_PAGE_BLOCKS']='0'
        # kmbin: QServe engine core requires max_num_batched_tokens > max_model_len
        #        QServe default is 262144
        qserve_max_num_batched_tokens = 8192
        cmd.extend([
            "--max-num-batched-tokens", str(qserve_max_num_batched_tokens),
            "--disable-log-stats",  # TODO: vllm/v1/core_qoq.py makes assertion error without this
        ])
    else:
        max_batched_tokens = cluster_config.max_num_batched_tokens
        cmd.extend([
            "--max-num-batched-tokens", str(max_batched_tokens),
        ])

    logdir = os.environ.get("VLLM_PROC_LOG", "log/vllm_proc")
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    name = job_config.name
    model_name = name.split("/")[-1]
    if logfile is None:
        logfile = f"{logdir}/{model_name}_sm{mps_percentage}.log"

    cmd_str = " ".join(cmd)
    logger.info(f"Start process cmd: {cmd_str}, Output log file: {logfile}")

    logfile_writer = open(logfile, "w")
    logfile_writer.write(f"Start process cmd: {cmd_str}\n")
    logfile_writer.write(f"Environment Variable: \n")
    for k, v in proc_env.items():
        logfile_writer.write(f"    {k}: {v}\n")
    proc = subprocess.Popen(
        cmd,
        env=proc_env,
        shell=False,
        stdout=logfile_writer,
        stderr=subprocess.STDOUT,
    )
    return proc
