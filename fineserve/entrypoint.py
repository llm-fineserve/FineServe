import argparse
import asyncio
import signal
import os
import torch.multiprocessing as mp
from fineserve.arg_utils import FineServeArgs
from fineserve.config import FineServeConfig
from fineserve.logger import get_logger
from fineserve.finemanager.resource_manager import FineServeManager
from fineserve.finesched.scheduler import FineServeScheduler

logger = get_logger()

def main_manager(config: FineServeConfig):
    manager = FineServeManager(config)
    # asyncio.run(manager.init())
    asyncio.run(manager.deploy())


def main_scheduler(config: FineServeConfig, manager_pid=None):
    scheduler = FineServeScheduler(config)
    scheduler.serve_models()
    asyncio.run(scheduler.schedule_loop())

    ## shutdown resource_manager
    if manager_pid:
        os.kill(manager_pid, signal.SIGTERM)


def main(args: argparse.Namespace):
    mp.set_start_method("spawn")
    fineserve_args = FineServeArgs.from_cli_args(args)
    config = fineserve_args.create_config()

    manager_process: mp.Process
    scheduler_process: mp.Process

    procs = []
    manager_pid = None
    if args.manager:
        manager_process = mp.Process(target=main_manager, args=(config, ))
        manager_process.start()
        procs.append(manager_process)
        manager_pid = manager_process.pid
    if args.scheduler:
        scheduler_process = mp.Process(target=main_scheduler, args=(config, manager_pid))
        scheduler_process.start()
        procs.append(scheduler_process)

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FineServe Entry Point')
    parser.add_argument("--manager",
                        action="store_true",
                        help="Launch resource manager process.")
    parser.add_argument("--scheduler",
                        action="store_true",
                        help="Launch scheduler process.")
    parser = FineServeArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
