import asyncio
import zmq
import zmq.asyncio
import pickle
import signal
from typing import Dict, List

from fineserve.config import FineServeConfig, FineServeJobConfig
from fineserve.kvslab.kv_slab_manager import AsyncKVSlabManager
from fineserve.logger import get_logger
from fineserve.utils.port_utils import get_manager_port, get_rank

logger = get_logger()

class ResourceManager:
    def __init__(self,
                 config: FineServeConfig,
                 local_rank: int, rank: int,
                 host="localhost", port=5555):
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.host = host
        self.port = port
        self.sync_port = port+1
        self.url = f"tcp://{self.host}:{self.port}"
        self.sync_url = f"tcp://{self.host}:{self.sync_port}"
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(self.url)
        self.sync_url = f"tcp://{self.host}:{self.port + 1}"
        self.sync_socket = self.ctx.socket(zmq.PUB)
        self.sync_socket.bind(self.sync_url)
        self.job_configs: list[FineServeJobConfig] = [
            job_config
            for job_config in self.config.job_configs
            if self.rank in job_config.placement
        ]
        
        self.num_placed_models = len(self.job_configs)
        self.kv_slab_manager = AsyncKVSlabManager(self.num_placed_models, self.config.max_num_batched_tokens)

    async def deploy(self):
        logger.info(f"[ResourceManager {self.local_rank}] Waiting {self.num_placed_models} engines on {self.url}...")
        await self.register_engines()
        
        logger.info(f"[ResourceManager {self.local_rank}] All engines connected. Initializing memory...")
        device = f"cuda:{self.local_rank}"
        await self.kv_slab_manager.init(device, self.job_configs)
        
        logger.info(f"[ResourceManager {self.local_rank}] Initialize finish.")
        await self.sync_socket.send_string("START")

        try:
            while True:
                identity, _, msg = await self.socket.recv_multipart()
                engine_id, op, args = pickle.loads(msg)
                logger.debug(f"[Local Rank {self.local_rank}] engine_id {engine_id} received request ({op}, {args})")
                await self.response(identity=identity, engine_id=engine_id, op=op, args=args)

        except asyncio.CancelledError:
            logger.info(f"[Local Rank {self.local_rank} Resource Manager] {self.url} is shutting down...")
        finally:
            self.socket.close()
            self.sync_socket.close()
            logger.info(f"[Local Rank {self.local_rank} Resource Manager] {self.url} is terminated.")
    
    async def register_engines(self):
        engine_connected = await self.kv_slab_manager.get_num_conn_engines()
        while engine_connected < self.num_placed_models:
            identity, _, msg = await self.socket.recv_multipart()
            engine_id, op, args = pickle.loads(msg)
            
            if op == "register":
                model_name, num_layers, tokens_per_block, token_size, scale_size = args
                engine_id = await self.kv_slab_manager.register_engine(model_name,
                                                           num_layers, tokens_per_block, token_size, 
                                                           scale_size, self.job_configs, self.local_rank)
                response = pickle.dumps(engine_id)
                await self.socket.send_multipart([identity, b'', response])
                engine_connected = await self.kv_slab_manager.get_num_conn_engines()

    async def response(self, identity, engine_id, op, args):
        if op == "alloc":
            block_size, num_blocks = args
            block_ids = await self.kv_slab_manager.alloc(engine_id, block_size, num_blocks)
            num_free_blocks = await self.kv_slab_manager.get_num_free_blocks(engine_id, block_size)
            response = pickle.dumps((block_ids, num_free_blocks))
            await self.socket.send_multipart([identity, b'', response])
        elif op == "free":
            block_size, block_ids = args
            await self.kv_slab_manager.free(engine_id, block_size, block_ids)
            num_free_blocks = await self.kv_slab_manager.get_num_free_blocks(engine_id, block_size)
            response = pickle.dumps(num_free_blocks)
            await self.socket.send_multipart([identity, b'', response])
        elif op == "get_shared_kv":
            shared_kv_info = await self.kv_slab_manager.get_shared_kv()
            response = pickle.dumps(shared_kv_info)
            await self.socket.send_multipart([identity, b'', response])
        elif op == "get_slab_size":
            slab_size = await self.kv_slab_manager.get_slab_size()
            response = pickle.dumps(slab_size)
            await self.socket.send_multipart([identity, b'', response])
        elif op == "get_engine_info":
            engine_info = await self.kv_slab_manager.get_engine_info(engine_id)
            response = pickle.dumps(engine_info)
            await self.socket.send_multipart([identity, b'', response])
        elif op == "get_format_info":
            format_info = await self.kv_slab_manager.get_format_info(*args)
            response = pickle.dumps(format_info)
            await self.socket.send_multipart([identity, b'', response])
        elif op == "get_num_free_blocks":
            block_size = args
            num_free_blocks = await self.kv_slab_manager.get_num_free_blocks(engine_id, block_size)
            response = pickle.dumps(num_free_blocks)
            await self.socket.send_multipart([identity, b'', response])
        elif op == "health_check":
            response = f"[Local Rank {self.local_rank}] Listening on {self.url}"
            await self.socket.send_multipart([identity, b'', response])

class FineServeManager:
    def __init__(self, config: FineServeConfig):
        self.config = config
        self.host = self.config.manager_host
        self.port = self.config.manager_port
        self.node_rank = self.config.node_rank
        self.nproc_per_node = self.config.nproc_per_node
        self.resource_managers: Dict[int, ResourceManager] = {}
        #  node_rank, rank, local_rank
        #      0           1        : node_rank
        #   0 1 2 3     0 1 2 3     : local_rank
        #   0 1 2 3     4 5 6 7     : rank (global_rank)
        
        for local_rank in range(self.nproc_per_node):
            rank = get_rank(self.node_rank, self.nproc_per_node, local_rank)
            port = get_manager_port(self.port, local_rank)
            self.resource_managers[rank] = ResourceManager(self.config,
                                                           local_rank,
                                                           rank,
                                                           self.host,
                                                           port)

    def signal_handler(self, sig, loop, tasks):
        for task in tasks:
            task.cancel()

    async def deploy(self):
        loop = asyncio.get_running_loop()
        tasks = [
            asyncio.create_task(manager.deploy()) for manager in self.resource_managers.values()
        ]

        # Register signal handlers for safe KVSlabManager termination
        loop.add_signal_handler(signal.SIGINT, self.signal_handler, signal.SIGINT, loop, tasks)
        loop.add_signal_handler(signal.SIGTERM, self.signal_handler, signal.SIGTERM, loop, tasks)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("deploy cancelled.")
        finally:
            logger.info("resource manager cleanup complete.")