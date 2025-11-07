from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    # Default values for server configuration
    DEFAULT_WARMUP_PROMPTS: int = 10
    DEFAULT_SAMPLING_TEMP: float = 0.8
    DEFAULT_SAMPLING_TOP_P: float = 0.95
    DEFAULT_WARMUP_MAX_TOKENS: int = 20
    DEFAULT_TIMEOUT_SECONDS: int = 30
    DEFAULT_MPS_PERCENTAGE: int = 100
    DEFAULT_MODEL_ID: int = 0
    DEFAULT_SLEEP_INTERVAL: float = 0.001
    SHARED_MEMORY_RETRY_INTERVAL: float = 1 / 5e4
    PROMPT_TOKEN_SIZE: int = 16
    PROMPT_VOCAB_SIZE: int = 24000


@dataclass
class ProfilingConfig:
    # Configuration for profiling
    ENABLE_PROFILER: bool = False
    
    # Profiler activity types
    PROFILER_ACTIVITIES: tuple = (
        "torch.profiler.ProfilerActivity.CPU",
        "torch.profiler.ProfilerActivity.CUDA"
    )