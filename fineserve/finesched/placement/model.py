import abc
from typing import Dict, List

from transformers import AutoConfig

from fineserve.finesched.placement.constants import *
from fineserve.utils.quant_format import get_quant


class LLM(abc.ABC):

    def __init__(self,
                 name: str,
                 model: str,
                 model_size_org: int,
                 qformat: str,
                 rate: float,
                 avg_input_len: float,
                 avg_output_len: float,
                 placement: List[int] = None,
                 slo: float = None):
        self.name = name
        self.model = model
        self.model_type = model.split("/")[-1]
        self.model_size_org = model_size_org
        self.activation_size_org = 3  # Base activation size in GB
        self.qformat = qformat
        self.quant = get_quant(qformat)
        self.rate = rate
        self.avg_input_len = avg_input_len
        self.avg_output_len = avg_output_len
        self.config = AutoConfig.from_pretrained(self.model)

        self.num_attention_heads = self.config.num_attention_heads
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.kv_cache_bytes = self.quant.kvcache.bit / 8

        self.tp_candidates: Dict[int, list[ThroughputConfig]] = {}
        if placement is None:
            placement = []
        self.placement: List[int] = placement
        self.min_mesh_size = None
        self.slo = slo
        self.slo_fixed = False
        self.tpt_config: ThroughputConfig = None

        # Determine base number of GPUs based on model size
        if self.model_size_org >= 60:
            self.base_ngpu = 4
        elif self.model_size_org >= 30:
            self.base_ngpu = 2
        else:
            self.base_ngpu = 1

        # Adjust for AWQ quantization format
        if self.qformat == 'awq':
            self.base_ngpu = max(int(self.base_ngpu/4), 1)

        self.rate_scale = 1

    @property
    @abc.abstractmethod
    def token_size(self):
        """
        key or value cache size of one token for each layer in byte.
        For Multi-Head Attention Llama:
            (number of attention heads) * (heads dim) * (bytes of dtype)
        For Grouped-Query Attention Llama:
            (number of kv heads) * (heads dim) * (bytes of dtype)
        """
        ...

    @property
    @abc.abstractmethod
    def kv_cache_size_per_token(self):
        """
        kv cache size of one token of all layers in gigabyte.
        For Llama:
            2 * (number of layers) * (token size) / (1024 ** 3)
        """
        ...

    def kv_cache_size_per_seq(self, seq_len=None):
        if seq_len is None:
            seq_len = self.avg_output_len + self.avg_input_len
        return self.kv_cache_size_per_token * seq_len

    def kv_cache_size_per_batch(self, batch_size=None, seq_len=None):
        if batch_size is None:
            batch_size = 1
        return self.kv_cache_size_per_seq(seq_len) * batch_size

    @property
    def model_size(self):
        return self.model_size_org * (self.quant.weights.bit / 8)

    @property
    def activation_size(self):
        return self.activation_size_org * (self.quant.activation.bit / 8)

    @property
    def activation_size_per_token(self):
        return 8.5 * self.hidden_size * (self.quant.activation.bit / 8) / (1024 ** 3)

    @property
    def placement_memory(self):
        num_proc = 2
        memory_per_proc = 0.8
        return self.model_size + self.activation_size + num_proc * memory_per_proc

    def place(self, rank):
        self.placement.append(rank)

    def query_candidate(self, ngpu: int):
        if ngpu in self.tp_candidates:
            return self.tp_candidates[ngpu]
        return ThroughputConfig(False, 0, 0, 0, 0, 0, 0, 0, 0, None)


class Llama(LLM):

    def __init__(self,
                 name: str,
                 model: str,
                 model_size_org: int,
                 qformat: str,
                 rate: float,
                 avg_input_len: float,
                 avg_output_len: float,
                 placement: List[int] = None,
                 slo: float = None):
        super().__init__(name,
                         model,
                         model_size_org,
                         qformat,
                         rate,
                         avg_input_len,
                         avg_output_len,
                         placement=placement,
                         slo=slo)

    @property
    def token_size(self):
        return self.num_key_value_heads * self.head_size * self.kv_cache_bytes

    @property
    def kv_cache_size_per_token(self):
        return (2 * self.num_hidden_layers * self.token_size) / (1024 ** 3)

    @property
    def activation_size_per_token(self):
        return 8.5 * self.hidden_size * (self.quant.activation.bit / 8) / (1024 ** 3)

    def __repr__(self):
        return (f"Llama("
                f"name={repr(self.name)}, "
                f"model_type={repr(self.model_type)}, "
                f"qformat={repr(self.qformat)}, "
                f"placement={repr(sorted(self.placement))}"
                f")")