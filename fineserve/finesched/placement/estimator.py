import abc
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from fineserve.finesched.placement.constants import *
from fineserve.finesched.placement.model import LLM
from fineserve.finesched.placement.profile import ProfileData
from fineserve.finesched.placement.utils import linear_interp


def estimate_kv_cache_size(total_memory: float, llms: List[LLM], portion: Optional[np.ndarray] = None) -> float:
    """Estimate available KV cache size after accounting for model and activation memory.

    Args:
        total_memory: Total available memory
        llms: List of LLM models
        portion: Portion of memory allocated to each model

    Returns:
        Available KV cache size in GB
    """
    placement_memory = np.array([llm.placement_memory for llm in llms])
    if portion is None:
        portion = np.ones_like(placement_memory)
    kv_cache_size = total_memory - np.sum(placement_memory * portion)
    return max(0.0, kv_cache_size)


class Estimator(abc.ABC):
    """Abstract base class for performance estimators."""

    def __init__(self, cost_file: str, memory_per_gpu: int = MEMORY_PER_GPU):
        """Initialize the estimator with cost data.

        Args:
            cost_file: Path to the cost CSV file
            memory_per_gpu: Memory per GPU in GB
        """
        self.cost_file = cost_file
        self.memory_per_gpu = memory_per_gpu
        self.pf = ProfileData()
        self.pf.read_cost_file(self.cost_file)

        self._raw_cost: Dict[str, Any] = {}
        self.model_cost: Dict[str, Any] = {}
        self._build_cost()

    def _build_cost(self) -> None:
        """Build cost dictionaries from profile data."""
        for record in self.pf.data.itertuples():
            model = getattr(record, MODEL)
            qformat = getattr(record, QFORMAT)
            tp_size = getattr(record, TP_SIZE)
            mps = getattr(record, MPS_PERCENTAGE)
            batch_size = getattr(record, BATCH_SIZE)
            input_len = getattr(record, INPUT_LEN)
            output_len = getattr(record, OUTPUT_LEN)
            ttft = getattr(record, TTFT)
            decoding_latency = getattr(record, DECODE_TIME)

            # Simplified nested dictionary creation
            cost_entry = self._get_or_create_nested_dict(
                self._raw_cost, [model, qformat, tp_size, mps, batch_size]
            )

            # Add prefill and decoding latencies
            self._append_to_nested_list(cost_entry, input_len, "prefill", ttft)
            self._append_to_nested_list(cost_entry, output_len, "decoding", decoding_latency)

        # Process raw costs into averaged model costs
        for model in self._raw_cost.keys():
            self.add_model_cost(model, self._raw_cost[model])

    def _get_or_create_nested_dict(self, container: dict, keys: List[Any]) -> dict:
        """Get or create a nested dictionary structure.

        Args:
            container: Dictionary to operate on
            keys: List of keys for nesting

        Returns:
            Nested dictionary at the specified path
        """
        current = container
        for key in keys:
            current = current.setdefault(key, {})
        return current

    def _append_to_nested_list(self, container: dict, key: Any, subkey: str, value: float) -> None:
        """Append a value to a nested list structure.

        Args:
            container: Dictionary to operate on
            key: Primary key
            subkey: Secondary key
            value: Value to append
        """
        container.setdefault(key, {}).setdefault(subkey, []).append(value)

    def add_model_cost(self, model: str, model_cost: dict) -> None:
        """Process raw cost data into averaged model costs.

        Args:
            model: Model name
            model_cost: Raw cost data
        """
        self.model_cost[model] = {}
        for qformat, qformat_cost in model_cost.items():
            self.model_cost[model][qformat] = {}
            for tp_size, tp_cost in qformat_cost.items():
                self.model_cost[model][qformat][tp_size] = {}
                for mps, mps_cost in tp_cost.items():
                    self.model_cost[model][qformat][tp_size][mps] = {}
                    for batch_size, bs_cost in mps_cost.items():
                        self.model_cost[model][qformat][tp_size][mps][batch_size] = {}
                        for seq_len, cost_info in bs_cost.items():
                            prefill_cost = cost_info.get("prefill")
                            decode_cost = cost_info.get("decoding")
                            self.model_cost[model][qformat][tp_size][mps][batch_size][seq_len] = {
                                "prefill": np.mean(prefill_cost) if prefill_cost else None,
                                "decoding": np.mean(decode_cost) if decode_cost else None
                            }

    @abc.abstractmethod
    def estimate_mps(self, llm: LLM, tp_size: int, prefill_mps: int) -> ThroughputConfig:
        """Estimate optimal MPS settings.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            prefill_mps: Prefill MPS percentage

        Returns:
            Throughput configuration
        """
        pass

    @abc.abstractmethod
    def estimate_throughputs(self, llm: LLM, tp_size: int, prefill_mps: int,
                             decoding_mps: int, cache_size: Optional[float] = None) -> ThroughputConfig:
        """Estimate throughputs for given configuration.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            prefill_mps: Prefill MPS percentage
            decoding_mps: Decoding MPS percentage
            cache_size: Available cache size

        Returns:
            Throughput configuration
        """
        pass


class FineServeEstimator(Estimator):
    """FineServe-specific performance estimator."""

    def __init__(self, cost_file: str, memory_per_gpu: int = MEMORY_PER_GPU,
                 max_batch_size: int = 32):
        """Initialize the FineServe estimator.

        Args:
            cost_file: Path to the cost CSV file
            memory_per_gpu: Memory per GPU in GB
            max_batch_size: Maximum batch size to consider
        """
        super().__init__(cost_file, memory_per_gpu)
        self.max_batch_size = max_batch_size

    def get_avg_latency_per_token(self, llm: LLM, tp_size: int, mps: int,
                                  batch_size: int, prefill: bool = True) -> Optional[float]:
        """Calculate average latency per token.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            mps: MPS percentage
            batch_size: Batch size
            prefill: Whether to calculate prefill latency

        Returns:
            Average latency per token or None if not available
        """
        key = "prefill" if prefill else "decoding"

        try:
            bs_cost = self.model_cost[llm.model_type][llm.qformat][tp_size][mps][batch_size]
        except KeyError:
            return None

        latency_per_token = [
            cost_info.get(key) / seq_len
            for seq_len, cost_info in bs_cost.items()
            if cost_info.get(key) is not None
        ]

        if not latency_per_token:
            return None

        return np.mean(latency_per_token)

    def get_avg_latency(self, llm: LLM, tp_size: int, mps: int, batch_size: int,
                        seq_len: float, prefill: bool = True) -> Optional[float]:
        """Calculate average latency for a sequence.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            mps: MPS percentage
            batch_size: Batch size
            seq_len: Sequence length
            prefill: Whether to calculate prefill latency

        Returns:
            Average latency or None if not available
        """
        latency_per_token = self.get_avg_latency_per_token(
            llm, tp_size, mps, batch_size, prefill=prefill
        )

        if not latency_per_token:
            return None

        return latency_per_token * seq_len

    def get_batch_and_avg_cost(self, llm: LLM, tp_size: int, prefill_mps: int,
                               decoding_mps: int, input_len: float, output_len: float) -> CostData:
        """Calculate batch sizes and associated costs.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            prefill_mps: Prefill MPS percentage
            decoding_mps: Decoding MPS percentage
            input_len: Input sequence length
            output_len: Output sequence length

        Returns:
            Cost data structure
        """
        ttft_batch_sizes, ttft_latencies = [], []
        decoding_batch_sizes, decoding_latencies = [], []
        throughput_batch_sizes, throughputs = [], []

        for batch_size in self.pf.all_categories[BATCH_SIZE]:
            # Calculate prefill latency
            prefill_latency = self.get_avg_latency(
                llm, tp_size, prefill_mps, batch_size, input_len, True
            )

            if prefill_latency:
                ttft_batch_sizes.append(batch_size)
                ttft_latencies.append(prefill_latency)

            # Calculate decoding latency
            decoding_latency = self.get_avg_latency(
                llm, tp_size, decoding_mps, batch_size, output_len, False
            )

            if decoding_latency:
                decoding_batch_sizes.append(batch_size)
                decoding_latencies.append(decoding_latency)

            # Calculate throughput if both latencies are available
            if prefill_latency and decoding_latency:
                throughput = batch_size / (prefill_latency + decoding_latency)
                throughput_batch_sizes.append(batch_size)
                throughputs.append(throughput)

        return CostData(
            llm.name, tp_size,
            prefill_mps, decoding_mps,
            input_len, output_len,
            ttft_latencies, ttft_batch_sizes,
            decoding_latencies, decoding_batch_sizes,
            throughputs, throughput_batch_sizes
        )

    def set_tp_candidates(self, llm: LLM, prefill_mps: int = 100, verbose: bool = False) -> None:
        """Set tensor parallelism candidates for the LLM.

        Args:
            llm: Language model
            prefill_mps: Prefill MPS percentage
            verbose: Whether to print verbose output
        """
        llm.tp_candidates = {}

        for tp_size in [1, 2, 4, 8]:
            throughput_config = self._estimate_tp_throughput_config(llm, tp_size)

            # Skip impossible configurations
            if throughput_config is None:
                continue

            llm.tp_candidates[tp_size] = throughput_config

        # Set minimum mesh size
        llm.min_mesh_size = min(llm.tp_candidates.keys()) if llm.tp_candidates else 8
        assert len(llm.tp_candidates) > 0, "No valid mesh size"

        if verbose:
            self._print_tp_candidates(llm)

    def _estimate_tp_throughput_config(self, llm: LLM, tp_size: int) -> Optional[ThroughputConfig]:
        """Estimate throughput configuration for a TP size.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size

        Returns:
            Throughput configuration or None if not possible
        """
        if tp_size not in self.pf.all_categories[TP_SIZE]:
            return None

        throughput_config = self.estimate_throughputs(
            llm, tp_size, prefill_mps=100, decoding_mps=100
        )

        return throughput_config

    def _print_tp_candidates(self, llm: LLM) -> None:
        """Print TP candidates for debugging.

        Args:
            llm: Language model
        """
        print(f"## LLM: {llm.name}, Model: {llm.model}, Rate: {llm.rate}, Candidates:")
        for _, config in llm.tp_candidates.items():
            print(f"### ngpu: {config.tp_size}, mps: {config.mps_percentage}, "
                  f"bs: {config.rate_min_batch_size}, tpt: {config.throughput:.3f}, "
                  f"can_satisfy: {config.satisfied}")

    def estimate_mps(self, llm: LLM, tp_size: int, prefill_mps: int) -> ThroughputConfig:
        """Estimate optimal MPS settings.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            prefill_mps: Prefill MPS percentage

        Returns:
            Throughput configuration
        """
        max_throughput_config = ThroughputConfig(
            False, tp_size, 1, 0, 0, 0, 0, 0, 0, None
        )

        if tp_size not in self.pf.all_categories[TP_SIZE]:
            return max_throughput_config

        for decoding_mps in self.pf.all_categories[MPS_PERCENTAGE]:
            throughput_config = self.estimate_throughputs(
                llm, tp_size, prefill_mps, decoding_mps
            )

            if throughput_config.satisfied:
                return throughput_config

            if throughput_config.throughput > max_throughput_config.throughput:
                max_throughput_config = throughput_config

        return max_throughput_config

    def estimate_throughputs(self, llm: LLM, tp_size: int, prefill_mps: int,
                             decoding_mps: int, cache_size: Optional[float] = None) -> Optional[ThroughputConfig]:
        """Estimate throughputs for given configuration.

        Args:
            llm: Language model
            tp_size: Tensor parallelism size
            prefill_mps: Prefill MPS percentage
            decoding_mps: Decoding MPS percentage
            cache_size: Available cache size

        Returns:
            Throughput configuration or None if not possible
        """
        cost_data = self.get_batch_and_avg_cost(
            llm, tp_size, prefill_mps, decoding_mps,
            llm.avg_input_len, llm.avg_output_len
        )

        # Return None if no data is available
        if not cost_data.ttft or not cost_data.throughput:
            return None

        # Initialize variables for iterative optimization
        data_parallel_size = 1
        request_rate = llm.rate
        satisfiable = False

        while not satisfiable:
            # Calculate maximum batch size that satisfies SLO
            slo_max_batch_size = np.floor(
                linear_interp(llm.slo, cost_data.ttft, cost_data.ttft_batch_size)
            )

            # Calculate minimum batch size that satisfies request rate
            rate_min_batch_size = np.ceil(
                linear_interp(request_rate, cost_data.throughput, cost_data.throughput_batch_size)
            )

            # Calculate available cache size
            if cache_size is None:
                cache_size = estimate_kv_cache_size(
                    self.memory_per_gpu * tp_size, [llm]
                )

            # Calculate maximum batch size constrained by cache
            cache_max_batch_size = np.floor_divide(
                cache_size, llm.kv_cache_size_per_seq()
            )

            # Determine actual maximum batch size
            max_batch_size = min(cache_max_batch_size, slo_max_batch_size)

            # Check if we can satisfy the request rate
            satisfiable = (rate_min_batch_size <= max_batch_size)
            current_batch_size = min(rate_min_batch_size, max_batch_size)

            # Special case for very low rates
            if satisfiable and rate_min_batch_size <= 0.0:
                current_batch_size = 1
                rate_min_batch_size = 1

            # Calculate latencies for current batch size
            ttft = linear_interp(
                current_batch_size, cost_data.ttft_batch_size, cost_data.ttft
            )
            decoding_latency = linear_interp(
                current_batch_size, cost_data.decoding_latency_batch_size,
                cost_data.decoding_latency
            )

            # Calculate throughput
            throughput = current_batch_size / (ttft + decoding_latency)

            # If satisfiable, we're done
            if satisfiable:
                break

            # Otherwise, increase data parallelism and reduce effective rate
            data_parallel_size += 1
            request_rate = llm.rate // data_parallel_size

            # Check if we've exhausted possibilities
            if request_rate <= 0.0:
                break

        return ThroughputConfig(
            satisfiable, tp_size, data_parallel_size, decoding_mps,
            int(rate_min_batch_size), slo_max_batch_size,
            throughput, ttft, decoding_latency, cost_data
        )