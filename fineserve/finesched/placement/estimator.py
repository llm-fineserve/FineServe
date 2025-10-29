import abc
from typing import Optional, List

import numpy as np

from fineserve.finesched.placement.constants import *
from fineserve.finesched.placement.model import LLM
from fineserve.finesched.placement.profile import ProfileData
from fineserve.finesched.placement.utils import linear_interp


def estimate_kv_cache_size(total_memory, llms: List[LLM], portion=None):
    # weight = sum([llm.model_size for llm in llms])
    # activation = sum([llm.activation_size for llm in llms])
    # kv_cache_size = total_memory * 0.98 - weight - activation
    placement_memory = np.array([llm.placement_memory for llm in llms])
    if portion is None:
        portion = np.ones_like(placement_memory)
    kv_cache_size = total_memory - np.sum(placement_memory * portion)
    return max(0, kv_cache_size)


class Estimator(abc.ABC):

    def __init__(self,
                 cost_file: str,
                 memory_per_gpu: int = MEMORY_PER_GPU):
        self.cost_file = cost_file
        self.memory_per_gpu = memory_per_gpu
        self.pf = ProfileData()
        self.pf.read_cost_file(self.cost_file)

        self._raw_cost = {}
        self.model_cost = {}
        self._build_cost()

    def _build_cost(self):
        for r in self.pf.data.itertuples():
            model = getattr(r, MODEL)
            qformat = getattr(r, QFORMAT)
            ngpu = getattr(r, TP_SIZE)
            mps = getattr(r, MPS_PERCENTAGE)
            batch_size = getattr(r, BATCH_SIZE)
            input_len = getattr(r, INPUT_LEN)
            output_len = getattr(r, OUTPUT_LEN)
            ttft = getattr(r, TTFT)
            decoding_latency = getattr(r, DECODE_TIME)

            cost = self._raw_cost \
                .setdefault(model, {}) \
                .setdefault(qformat, {}) \
                .setdefault(ngpu, {}) \
                .setdefault(mps, {}) \
                .setdefault(batch_size, {})

            cost.setdefault(input_len, {}) \
                .setdefault("prefill", []) \
                .append(ttft)
            cost.setdefault(output_len, {}) \
                .setdefault("decoding", []) \
                .append(decoding_latency)

        for model in self._raw_cost.keys():
            self.add_model_cost(model, self._raw_cost[model])

    def add_model_cost(self, model: str, model_cost: dict):
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
    def estimate_mps(self,
                     llm: LLM,
                     tp: int,
                     prefill_mps: int) -> ThroughputConfig:
        ...

    @abc.abstractmethod
    def estimate_throughputs(self,
                            llm: LLM,
                            tp_size: int,
                            prefill_mps: int,
                            decoding_mps: int,
                            cache_size: Optional[float] = None) -> ThroughputConfig:
        ...


class FineServeEstimator(Estimator):

    def __init__(self,
                 cost_file: str,
                 memory_per_gpu: int = MEMORY_PER_GPU):
        super().__init__(cost_file, memory_per_gpu)
        self.MAX_BATCH_SIZE = 32 # TODO: make this parameter configurable with yaml file

    def get_avg_latency_per_token(self, llm: LLM, tp_size, mps, batch_size, prefill=True):
        key = "prefill" if prefill else "decoding"
        try:
            bs_cost = self.model_cost[llm.model_type][llm.qformat][tp_size][mps][batch_size]
        except KeyError as e:
            return None
        latency_per_token = [cost_info.get(key) / seq_len
                             for seq_len, cost_info in bs_cost.items()
                             if cost_info.get(key) is not None]
        if not latency_per_token:
            return None
        return np.mean(latency_per_token)

    def get_avg_latency(self, llm: LLM, tp_size, mps, batch_size, seq_len, prefill=True):
        latency_per_token = self.get_avg_latency_per_token(llm, tp_size, mps, batch_size,
                                                           prefill=prefill)
        if not latency_per_token:
            return None
        avg_latency = np.mean(latency_per_token) * seq_len
        return avg_latency

    def get_batch_and_avg_cost(self,
                               llm: LLM,
                               tp_size: int,
                               prefill_mps: int,
                               decoding_mps: int,
                               input_len: float,
                               output_len: float):
        ttft_bs_data = []
        ttft_data = []
        dec_lat_bs_data = []
        dec_lat_data = []
        tpt_bs_data = []
        tpt_data = []
        for batch_size in self.pf.all_categories[BATCH_SIZE]:
            pre_lat = self.get_avg_latency(llm,
                                           tp_size,
                                           prefill_mps,
                                           batch_size,
                                           input_len,
                                           True)
            if pre_lat:
                ttft_bs_data.append(batch_size)
                ttft_data.append(pre_lat)

            dec_lat = self.get_avg_latency(llm,
                                           tp_size,
                                           decoding_mps,
                                           batch_size,
                                           output_len,
                                           False)
            if dec_lat:
                dec_lat_bs_data.append(batch_size)
                dec_lat_data.append(dec_lat)

            if pre_lat and dec_lat:
                tpt = batch_size / (pre_lat + dec_lat)
                tpt_bs_data.append(batch_size)
                tpt_data.append(tpt)

        data = CostData(llm.name, tp_size,
                        prefill_mps, decoding_mps,
                        input_len,
                        output_len,
                        ttft_data, ttft_bs_data,
                        dec_lat_data, dec_lat_bs_data,
                        tpt_data, tpt_bs_data)
        return data


    def set_tp_candidates(self,
                          llm: LLM,
                          prefill_mps: int = 100,
                          verbose: bool = False):
        llm.tp_candidates = {}
        for tp_size in [1, 2, 4, 8]:
            tpt_config = self.estimate_tp_trpt_config(llm, tp_size)
            ## if impossible to satisfy, then skip this tp
            if tpt_config is None:
                #print(f"Impossible to setup model {llm.name} for TP{ngpu}")
                continue
            llm.tp_candidates[tp_size] = tpt_config

        llm.min_mesh_size = 8
        if llm.tp_candidates:
            llm.min_mesh_size = min([ngpu for ngpu in llm.tp_candidates])
        assert len(llm.tp_candidates) > 0, "No valid mesh size"

        if verbose:
            print(f"## LLM: {llm.name}, Model: {llm.model}, "
                  f"Rate: {llm.rate}, Candidates:")
            for _, c in llm.tp_candidates.items():
                ngpu = c.tp_size
                mps = c.mps_percentage
                bs = c.batch_size
                tpt = c.throughput
                sfy = c.satisfied
                print(f"### ngpu: {ngpu}, mps: {mps}, bs: {bs}, "
                      f"tpt: {tpt:.3f}, can_satisfy: {sfy}")

    def estimate_tp_trpt_config(self,
                                llm: LLM,
                                tp_size: int):

        if tp_size not in self.pf.all_categories[TP_SIZE]:
            return None

        tpt_config = self.estimate_throughputs(llm,
                                              tp_size,
                                              prefill_mps=100,
                                              decoding_mps=100)

        if tpt_config is None:
            return None

        return tpt_config

    def estimate_mps(self,
                     llm: LLM,
                     tp_size: int,
                     prefill_mps: int) -> ThroughputConfig:
        max_tpt_config = ThroughputConfig(False, tp_size, 1, 0, 0, 0, 0, 0, 0, None)
        if tp_size not in self.pf.all_categories[TP_SIZE]:
            return max_tpt_config

        for decoding_mps in self.pf.all_categories[MPS_PERCENTAGE]:
            tpt_config = self.estimate_throughputs(llm,
                                                  tp_size,
                                                  prefill_mps,
                                                  decoding_mps)
            if tpt_config.satisfied:
                return tpt_config

            if tpt_config.throughput > max_tpt_config.throughput:
                max_tpt_config = tpt_config

        return max_tpt_config

    def estimate_throughputs(self,
                            llm: LLM,
                            tp_size: int,
                            prefill_mps: int,
                            decoding_mps: int,
                            cache_size: Optional[float] = None) -> ThroughputConfig:
        data = self.get_batch_and_avg_cost(llm,
                                           tp_size,
                                           prefill_mps,
                                           decoding_mps,
                                           llm.avg_input_len,
                                           llm.avg_output_len)
        ## data related to tp does not exist
        if not data.ttft or not data.throughput:
            return None
        dp_size=1
        the_rate=llm.rate
        good_to_go=False
        can_satisfy=False
        while not good_to_go:
            # maximum batch size which satisfies the SLO
            slo_bs = np.floor(linear_interp(llm.slo, data.ttft, data.ttft_batch_size))
            # minumum batch size which satisfies the request rate
            rate_bs = np.ceil(linear_interp(the_rate, data.throughput, data.throughput_batch_size))
            if cache_size is None:
                cache_size = estimate_kv_cache_size(self.memory_per_gpu * tp_size, [llm])
            cache_bs = np.floor_divide(cache_size, llm.kv_cache_size_per_seq()) # cache_bs : max batch size related kv cache
            max_bs = min(cache_bs, slo_bs)

            can_satisfy = (rate_bs <= max_bs)
            cur_bs = min(rate_bs, max_bs)

            # cases where rate is very small, but can be satisfied
            if can_satisfy and rate_bs <= 0.0:
                cur_bs=1
                rate_bs=1

            ttft = linear_interp(cur_bs, data.ttft_batch_size, data.ttft)
            dec_lat = linear_interp(cur_bs, data.decoding_latency_batch_size, data.decoding_latency)
            tpt_cal = cur_bs / (ttft + dec_lat)

            if can_satisfy:
                good_to_go=True
            else:
                dp_size = dp_size +1
                the_rate = llm.rate // dp_size
            # check whether this is end of the line (this tp setting has no way to satisfy the rate)
            if the_rate <= 0.0:
                break

        return ThroughputConfig(can_satisfy, tp_size, dp_size, decoding_mps, int(rate_bs), slo_bs,
                                tpt_cal, ttft, dec_lat, data)
