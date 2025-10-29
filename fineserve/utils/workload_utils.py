"""Workload definition
Borrowed from https://github.com/alpa-projects/mms/blob/main/alpa_serve/simulator/workload.py
"""
from transformers import AutoTokenizer
import argparse
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict, namedtuple
import dataclasses
import json
import random
import numpy as np
from typing import Any, List, Tuple, Sequence, Dict, Optional

import numpy as np
import yaml
import pickle
import os

from fineserve.logger import get_logger

logger = get_logger()

## GLOBAL DEFAULT CONSTANTS
DEFAULT_WARMUP = 10
DEFAULT_DATASET_PATH = "/home/jovyan/shared-dir/hpclab/hub//ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_TOKENIZER_PATH = "/home/jovyan/shared-dir/hpclab/hub/llama-3.1-8b"
eps = 1e-6


def to_str_round(x: Any, decimal: int = 6):
    """logger.info a python object but round all floating point numbers."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        tmp_str = ", ".join([to_str_round(y, decimal=decimal) for y in x])
        return "[" + tmp_str + "]"
    if isinstance(x, dict):
        return str({k: to_str_round(v, decimal=decimal) for k, v in x.items()})
    if isinstance(x, (int, np.int32, np.int64)):
        return str(x)
    if isinstance(x, (float, np.float32, np.float64)):
        format_str = f"%.{decimal}f"
        return format_str % x
    if x is None:
        return str(x)
    raise ValueError("Invalid value: " + str(x))


@dataclasses.dataclass
class Request:
    """A single request."""
    model_name: str
    slo: Optional[float]
    idx: int
    time_stamp: Dict  # debug only
    data: Any
    arrival_time: float = None  # This will be filled later
    submit_time: float = None  # This will be filled later
    prefill_end_time: float = None  # This will be filled later
    decode_submit_time: float = None  # This will be filled later
    end_time: float = None  # This will be filled later
    is_prefill: bool = True
    output: str = None
    output_idx: int = 0
    output_tokens: Optional[List[int]] = None
    dropped: bool = False
    aborted: bool = False

    def __lt__(self, other):
        return self.idx < other.idx

    def __hash__(self):
        return hash(self.idx)


PerModelStatsResult = namedtuple(
    "PerModelStatsResult",
    ("name", "num_requests", "goodput", "throughput", "latency_mean",
     "latency_std", "latency_p90", "latency_p99", "latency", "request_starts",
     "request_finishes"))

PerDeviceStatsResult = namedtuple("PerDeviceStatsResult", ("num_requests",))


@dataclasses.dataclass
class StatsResult:
    per_model_stats: List[PerModelStatsResult]
    group_num_requests: List[int]
    goodput: float
    latency_mean: float
    num_requests: int
    request_rate: float


class ArrivalProcess(ABC):

    @abstractmethod
    def rate(self):
        """Return the mean arrival rate."""
        raise NotImplementedError()

    @abstractmethod
    def cv(self):
        """Return the coefficient of variation of the gap between
        the requests."""
        raise NotImplementedError()

    @abstractmethod
    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def generate_workload(self,
                          model_name: str,
                          start: float,
                          duration: float,
                          slo: Optional[float] = None,
                          seed: int = 0):
        """Generate a workload with the arrival process.

        Args:
            model_name (str): Name of the model.
            start (float): The start time of the workload.
            duration (float): The duration of the workload.
            slo (Optional[float]): The service level objective of each model.
            seed (int): The random seed.
        """
        raise NotImplementedError()

    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"rate={self.rate()}, "
                f"cv={self.cv()})")

    def params(self):
        return self.rate(), self.cv()


class DeterministicProcess(ArrivalProcess):
    """Deterministic arrival process."""

    def __init__(self, arrival_rate: float):
        """Create a deterministic arrival process.

        Args:
            arrival_rate (float): The arrival rate of the process. The gap
                between the requests is 1 / arrival_rate seconds.
        """
        self.rate_ = arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return 0

    def generate_arrivals(self,
                          start: float,
                          duration: float,
                          num_requests: Optional[int] = None,
                          seed: int = 0):
        pass

    def generate_workload(self,
                          model_name: str,
                          start: float,
                          duration: float,
                          num_requests: Optional[int] = None,
                          slo: Optional[float] = None,
                          seed: int = 0):
        if num_requests is None:
            n_requests = max(int(duration * self.rate_), 1)
        else:
            n_requests = num_requests
        interval = 1 / self.rate_
        ticks = [start + i * interval for i in range(n_requests)]
        return Workload(
            ticks,
            [Request(model_name, slo, i, {}, None) for i in range(n_requests)])


class GammaProcess(ArrivalProcess):
    """Gamma arrival process."""

    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.

        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_

    def generate_arrivals(self,
                          start: float,
                          duration: float,
                          num_requests: Optional[int] = None,
                          seed: int = 0):
        np.random.seed(seed)

        if num_requests is None:
            batch_size = max(int(self.rate_ * duration * 1.2), 1)
        else:
            batch_size = num_requests
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        pt = 0

        ticks = []
        cur = start + intervals[0]
        end = start + duration
        while cur < end:
            ticks.append(cur)

            pt += 1
            if pt >= batch_size:
                if num_requests is not None:
                    break
                intervals = np.random.gamma(self.shape,
                                            self.scale,
                                            size=batch_size)
                pt = 0

            cur += intervals[pt]

        return ticks

    def generate_workload(self,
                          model_name: str,
                          start: float,
                          duration: float,
                          num_requests: Optional[int] = None,
                          slo: Optional[float] = None,
                          seed: int = 0):
        ticks = self.generate_arrivals(start, duration, num_requests, seed)
        return Workload(
            ticks,
            [Request(model_name, slo, i, {}, None) for i in range(len(ticks))])


class PoissonProcess(GammaProcess):
    """Poisson arrival process."""

    def __init__(self, arrival_rate: float):
        """Initialize a Poisson arrival process.

        Args:
            arrival_rate: The mean arrival rate.
        """
        super().__init__(arrival_rate, 1)


class Workload:
    """A sorted list of requests."""

    def __init__(self,
                 arrivals: List[float],
                 requests: List[Request],
                 workload_infos: Optional[Dict[str, Any]] = None):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests
        self.workload_infos = workload_infos

        self.enable_simulator_cache = False
        self.cached_data = None

        if len(self.arrivals) > 1:
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self.rate = 1 / (np.mean(intervals) + eps)
            self.cv = np.std(intervals) * self.rate
        else:
            self.rate = 0
            self.cv = 0

    @staticmethod
    def from_workload_file(workload_file: str):
        with open(workload_file) as f:
            workload_json = json.load(f)
        arrivals = workload_json["arrivals"]
        requests = [Request(**r) for r in workload_json["requests"]]
        workload_infos = workload_json.get("info", None)
        # convert to numpy array
        for req in requests:
            if isinstance(req.data[0], list):
                req.data = (np.array(req.data[0]), req.data[1], req.data[2])
        return Workload(arrivals, requests, workload_infos)

    def split_round_robin(self, number: int):
        rets = []
        for i in range(number):
            rets.append(self[i::number])
        return rets

    def split_time_interval(self, interval: float):
        if len(self.arrivals) < 1:
            return []

        ws = []
        start_i = 0
        start_time = self.arrivals[start_i]
        for i in range(len(self.arrivals)):
            if self.arrivals[i] > start_time + interval:
                ws.append(self[start_i:i])
                start_i = i
                start_time = self.arrivals[i]

        ws.append(self[start_i:])
        return ws

    def num_model_requests(self, model_name: str):
        return len([r for r in self.requests if r.model_name == model_name])

    def split_by_model(self, model_name: str):
        if len(self.arrivals) < 1:
            return []

        arrivals = []
        requests = []
        workload_infos = self.workload_infos
        for i in range(len(self.arrivals)):
            if self.requests[i].model_name == model_name:
                req = deepcopy(self.requests[i])
                req.idx = len(arrivals)
                arrivals.append(self.arrivals[i])
                requests.append(req)

        return Workload(arrivals, requests, workload_infos)

    def split_by_models(self, models: List[str]):
        if len(self.arrivals) < 1:
            return []

        arrivals = []
        requests = []
        workload_infos = self.workload_infos
        for i in range(len(self.arrivals)):
            if self.requests[i].model_name in models:
                req = deepcopy(self.requests[i])
                req.idx = len(arrivals)
                arrivals.append(self.arrivals[i])
                requests.append(req)

        return Workload(arrivals, requests, workload_infos)

    @classmethod
    def empty(cls):
        return cls([], [])

    @classmethod
    def merge(cls, *args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].idx = i

        return cls(arrivals, requests)

    def __getitem__(self, key):
        if isinstance(key, slice):
            arrivals = self.arrivals.__getitem__(key)
            requests = self.requests.__getitem__(key)
            return Workload(arrivals, requests)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return Workload.merge(self, other)

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (f"Workload(len={len(self)}, "
                f"rate={self.rate:.2f}, "
                f"CV={self.cv:.2f}, "
                f"tstamps={to_str_round(self.arrivals[:20])} ...)")


def sample_requests(
        num_requests: int,
        tokenizer: AutoTokenizer,
        dataset_path: str,
        seqlen_distribution: Optional[Tuple[int, int]] = None,
        tokenized_cache_path: str = None,
) -> List[Tuple[list[int], int, int]]:
    '''
    return: (encoded_prompt, intput_len, output_len)
    '''
    if tokenized_cache_path and os.path.exists(tokenized_cache_path):
        with open(tokenized_cache_path, "rb") as fp:
            tokenized_dataset: list[tuple[str, list[int],
            int]] = pickle.load(fp)
    else:
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data["conversations"][0]["value"],
                    data["conversations"][1]["value"]) for data in dataset]

        # Tokenize the prompts and completions.
        prompts: list[str] = [prompt for prompt, _ in dataset]
        prompt_token_ids: list[list[int]] = tokenizer(prompts, truncation=True).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append(
                (prompts[i], prompt_token_ids[i], output_len))
        if tokenized_cache_path:
            os.makedirs(os.path.dirname(tokenized_cache_path), exist_ok=True)
            with open(tokenized_cache_path, "wb") as fp:
                pickle.dump(tokenized_dataset, fp)

    min_seq_len = 0
    max_seq_len = np.inf
    if seqlen_distribution is not None:
        min_seq_len, max_seq_len = seqlen_distribution

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_id, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_id)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 2048 or prompt_len + output_len > 4096:
            # Prune too long sequences.
            continue
        total_len = prompt_len + output_len
        if total_len < min_seq_len or total_len > max_seq_len:
            continue
        filtered_dataset.append((prompt_token_id, prompt_len, output_len))

    # Sample the requests.
    if num_requests > len(filtered_dataset):
        sampled_requests = random.choices(filtered_dataset, k=num_requests)
    else:
        sampled_requests = random.sample(filtered_dataset, k=num_requests)
    return sampled_requests


def generate_workload_requests(num_requests: int,
                               prompt_len: int,
                               output_len: int,
                               tokenizer: AutoTokenizer,
                               distribution: str = "fixed"):
    if distribution == "fixed":
        prompt_lens = [prompt_len] * num_requests
        output_lens = [output_len] * num_requests
    elif distribution == "uniform":
        prompt_lens = np.random.uniform(prompt_len // 2,
                                        prompt_len + prompt_len // 2,
                                        size=num_requests)
        output_lens = np.random.uniform(output_len // 2,
                                        output_len + output_len // 2,
                                        size=num_requests)
    request_datasets = []
    for i in range(num_requests):
        cur_prompt_len = int(prompt_lens[i])
        cur_output_len = int(output_lens[i])
        prompt = np.random.randint(0, 24000, size=cur_prompt_len).tolist()
        prompt = tokenizer.decode(prompt)
        request_datasets.append((prompt, cur_prompt_len, cur_output_len))
    return request_datasets


def get_workload(
        models: List[str],
        arrival_rates: List[float],
        model_paths: List[str],
        start: int,
        duration: int,
        distribution: str = "poisson",
        seed: int = 0,
        dataset_path: str = DEFAULT_DATASET_PATH,
        num_requests: Optional[List[int]] = None,
        sampled_requests=None,
        prompt_distribution=None,
        prompt_lens=None,
        output_lens=None,
        tokenized_cache_path: str = None,
) -> Workload:
    """Generate a workload with the given models and arrival rates."""
    workloads = []
    for i, (model_name, arrival_rate,
            nreq) in enumerate(zip(models, arrival_rates, num_requests)):
        assert arrival_rate >= 0
        if distribution == "poisson":
            process = PoissonProcess
        elif distribution == "uniform":
            process = DeterministicProcess
        else:
            raise ValueError(f"Unknown arrival process: {distribution}")
        w = process(arrival_rate).generate_workload(model_name,
                                                    start=start,
                                                    duration=duration,
                                                    num_requests=nreq,
                                                    seed=seed)
        if sampled_requests is not None:
            for req_idx in range(len(w)):
                w.requests[req_idx].data = sampled_requests[i][req_idx]

        if prompt_lens is not None and output_lens is not None:
            if isinstance(prompt_lens, int):
                prompt_len = prompt_lens
                output_len = output_lens
            else:
                prompt_len = prompt_lens[i]
                output_len = output_lens[i]
            #logger.info(f"Replace {model_name} workload with prompt {prompt_len} "
            #      f"output {output_len} distribution {prompt_distribution}")
            model_path = None
            if model_paths is not None:
                model_path = model_paths[i]
            w = replace_long_workloads(w,
                                       model_path,
                                       prompt_len,
                                       output_len,
                                       distribution=prompt_distribution)
        workloads.append(w)
        seed += random.randint(1, 100)

    workload = Workload.merge(*workloads)

    if sampled_requests is None:
        # sample requests to fill the workload
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)
        requests = sample_requests(len(workload),
                                   tokenizer,
                                   dataset_path,
                                   tokenized_cache_path=tokenized_cache_path)

        # fill requests
        for i, request in enumerate(requests):
            # # encode the prompt
            # prompt_tokens = tokenizer(request[0]).input_ids
            workload.requests[i].data = request
    return workload


def replace_long_workloads(workload,
                           tokenizer_path,
                           prompt_len,
                           output_len,
                           distribution: str = "fixed"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    num_requests = len(workload)
    workload_reqs = generate_workload_requests(num_requests,
                                               prompt_len,
                                               output_len,
                                               tokenizer,
                                               distribution=distribution)

    for i in range(num_requests):
        workload.requests[i].data = workload_reqs[i]
    return workload


def generate_workload(
        workload_infos,
        output_file,
        model_paths=None,
        num_requests=1500,
        sampled_requests=None,
        duration=2000,
        distribution="poisson",
        prompt_distribution=None,
        prompt_len=0,
        output_len=0,
        dataset=None,
):
    start = 0
    logger.info(workload_infos)
    if isinstance(num_requests, int):
        num_requests = [num_requests] * len(workload_infos)
    models = [model for model, _ in workload_infos]
    arrival_rates = [rate for _, rate in workload_infos]
    w = get_workload(models,
                     arrival_rates,
                     model_paths,
                     start,
                     duration,
                     dataset_path=dataset,
                     num_requests=num_requests,
                     sampled_requests=sampled_requests,
                     prompt_distribution=prompt_distribution,
                     prompt_lens=prompt_len,
                     output_lens=output_len)

    workload_num_requests = []
    for model, rate in workload_infos:
        workload_num_requests.append(w.num_model_requests(model))
    workload_json = {
        "info": {
            "rates": workload_infos,
            "start": start,
            "duration": duration,
            "num_requests": workload_num_requests,
            "distribution": distribution,
            "prompt_len": prompt_len,
            "output_len": output_len,
        },
        "arrivals": w.arrivals.tolist(),
        "requests": [dataclasses.asdict(r) for r in w.requests]
    }

    logger.info(f"Save workload to {output_file}")
    with open(output_file, "w") as f:
        json.dump(workload_json, f)


def get_workloads_info_from_yaml(models_yaml: str) -> List[Tuple[str, float, str]]:
    with open(models_yaml, "r") as fp:
        model_group = yaml.safe_load(fp)

    models = model_group["models"]
    ret_list = []
    model_id = [model["name"] for model in models]
    rate_list = [model["rate"] for model in models]
    model_path = [model["model"] for model in models]
    for i in range(len(model_id)):
        ret_list.append((model_id[i], rate_list[i], model_path[i]))
    return ret_list
    # return [(id, rate, model_path) for id, rate, model_path in zip(model_id, rate_list, model_path)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uneven_distribution", type=bool, default=False)
    #parser.add_argument("--workload_info_from_yaml", type=bool, default=False)
    parser.add_argument("--dataset-source", type=str,
                        default=DEFAULT_DATASET_PATH,
                        help="the dataset source")
    parser.add_argument("--model-yaml", type=str, default="examples/finserve-1models.yaml",
                        help="the model yaml to generate the workload, refer to `examples/fineserve-1models.yaml`")
    parser.add_argument("--output-file", type=str, default="output.json", help="the output json file")
    parser.add_argument("--overwrite-rate", type=float, default=None, help="the rate to be overwritten, all models will have this rate (optional)")
    parser.add_argument("--nreq", type=int, default=200, help="the number of requests to overwrite (default=200)")
    parser.add_argument("--duration-min", type=int, default=10,
                        help="duration in minutes, this will generate approximately arg * 60 requests (default=10)")
    parser.add_argument("--rate-multiply", type=float, default=None, help="the number to be multiple to all rates")
    args = parser.parse_args()

    dataset = args.dataset_source
    num_requests = args.nreq
    duration = args.duration_min * 60 ## change from minutes to seconds
    output_file = args.output_file

    distribution = "poisson"
    prompt_distribution = "uniform"
    prompt_len = 768
    output_len = 256

    uneven_distribution = args.uneven_distribution

    workload_infos = get_workloads_info_from_yaml(args.model_yaml)
    if args.overwrite_rate:
        logger.info(f"all rate will be replaced with { args.overwrite_rate}")
        rate_dist = [args.overwrite_rate for v in workload_infos]
    else:
        rate_dist = [v[1] for v in workload_infos]

    if args.rate_multiply:
        logger.info(f"All rates will be multiplied by {args.rate_multiply}")
        rate_dist = [rate*args.rate_mutliply for rate in rate_dist]
    #logger.info(f"final rate_dist: {rate_dist}")
    num_models = len(workload_infos)
    names = [v[0] for v in workload_infos]

    # first sample requests for each model
    max_rate = max(rate_dist)
    sampled_requests = []
    model_paths = []

    for i in range(len(rate_dist)):
        model_path = workload_infos[i][2]
        model_paths.append(model_path)

    ## cap by comparing given num_requests and max_rate * duration
    capped_num_requests = min(num_requests, int(max_rate * duration))
    num_requests_dist=[]
    dispatch_duration = capped_num_requests / max_rate * 1.02
    for rate in rate_dist:

        num_requests_dist.append(max(int(rate * dispatch_duration), 1))

    updated_workload_infos = [(names[model_id], rate)
                              for model_id, rate in enumerate(rate_dist)]
    logger.info(f"Sample total {sum(num_requests_dist)} requests")
    logger.info(f"updated workload_infos: {updated_workload_infos} ")

    logger.info(f"final num_requests_dist: {num_requests_dist}")
    generate_workload(updated_workload_infos,
                      output_file,
                      model_paths=model_paths,
                      num_requests=num_requests_dist,
                      duration=duration,
                      distribution=distribution,
                      prompt_distribution=prompt_distribution,
                      prompt_len=prompt_len,
                      output_len=output_len,
                      dataset=dataset)