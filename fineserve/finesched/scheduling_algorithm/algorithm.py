import abc
import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from fineserve.logger import get_logger
from fineserve.utils.workload_utils import Request

logger = get_logger()


@dataclass
class ScheduleResult:
    scheduled: list[Request] = field(default_factory=list)
    dropped: list[Request] = field(default_factory=list)


class SchedulingAlgorithm(abc.ABC):

    def __init__(self):
        pass

    @classmethod
    def schedule(cls, sched: "FineServeScheduler", model, *args,
                 **kwargs) -> ScheduleResult:
        st = time.time()
        sched_rlt = cls.algorithm(sched, model, *args,
                                  num_enqueued_seqs_threshold=3, **kwargs)
        et = time.time()

        schedule_time = et - st
        num_running_seqs = curr_num_running_seqs(sched, model)
        num_enqueued_seqs = curr_num_enqueued_seqs(sched, model)
        num_batched_tokens = curr_num_batched_tokens(sched, model)
        sched._max_num_seqs = max(sched._max_num_seqs, num_running_seqs + num_enqueued_seqs)
        sched._max_num_batched_tokens = max(sched._max_num_batched_tokens, num_batched_tokens)
        if sched_rlt.scheduled or sched_rlt.dropped:
            logger.info(f"running: {num_running_seqs}")
            logger.info(f"enqueued: {num_enqueued_seqs}")
            logger.info(f"num_batched_tokens: {num_batched_tokens}")
            logger.info(f"scheduled this step: {[r.idx for r in sched_rlt.scheduled]}")
            logger.info(f"dropped this step: {[r.idx for r in sched_rlt.dropped]}")
            logger.info(f"schedule time: {schedule_time * 1000} ms")
        return sched_rlt

    @classmethod
    @abc.abstractmethod
    def algorithm(cls, sched: "FineServeScheduler", model,
                  *args, **kwargs) -> ScheduleResult:
        ...


class FCFS(SchedulingAlgorithm):

    def __init__(self):
        super().__init__()

    @classmethod
    def algorithm(cls, sched: "FineServeScheduler", model, *args,
                  num_enqueued_seqs_threshold=3, sort_fn=None, **kwargs) -> ScheduleResult:
        ## SBCHOI
        max_num_seqs = sched.max_num_seqs[model]
        max_num_batched_tokens = sched.fineserve_config.max_num_batched_tokens
        waiting_queue = sched.waiting[model]
        if sort_fn is None:
            sort_fn = lambda r: r.arrival_time
        waiting_queue.sort(key=sort_fn)
        num_running_seqs = curr_num_running_seqs(sched, model)
        num_enqueued_seqs = curr_num_enqueued_seqs(sched, model)
        num_batched_tokens = curr_num_batched_tokens(sched, model)
        num_total_seqs = num_running_seqs + num_enqueued_seqs
        sched_rlt = ScheduleResult()

        if not waiting_queue:
            return sched_rlt
        if num_total_seqs > max_num_seqs:
            return sched_rlt
        if num_batched_tokens > max_num_batched_tokens:
            return sched_rlt
        if (num_enqueued_seqs_threshold is not None and
                num_enqueued_seqs > num_enqueued_seqs_threshold):
            return sched_rlt

        scheduled = []
        for request in waiting_queue:
            num_prompt_tokens = request.data[1]
            num_batched_tokens += num_prompt_tokens
            num_total_seqs += 1
            scheduled.append(request)

            if num_batched_tokens > max_num_batched_tokens:
                break

            if num_total_seqs > max_num_seqs:
                break

        sched_rlt.scheduled = scheduled
        return sched_rlt


class ShortestJobFirst(SchedulingAlgorithm):

    def __init__(self):
        super().__init__()

    @classmethod
    def algorithm(cls, sched: "FineServeScheduler", model, *args,
                  **kwargs) -> ScheduleResult:
        sort_fn = lambda r: r.data[1]   # sort by input length
        return FCFS.algorithm(sched, model, *args,
                              sort_fn=sort_fn, **kwargs)


class LongestSlowdownFirst(SchedulingAlgorithm):

    def __init__(self):
        super().__init__()

    @classmethod
    def algorithm(cls, sched: "FineServeScheduler", model, *args,
                  **kwargs) -> ScheduleResult:
        curr_time = sched.get_tick()
        num_batched_tokens = curr_num_batched_tokens(sched, model)
        sort_fn = lambda r: (-waiting_time(r, curr_time) /
                             (num_batched_tokens + r.data[1]))   # sort reversely by slowdown
        # sort_fn = lambda r: (-waiting_time(r, curr_time) /
        #                      r.data[1])   # sort reversely by slowdown
        return FCFS.algorithm(sched, model, *args,
                              sort_fn=sort_fn, **kwargs)


class SloAwareAdaptiveBatching(SchedulingAlgorithm):

    def __init__(self):
        super().__init__()

    @classmethod
    def deadlines(cls, waiting_queue, slo=None):
        return np.array([deadline(r, slo) for r in waiting_queue])

    @classmethod
    def future_num_batched_tokens(cls, waiting_queue, cum=True):
        if cum:
            return cls.future_num_batched_tokens_cum(waiting_queue)
        else:
            return cls.future_num_batched_tokens_ind(waiting_queue)

    @classmethod
    def future_num_batched_tokens_cum(cls, waiting_queue):
        return np.sum([r.data[1] for r in waiting_queue])

    @classmethod
    def future_num_batched_tokens_ind(cls, waiting_queue):
        return np.array([r.data[1] for r in waiting_queue])

    @classmethod
    def met_deadlines(cls, sched: "FineServeScheduler", model, waiting_queue,
                      slo=None, curr=None, cum=True):
        if curr is None:
            curr = sched.get_tick()
        _ttft_per_token = sched._ttft_per_token[model]
        deadlines = cls.deadlines(waiting_queue, slo=slo)
        num_batched_tokens = curr_num_batched_tokens(sched, model)
        future_num_batched_tokens = (
                cls.future_num_batched_tokens(waiting_queue, cum=cum) +
                num_batched_tokens
        )
        ttft_b = _ttft_per_token * future_num_batched_tokens
        resp_times = curr + ttft_b
        return resp_times < deadlines

    @classmethod
    def algorithm(cls, sched: "FineServeScheduler", model, *args,
                  num_enqueued_seqs_threshold=3, **kwargs) -> ScheduleResult:
        curr_time = sched.get_tick()
        llm = sched._served_llms[model]
        max_num_seqs = sched.max_num_seqs[model]
        max_num_batched_tokens = sched.fineserve_config.max_num_batched_tokens
        max_num_partial_prefills = sched.fineserve_config.max_num_partial_prefills
        waiting_queue = sched.waiting[model]
        num_running_seqs = curr_num_running_seqs(sched, model)
        num_enqueued_seqs = curr_num_enqueued_seqs(sched, model)
        num_batched_tokens = curr_num_batched_tokens(sched, model)
        num_total_seqs = num_running_seqs + num_enqueued_seqs
        sched_rlt = ScheduleResult()

        # control
        _curr_time = curr_time
        _running = num_total_seqs
        _num_batched_tokens = num_batched_tokens
        _num_enqueued_seqs_threshold = num_enqueued_seqs_threshold

        if not waiting_queue:
            return sched_rlt
        if num_total_seqs > max_num_seqs:
            return sched_rlt
        if num_batched_tokens > max_num_batched_tokens:
            return sched_rlt
        if num_enqueued_seqs > _num_enqueued_seqs_threshold:
            return sched_rlt

        st = time.time()
        slo = llm.slo
        met_deadlines = cls.met_deadlines(sched, model, waiting_queue,
                                          slo=slo, curr=_curr_time, cum=False)

        requests_attain_slo = [r for r, s in zip(waiting_queue, met_deadlines) if s]
        logger.debug(f"waiting requests attained slo: {[r.idx for r in requests_attain_slo]}")
        requests_not_attain_slo = [r for r, s in zip(waiting_queue, met_deadlines) if not s]
        logger.debug(f"waiting requests not attained slo: {[r.idx for r in requests_not_attain_slo]}")

        et = time.time()
        logger.debug(f"done | slo attained reqs: {(et - st) * 1000} ms")
        st = time.time()

        sched_rlt.dropped = requests_not_attain_slo
        if not requests_attain_slo:
            logger.debug(f"there is no waiting requests attained slo")
            return sched_rlt

        # batch size 탐색
        waiting_queue = requests_attain_slo

        et = time.time()
        logger.debug(f"done | init data: {(et - st) * 1000} ms")
        st = time.time()

        requests = []
        for r in waiting_queue:
            heapq.heappush(requests, (-r.data[1], r))

        while waiting_queue:
            met_deadlines = cls.met_deadlines(sched, model, waiting_queue,
                                              slo=slo, curr=_curr_time, cum=True)
            if np.all(met_deadlines):
                break
            _, r = heapq.heappop(requests)
            waiting_queue.remove(r)

        if not waiting_queue:
            raise RuntimeError("something was wrong: waiting_queue must not be empty")

        et = time.time()
        logger.debug(f"done | schedule: {(et - st) * 1000} ms")
        st = time.time()

        scheduled = []
        ns = 0
        bt = 0
        for r in waiting_queue:
            scheduled.append(r)
            ns += 1
            bt += r.data[1]
            if ns + _running > max_num_seqs:
                break
            if bt + _num_batched_tokens > max_num_batched_tokens:
                break

        sched_rlt.scheduled = scheduled

        logger.debug(f"after schedule: ")
        logger.debug(f">> running: {num_running_seqs}")
        logger.debug(f">> enqueued: {num_enqueued_seqs + len(scheduled)}")
        logger.debug(f">> num_total_seqs: {num_total_seqs + len(scheduled)}")
        logger.debug(f">> num_batched_tokens: {num_batched_tokens + sum([r.data[1] for r in scheduled])}")
        return sched_rlt


class Prism(SloAwareAdaptiveBatching):

    def __init__(self):
        super().__init__()

    @classmethod
    def deadlines(cls, waiting_queue, slo=None):
        return np.array([deadline(r, slo) for r in waiting_queue])

    @classmethod
    def future_num_batched_tokens_cum(cls, waiting_queue):
        return np.cumulative_sum([r.data[1] for r in waiting_queue])


def curr_num_enqueued_seqs(sched: "FineServeScheduler", model):
    return len(sched.in_prefill[model])


def curr_num_running_seqs(sched: "FineServeScheduler", model):
    return len(sched.executing[model])


def curr_num_batched_tokens(sched: "FineServeScheduler", model):
    decoding_tokens = curr_num_running_seqs(sched, model)
    num_batched_tokens = sum([sched.workload.requests[i].data[1]
                              for i in sched.in_prefill[model]]) + decoding_tokens
    return num_batched_tokens


def waiting_time(r: Request, curr_time):
    return curr_time - r.arrival_time


def deadline(r: Request, req_slo=None):
    slo = r.slo if req_slo is None else req_slo
    return r.arrival_time + slo


def get_scheduling_algorithm(key: str) -> SchedulingAlgorithm:
    if key not in __ALL_SCHEDULING_ALGORITHMS:
        raise RuntimeError(f"Unknown schedule approach {key})")
    return __ALL_SCHEDULING_ALGORITHMS[key]


__ALL_SCHEDULING_ALGORITHMS = {
    "fcfs": FCFS,
    "sjf": ShortestJobFirst,
    "lsf": LongestSlowdownFirst,
    "saab": SloAwareAdaptiveBatching,
    "prism": Prism,
}