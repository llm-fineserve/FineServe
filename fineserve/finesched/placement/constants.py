import re
from collections import namedtuple

METRIC_LOG_FORMAT = re.compile(r"i(?P<input_len>\d+)-o(?P<output_len>\d+)-latency_profile*")

# used in placement
SLO_BUFFER_S = 0.004    # BUFFER due to scheduler overhead
SLO_SCALE = 50          # ttft_slo = ttft_latency * slo_scale
MEMORY_PER_GPU = 80     # GB
NUM_TOKENS_PER_BLOCK = 16

# profile data columns' name
MODEL = "model"
QFORMAT = "qformat"
TP_SIZE = "tp"
MPS_PERCENTAGE = "mps_percentage"
BATCH_SIZE = "batch_size"
INPUT_LEN = "input_len"
OUTPUT_LEN = "output_len"
SEQUENCE_LEN = "seq_len"
PREFILL_TIME = "prefill_time"
DECODE_TIME = "decode_time"
TTFT = "ttft"
TPOT = "tpot"
E2E = "e2e"

PROFILE_DATA_COLUMNS_CATEGORICAL = [
    MODEL,
    QFORMAT,
    TP_SIZE,
    MPS_PERCENTAGE,
    BATCH_SIZE,
    INPUT_LEN,
    OUTPUT_LEN,
    SEQUENCE_LEN,
]
PROFILE_DATA_COLUMNS_NON_CATEGORICAL = [
    PREFILL_TIME,
    DECODE_TIME,
    TTFT,
    TPOT,
    E2E,
]
PROFILE_DATA_COLUMNS = PROFILE_DATA_COLUMNS_CATEGORICAL + PROFILE_DATA_COLUMNS_NON_CATEGORICAL

ThroughputConfig = namedtuple("ThroughputConfig",
                              ["satisfied", "tp_size", "dp_size", "mps_percentage", "rate_min_batch_size", "slo_max_batch_size",
                               "throughput", "prefill_latency", "decoding_latency", "cost_data"])

CostData = namedtuple("CostData",
                      ["model_name", "tp_size",
                       "prefill_mps", "decoding_mps",
                       "input_len", "output_len",
                       "ttft", "ttft_batch_size",
                       "decoding_latency", "decoding_latency_batch_size",
                       "throughput", "throughput_batch_size"])
