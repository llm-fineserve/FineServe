from collections import namedtuple
from enum import Enum

DTypeInfo = namedtuple("DTypeInfo", ["name", "bit"])
QuantInfo = namedtuple("QuantFormatInfo",
                       ["name", "vllm_option", "weights", "activation", "kvcache"])


class DType(Enum):
    fp16 = DTypeInfo("fp16", 16)
    fp8 = DTypeInfo("fp8", 8)
    int4 = DTypeInfo("int4", 4)


def get_dtype(dtype: str) -> DTypeInfo:
    try:
        d = DType[dtype]
        return d.value
    except KeyError as e:
        raise e


class Quant(Enum):
    org = QuantInfo("org",
                    None, # None = follow options specified in config.json in the model directory
                    get_dtype("fp16"),
                    get_dtype("fp16"),
                    get_dtype("fp16"))
    fp8 = QuantInfo("fp8",
                    None,
                    get_dtype("fp8"),
                    get_dtype("fp8"),
                    get_dtype("fp16"))
    fp8_kv8 = QuantInfo("fp8_kv8",
                        "--kv-cache-dtype fp8 --dtype auto ",
                        get_dtype("fp8"),
                        get_dtype("fp8"),
                        get_dtype("fp8"))
    awq = QuantInfo("awq",
                    None,
                    get_dtype("int4"),
                    get_dtype("fp16"),
                    get_dtype("fp16"))
    qoq = QuantInfo("qoq",
                    "--quantization qoq ",
                    get_dtype("int4"),
                    get_dtype("fp8"),
                    get_dtype("int4"))


def get_quant(qformat: str) -> QuantInfo:
    try:
        q = Quant[qformat]
        return q.value
    except KeyError as e:
        raise e
