import argparse
import os

import pandas as pd

from fineserve.finesched.placement.constants import *


class ProfileData:
    def __init__(self):
        self.data = pd.DataFrame()
        self.column_dict = {
            "model": MODEL,
            "qformat": QFORMAT,
            "tp": TP_SIZE,
            "mps_percentage": MPS_PERCENTAGE,
            "batch_size": BATCH_SIZE,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "seq_len": SEQUENCE_LEN,
            "prefill_time": PREFILL_TIME,
            "decode_time": DECODE_TIME,
            "ttft": TTFT,
            "tpot": TPOT,
            "e2e": E2E,
        }
        self.all_categories = {}

    def query(self, q):
        return self.data.query(q, engine="python")

    def get(self,
            model: str = None,
            qformat: str = None,
            tp: int = None,
            mps_percentage: int = None,
            batch_size: int = None,
            input_len: int = None,
            output_len: int = None,
            seq_len: int = None,
            verbose: bool = False):
        q = self.build_query(
            model=model,
            qformat=qformat,
            tp=tp,
            mps_percentage=mps_percentage,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            seq_len=seq_len
        )
        if verbose:
            print(f"query: {q}")
        return self.query(q)

    def build_query(self, **kwargs):
        def to_value(v):
            if isinstance(v, int):
                return str(v)
            else:
                return f"'{v}'"

        ql = [
            f"{self.column_dict[k]}=={to_value(v)}"
            for k, v in kwargs.items() if k in self.column_dict and v is not None
        ]
        q = " and ".join(ql)
        return q

    def read_cost_file(self, cost_file: str):
        df = pd.read_csv(cost_file)
        # df[PROFILE_DATA_COLUMNS_CATEGORY] = df[PROFILE_DATA_COLUMNS_CATEGORY].astype("category")
        df[PROFILE_DATA_COLUMNS_NON_CATEGORICAL] = df[PROFILE_DATA_COLUMNS_NON_CATEGORICAL].astype(float)
        self.data = df
        self.all_categories = {
            k: sorted(set(df[k])) for k in PROFILE_DATA_COLUMNS_CATEGORICAL
        }
        return df


def build_cost_file(cost_file: str, profile_log_dir: str):
    df = pd.DataFrame()
    for f in os.listdir(profile_log_dir):
        m = METRIC_LOG_FORMAT.match(f)
        if not m:
            continue

        tmp = pd.read_csv(os.path.join(profile_log_dir, f))
        tmp[INPUT_LEN] = int(m.group(INPUT_LEN))
        tmp[OUTPUT_LEN] = int(m.group(OUTPUT_LEN))
        tmp[SEQUENCE_LEN] = tmp[INPUT_LEN] + tmp[OUTPUT_LEN]
        df = pd.concat((df, tmp))

    df.rename(columns={c: c.replace("-", "_") for c in df.columns}, inplace=True)
    df = df[PROFILE_DATA_COLUMNS]
    df.to_csv(cost_file, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-file", type=str, default="examples/placement/cost.csv")
    parser.add_argument("--profile-log-dir", type=str, default="examples/placement/profile_log")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    build_cost_file(args.cost_file, args.profile_log_dir)
    pf = ProfileData()
    pf.read_cost_file(args.cost_file)

    print("####  Read Profile Data  ####")
    print(">> Head: ")
    print(pf.data.head(10))
    print(">> Dtypes: ")
    print(pf.data.dtypes)
    print(">> Describe: ")
    print(pf.data.describe())
    print()

    print("####  Query  ####")
    print(f">> Query: ")
    rtn = pf.get(qformat='fp8', tp=4, verbose=True)
    print(f">> Result: ")
    print(rtn.describe())
    print()

    print("####  Timeit  ####")
    import timeit
    num = 1000
    t1 = timeit.timeit(f"pf.get(qformat='fp8', tp=4)", globals=globals(),  number=num)
    print(f">> querying performance: {t1/num * 1000:.3f} ms")
