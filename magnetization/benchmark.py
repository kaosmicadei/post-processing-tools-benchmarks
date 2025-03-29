from collections import Counter
from datetime import datetime, timedelta
import json
import numpy as np

import magnetization as mg


keys = [f"{x:016b}" for x in range(2**16)]
values = map(int, np.random.uniform(0, 200, 2**16))
samples = dict(zip(keys, values))
sample_data = {
    "bitstring_size": 16,
    "counts": samples,
}

json_str = json.dumps(sample_data)


def total_magnetization(json_data: str) -> float:
    data = json.loads(json_data)
    total_mag = 0
    total_count = 0
    for key, value in data["counts"].items():
        count_bits = Counter(key)
        zeros, ones = count_bits["0"], count_bits["1"]
        total_mag += (ones - zeros) * value
        total_count += value
    return total_mag / total_count if total_count != 0 else 0


def benchmark(f, json_data: str, runs: int) -> timedelta:
    start = datetime.now()
    for _ in range(runs):
        f(json_data)
    end = datetime.now()
    print(f"Elapsed time: {end - start}")
    return end - start


if __name__ == "__main__":
    RUNS = 100
    print("Computing magnetization with pure Python...")
    d1 = benchmark(total_magnetization, json_str, RUNS)
    print("Computing magnetization with Rust...")
    d2 = benchmark(mg.total_magnetization, json_str, RUNS)
    print(f"Speedup: {d1 / d2:.2f}x")
