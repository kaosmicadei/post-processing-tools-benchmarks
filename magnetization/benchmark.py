from collections import Counter
from datetime import datetime
import json
import numpy as np

import magnetization as mg


keys = [f"{x:016b}" for x in range(2**16)]
values = map(int, np.random.uniform(0, 200, 2**16))
samples = dict(zip(keys, values))

json_str = json.dumps(samples, indent=2)


def total_magnetization(json_data: str) -> float:
    data = json.loads(json_data)
    total_magnetization = 0
    total_count = 0
    for key, value in data.items():
        count_bits = Counter(key)
        zeros, ones = count_bits["0"], count_bits["1"]
        total_magnetization += (ones - zeros) * value
        total_count += value
    return total_magnetization / total_count if total_count != 0 else 0


def benchmark(f, json_data: str, runs: int):
    start = datetime.now()
    for _ in range(runs):
        result = f(json_data)
    end = datetime.now()
    print(f"Result: {result}")
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    runs = 100
    benchmark(total_magnetization, json_str, runs)
    benchmark(mg.total_magnetization, json_str, runs)
