from datetime import datetime
import numpy as np
import confusion_matrix


def apply(m, data):
    N = int(np.log2(len(data)))
    res = np.reshape(data, [2] * N)
    for i in range(N):
        res = np.tensordot(m, res, axes=(1, i))
    return res.transpose().flatten()

# m = np.array([[1, 2], [3, 4]], dtype=np.float32)
# data = np.array(range(1, 9), dtype=np.float32)
# result1 = apply(m, data)

m = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
data = np.random.uniform(0, 1000, 2**25)
data /= np.sum(data)
data = np.array(data, dtype=np.float32)

start = datetime.now()
result1 = apply(m, data)
end = datetime.now()
d1 = end - start
print(result1)
print(f"Elapsed time: {d1}")


start = datetime.now()
result2 = confusion_matrix.apply(m.T, data)
end = datetime.now()
d2 = end - start
print(result2)
print(f"Elapsed time: {d2}")

print(f"Speedup: {d1 / d2}")
