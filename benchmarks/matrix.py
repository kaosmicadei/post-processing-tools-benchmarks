import sys
import logging
from datetime import datetime

from functools import reduce
import numpy as np
import tensor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Avoid using np.kron with large number of qubits since it uses a lot of memory.
KRON_LIMIT = 15


def apply(m, data):
    """
    Equivalente to `np.kron([m] * N) @ data`, but avoiding the memory overhead
    like np.kron.
    """
    N = int(np.log2(len(data)))
    res = np.reshape(data, [2] * N)
    for i in range(N):
        res = np.tensordot(m, res, axes=(1, i))
    return res.transpose().flatten()


# Example usage.
# m = np.array([[1, 2], [3, 4]], dtype=np.float32)
# data = np.array(range(1, 9), dtype=np.float32)


try:
    N = int(sys.argv[1])
except IndexError:
    N = 8
logger.info("Running for %d qubits", N)

# Example of confusing matrix.
m = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)

# Generate random data.
data = np.random.uniform(0, 1000, 2**N)
data /= np.sum(data)
data = np.array(data, dtype=np.float32)

# Using Kronecker product.
if N <= KRON_LIMIT:
    logger.info("Running with np.kron.")
    start = datetime.now()
    result0 = np.dot(reduce(np.kron, [m] * N), data)
    d0 = datetime.now() - start
    logger.info("np.kron: %s", d0)


# Using tensordot.
logger.info("Running with np.tensordot.")
start = datetime.now()
result1 = apply(m, data)
d1 = datetime.now() - start
if N <= KRON_LIMIT:
    assert np.allclose(result0, result1)
    logger.info("np.tensordot: %s (%.5fx np.kron)", d1, d1 / d0)
else:
    logger.info("np.tensordot: %s", d1)

# Using Rust implementation.
logger.info("Running with tensor.mat_vec_multiply.")
start = datetime.now()
result2 = tensor.mat_vec_multiply(m, data)
d2 = datetime.now() - start
if N <= KRON_LIMIT:
    assert np.allclose(result0, result2)
    logger.info("tensor.mat_vec_multiply: %s (%.5fx np.kron)", d2, d2 / d0)
else:
    assert np.allclose(result1, result2)
    logger.info("tensor.mat_vec_multiply: %s (%.2fx np.tensordot)", d2, d2 / d1)
