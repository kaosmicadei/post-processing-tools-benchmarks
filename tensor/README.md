# The Problem
Given a bitstring distribution $`\boldsymbol{b}`$, we need to apply the inverse of a $`2 \times 2`$ confusion matrix $`\boldsymbol{C}`$ in order to correct the measurements.

For example, let's take $`\boldsymbol{b}`$ as the bitstring distribution of a three qubits system, and $`\boldsymbol{M} = \boldsymbol{C}^{-1}`$.

```math
\boldsymbol{b}^\intercal = 
\begin{pmatrix}
b_{000} & b_{010} & b_{001} & b_{011} &
b_{100} & b_{110} & b_{101} & b_{111} 
\end{pmatrix}
```

The standard procedure would be to compute
```math
\left( \boldsymbol{M} \otimes \boldsymbol{M} \otimes  \boldsymbol{M} \right) \boldsymbol{b}
```

But that can consume too much time and memory to compute.

# Solution 1
The first approach is to the vector $`\boldsymbol{b}`$ into a three indices tensor $`B_{ijk}`$, and then computing the multiplication iteratively contracting one index at time. Then, finally, converting the the final tensor back to a vector.

```python
import numpy as np

def multiply(m, data):
    N = int(np.log2(len(data)))
    res = np.reshape(data, [2] * N)
    for i in range(N):
        res = np.tensordot(m, res, axes=(1, i))
    return res.transpose().flatten()
```

# Solution 2
Another option is to create a $`2 \times 2^{N-1}`$ view $`\boldsymbol{B}`$ of the data
```math
\boldsymbol{B} =
\begin{bmatrix}
b_{000} & b_{010} & b_{100} & b_{110} \\ 
b_{001} & b_{011} & b_{101} & b_{111}  
\end{bmatrix} 
```
so,
```math
\boldsymbol{M} \cdot \boldsymbol{B}
\equiv
\left( \mathbb{I}_2 \otimes \mathbb{I}_2 \otimes \boldsymbol{M}  \right)
\cdot \boldsymbol{b} .
```

To compute all the multiplications, we multiple $`\boldsymbol{M} \cdot \boldsymbol{B}`$ iteratively reordering the elements of the view at each step.

```math
\begin{gather*}
\boldsymbol{B}' =
\boldsymbol{M} \cdot
\begin{bmatrix}
b_{000} & b_{010} & b_{100} & b_{110} \\ 
b_{001} & b_{011} & b_{101} & b_{111}  
\end{bmatrix}
\\[1em]
\boldsymbol{B}'' =
\boldsymbol{M} \cdot
\begin{bmatrix}
b'_{000} & b'_{001} & b'_{100} & b'_{101} \\ 
b'_{010} & b'_{011} & b'_{110} & b'_{111}  
\end{bmatrix}
\\[1em]
\boldsymbol{B}''' =
\boldsymbol{M} \cdot
\begin{bmatrix}
b''_{000} & b''_{001} & b''_{010} & b''_{011} \\ 
b''_{100} & b''_{101} & b''_{110} & b''_{111}  
\end{bmatrix}
\end{gather*}
```

The final result is given by $`\text{vec}(\boldsymbol{B}''')`$.
