import numpy as np

def quantize_regionwise(normalized, bins_per_vertex):
    q = np.empty_like(normalized, dtype=np.int64)
    for i in range(normalized.shape[0]):
        b = int(bins_per_vertex[i])
        q[i] = np.floor(normalized[i] * (b - 1)).astype(int)
    return q

def dequantize_regionwise(q, bins_per_vertex):
    dq = np.empty_like(q, dtype=np.float64)
    for i in range(q.shape[0]):
        b = int(bins_per_vertex[i])
        dq[i] = q[i].astype(float) / (b - 1)
    return dq
