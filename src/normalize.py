import numpy as np

def minmax_normalize(vertices):
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    scale = vmax - vmin
    scale[scale == 0] = 1e-9
    normalized = (vertices - vmin) / scale
    meta = {'type':'minmax', 'vmin':vmin.tolist(), 'vmax':vmax.tolist()}
    return normalized, meta

def unit_sphere_normalize(vertices):
    norms = np.linalg.norm(vertices, axis=1)
    max_norm = norms.max() if norms.max() > 0 else 1e-9
    normalized = vertices / max_norm
    # shift to [0,1] for quantization convenience
    normalized = (normalized + 1.0) / 2.0
    meta = {'type':'unit_sphere', 'scale': float(max_norm), 'shifted': True}
    return normalized, meta

def denormalize(normalized, meta):
    arr = np.asarray(normalized)
    if meta['type'] == 'minmax':
        vmin = np.array(meta['vmin'])
        vmax = np.array(meta['vmax'])
        return arr * (vmax - vmin) + vmin
    elif meta['type'] == 'unit_sphere':
        scaled = arr * 2.0 - 1.0
        return scaled * float(meta['scale'])
    else:
        raise ValueError("Unknown normalization meta")
