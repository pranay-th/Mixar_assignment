import numpy as np
from scipy.spatial import cKDTree
import trimesh

def axis_mse_mae(orig, recon):
    diff = orig - recon
    mse_axis = np.mean(diff**2, axis=0)
    mae_axis = np.mean(np.abs(diff), axis=0)
    return mse_axis, mae_axis

def chamfer_distance(A, B):
    treeB = cKDTree(B)
    dA, _ = treeB.query(A)
    treeA = cKDTree(A)
    dB, _ = treeA.query(B)
    return float(np.mean(dA**2) + np.mean(dB**2))

def hausdorff_distance(A, B):
    treeB = cKDTree(B)
    dA, _ = treeB.query(A)
    treeA = cKDTree(A)
    dB, _ = treeA.query(B)
    return float(max(dA.max(), dB.max()))

def normal_angle_error(orig_mesh: trimesh.Trimesh, recon_mesh: trimesh.Trimesh):
    # ensure vertex normals exist
    try:
        on = orig_mesh.vertex_normals
        rn = recon_mesh.vertex_normals
    except Exception:
        orig_mesh.rezero()
        recon_mesh.rezero()
        on = orig_mesh.vertex_normals
        rn = recon_mesh.vertex_normals
    dots = (on * rn).sum(axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return float(np.mean(angles)), float(np.median(angles))
