import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_axis_errors(mse_axis, mae_axis, out_path):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title('MSE per axis'); plt.bar(['x','y','z'], mse_axis)
    plt.subplot(1,2,2); plt.title('MAE per axis'); plt.bar(['x','y','z'], mae_axis)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def save_summary_csv(rows, path):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

import os
import trimesh


def render_and_save_mesh_comparison(orig_path: str,
                                    recon_path: str,
                                    out_path: str,
                                    resolution=(1024, 768),
                                    show_axes: bool = False):
    """
    Render original and reconstructed meshes side-by-side and save a PNG.

    Works headlessly.  Falls back to a matplotlib 2-D projection if trimesh
    off-screen rendering is unavailable.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- load meshes ---
    orig = trimesh.load(orig_path, force="mesh")
    recon = trimesh.load(recon_path, force="mesh")

    # color hints
    try:
        orig.visual.vertex_colors = np.tile([180, 200, 255, 255], (len(orig.vertices), 1))
        recon.visual.vertex_colors = np.tile([255, 200, 200, 255], (len(recon.vertices), 1))
    except Exception:
        pass

    # --- align centers, place side-by-side ---
    try:
        orig_center = orig.bounds.mean(axis=0)
        recon_center = recon.bounds.mean(axis=0)
        orig.apply_translation(-orig_center)
        recon.apply_translation(-recon_center)
        width_orig = orig.bounds[1][0] - orig.bounds[0][0]
        width_recon = recon.bounds[1][0] - recon.bounds[0][0]
        gap = max(width_orig, width_recon) * 0.6 + 1e-4
        recon.apply_translation([width_orig + gap, 0, 0])
    except Exception:
        pass

    # --- try trimesh off-screen render first ---
    try:
        scene = trimesh.Scene()
        scene.add_geometry(orig, node_name="original")
        scene.add_geometry(recon, node_name="reconstructed")
        scene.camera.resolution = resolution
        img_bytes = scene.save_image(resolution=resolution, visible=True)
        if img_bytes:
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            return out_path
    except Exception:
        pass

    # --- fallback: matplotlib 2-D projection (XZ plane) ---
    orig_v = np.asarray(orig.vertices)
    recon_v = np.asarray(recon.vertices)
    if orig_v.size == 0 or recon_v.size == 0:
        raise RuntimeError("Empty vertex arrays in fallback renderer")

    orig_v = orig_v - orig_v.mean(axis=0)
    recon_v = recon_v - recon_v.mean(axis=0)

    # use np.ptp instead of arr.ptp for NumPy 2 compatibility
    orig_range = np.ptp(orig_v[:, 0])
    recon_range = np.ptp(recon_v[:, 0])
    gap = max(orig_range, recon_range) * 0.6 + 1e-4
    recon_v = recon_v + np.array([orig_range + gap, 0, 0])

    fig = plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(orig_v[:, 0], orig_v[:, 2], s=0.6, label="original",
               alpha=0.7, c="blue", linewidths=0)
    ax.scatter(recon_v[:, 0], recon_v[:, 2], s=0.6, label="recon",
               alpha=0.7, c="red", linewidths=0)
    if show_axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
    else:
        ax.set_axis_off()
    ax.set_aspect("equal", "box")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path
