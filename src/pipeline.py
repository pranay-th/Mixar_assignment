#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import trimesh

from src.utils import ensure_dir, init_logger, dump_json
from src.normalize import minmax_normalize, unit_sphere_normalize, denormalize
from src.density import compute_local_density, region_bins_by_kmeans
from src.quantize import quantize_regionwise, dequantize_regionwise
from src.metrics import axis_mse_mae, chamfer_distance, hausdorff_distance, normal_angle_error
from src.visualize import plot_axis_errors, save_summary_csv, render_and_save_mesh_comparison
from src.seam_tokenizer import extract_and_tokenize 


def save_mesh_as_obj(vertices, faces, out_path):
    """Save vertices + faces as .obj using trimesh."""
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    m.export(out_path)


def process_mesh(path, out_dir, args, logger):
    """Run full preprocessing, quantization, reconstruction, and seam tokenization for one mesh."""
    mesh_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Processing {mesh_name}")

    # Load mesh
    mesh = trimesh.load(path, force='mesh')
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # Mesh statistics
    stats = {
        'n_vertices': int(verts.shape[0]),
        'min': verts.min(axis=0).tolist(),
        'max': verts.max(axis=0).tolist(),
        'mean': verts.mean(axis=0).tolist(),
        'std': verts.std(axis=0).tolist()
    }
    logger.info(f"Stats: {stats}")

    mesh_out = os.path.join(out_dir, mesh_name)
    ensure_dir(mesh_out)
    dump_json(os.path.join(mesh_out, 'stats.json'), stats)

    # Two normalization methods
    normings = [
        ('minmax',) + minmax_normalize(verts),
        ('unit_sphere',) + unit_sphere_normalize(verts)
    ]

    # Compute density and adaptive bins (used for both methods)
    density = compute_local_density(verts, k=args.k_density)
    bins_per_vertex, clusters, bins_per_cluster = region_bins_by_kmeans(
        density, base_bins=args.base_bins, n_regions=args.clusters, alpha=args.alpha)
    np.save(os.path.join(mesh_out, 'bins_per_vertex.npy'), bins_per_vertex)
    np.save(os.path.join(mesh_out, 'clusters.npy'), clusters)
    np.save(os.path.join(mesh_out, 'bins_per_cluster.npy'), bins_per_cluster)

    summary_rows = []

    # --- normalization loops ---
    for name, normalized, meta in normings:
        logger.info(f"-- method {name}")

        # Quantize and reconstruct
        q = quantize_regionwise(normalized, bins_per_vertex)
        np.save(os.path.join(mesh_out, f"{name}_q.npy"), q)

        deq = dequantize_regionwise(q, bins_per_vertex)
        recon = denormalize(deq, meta)
        recon_path = os.path.join(mesh_out, f"{name}_recon.obj")
        save_mesh_as_obj(recon, faces, recon_path)

        # Error metrics
        mse_axis, mae_axis = axis_mse_mae(verts, recon)
        chamfer = chamfer_distance(verts, recon)
        haus = hausdorff_distance(verts, recon)
        recon_mesh = trimesh.Trimesh(vertices=recon, faces=faces, process=False)
        mean_angle, med_angle = normal_angle_error(mesh, recon_mesh)

        # Plots
        axis_err_path = os.path.join(mesh_out, f"{name}_axis_err.png")
        plot_axis_errors(mse_axis, mae_axis, axis_err_path)

        # 3D comparison render
        try:
            out_img_path = os.path.join(mesh_out, f"{name}_compare.png")
            render_and_save_mesh_comparison(path, recon_path, out_img_path, resolution=(1200, 800))
            logger.info(f"Saved comparison image: {out_img_path}")
        except Exception as e:
            logger.warning(f"Render comparison failed for {mesh_name} {name}: {e}")

        # Record summary
        row = {
            'mesh': mesh_name,
            'method': name,
            'mse_x': float(mse_axis[0]), 'mse_y': float(mse_axis[1]), 'mse_z': float(mse_axis[2]),
            'mae_x': float(mae_axis[0]), 'mae_y': float(mae_axis[1]), 'mae_z': float(mae_axis[2]),
            'chamfer_sq': float(chamfer), 'hausdorff': float(haus),
            'normal_mean_deg': float(mean_angle)
        }
        summary_rows.append(row)
        logger.info(f"Result row: {row}")

    # Save summary CSV for this mesh
    save_summary_csv(summary_rows, os.path.join(mesh_out, 'summary.csv'))

    # --- Enhanced seam tokenization ---
    try:
        seam_vis_path = os.path.join(mesh_out, "seams_visualization.ply")
        seam_idxs, seam_mask, paths, tokens = extract_and_tokenize(
            mesh,
            dihedral_deg_thresh=30.0,
            curvature_percentile=90.0,
            k_curv=16,
            save_vis=seam_vis_path
        )

        token_path = os.path.join(mesh_out, "seams_tokens.json")
        dump_json(token_path, tokens)
        logger.info(f"Saved seam tokens: {token_path} ({len(tokens)} sequences)")
        logger.info(f"Saved seam visualization: {seam_vis_path}")
        logger.info(f"{mesh_name}: {len(tokens)} seam sequences, {len(seam_idxs)} seam edges")
    except Exception as e:
        logger.warning(f"Enhanced seam tokenization failed for {mesh_name}: {e}")

    return os.path.join(mesh_out, 'summary.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='meshes')
    parser.add_argument('--out_dir', default='results')
    parser.add_argument('--clusters', type=int, default=4)
    parser.add_argument('--k_density', type=int, default=16)
    parser.add_argument('--base_bins', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    logger = init_logger(args.out_dir)

    mesh_files = [os.path.join(args.input_dir, f)
                  for f in sorted(os.listdir(args.input_dir))
                  if f.endswith('.obj')]

    if not mesh_files:
        logger.error("No .obj files in input_dir")
        return

    logger.info(f"Found {len(mesh_files)} meshes")

    for m in mesh_files:
        try:
            process_mesh(m, args.out_dir, args, logger)
        except Exception as e:
            logger.exception(f"Error processing {m}: {e}")


if __name__ == '__main__':
    main()
