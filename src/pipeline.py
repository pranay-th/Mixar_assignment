#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import trimesh
from src.utils import ensure_dir, init_logger, dump_json
from src.normalize import minmax_normalize, unit_sphere_normalize, denormalize
from src.density import compute_local_density, region_bins_by_kmeans
from src.quantize import quantize_regionwise, dequantize_regionwise
from src.metrics import axis_mse_mae, chamfer_distance, hausdorff_distance, normal_angle_error
from src.seam_tokenizer import detect_seams, seam_loops_from_edges, tokens_from_loops
from src.visualize import plot_axis_errors, save_summary_csv

def save_mesh_as_obj(vertices, faces, out_path):
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    m.export(out_path)

def process_mesh(path, out_dir, args, logger):
    mesh_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Processing {mesh_name}")
    mesh = trimesh.load(path, force='mesh')
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
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

    normings = [
        ('minmax',) + minmax_normalize(verts),
        ('unit_sphere',) + unit_sphere_normalize(verts)
    ]
    summary_rows = []
    density = compute_local_density(verts, k=args.k_density)
    bins_per_vertex, clusters, bins_per_cluster = region_bins_by_kmeans(
        density, base_bins=args.base_bins, n_regions=args.clusters, alpha=args.alpha)
    np.save(os.path.join(mesh_out,'bins_per_vertex.npy'), bins_per_vertex)
    np.save(os.path.join(mesh_out,'clusters.npy'), clusters)
    np.save(os.path.join(mesh_out,'bins_per_cluster.npy'), bins_per_cluster)

    for name, normalized, meta in normings:
        logger.info(f"-- method {name}")
        q = quantize_regionwise(normalized, bins_per_vertex)
        np.save(os.path.join(mesh_out, f"{name}_q.npy"), q)
        deq = dequantize_regionwise(q, bins_per_vertex)
        recon = denormalize(deq, meta)
        recon_path = os.path.join(mesh_out, f"{name}_recon.obj")
        save_mesh_as_obj(recon, faces, recon_path)

        mse_axis, mae_axis = axis_mse_mae(verts, recon)
        chamfer = chamfer_distance(verts, recon)
        haus = hausdorff_distance(verts, recon)
        recon_mesh = trimesh.Trimesh(vertices=recon, faces=faces, process=False)
        mean_angle, med_angle = normal_angle_error(mesh, recon_mesh)

        plot_axis_errors(mse_axis, mae_axis, os.path.join(mesh_out, f"{name}_axis_err.png"))
        row = {
            'mesh': mesh_name, 'method': name,
            'mse_x': float(mse_axis[0]), 'mse_y': float(mse_axis[1]), 'mse_z': float(mse_axis[2]),
            'mae_x': float(mae_axis[0]), 'mae_y': float(mae_axis[1]), 'mae_z': float(mae_axis[2]),
            'chamfer_sq': float(chamfer), 'hausdorff': float(haus),
            'normal_mean_deg': float(mean_angle)
        }
        summary_rows.append(row)
        logger.info(f"Result row: {row}")

    import csv
    save_summary_csv(summary_rows, os.path.join(mesh_out, 'summary.csv'))

    # seam tokenization
    seam_idxs, mask = detect_seams(mesh)
    loops = seam_loops_from_edges(mesh, seam_idxs)
    tokens = tokens_from_loops(mesh, loops)
    dump_json(os.path.join(mesh_out, 'seams_tokens.json'), tokens)
    logger.info(f"Found {len(tokens)} seam token sequences")

    return os.path.join(mesh_out, 'summary.csv')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='meshes')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--clusters', type=int, default=4)
    p.add_argument('--k_density', type=int, default=16)
    p.add_argument('--base_bins', type=int, default=1024)
    p.add_argument('--alpha', type=float, default=1.0)
    args = p.parse_args()
    ensure_dir(args.out_dir)
    logger = init_logger(args.out_dir)
    mesh_files = [os.path.join(args.input_dir,f) for f in sorted(os.listdir(args.input_dir)) if f.endswith('.obj')]
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
