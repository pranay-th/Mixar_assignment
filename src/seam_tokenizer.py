"""
src/seam_tokenizer.py
Robust seam detection + tokenization for meshes.

Outputs:
 - seam_edge_indices : np.ndarray of edge indices flagged as seams
 - tokens_from_loops(mesh, loops) -> list of token sequences; each token is a small dict

Seam detection combines:
 - boundary edges
 - high dihedral edges (face adjacency angle)
 - high curvature edges (vertex normal variance -> threshold by percentile)

Also includes helper to write a visualization mesh with seam vertices colored.
"""

import numpy as np
from collections import defaultdict, deque
from scipy.spatial import cKDTree
import trimesh

def compute_vertex_normal_variance(mesh: trimesh.Trimesh, k: int = 16):
    """
    Proxy curvature: for each vertex compute variance of neighbor vertex normals.
    Returns array shape (n_vertices,) with scalar curvature proxy (higher => sharper).
    """
    verts = np.asarray(mesh.vertices)
    # ensure vertex normals exist
    try:
        normals = np.asarray(mesh.vertex_normals)
    except Exception:
        o3d = None
        normals = np.zeros_like(verts)  # fallback zeros
    # build kdtree on vertices
    tree = cKDTree(verts)
    k_q = min(k, len(verts)-1)
    if k_q < 1:
        return np.zeros(len(verts))
    dists, idxs = tree.query(verts, k=k_q+1)  # includes self
    # compute variance of normals in neighborhood excluding self
    var = np.zeros(len(verts), dtype=float)
    for i in range(len(verts)):
        neigh = idxs[i,1:] if idxs.ndim == 2 else idxs[i+1:]  # defensive
        neigh_norms = normals[neigh]
        # variance across components
        v = np.mean((neigh_norms - neigh_norms.mean(axis=0))**2)
        var[i] = float(v)
    # normalize to [0,1]
    if var.max() > 0:
        var = (var - var.min()) / (var.max() - var.min() + 1e-12)
    return var

def detect_seams(mesh: trimesh.Trimesh,
                 dihedral_deg_thresh: float = 30.0,
                 curvature_percentile: float = 90.0,
                 k_curv: int = 16):
    """
    Robust seam detection combining multiple signals.
    Returns: (seam_edge_indices (np.array), seam_mask (bool array length n_edges))
    """
    edges = mesh.edges_sorted
    n_edges = len(edges)
    seam_mask = np.zeros(n_edges, dtype=bool)
    edge_index_by_tuple = {tuple(e): i for i, e in enumerate(edges)}

    # 1) boundary edges (safe method)
    try:
        boundary = mesh.edges_boundary
        if boundary is not None and len(boundary) > 0:
            for be in boundary:
                key = tuple(np.sort(be))
                idx = edge_index_by_tuple.get(key)
                if idx is not None:
                    seam_mask[idx] = True
    except Exception:
        pass

    # 2) dihedral edges via face_adjacency_edges and face_adjacency_angles
    try:
        adj_edges = getattr(mesh, 'face_adjacency_edges', None)
        adj_angles = getattr(mesh, 'face_adjacency_angles', None)
        if adj_edges is not None and adj_angles is not None and len(adj_edges) == len(adj_angles):
            # adj_edges can be edge indices
            for eidx, angle in zip(adj_edges, adj_angles):
                try:
                    ei = int(eidx)
                    if 0 <= ei < n_edges and np.degrees(angle) >= dihedral_deg_thresh:
                        seam_mask[ei] = True
                except Exception:
                    # sometimes eidx not directly indexable; try mapping via vertices:
                    pass
    except Exception:
        pass

    # 3) curvature-based edges: mark edges touching high-curvature vertices
    try:
        var = compute_vertex_normal_variance(mesh, k=k_curv)
        # threshold at percentile
        thresh = np.percentile(var, curvature_percentile)
        high_v = var >= thresh
        # mark edges where either vertex is high curvature
        for ei, (a,b) in enumerate(edges):
            if high_v[int(a)] or high_v[int(b)]:
                seam_mask[ei] = True
    except Exception:
        pass

    # Final seam indices
    seam_edge_indices = np.nonzero(seam_mask)[0]
    return seam_edge_indices, seam_mask

# -------------------------
# Graph + loop extraction
# -------------------------
def build_edge_vertex_map(mesh: trimesh.Trimesh):
    """
    Returns mapping: edge_index -> (v0, v1)
    """
    return {i: tuple(e) for i, e in enumerate(mesh.edges_sorted)}

def build_vertex_to_edge_map(edge_map):
    """
    From edge_map {eidx:(a,b)} -> v2e {v: set(eidx)}
    """
    v2e = defaultdict(set)
    for eidx, (a,b) in edge_map.items():
        v2e[int(a)].add(int(eidx))
        v2e[int(b)].add(int(eidx))
    return v2e

def extract_seam_paths(mesh: trimesh.Trimesh, seam_edge_indices):
    """
    From seam edges, build continuous paths (open or closed).
    Returns list of paths; each path is a list of edge indices in order.
    """
    if len(seam_edge_indices) == 0:
        return []

    edge_map = build_edge_vertex_map(mesh)
    v2e = build_vertex_to_edge_map({int(e): tuple(mesh.edges_sorted[int(e)]) for e in seam_edge_indices})

    # faster: construct adjacency for seam edges only
    seam_set = set(int(e) for e in seam_edge_indices)
    # vertex -> seam-neighbor edges
    v2se = defaultdict(list)
    for eidx in seam_set:
        a,b = edge_map[eidx]
        v2se[int(a)].append(int(eidx))
        v2se[int(b)].append(int(eidx))

    visited = set()
    paths = []

    # helper to walk from an edge and vertex
    def walk_from(start_edge, start_vertex):
        path = []
        cur_edge = start_edge
        cur_vertex = start_vertex
        while True:
            if cur_edge in visited:
                break
            visited.add(cur_edge)
            path.append(cur_edge)
            # find next edge sharing the opposite vertex
            a,b = edge_map[cur_edge]
            next_vertex = int(b) if int(a) == cur_vertex else int(a)
            # neighbors edges at next_vertex
            neigh_edges = [e for e in v2se[next_vertex] if e not in visited]
            if not neigh_edges:
                break
            # choose neighbor whose other vertex continues the path (prefer degree 2)
            # pick the one that is not cur_edge and not visited
            cur_edge = neigh_edges[0]
            cur_vertex = next_vertex
        return path

    # First, start walks from vertices with degree != 2 (endpoints), to capture open paths
    vertex_degrees = {v: len(v2se[v]) for v in v2se}
    end_vertices = [v for v,deg in vertex_degrees.items() if deg != 2]
    # Walk from each incident seam edge at end vertices
    for v in end_vertices:
        for e in list(v2se[v]):
            if e in visited:
                continue
            p = walk_from(e, v)
            if p:
                paths.append(p)

    # Finally, any remaining cycles (closed loops)
    for e in list(seam_set):
        if e in visited:
            continue
        # pick an endpoint (one of its vertices)
        a,b = edge_map[e]
        p = walk_from(e, int(a))
        if p:
            paths.append(p)

    return paths

# -------------------------
# Token encoding
# -------------------------
def tokens_from_paths(mesh: trimesh.Trimesh, paths):
    """
    Convert paths (edge-index lists) into token sequences.
    Token format (list of dicts):
      {'t':'S','v':start_vertex}
      {'t':'V','v':vertex_index, 'e':edge_index, 'l':edge_length, 'd':dihedral_deg}
      ...
      {'t':'E'}
    The first 'S' token uses the first vertex of the first edge in the path.
    """
    edge_map = build_edge_vertex_map(mesh)
    # precompute edge lengths and dihedral angles if available
    edge_lengths = []
    for eidx in range(len(mesh.edges_sorted)):
        a,b = mesh.edges_sorted[eidx]
        a_pt = mesh.vertices[int(a)]; b_pt = mesh.vertices[int(b)]
        edge_lengths.append(float(np.linalg.norm(np.asarray(a_pt)-np.asarray(b_pt))))
    # dihedral: map adjacency edges -> angles (radians) via face_adjacency_edges and face_adjacency_angles
    dihedral_map = {}
    try:
        adj_edges = getattr(mesh, 'face_adjacency_edges', None)
        adj_angles = getattr(mesh, 'face_adjacency_angles', None)
        if adj_edges is not None and adj_angles is not None:
            for eidx, ang in zip(adj_edges, adj_angles):
                try:
                    dihedral_map[int(eidx)] = float(np.degrees(ang))
                except Exception:
                    continue
    except Exception:
        pass

    token_seqs = []
    for path in paths:
        if not path:
            continue
        # derive vertex sequence by walking edges in order
        vert_seq = []
        for i, eidx in enumerate(path):
            a,b = edge_map[eidx]
            a=int(a); b=int(b)
            if i == 0:
                vert_seq.extend([a,b])
            else:
                # if last vertex equals a, append b; if last equals b, append a; else append a
                last = vert_seq[-1]
                if last == a:
                    vert_seq.append(b)
                elif last == b:
                    vert_seq.append(a)
                else:
                    vert_seq.append(a)

        # compress into tokens
        toks = []
        toks.append({'t':'S','v':int(vert_seq[0])})
        # map edges to their associated dihedral/length
        for i in range(len(path)):
            eidx = int(path[i])
            v = int(vert_seq[i+1])  # vertex reached after traversing edge i
            d = dihedral_map.get(eidx, None)
            l = float(edge_lengths[eidx]) if eidx < len(edge_lengths) else None
            tok = {'t':'V', 'v':v, 'e':eidx}
            if l is not None:
                tok['l'] = round(l, 6)
            if d is not None:
                tok['d'] = round(d, 3)
            toks.append(tok)
        toks.append({'t':'E'})
        token_seqs.append(toks)
    return token_seqs

# -------------------------
# Helper: write seam-highlighted mesh for quick visual check
# -------------------------
def save_seam_visualization(mesh: trimesh.Trimesh, seam_mask, out_path):
    """
    Create a copy of mesh with seam vertices colored (red) and save as PLY/OBJ.
    seam_mask: boolean array length n_edges
    """
    # determine vertices in seam edges
    edges = mesh.edges_sorted
    seam_edges = [edges[i] for i, m in enumerate(seam_mask) if m]
    seam_verts = set()
    for a,b in seam_edges:
        seam_verts.add(int(a)); seam_verts.add(int(b))

    # create copy and color vertices
    m = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    # default color
    n = len(m.vertices)
    colors = np.tile(np.array([200,200,200,255], dtype=np.uint8), (n,1))
    for v in seam_verts:
        if 0 <= v < n:
            colors[int(v)] = np.array([255,60,60,255], dtype=np.uint8)
    try:
        m.visual.vertex_colors = colors
    except Exception:
        pass
    # save
    m.export(out_path)

# -------------------------
# Top-level convenience: full pipeline per mesh
# -------------------------
def extract_and_tokenize(mesh: trimesh.Trimesh,
                         dihedral_deg_thresh: float = 30.0,
                         curvature_percentile: float = 90.0,
                         k_curv: int = 16,
                         save_vis: str = None):
    """
    High-level convenience: detect seams, extract paths, produce tokens, optionally save visualization.
    Returns: (seam_edge_indices, seam_mask, paths, token_seqs)
    """
    seam_idxs, seam_mask = detect_seams(mesh, dihedral_deg_thresh=dihedral_deg_thresh,
                                       curvature_percentile=curvature_percentile, k_curv=k_curv)
    paths = extract_seam_paths(mesh, seam_idxs)
    tokens = tokens_from_paths(mesh, paths)
    if save_vis:
        try:
            save_seam_visualization(mesh, seam_mask, save_vis)
        except Exception:
            pass
    return seam_idxs, seam_mask, paths, tokens
