import numpy as np
import trimesh
from collections import defaultdict

def detect_seams(mesh: trimesh.Trimesh, dihedral_deg_thresh=60.0):
    # boundary edges appear once; trimesh stores edges and edges_face
    edges = mesh.edges_sorted
    ef = mesh.edges_face
    boundary_mask = (ef[:,1] == -1) | (ef[:,0] == -1)
    # dihedral edges:
    try:
        dihedral = mesh.face_adjacency_angles
        adj_edges = mesh.face_adjacency_edges
        high = np.zeros(len(edges), dtype=bool)
        for idx_edge, angle in zip(adj_edges, dihedral):
            if np.degrees(angle) > dihedral_deg_thresh:
                high[idx_edge] = True
    except Exception:
        high = np.zeros(len(edges), dtype=bool)
    seam_mask = boundary_mask | high
    idxs = np.nonzero(seam_mask)[0]
    return idxs, seam_mask

def seam_loops_from_edges(mesh: trimesh.Trimesh, seam_edge_indices):
    edges = mesh.edges_sorted[seam_edge_indices]
    vmap = defaultdict(list)
    for i, eidx in enumerate(seam_edge_indices):
        a,b = edges[i]
        vmap[int(a)].append((int(b), int(eidx)))
        vmap[int(b)].append((int(a), int(eidx)))
    loops = []
    visited = set()
    for i, eidx in enumerate(seam_edge_indices):
        if int(eidx) in visited:
            continue
        start_edge = edges[i]
        start_v = int(start_edge[0])
        cur_v = start_v
        loop = []
        while True:
            neigh = vmap.get(cur_v, [])
            next_found = False
            for nb_v, nb_eidx in neigh:
                if int(nb_eidx) not in visited:
                    visited.add(int(nb_eidx))
                    loop.append(int(nb_eidx))
                    cur_v = nb_v
                    next_found = True
                    break
            if not next_found:
                break
            if cur_v == start_v:
                break
        if loop:
            loops.append(loop)
    return loops

def tokens_from_loops(mesh: trimesh.Trimesh, loops):
    tokens = []
    for loop in loops:
        vert_seq = []
        for eidx in loop:
            a,b = mesh.edges_sorted[eidx]
            if not vert_seq:
                vert_seq.extend([int(a), int(b)])
            else:
                if vert_seq[-1] == a:
                    vert_seq.append(int(b))
                elif vert_seq[-1] == b:
                    vert_seq.append(int(a))
                else:
                    vert_seq.append(int(a))
        if len(vert_seq) < 2:
            continue
        t = [{'t':'S','v':int(vert_seq[0])}]
        for v in vert_seq[1:]:
            t.append({'t':'V','v':int(v)})
        t.append({'t':'E'})
        tokens.append(t)
    return tokens
