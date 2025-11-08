import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

def compute_local_density(vertices, k=16):
    tree = cKDTree(vertices)
    dists, _ = tree.query(vertices, k=k+1)
    avg = dists[:,1:].mean(axis=1)
    density = 1.0 / (avg + 1e-9)
    nd = (density - density.min()) / (density.max() - density.min() + 1e-12)
    return nd

def region_bins_by_kmeans(norm_density, base_bins=1024, n_regions=4, alpha=1.0, min_bins=64, max_bins=4096, random_state=0):
    # cluster 1D density into regions
    kmeans = KMeans(n_clusters=n_regions, random_state=random_state)
    clusters = kmeans.fit_predict(norm_density.reshape(-1,1))
    centers = kmeans.cluster_centers_.reshape(-1)
    cn = (centers - centers.min()) / (centers.max() - centers.min() + 1e-12)
    bins_per_cluster = np.clip((1.0 + alpha * cn) * base_bins, min_bins, max_bins).astype(int)
    bins_per_vertex = bins_per_cluster[clusters]
    return bins_per_vertex, clusters, bins_per_cluster
