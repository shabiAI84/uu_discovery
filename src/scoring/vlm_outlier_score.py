import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_outlier_score(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Simple kNN-based outlier score.
    For each point, compute mean distance to k nearest neighbors.
    Larger distance = more "novel".
    """
    if embeddings.shape[0] <= k:
        raise ValueError("Not enough samples for kNN outlier scoring.")

    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    # Ignore distance to self if k includes it; mean over neighbors
    scores = distances.mean(axis=1)
    return scores
