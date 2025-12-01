import numpy as np
import hdbscan


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int | None = None
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN.
    Returns:
        labels: (N,) array; -1 = noise
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(embeddings)
    return labels
