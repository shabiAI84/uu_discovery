import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_outlier_flag(scores: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """
    Turn continuous outlier scores into a binary flag.
    Mark top quantile as "outliers".
    """
    threshold = np.quantile(scores, quantile)
    return scores >= threshold
