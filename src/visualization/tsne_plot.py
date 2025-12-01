import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Optional[str] = None,
    title: str = "TSNE of Embeddings"
):
    """
    Compute a TSNE projection and plot clusters.
    """
    tsne = TSNE(n_components=2, perplexity=30, init="random", learning_rate="auto")
    emb2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            lab_name = "noise"
        else:
            lab_name = f"cluster {lab}"
        plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=10, alpha=0.7, label=lab_name)

    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200)
    else:
        plt.show()
