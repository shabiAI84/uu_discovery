import os
import numpy as np

from src.embedding.clip_encoder import CLIPEncoder
from src.scoring.vlm_outlier_score import knn_outlier_score
from src.scoring.caption_entropy import dummy_caption_entropy
from src.scoring.uncertainty_mc_dropout import dummy_mc_dropout_uncertainty
from src.clustering.hdbscan_cluster import cluster_embeddings
from src.visualization.tsne_plot import plot_tsne


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)


def main(
    image_dir: str = "data/images",
    model_name: str = "ViT-B/32",
    tsne_out: str = "outputs/tsne_clusters.png"
):
    ensure_dirs()

    encoder = CLIPEncoder(model_name=model_name)
    image_paths = encoder.collect_image_paths(image_dir)

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}. Put some JPG/PNG files there.")

    embeddings = encoder.encode_image_paths(image_paths)

    # Scores
    outlier_scores = knn_outlier_score(embeddings, k=5)
    caption_entropy = dummy_caption_entropy(len(image_paths))
    mc_uncertainty = dummy_mc_dropout_uncertainty(len(image_paths))

    # Combine scores – simple weighted sum
    combined_score = (
        0.6 * outlier_scores +
        0.2 * caption_entropy +
        0.2 * mc_uncertainty
    )

    labels = cluster_embeddings(embeddings, min_cluster_size=15)

    # Save artifacts for browser
    np.save("outputs/embeddings.npy", embeddings)
    np.save("outputs/labels.npy", labels)
    np.save("outputs/outlier_scores.npy", combined_score)
    np.save("outputs/image_paths.npy", np.array(image_paths))

    # TSNE plot for sanity
    plot_tsne(
        embeddings,
        labels,
        out_path=tsne_out,
        title="Embeddings TSNE colored by HDBSCAN cluster"
    )
    print(f"Saved TSNE plot to {tsne_out}")
    print("Done. Now run: streamlit run src/visualization/cluster_browser.py")


if __name__ == "__main__":
    main()
