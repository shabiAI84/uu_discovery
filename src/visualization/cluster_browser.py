import os
import numpy as np
import streamlit as st
from PIL import Image


def load_numpy(path: str):
    return np.load(path, allow_pickle=True)


def main():
    st.title("Mini Data Engine – Cluster Browser")

    st.markdown(
        "Browse clusters and outliers discovered by the unknown–unknown pipeline."
    )

    # Expect these files from the pipeline:
    # - outputs/embeddings.npy
    # - outputs/labels.npy
    # - outputs/outlier_scores.npy
    # - outputs/image_paths.npy
    base_dir = "outputs"
    emb_path = os.path.join(base_dir, "embeddings.npy")
    lab_path = os.path.join(base_dir, "labels.npy")
    score_path = os.path.join(base_dir, "outlier_scores.npy")
    img_path = os.path.join(base_dir, "image_paths.npy")

    if not all(os.path.exists(p) for p in [emb_path, lab_path, score_path, img_path]):
        st.error("Run the discovery pipeline first to generate outputs/*.npy.")
        return

    labels = load_numpy(lab_path)
    scores = load_numpy(score_path)
    image_paths = load_numpy(img_path)

    unique_clusters = np.unique(labels)
    st.sidebar.header("Filters")

    show_noise = st.sidebar.checkbox("Show noise (label -1)", value=True)
    cluster_options = [int(c) for c in unique_clusters if c != -1]
    selected_cluster = st.sidebar.selectbox(
        "Select cluster (or 'Outliers only')",
        options=["Outliers only"] + cluster_options
    )

    sort_by = st.sidebar.radio("Sort images by", options=["Outlier score (desc)", "Path (asc)"])

    if selected_cluster == "Outliers only":
        # Pick top-k outliers
        k = st.sidebar.slider("Top-k outliers", min_value=10, max_value=200, value=50, step=10)
        idx_sorted = np.argsort(scores)[::-1]
        selected_indices = idx_sorted[:k]
    else:
        cl = int(selected_cluster)
        mask = labels == cl
        selected_indices = np.where(mask)[0]

    if not show_noise:
        noise_mask = labels[selected_indices] != -1
        selected_indices = selected_indices[noise_mask]

    if sort_by == "Outlier score (desc)":
        selected_indices = selected_indices[np.argsort(scores[selected_indices])[::-1]]
    else:
        selected_indices = selected_indices[np.argsort(image_paths[selected_indices])]

    st.write(f"Showing {len(selected_indices)} images.")

    n_cols = 5
    for i in range(0, len(selected_indices), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j >= len(selected_indices):
                break
            idx = selected_indices[i + j]
            p = image_paths[idx]
            scr = scores[idx]
            lab = labels[idx]
            try:
                img = Image.open(p).convert("RGB")
                with col:
                    st.image(img, use_column_width=True)
                    st.caption(f"cluster={lab}, score={scr:.3f}")
            except Exception:
                continue


if __name__ == "__main__":
    main()
