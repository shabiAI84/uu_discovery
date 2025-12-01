import os
from typing import List

import torch
import clip
from PIL import Image
from tqdm import tqdm


class CLIPEncoder:
    """
    Simple wrapper around OpenAI CLIP for image embeddings.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    def encode_image_paths(self, image_paths: List[str], batch_size: int = 32):
        """
        Encode a list of image file paths into CLIP embeddings.
        Returns:
            embeddings: (N, D) float32 numpy array
        """
        all_embeddings = []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
                batch_paths = image_paths[i:i + batch_size]
                images = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                    except Exception:
                        # Skip unreadable images
                        continue
                    images.append(self.preprocess(img))

                if not images:
                    continue

                image_tensor = torch.stack(images).to(self.device)
                feats = self.model.encode_image(image_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu())

        if not all_embeddings:
            raise RuntimeError("No embeddings could be computed. Check your images.")

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return embeddings

    @staticmethod
    def collect_image_paths(root_dir: str, exts: tuple = (".jpg", ".jpeg", ".png")) -> List[str]:
        paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, f))
        paths.sort()
        return paths
