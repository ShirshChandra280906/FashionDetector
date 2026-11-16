"""Image embedders: a colour-histogram baseline and CLIP-family models.

All embedders share one interface:
    emb = embedder.embed(list_of_PIL_images)  # -> float32 array (N, D), L2-normalised
Because outputs are L2-normalised, inner product == cosine similarity, which
is what the FAISS index uses.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Hugging Face model ids for the CLIP-family models we compare.
CLIP_MODELS = {
    "clip": "openai/clip-vit-base-patch32",  # general-purpose CLIP (ViT-B/32)
    "fashionclip": "patrickjohncyh/fashion-clip",  # CLIP ViT-B/32 fine-tuned on fashion products
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def pad_to_square(img: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    """Pad (not crop) to a square so tall items like trousers are not cut off.
    White fill matches the studio background of catalog images."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    return canvas


class ColorHistogramEmbedder:
    """Baseline: an HSV colour histogram (no learning, no notion of shape)."""

    name = "color_hist"

    def __init__(self, bins=(16, 4, 4), size: int = 128):
        self.bins, self.size = bins, size

    @property
    def dim(self) -> int:
        return int(np.prod(self.bins))

    def _one(self, img: Image.Image) -> np.ndarray:
        hsv = np.asarray(img.convert("RGB").resize((self.size, self.size)).convert("HSV")).reshape(-1, 3)
        hist, _ = np.histogramdd(hsv, bins=self.bins, range=[(0, 256)] * 3)
        hist = hist.ravel() / max(hist.sum(), 1)
        return np.sqrt(hist)  # Hellinger mapping: cosine on sqrt-histograms ~ Bhattacharyya

    def embed(self, images: list[Image.Image], batch_size: int = 0) -> np.ndarray:
        return l2_normalize(np.stack([self._one(im) for im in images])) if images else np.zeros((0, self.dim), np.float32)


class ClipEmbedder:
    """Image tower of a CLIP-style model loaded from Hugging Face."""

    def __init__(self, key: str = "fashionclip", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from src.config import best_device

        self.name = key
        self.model_id = CLIP_MODELS.get(key, key)
        self.device = device or best_device()
        self.torch = torch
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim

    def embed(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = [pad_to_square(im) for im in images[i : i + batch_size]]
                pix = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
                feats = self.model.get_image_features(pixel_values=pix)
                if not isinstance(feats, self.torch.Tensor):  # some transformers versions wrap the output
                    emb = getattr(feats, "image_embeds", None)
                    feats = emb if emb is not None else feats.pooler_output
                out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


def get_embedder(name: str, device: str | None = None):
    """Factory used by scripts and the app: 'color_hist', 'clip' or 'fashionclip'."""
    if name == "color_hist":
        return ColorHistogramEmbedder()
    return ClipEmbedder(name, device=device)
