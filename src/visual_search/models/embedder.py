"""Embedding adapters."""

from __future__ import annotations

import numpy as np
from PIL import Image

from visual_search.models.base import Embedder
from visual_search.utils import image_to_vector, hash_text_to_vector, normalize_vector, ensure_rgb


class MockEmbedder:
    name = "mock-embedder"
    is_mock = True

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_image(self, image: Image.Image) -> np.ndarray:
        vec = image_to_vector(image, dim=self.dim)
        return normalize_vector(vec)

    def embed_text(self, text: str) -> np.ndarray:
        vec = hash_text_to_vector(text, dim=self.dim)
        return normalize_vector(vec)


class ClipEmbedder:
    name = "clip"
    is_mock = False

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise RuntimeError("open_clip and torch are required for ClipEmbedder") from exc

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
        self.dim = self.model.text_projection.shape[1]
        self._torch = torch

    def embed_image(self, image: Image.Image) -> np.ndarray:
        img = ensure_rgb(image)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feat = self.model.encode_image(tensor)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return normalize_vector(vec)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with self._torch.no_grad():
            feat = self.model.encode_text(tokens)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        return normalize_vector(vec)
