"""Embedding adapters — mock, vanilla CLIP, and fine-tuned CLIP."""

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
    """Vanilla (pre-trained) CLIP embedder — no fine-tuned weights."""

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


class ClipFinetunedEmbedder:
    """Fine-tuned CLIP embedder — loads weights from ``clip_best.pt``.

    The checkpoint is expected to contain ``model_state`` produced by the
    fine-tuning loop in the ablation notebooks (InfoNCE on last-N visual
    blocks).
    """

    name = "clip-finetuned"
    is_mock = False

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        weights_path: str = "",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise RuntimeError("open_clip and torch are required for ClipFinetunedEmbedder") from exc

        self.device = device
        self._torch = torch

        # Create base model architecture
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Load fine-tuned weights if provided
        if weights_path:
            ckpt = torch.load(weights_path, map_location="cpu")
            state_dict = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state_dict)
            epoch = ckpt.get("epoch", "?")
            loss = ckpt.get("loss", "?")
            print(f"[ClipFinetunedEmbedder] Loaded fine-tuned weights "
                  f"(epoch={epoch}, loss={loss})")

        self.model.to(self.device).eval()
        self.dim = self.model.text_projection.shape[1]

    def embed_image(self, image: Image.Image) -> np.ndarray:
        img = ensure_rgb(image)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with self._torch.no_grad(), self._torch.amp.autocast(
            self.device, enabled=self.device != "cpu"
        ):
            feat = self.model.encode_image(tensor)
        vec = feat[0].detach().cpu().float().numpy().astype(np.float32)
        return normalize_vector(vec)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with self._torch.no_grad(), self._torch.amp.autocast(
            self.device, enabled=self.device != "cpu"
        ):
            feat = self.model.encode_text(tokens)
        vec = feat[0].detach().cpu().float().numpy().astype(np.float32)
        return normalize_vector(vec)

    def embed_images_batch(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Encode a list of PIL images in batches. Returns (N, dim) array."""
        all_vecs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = self._torch.stack([self.preprocess(ensure_rgb(img)) for img in batch]).to(self.device)
            with self._torch.no_grad(), self._torch.amp.autocast(
                self.device, enabled=self.device != "cpu"
            ):
                feats = self.model.encode_image(tensors)
            vecs = feats.detach().cpu().float().numpy().astype(np.float32)
            # Normalize each vector
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
            all_vecs.append(vecs / norms)
        return np.concatenate(all_vecs, axis=0)

    def embed_texts_batch(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode a list of text strings in batches. Returns (N, dim) array."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with self._torch.no_grad(), self._torch.amp.autocast(
                self.device, enabled=self.device != "cpu"
            ):
                feats = self.model.encode_text(tokens)
            vecs = feats.detach().cpu().float().numpy().astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
            all_vecs.append(vecs / norms)
        return np.concatenate(all_vecs, axis=0)

