"""Small utilities used across the demo."""

from __future__ import annotations

from typing import Iterable
import numpy as np
from PIL import Image


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _hist_channel(values: np.ndarray, bins: int) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=(0, 255), density=True)
    return hist.astype(np.float32)


def image_to_vector(image: Image.Image, dim: int = 128, bins: int = 16) -> np.ndarray:
    rgb = np.asarray(ensure_rgb(image))
    if rgb.size == 0:
        return np.zeros(dim, dtype=np.float32)

    r = rgb[:, :, 0].reshape(-1)
    g = rgb[:, :, 1].reshape(-1)
    b = rgb[:, :, 2].reshape(-1)

    hist = np.concatenate([
        _hist_channel(r, bins),
        _hist_channel(g, bins),
        _hist_channel(b, bins),
    ])
    mean = np.array([r.mean(), g.mean(), b.mean()], dtype=np.float32) / 255.0
    std = np.array([r.std(), g.std(), b.std()], dtype=np.float32) / 255.0
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    gray_hist = _hist_channel(gray, bins)

    base = np.concatenate([hist, gray_hist, mean, std])
    if base.size >= dim:
        return base[:dim].astype(np.float32)

    reps = int(np.ceil(dim / base.size))
    tiled = np.tile(base, reps)[:dim]
    return tiled.astype(np.float32)


def hash_text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    for token in text.lower().split():
        idx = abs(hash(token)) % dim
        vec[idx] += 1.0
    return vec


def pick_first_existing(paths: Iterable[str]) -> str:
    for path in paths:
        if path:
            return path
    return ""
