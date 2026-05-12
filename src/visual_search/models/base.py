"""Base interfaces for model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from PIL import Image
import numpy as np


@dataclass
class DetectionResult:
    crop: Image.Image
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str
    class_id: int | None = None
    class_name: str | None = None


class Detector(Protocol):
    name: str
    is_mock: bool

    def detect_and_crop(self, image: Image.Image, threshold: float = 0.5) -> list[DetectionResult]:
        ...


class Captioner(Protocol):
    name: str
    is_mock: bool

    def caption(self, image: Image.Image) -> str:
        ...


class Embedder(Protocol):
    name: str
    is_mock: bool
    dim: int

    def embed_image(self, image: Image.Image) -> np.ndarray:
        ...

    def embed_text(self, text: str) -> np.ndarray:
        ...


class Reranker(Protocol):
    name: str
    is_mock: bool

    def rerank(self, query_image: Image.Image, candidates: list[dict]) -> list[dict]:
        ...
