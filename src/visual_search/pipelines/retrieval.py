"""Retrieval pipeline for query-by-image search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image

from visual_search.models.base import Embedder, Reranker
from visual_search.retrieval.index import VectorIndex
from visual_search.utils import normalize_vector, ensure_rgb


@dataclass
class RetrievalResult:
    rank: int
    score: float
    item_id: str
    image_path: str
    caption: str
    category: str


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        reranker: Reranker,
        alpha: float = 0.5,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.alpha = alpha
        self.records: list = []
        self.missing_images = 0

    def _load_image(self, path_str: str) -> Image.Image:
        path = Path(path_str)
        if not path.exists():
            self.missing_images += 1
            return Image.new("RGB", (224, 224), color=(0, 0, 0))
        try:
            return ensure_rgb(Image.open(path))
        except OSError:
            self.missing_images += 1
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def build_index(self, records: Iterable) -> None:
        vectors: list[np.ndarray] = []
        self.records = []
        self.missing_images = 0

        for record in records:
            image = self._load_image(getattr(record, "image_path"))
            caption = getattr(record, "caption", "") or ""
            img_vec = self.embedder.embed_image(image)
            if caption:
                txt_vec = self.embedder.embed_text(caption)
            else:
                txt_vec = np.zeros(self.embedder.dim, dtype=np.float32)
            fused = self.alpha * img_vec + (1.0 - self.alpha) * txt_vec
            vectors.append(normalize_vector(fused))
            self.records.append(record)

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            self.index.build(matrix)

    def search(self, query_image: Image.Image, top_k: int) -> list[RetrievalResult]:
        if not self.records:
            return []

        query_vec = self.embedder.embed_image(query_image)
        hits = self.index.search(query_vec, top_k)
        results: list[dict] = []

        for idx, score in hits:
            record = self.records[idx]
            results.append(
                {
                    "rank": 0,
                    "score": score,
                    "item_id": getattr(record, "item_id", ""),
                    "image_path": getattr(record, "image_path", ""),
                    "caption": getattr(record, "caption", ""),
                    "category": getattr(record, "category", ""),
                }
            )

        results = self.reranker.rerank(query_image, results)
        final: list[RetrievalResult] = []
        for rank, item in enumerate(results, start=1):
            final.append(
                RetrievalResult(
                    rank=rank,
                    score=float(item.get("score", 0.0)),
                    item_id=str(item.get("item_id", "")),
                    image_path=str(item.get("image_path", "")),
                    caption=str(item.get("caption", "")),
                    category=str(item.get("category", "")),
                )
            )
        return final
