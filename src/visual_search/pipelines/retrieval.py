"""Retrieval pipeline for query-by-image search.

Matches the Condition C pipeline from c-abilation-p.ipynb:
  1. Fine-tuned CLIP encodes query image
  2. HNSW index retrieves top-K candidates via fused embeddings
  3. BLIP ITM re-ranks candidates using image-text matching

Includes embedding caching so the index only needs to be built once.
"""

from __future__ import annotations

import hashlib
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
    itm_score: float = 0.0


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        reranker: Reranker,
        alpha: float = 0.5,
        cache_dir: str = "",
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.alpha = alpha
        self.records: list = []
        self.missing_images = 0
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, n_records: int) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{self.embedder.name}_a{self.alpha}_{n_records}"
        h = hashlib.md5(tag.encode()).hexdigest()[:12]
        return self.cache_dir / f"gallery_emb_{h}.npy"

    def build_index(self, records: Iterable, progress_callback=None) -> None:
        """Build the HNSW index from gallery records.

        Uses batched CLIP encoding for speed. Caches embeddings to disk.
        """
        self.records = list(records)
        self.missing_images = 0
        n = len(self.records)

        # Try loading cached embeddings
        cp = self._cache_path(n)
        if cp and cp.exists():
            matrix = np.load(str(cp))
            if len(matrix) == n:
                self.index.build(matrix)
                if progress_callback:
                    progress_callback(1.0, f"Loaded cached index ({n:,} items)")
                return

        dim = self.embedder.dim
        has_batch = hasattr(self.embedder, "embed_images_batch")

        # --- Collect all images and captions ---
        if progress_callback:
            progress_callback(0.0, "Loading gallery images…")

        images: list[Image.Image] = []
        captions: list[str] = []
        img_valid: list[bool] = []

        for i, record in enumerate(self.records):
            caption = getattr(record, "caption", "") or ""
            captions.append(caption)
            img_path = getattr(record, "image_path", "")
            if img_path and Path(img_path).exists():
                try:
                    images.append(ensure_rgb(Image.open(img_path)))
                    img_valid.append(True)
                except OSError:
                    images.append(Image.new("RGB", (224, 224)))
                    img_valid.append(False)
                    self.missing_images += 1
            else:
                images.append(Image.new("RGB", (224, 224)))
                img_valid.append(False)
                if img_path:
                    self.missing_images += 1

        # --- Batch encode images ---
        if progress_callback:
            progress_callback(0.1, f"Encoding {n:,} images (batched)…")

        if has_batch:
            img_matrix = self.embedder.embed_images_batch(images, batch_size=64)
        else:
            vecs = []
            for i, img in enumerate(images):
                vecs.append(self.embedder.embed_image(img))
                if progress_callback and i % 200 == 0:
                    progress_callback(0.1 + 0.5 * i / n, f"Encoding images ({i:,}/{n:,})…")
            img_matrix = np.vstack(vecs)

        if progress_callback:
            progress_callback(0.6, f"Encoding {n:,} captions (batched)…")

        # --- Batch encode captions ---
        non_empty = [c for c in captions if c.strip()]
        if has_batch and non_empty:
            txt_matrix = self.embedder.embed_texts_batch(captions, batch_size=128)
        else:
            tvecs = []
            for c in captions:
                if c.strip():
                    tvecs.append(self.embedder.embed_text(c))
                else:
                    tvecs.append(np.zeros(dim, dtype=np.float32))
            txt_matrix = np.vstack(tvecs)

        if progress_callback:
            progress_callback(0.9, "Fusing embeddings & building index…")

        # --- Fuse ---
        fused = np.zeros_like(img_matrix)
        for i in range(n):
            if img_valid[i]:
                fused[i] = self.alpha * img_matrix[i] + (1.0 - self.alpha) * txt_matrix[i]
            else:
                fused[i] = txt_matrix[i]

        norms = np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-8)
        matrix = (fused / norms).astype(np.float32)

        self.index.build(matrix)
        if cp:
            np.save(str(cp), matrix)

        if progress_callback:
            progress_callback(1.0, f"Index built ({n:,} items)")


    def search(self, query_image: Image.Image, top_k: int) -> list[RetrievalResult]:
        """Run the full retrieval pipeline:

        1. Encode query image with fine-tuned CLIP
        2. HNSW nearest-neighbor search
        3. BLIP ITM re-ranking on top-K candidates
        """
        if not self.records:
            return []

        query_vec = self.embedder.embed_image(query_image)
        hits = self.index.search(query_vec, top_k)
        candidates: list[dict] = []

        for idx, score in hits:
            record = self.records[idx]
            candidates.append(
                {
                    "rank": 0,
                    "score": score,
                    "item_id": getattr(record, "item_id", ""),
                    "image_path": getattr(record, "image_path", ""),
                    "caption": getattr(record, "caption", ""),
                    "category": getattr(record, "category", ""),
                }
            )

        # BLIP ITM re-ranking (matches notebook's online query flow)
        candidates = self.reranker.rerank(query_image, candidates)

        final: list[RetrievalResult] = []
        for rank, item in enumerate(candidates, start=1):
            final.append(
                RetrievalResult(
                    rank=rank,
                    score=float(item.get("score", 0.0)),
                    item_id=str(item.get("item_id", "")),
                    image_path=str(item.get("image_path", "")),
                    caption=str(item.get("caption", "")),
                    category=str(item.get("category", "")),
                    itm_score=float(item.get("itm_score", 0.0)),
                )
            )
        return final
