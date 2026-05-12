"""Vector index backends for nearest-neighbor search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


class VectorIndex(Protocol):
    name: str

    def build(self, vectors: np.ndarray) -> None:
        ...

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        ...


@dataclass
class HnswConfig:
    m: int = 32
    ef: int = 64
    ef_construction: int = 200


class HnswIndex:
    name = "hnsw"

    def __init__(self, dim: int, config: HnswConfig | None = None) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise RuntimeError("hnswlib is not installed") from exc

        self._hnswlib = hnswlib
        self.dim = dim
        self.config = config or HnswConfig()
        self.index = self._hnswlib.Index(space="cosine", dim=dim)

    def build(self, vectors: np.ndarray) -> None:
        self.index.init_index(
            max_elements=len(vectors),
            ef_construction=self.config.ef_construction,
            M=self.config.m,
        )
        self.index.add_items(vectors, np.arange(len(vectors)))
        self.index.set_ef(self.config.ef)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query_vec = query.reshape(1, -1)
        labels, distances = self.index.knn_query(query_vec, k=top_k)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            results.append((int(idx), float(1.0 - dist)))
        return results


class BruteForceIndex:
    name = "brute-force"

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.vectors is None:
            return []
        scores = self.vectors @ query
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]
