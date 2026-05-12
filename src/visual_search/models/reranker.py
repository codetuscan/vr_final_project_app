"""Reranker adapters."""

from __future__ import annotations

from PIL import Image

from visual_search.models.base import Reranker


class NoOpReranker:
    name = "noop-reranker"
    is_mock = True

    def rerank(self, query_image: Image.Image, candidates: list[dict]) -> list[dict]:
        return candidates
