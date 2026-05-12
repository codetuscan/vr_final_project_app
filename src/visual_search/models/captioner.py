"""Captioner adapters."""

from __future__ import annotations

from PIL import Image

from visual_search.models.base import Captioner


class MockCaptioner:
    name = "mock-captioner"
    is_mock = True

    def caption(self, image: Image.Image) -> str:
        return "mock caption for clothing item"
