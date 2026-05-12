"""Detector adapters."""

from __future__ import annotations

import random
from typing import Optional
from PIL import Image

from visual_search.models.base import DetectionResult
from visual_search.utils import ensure_rgb


class MockDetector:
    name = "mock-detector"
    is_mock = True

    def __init__(self, pad: int = 10, jitter: float = 0.08) -> None:
        self.pad = pad
        self.jitter = jitter

    def _center_crop(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
        img = ensure_rgb(image)
        width, height = img.size
        crop_w = int(width * 0.7)
        crop_h = int(height * 0.7)
        dx = int((random.random() * 2 - 1) * self.jitter * width)
        dy = int((random.random() * 2 - 1) * self.jitter * height)

        x1 = max(0, int((width - crop_w) / 2 + dx))
        y1 = max(0, int((height - crop_h) / 2 + dy))
        x2 = min(width, x1 + crop_w)
        y2 = min(height, y1 + crop_h)

        crop = img.crop((x1, y1, x2, y2))
        return crop, (x1, y1, x2, y2)

    def detect_and_crop(self, image: Image.Image, threshold: float = 0.5) -> list[DetectionResult]:
        crop, bbox = self._center_crop(image)
        return [DetectionResult(crop=crop, bbox=bbox, confidence=0.5, source=self.name, class_id=None, class_name=None)]


class YoloDetector:
    name = "yolo"
    is_mock = False

    def __init__(self, weights_path: str, device: str = "cpu", pad: int = 10) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc

        self.model = YOLO(weights_path)
        self.device = device
        self.pad = pad

    def _fallback(self, image: Image.Image) -> list[DetectionResult]:
        crop, bbox = MockDetector(pad=self.pad)._center_crop(image)
        return [DetectionResult(crop=crop, bbox=bbox, confidence=0.0, source="fallback", class_id=None, class_name=None)]

    def detect_and_crop(self, image: Image.Image, threshold: float = 0.5) -> list[DetectionResult]:
        img = ensure_rgb(image)
        results = self.model.predict(img, device=self.device, verbose=False)
        if not results:
            return self._fallback(img)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._fallback(img)

        best_boxes = []
        if hasattr(boxes, "cls") and hasattr(boxes, "conf"):
            cls_to_best_box = {}
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < threshold:
                    continue
                cls_id = int(boxes.cls[i])
                if cls_id not in cls_to_best_box or conf > cls_to_best_box[cls_id][1]:
                    cls_to_best_box[cls_id] = (i, conf)
            
            for cls_id, (i, conf) in cls_to_best_box.items():
                best_boxes.append(boxes[i])
        
        if not best_boxes:
            return self._fallback(img)

        results_list = []
        width, height = img.size
        for box in best_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(width, x2 + self.pad)
            y2 = min(height, y2 + self.pad)

            crop = img.crop((x1, y1, x2, y2))
            confidence = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            class_id = int(box.cls[0]) if hasattr(box, "cls") else None
            class_name = self.model.names.get(class_id) if class_id is not None and hasattr(self.model, "names") else None
            results_list.append(DetectionResult(
                crop=crop, 
                bbox=(x1, y1, x2, y2), 
                confidence=confidence, 
                source=self.name,
                class_id=class_id,
                class_name=class_name
            ))

        return results_list
