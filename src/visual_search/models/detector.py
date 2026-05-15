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
    """YOLO-based garment detector with cross-class NMS.

    Detection flow (Section 9.2):
      1. Run YOLO inference with a low confidence floor (0.10) to avoid
         missing genuine multi-garment images.
      2. Per-class best-box selection: retain only the single highest-
         confidence box for each detected class.
      3. Cross-class NMS: compare retained boxes pairwise across classes
         using IoU.  If IoU > 0.60, suppress the lower-confidence box.
         This prevents overlapping upper / fullbody crops from confusing
         the retrieval stage.
    """

    name = "yolo"
    is_mock = False

    # Section 9.2 thresholds
    _CONF_FLOOR: float = 0.10        # deliberately low internal floor
    _CROSS_CLASS_IOU_THRESH: float = 0.60

    def __init__(self, weights_path: str, device: str = "cpu", pad: int = 10) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc

        self.model = YOLO(weights_path)
        self.device = device
        self.pad = pad

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(box_a: tuple[int, int, int, int],
             box_b: tuple[int, int, int, int]) -> float:
        """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    def _fallback(self, image: Image.Image) -> list[DetectionResult]:
        crop, bbox = MockDetector(pad=self.pad)._center_crop(image)
        return [DetectionResult(crop=crop, bbox=bbox, confidence=0.0, source="fallback", class_id=None, class_name=None)]

    # ------------------------------------------------------------------
    # Main detection entry-point
    # ------------------------------------------------------------------
    def detect_and_crop(self, image: Image.Image, threshold: float = 0.5) -> list[DetectionResult]:
        img = ensure_rgb(image)
        results = self.model.predict(img, device=self.device, verbose=False)
        if not results:
            return self._fallback(img)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._fallback(img)

        # Use the lower of the UI threshold and the internal floor so that
        # genuine detections are never dropped prematurely.
        effective_threshold = min(threshold, self._CONF_FLOOR)

        # ---- Step 1: Per-class best-box selection ----
        # For each class retain only the single highest-confidence box.
        best_per_class: dict[int, tuple[int, float]] = {}  # cls_id -> (box_idx, conf)
        if hasattr(boxes, "cls") and hasattr(boxes, "conf"):
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < effective_threshold:
                    continue
                cls_id = int(boxes.cls[i])
                if cls_id not in best_per_class or conf > best_per_class[cls_id][1]:
                    best_per_class[cls_id] = (i, conf)

        if not best_per_class:
            return self._fallback(img)

        # Build candidate list *before* padding (raw YOLO coords) for IoU
        # comparison, since padding is a display concern.
        candidates: list[dict] = []
        for cls_id, (i, conf) in best_per_class.items():
            raw_xyxy = tuple(map(int, boxes[i].xyxy[0].tolist()))
            candidates.append({
                "cls_id": cls_id,
                "conf": conf,
                "box_idx": i,
                "raw_bbox": raw_xyxy,
            })

        # ---- Step 2-3: Cross-class NMS ----
        # Sort by confidence descending so the higher-confidence box survives.
        candidates.sort(key=lambda c: c["conf"], reverse=True)
        keep: list[dict] = []
        for cand in candidates:
            suppressed = False
            for kept in keep:
                if self._iou(cand["raw_bbox"], kept["raw_bbox"]) > self._CROSS_CLASS_IOU_THRESH:
                    # cand has lower (or equal) confidence → suppress it
                    suppressed = True
                    break
            if not suppressed:
                keep.append(cand)

        if not keep:
            return self._fallback(img)

        # ---- Build final DetectionResult list ----
        width, height = img.size
        results_list: list[DetectionResult] = []
        for cand in keep:
            rx1, ry1, rx2, ry2 = cand["raw_bbox"]
            x1 = max(0, rx1 - self.pad)
            y1 = max(0, ry1 - self.pad)
            x2 = min(width, rx2 + self.pad)
            y2 = min(height, ry2 + self.pad)

            crop = img.crop((x1, y1, x2, y2))
            class_id = cand["cls_id"]
            class_name = (
                self.model.names.get(class_id)
                if class_id is not None and hasattr(self.model, "names")
                else None
            )
            results_list.append(DetectionResult(
                crop=crop,
                bbox=(x1, y1, x2, y2),
                confidence=cand["conf"],
                source=self.name,
                class_id=class_id,
                class_name=class_name,
            ))

        # Sort by confidence descending for display
        results_list.sort(key=lambda r: r.confidence, reverse=True)
        return results_list
