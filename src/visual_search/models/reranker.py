"""Reranker adapters — NoOp and BLIP-ITM (from c-abilation-p notebook)."""

from __future__ import annotations

from PIL import Image

from visual_search.models.base import Reranker


class NoOpReranker:
    """Pass-through reranker — returns candidates unchanged."""

    name = "noop-reranker"
    is_mock = True

    def rerank(self, query_image: Image.Image, candidates: list[dict]) -> list[dict]:
        return candidates


class BlipItmReranker:
    """BLIP Image-Text Matching reranker.

    Uses ``Salesforce/blip-itm-base-coco`` to compute ITM scores between
    the query image and each candidate's caption, then reorders candidates
    by descending ITM match probability.

    This mirrors the ``itm_rerank`` function from the ablation-C notebook.
    """

    name = "blip-itm"
    is_mock = False

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from transformers import BlipProcessor, BlipForImageTextRetrieval
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for BlipItmReranker"
            ) from exc

        self._torch = torch
        self._F = F
        self.device = device
        self.batch_size = batch_size

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.model = BlipForImageTextRetrieval.from_pretrained(
            "Salesforce/blip-itm-base-coco",
            torch_dtype=torch.float16,
        ).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        print("[BlipItmReranker] Loaded BLIP-ITM model (Salesforce/blip-itm-base-coco)")

    def rerank(self, query_image: Image.Image, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using BLIP ITM scores.

        Candidates with valid captions get an ITM score; those without
        captions keep their original ordering (score = 0).
        """
        if not candidates:
            return candidates

        # Separate candidates into those with and without captions
        valid_caps: list[str] = []
        valid_indices: list[int] = []
        for i, cand in enumerate(candidates):
            cap = cand.get("caption", "")
            if cap and cap.strip():
                valid_caps.append(cap.strip())
                valid_indices.append(i)

        if not valid_caps:
            return candidates

        # Ensure query image is RGB
        from visual_search.utils import ensure_rgb
        qimg = ensure_rgb(query_image)

        # Compute ITM scores in batches
        scores: list[float] = []
        for i in range(0, len(valid_caps), self.batch_size):
            batch_caps = valid_caps[i : i + self.batch_size]
            try:
                inp = self.processor(
                    images=[qimg] * len(batch_caps),
                    text=batch_caps,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                inp["pixel_values"] = inp["pixel_values"].half()

                with self._torch.no_grad():
                    out = self.model(**inp, use_itm_head=True)
                    probs = self._F.softmax(out.itm_score, dim=1)[:, 1]
                    scores.extend(probs.cpu().numpy().tolist())
            except Exception:
                scores.extend([0.0] * len(batch_caps))

        # Assign scores — candidates without captions get 0
        import numpy as np
        full_scores = np.zeros(len(candidates), dtype=np.float32)
        for score, idx in zip(scores, valid_indices):
            full_scores[idx] = score

        # Sort by ITM score descending (stable to preserve original order for ties)
        ranked_order = np.argsort(-full_scores, kind="stable")

        reranked = []
        for new_rank, orig_idx in enumerate(ranked_order):
            cand = dict(candidates[orig_idx])
            cand["itm_score"] = float(full_scores[orig_idx])
            reranked.append(cand)

        return reranked
