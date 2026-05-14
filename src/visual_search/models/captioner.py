"""Captioner adapters — mock and BLIP-2."""

from __future__ import annotations

from PIL import Image

from visual_search.models.base import Captioner


class MockCaptioner:
    """Returns a placeholder caption."""

    name = "mock-captioner"
    is_mock = True

    def caption(self, image: Image.Image) -> str:
        return "mock caption for clothing item"


class Blip2Captioner:
    """BLIP-2 captioner using ``Salesforce/blip2-opt-2.7b``.

    This mirrors the ``caption_batch_fast`` function from the ablation-C
    notebook.  Uses the same prompt template for clothing descriptions.
    """

    name = "blip2"
    is_mock = False

    PROMPT = "Question: Describe this clothing item including color, type, and style. Answer:"

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers, torch and accelerate are required for Blip2Captioner"
            ) from exc

        self._torch = torch
        self.device = device

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        print("[Blip2Captioner] Loaded BLIP-2 model (Salesforce/blip2-opt-2.7b)")

    def caption(self, image: Image.Image) -> str:
        from visual_search.utils import ensure_rgb

        img = ensure_rgb(image)
        inp = self.processor(
            images=img,
            text=self.PROMPT,
            return_tensors="pt",
            padding=True,
        ).to(self.device, self._torch.float16)

        with self._torch.no_grad():
            out = self.model.generate(
                **inp, max_new_tokens=20, do_sample=False, num_beams=1
            )

        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        return text.strip()
