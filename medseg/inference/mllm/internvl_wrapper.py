"""InternVL grounding wrapper.
    InternVL grounding 封装器。

# Reference: https://github.com/OpenGVLab/InternVL
# Reference: https://huggingface.co/OpenGVLab/InternVL2_5-8B
# Paper: https://arxiv.org/abs/2312.14238  (InternVL, CVPR 2024)
# Paper: https://arxiv.org/abs/2412.05271  (InternVL 2.5)

InternVL 2 / 2.5 (OpenGVLab) supports natural-language visual grounding.
The native output markup is::

    <ref>{class_name}</ref><box>[[x1, y1, x2, y2]]</box>

Coordinates are absolute integers in ``[0, 1000]``; we divide by 1000
to map into the project-wide normalised :class:`BBox` space.
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from medseg.inference.mllm.base import MLLMGrounder, BBox

logger = logging.getLogger(__name__)


_INTERNVL_BOX_RE = re.compile(
    r"<box>\s*\[?\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?\]?\s*</box>"
)


class InternVLGrounder(MLLMGrounder):
    """InternVL 系列 grounding wrapper（推理-only）。"""

    DEFAULT_PROMPT = (
        "<image>\nPlease provide the bounding box coordinates of the {class_name} "
        "in this medical image, in the format <box>[[x1, y1, x2, y2]]</box>."
    )

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: str | None = None,
        max_new_tokens: int = 128,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template or self.DEFAULT_PROMPT,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            from transformers.modeling_utils import PreTrainedModel
            from transformers.generation.utils import GenerationMixin

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }

            # Trigger dynamic module loading via AutoConfig so InternVL classes
            # are available in sys.modules under transformers_modules.*
            AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)

            # Patch 1: InternVL2.5 remote code omits post_init() which sets
            # all_tied_weights_keys (required by transformers 5.x).
            # Patch _move_missing_keys_from_meta_to_device to tolerate absence.
            _orig_move = PreTrainedModel._move_missing_keys_from_meta_to_device
            def _safe_move(self_m, *a, **kw):
                if not hasattr(self_m, "all_tied_weights_keys"):
                    self_m.all_tied_weights_keys = {}
                return _orig_move(self_m, *a, **kw)
            PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move

            try:
                model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
                    trust_remote_code=True,
                )
            finally:
                PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move

            # Patch 2: Add GenerationMixin to sub-models that lack generate().
            # InternLM2ForCausalLM doesn't inherit GenerationMixin in transformers 5.x.
            # Also set generation_config which is only set in __init__ when can_generate() is True.
            lm = getattr(model, "language_model", None)
            if lm is not None and not hasattr(lm, "generate"):
                lm_cls = type(lm)
                if not issubclass(lm_cls, GenerationMixin):
                    lm_cls.__bases__ = (GenerationMixin,) + lm_cls.__bases__
                    logger.debug(f"Added GenerationMixin to {lm_cls.__name__}")
            if lm is not None and not hasattr(lm, "generation_config"):
                from transformers import GenerationConfig
                try:
                    lm.generation_config = GenerationConfig.from_model_config(lm.config)
                except Exception:
                    lm.generation_config = GenerationConfig()

            self.model = model.eval().to(self.device)
            self.processor = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True, use_fast=False
            )
            logger.info(f"InternVL loaded: {self.model_id} on {self.device}")
        except Exception:
            # Strict: no mock fallback on load failure.
            raise

    # ------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        class_name: str,
    ) -> List[BBox]:
        boxes: List[BBox] = []
        for m in _INTERNVL_BOX_RE.finditer(response):
            x1, y1, x2, y2 = (int(v) for v in m.groups())
            boxes.append(
                BBox(
                    x1=max(0.0, x1 / 1000.0),
                    y1=max(0.0, y1 / 1000.0),
                    x2=min(1.0, x2 / 1000.0),
                    y2=min(1.0, y2 / 1000.0),
                    score=1.0,
                    label=class_name,
                )
            )
        return boxes

    # ------------------------------------------------------------
    def _preprocess_image(self, image: np.ndarray):
        """Convert numpy image to InternVL pixel_values tensor.

        InternVL's chat() expects pixel_values as a tensor of shape
        (num_patches, C, image_size, image_size).  We use a single patch
        (no dynamic resolution tiling) which is sufficient for detection.
        """
        import torch
        import torchvision.transforms as T
        from PIL import Image

        # Get image_size from model config (default 448 for InternVL2.5-8B)
        img_size = 448
        if hasattr(self, "model") and self.model is not None:
            try:
                cfg = self.model.config
                img_size = getattr(cfg, "force_image_size", None) or getattr(cfg.vision_config, "image_size", 448)
            except Exception:
                pass

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image).convert("RGB")

        transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        # Shape: (1, C, H, W) — single patch
        pixel_values = transform(pil_img).unsqueeze(0)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return pixel_values.to(device=device, dtype=dtype)

    # ------------------------------------------------------------
    def _detect_single_class(self, image: np.ndarray, class_name: str) -> List[BBox]:
        if self.mock_mode:
            return self._mock_detect_single_class(image, class_name)
        if self.model is None:
            raise RuntimeError(
                "InternVL model is None but mock_mode is False; "
                "_load_model() did not raise — call site is inconsistent."
            )

        pixel_values = self._preprocess_image(image)
        # InternVL exposes a custom .chat() method (trust_remote_code).
        question = self.prompt_template.format(class_name=class_name)
        generation_config = dict(
            max_new_tokens=self.max_new_tokens, do_sample=False
        )
        response = self.model.chat(
            self.processor,
            pixel_values,
            question,
            generation_config,
        )
        return self._parse_response(response, class_name)
