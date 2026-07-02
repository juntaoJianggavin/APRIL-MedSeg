# SAM-Med2D MaskGenerator wrapper (pipeline segmenter 端)
# SAM-Med 2D MaskGenerator 封装器 ( pipeline segmenter side ) / SAM-Med2D MaskGenerator wrapper (pipeline segmenter side)
# 参考: https: / / github. com / OpenGVLab / SAM-Med 2D / Reference: https://github.com/OpenGVLab/SAM-Med2D
# Paper: https://arxiv.org/abs/2308.16184
"""SAM-Med2D mask generator for the MLLM grounding pipeline.
    SAM-Med 2D 掩码 generator for the MLLM grounding pipeline。

与 SAM2MaskGenerator / MedSAMMaskGenerator 实现相同的 predict_from_boxes 接口，
可作为 pipeline.py 中 detector → segmenter 的 segmenter 端使用。
Implements the same predict_from_boxes interface as SAM2MaskGenerator /
MedSAMMaskGenerator, usable as the segmenter in pipeline.py.

用法 / Usage (yaml):
    mask_generator:
      type: sammed2d
      checkpoint: ./weights/sam-med2d_b.pth   # 或自动下载 / or auto-download
      device: cuda
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SAMMed2DMaskGenerator:
    """SAM-Med2D box-prompt mask generator。
        SAM-Med 2D box-prompt 掩码 generator 。。
    SAM-Med2D box-prompt mask generator.

    接口与 SAM2MaskGenerator / MedSAMMaskGenerator 一致：
    Interface matches SAM2MaskGenerator / MedSAMMaskGenerator:
        predict_from_boxes(image, boxes) -> (N, H, W) uint8 masks
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cuda",
        image_size: int = 256,
        **kwargs,
    ):
        self.device = device
        self.image_size = image_size
        self.model = None
        self.mock_mode = False

        # Load SAM-Med2D weights
        # SAM-Med2D fine-tunes at image_size=256 (not the default 1024), so we
        # must construct the Sam model with img_size=256 for the checkpoint
        # state_dict to load without size mismatches.
        from functools import partial

        import torch.nn as nn
        from segment_anything import SamPredictor
        from segment_anything.modeling.image_encoder import ImageEncoderViT
        from segment_anything.modeling.mask_decoder import MaskDecoder
        from segment_anything.modeling.prompt_encoder import PromptEncoder
        from segment_anything.modeling.sam import Sam
        from segment_anything.modeling.transformer import TwoWayTransformer

        if not checkpoint:
            from medseg.utils.weight_downloader import ensure_weight
            checkpoint = str(ensure_weight("sam_med2d_vit_b"))

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"SAM-Med2D checkpoint not found: {checkpoint}. "
                f"Download from https://github.com/OpenGVLab/SAM-Med2D"
            )

        _ENCODER_CONFIGS = {
            "vit_b": dict(embed_dim=768, depth=12, num_heads=12,
                          global_attn_indexes=[2, 5, 8, 11]),
            "vit_l": dict(embed_dim=1024, depth=24, num_heads=16,
                          global_attn_indexes=[7, 15, 23, 31]),
            "vit_h": dict(embed_dim=1280, depth=32, num_heads=16,
                          global_attn_indexes=[7, 15, 23, 31]),
        }

        enc_cfg = _ENCODER_CONFIGS[model_type]
        prompt_embed_dim = 256
        vit_patch_size = 16
        img_embedding_size = image_size // vit_patch_size

        sam = Sam(
            image_encoder=ImageEncoderViT(
                depth=enc_cfg["depth"],
                embed_dim=enc_cfg["embed_dim"],
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_heads=enc_cfg["num_heads"],
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=enc_cfg["global_attn_indexes"],
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(img_embedding_size, img_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
        sam.eval()

        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        sam.load_state_dict(ckpt, strict=False)
        sam = sam.to(device).eval()
        self.predictor = SamPredictor(sam)
        logger.info(f"SAM-Med2D loaded: {model_type} from {checkpoint} on {device} (image_size={image_size})")

    def predict_from_boxes(
        self,
        image: np.ndarray,
        boxes: List,
    ) -> np.ndarray:
        """对一张图 + 若干 bbox 生成 mask。
        Generate masks for an image given bounding boxes.

        Args:
            image: (H, W, 3) uint8 RGB
            boxes: BBox 列表（归一化坐标）/ List of BBox (normalised coords)

        Returns:
            masks: (N, H, W) uint8 二值 mask / binary masks
        """
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)

        if self.mock_mode or self.predictor is None:
            return self._mock_predict(image, boxes)

        self.predictor.set_image(image)

        masks_out = []
        for b in boxes:
            x1, y1, x2, y2 = b.to_pixel(w, h)
            box_np = np.array([[x1, y1, x2, y2]])
            mask_logits, _, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False,
            )
            mask = (mask_logits[0] > 0).astype(np.uint8)
            masks_out.append(mask)

        return np.stack(masks_out, axis=0) if masks_out else np.zeros((0, h, w), dtype=np.uint8)

    def _mock_predict(self, image, boxes):
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), dtype=np.uint8)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.to_pixel(w, h)
            masks[i, max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 1
        return masks
