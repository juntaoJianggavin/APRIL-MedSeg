"""LLM4Seg bottleneck — Tang et al., MICCAI 2025.
    LLM4Seg 瓶颈层。

Reference:
    Tang et al., "Pre-Trained LLM is a Semantic-Aware and Generalizable
    Segmentation Booster", MICCAI 2025.
    https://github.com/FengheTan9/LLM4Seg
    arXiv: 2506.18034

Idea: use the last K transformer blocks of a *frozen* pretrained LLM
(Llama / Qwen / Phi / etc.) as a bottleneck refiner for medical image
segmentation. The LLM blocks reason over the flattened spatial tokens at
the bottleneck and the residual is added back to the visual feature.

Procedure:
    1. encoder feature  (B, C, H, W)
    2. 1x1 conv in_proj : C -> D  (D = LLM hidden_size)
    3. flatten + LayerNorm -> tokens (B, N, D) with N = H*W
    4. K frozen LLM transformer blocks (last K of the LLM)
    5. LayerNorm + 1x1 conv out_proj : D -> C
    6. residual add to the original feature
    7. return (B, C, H, W)

The LLM is loaded via ``transformers.AutoModel`` with automatic HF endpoint
fallback (official ``huggingface.co`` first, then ``hf-mirror.com`` on
network errors). See :mod:`medseg.utils.hf_hub`.
No silent fallback to a different model — if the requested LLM cannot be
loaded, a clear RuntimeError is raised.
"""
# Source: https://github.com/FengheTan9/LLM4Seg (MICCAI 2025)

from __future__ import annotations
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import BOTTLENECK_REGISTRY
from medseg.utils.hf_hub import call_with_hf_fallback


def _load_llm_blocks(llm_model: str, num_layers: int, dtype):
    """Load the last ``num_layers`` decoder blocks of an LLM via transformers.
        Load the last ``num_layers`` 解码器。

    Returns (block_list: nn.ModuleList, hidden_size: int).
    Raises RuntimeError on failure (no silent fallback).
    """
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError as e:
        raise RuntimeError(
            "LLM4SegBottleneck requires the `transformers` library. "
            "Install with: pip install transformers"
        ) from e

    try:
        config = call_with_hf_fallback(
            AutoConfig.from_pretrained,
            llm_model,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch config for LLM '{llm_model}'. "
            f"Tried official Hugging Face Hub and mirror fallback. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    # 加载 the 模型 with the requested dtype to 保存 memory; we only use the blocks / Load the model with the requested dtype to save memory; we only use the blocks
    try:
        model = call_with_hf_fallback(
            AutoModel.from_pretrained,
            llm_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LLM weights '{llm_model}'. "
            f"Either: (a) ensure network access to Hugging Face Hub (mirror fallback is automatic), "
            f"(b) pre-cache the weights into ~/.cache/huggingface/hub, "
            f"or (c) supply a local cache via HF_HOME env var. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    # Locate the Transformer 块 list. Architectures vary in attribute names / Locate the transformer block list. Architectures vary in attribute names.
    blocks = None
    for attr_chain in (("layers",), ("model", "layers"), ("transformer", "h"),
                       ("transformer", "blocks"), ("encoder", "layer"), ("h",)):
        cur = model
        ok = True
        for a in attr_chain:
            if hasattr(cur, a):
                cur = getattr(cur, a)
            else:
                ok = False
                break
        if ok and isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
            blocks = cur
            break
    if blocks is None:
        raise RuntimeError(
            f"Could not locate transformer block list inside '{llm_model}'. "
            "Set `transformer_attr_chain=['...']` kwarg if your model uses a "
            "non-standard attribute path."
        )

    selected = nn.ModuleList(list(blocks)[-num_layers:])
    hidden_size = int(getattr(config, "hidden_size",
                              getattr(config, "n_embd",
                                      getattr(config, "d_model", -1))))
    if hidden_size <= 0:
        raise RuntimeError(
            f"Could not infer hidden_size from config of '{llm_model}'."
        )
    return selected, hidden_size


# ---------------------------------------------------------------------------
# Source-code presets — Tang et al. ( MICCAI 2025 ) 来源 uses DeepSeek-R 1-Distill / Source-code presets — Tang et al. (MICCAI 2025) source uses DeepSeek-R1-Distill
# variants of Qwen and Llama as the two LLM families. Pass ` llm _ 模型 = ' deepseek _ qwen ' ` / variants of Qwen and Llama as the two LLM families. Pass `llm_model='deepseek_qwen'`
# ( alias ' qwen ' ) or ` llm _ 模型 = ' deepseek _ llama ' ` ( alias ' llama ' ), OR any HF 模型 id / (alias 'qwen') or `llm_model='deepseek_llama'` (alias 'llama'), OR any HF model id.
# ---------------------------------------------------------------------------
LLM_PRESETS = {
    # DeepSeek-R 1-Distill-Qwen ( as used in the LLM4Seg 来源 ) — the primary ' qwen ' option / DeepSeek-R1-Distill-Qwen (as used in the LLM4Seg source) — the primary 'qwen' option
    'qwen':           'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek_qwen':  'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek_qwen_1_5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek_qwen_7b':   'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'deepseek_qwen_14b':  'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',

    # DeepSeek-R 1-Distill-Llama ( matching family choice in the 来源 ) / DeepSeek-R1-Distill-Llama (matching family choice in the source)
    'llama':              'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek_llama':     'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek_llama_8b':  'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek_llama_70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',

    # Alternates if users explicitly want non-distilled Qwen / Llama
    'qwen_base':          'Qwen/Qwen2.5-1.5B',
    'llama_base':         'meta-llama/Llama-3.2-1B',
}


@BOTTLENECK_REGISTRY.register("llm4seg")
class LLM4SegBottleneck(nn.Module):
    """LLM4Seg (Tang et al., MICCAI 2025) bottleneck.
        LLM4Seg (Tang et al., MICCAI 2025) 瓶颈层。

    The official source code uses DeepSeek-R1-Distill family in two flavors:
        - DeepSeek-R1-Distill-Qwen-1.5B  — preset key 'qwen' / 'deepseek_qwen'
        - DeepSeek-R1-Distill-Llama-8B   — preset key 'llama' / 'deepseek_llama'

    Args:
        in_channels: encoder bottleneck feature channels.
        llm_model: HF model identifier of the LLM to use. Accepts:
            (a) Source-preset key (matches the paper's source code):
                'qwen' / 'deepseek_qwen' / 'deepseek_qwen_1_5b' / '_7b' / '_14b'
                'llama' / 'deepseek_llama' / 'deepseek_llama_8b' / '_70b'
                Also 'qwen_base' / 'llama_base' for non-distilled originals.
            (b) Any explicit HF model id, e.g. 'microsoft/Phi-3-mini-4k-instruct'.
            Default: 'qwen' -> 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'.
        num_llm_layers: how many of the LAST LLM transformer blocks to use
            (default 4). Fewer = lighter; more = stronger reasoning.
        freeze_llm: keep LLM weights frozen (default True — only the in/out
            projection convs train).
        dtype: float dtype for the LLM ('float32' / 'float16' / 'bfloat16').
            Default 'float32' to avoid AMP mismatch with the visual backbone.
        use_residual: add the input feature back as a residual (default True).
        kwargs: forwarded to AutoModel.from_pretrained where applicable.
    """

    def __init__(
        self,
        in_channels: int,
        llm_model: str = "qwen",
        num_llm_layers: int = 4,
        freeze_llm: bool = True,
        dtype: str = "float32",
        use_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        # Resolve source-preset key - > HF 模型 id / Resolve source-preset key -> HF model id
        resolved_llm = LLM_PRESETS.get(llm_model, llm_model)
        self.llm_model = resolved_llm
        self.num_llm_layers = num_llm_layers
        self.use_residual = use_residual

        torch_dtype = {
            "float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "fp32": torch.float32, "fp16": torch.float16,
        }.get(dtype, torch.float32)

        # 加载 frozen LLM blocks ( use the resolved 模型 id ) / Load frozen LLM blocks (use the resolved model id)
        self.llm_blocks, self.hidden_dim = _load_llm_blocks(
            resolved_llm, num_llm_layers, torch_dtype
        )
        if freeze_llm:
            for p in self.llm_blocks.parameters():
                p.requires_grad = False

        # Visual <-> LLM projections (trainable)
        self.in_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.in_norm = nn.LayerNorm(self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Conv2d(self.hidden_dim, in_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # start as identity (residual + zero refinement)

        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def _llm_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run 标记 ( B, N, D ) through the LLM blocks. Handles common 块 APIs。
            Run tokens (B, N, D) through the LLM blocks. Handles common block APIs."""
        x = tokens
        # Cast to LLM dtype if blocks have a different dtype
        block_dtype = next(self.llm_blocks.parameters()).dtype
        x_in = x.to(block_dtype)

        for blk in self.llm_blocks:
            try:
                # Llama / Qwen / Phi style: 返回 tuple ( hidden _ states,... ) / Llama / Qwen / Phi style: returns tuple (hidden_states, ...)
                out = blk(x_in)
            except TypeError:
                # Some blocks need 注意力 _ 掩码, position _ ids etc / Some blocks need attention_mask, position_ids etc.
                # Fall back to passing an all-ones 注意力 掩码 / Fall back to passing an all-ones attention mask
                B, N, _ = x_in.shape
                attn_mask = torch.ones((B, N), device=x_in.device, dtype=torch.long)
                out = blk(x_in, attention_mask=attn_mask)
            if isinstance(out, tuple):
                x_in = out[0]
            elif isinstance(out, dict):
                x_in = out.get("hidden_states", out.get("last_hidden_state", x_in))
            else:
                x_in = out
        # Cast back to the original 输入 dtype / Cast back to the original input dtype
        return x_in.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x if self.use_residual else None

        # 1x1 in-proj + 展平 / 1x1 in-proj + flatten
        h = self.in_proj(x)                            # (B, D, H, W)
        h = h.flatten(2).transpose(1, 2)               # (B, N, D)
        h = self.in_norm(h)

        # Frozen LLM reasoning
        h = self._llm_forward(h)

        h = self.out_norm(h)
        # 重塑 back + out-proj / Reshape back + out-proj
        h = h.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        h = self.out_proj(h)                           # (B, C, H, W)

        if residual is not None:
            return residual + h
        return h
