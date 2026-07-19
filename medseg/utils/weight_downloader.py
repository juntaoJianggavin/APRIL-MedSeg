"""Unified weight downloader for medseg.
    统一的 权重 downloader for medseg。

Each method in the project that needs a non-trivial pretrained checkpoint
(MedSAM, GroundingDINO, GloVe, etc.) registers a :class:`WeightSource`
here. At runtime callers invoke :func:`ensure_weight` with the registry
key; the downloader tries every URL in order (HuggingFace Hub first,
then direct HTTP mirrors), caches the result under
``$MEDSEG_WEIGHT_CACHE`` (or ``~/.cache/medseg/weights``), and returns the
local path.

When **every** automated source fails, the function raises
:class:`WeightDownloadError` carrying:
    * the canonical manual download URL,
    * the exact target path the user should place the file at,
    * any extra license / token instructions.

So a yaml that names a method requiring weights either Just Works (auto
download) or fails with an actionable message telling the user where to
get the file and where to put it. Per project policy there is no silent
substitution of a different checkpoint or random initialisation.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 数据 classes / Data classes
# ----------------------------------------------------------------------


@dataclass
class WeightSource:
    """A single registered checkpoint, with auto-download + manual fallback URL.
        A single registered 检查点, with auto-download + manual fallback URL。

    Attributes
    ----------
    name
        Registry key (e.g. ``"medsam_vit_b"``).
    filename
        File name written into the cache directory.
    sources
        Ordered list of auto-downloadable URLs / HF specs. Each entry is
        a callable ``(target_path: Path) -> None`` that fetches the file
        when invoked. Helpers :func:`_hf_file`, :func:`_http` produce
        these.
    sha256
        Optional sha256 hex digest of the expected file.
    manual_url
        Human-facing URL to show the user when every automatic source
        fails (e.g. the official GitHub release page or Google Drive
        folder).
    manual_instructions
        Extra multi-line text shown to the user on failure (license /
        HF token / decompression instructions).
    size_mb
        Rough file size for the log message; ``None`` if unknown.
    """

    name: str
    filename: str
    sources: List[Callable[[Path], None]] = field(default_factory=list)
    sha256: Optional[str] = None
    manual_url: str = ""
    manual_instructions: str = ""
    size_mb: Optional[int] = None


class WeightDownloadError(RuntimeError):
    """Raised when a required checkpoint cannot be auto-downloaded.
        Raised when a required 检查点 cannot be auto-downloaded。

    The message always contains the canonical manual download URL and the
    exact target path so the user can drop the file in by hand.
    """


# ----------------------------------------------------------------------
# Cache root
# ----------------------------------------------------------------------


def default_cache_root() -> Path:
    """返回 ` ` $ MEDSEG _ 权重 _ CACHE ` ` or ` ` ~ /. cache / medseg / 权重 ` `。
        Return ``$MEDSEG_WEIGHT_CACHE`` or ``~/.cache/medseg/weights``."""
    env = os.environ.get("MEDSEG_WEIGHT_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "medseg" / "weights"


# ----------------------------------------------------------------------
# 来源 helpers — each 返回 a callable that performs one fetch / Source helpers — each returns a callable that performs one fetch
# ----------------------------------------------------------------------


def _hf_file(repo_id: str, filename: str, repo_type: str = "model") -> Callable[[Path], None]:
    """Build a fetcher that pulls ``filename`` from a HuggingFace repo."""

    def _fetch(target: Path) -> None:
        from medseg.utils.hf_hub import hf_hub_download_file

        local = hf_hub_download_file(
            repo_id,
            filename,
            repo_type=repo_type,
            cache_dir=target.parent.parent,
            local_dir=target.parent,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may write to a different filename if HF caches by
        # blob hash; make sure the final path matches what the caller wants.
        if Path(local) != target:
            try:
                Path(local).replace(target)
            except OSError:
                # Cross-device rename: fall back to copy + unlink.
                import shutil
                shutil.copyfile(local, target)
                Path(local).unlink(missing_ok=True)

    _fetch.__name__ = f"hf_file:{repo_id}/{filename}"
    return _fetch


def _http(url: str) -> Callable[[Path], None]:
    """Build a fetcher that downloads ` ` url ` ` to the 目标 path。
        Build a fetcher that downloads ``url`` to the target path."""

    def _fetch(target: Path) -> None:
        import urllib.request

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    if total and read % (16 * 1024 * 1024) < 1024 * 1024:
                        pct = 100.0 * read / total
                        logger.info(f"  {url}: {read/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)")
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    _fetch.__name__ = f"http:{url}"
    return _fetch


def _http_zip_extract(url: str, inner_file: str) -> Callable:
    """Download a zip archive from * url * and 提取 * inner _ file *。
        Download a zip archive from *url* and extract *inner_file*."""
    def _fetch(target: Path) -> None:
        import tempfile, zipfile, urllib.request, shutil
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            urllib.request.urlretrieve(url, tmp_path)
            with zipfile.ZipFile(tmp_path) as zf:
                candidates = [n for n in zf.namelist() if n.endswith(inner_file)]
                if not candidates:
                    raise RuntimeError(
                        f"File '{inner_file}' not found in zip {url}. "
                        f"Available: {[n for n in zf.namelist() if n.endswith(('.pth','.txt'))]}"
                    )
                with zf.open(candidates[0]) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    _fetch.__name__ = f"http_zip:{url}/{inner_file}"
    return _fetch


def _gdrive_folder_file(folder_id: str, filename: str) -> Callable:
    """Download a single file from a public Google Drive folder.

    Uses ``gdown`` for reliable large-file downloads with confirmation
    token handling.
    """
    def _fetch(target: Path) -> None:
        try:
            import gdown
        except ImportError:
            raise RuntimeError(
                "gdown is required for Google Drive downloads. "
                "Install with: pip install gdown"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id=&export=download"
        # Search the folder for the file ID
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        # Use gdown with folder download
        gdown.download_folder(folder_url, output=str(target.parent),
                              quiet=False, remaining_ok=True)
        # Find the downloaded file
        candidates = list(target.parent.rglob(filename))
        if not candidates:
            # Try with the same name
            candidates = list(target.parent.rglob("*.pth"))
        if not candidates:
            raise RuntimeError(
                f"File '{filename}' not found in downloaded folder {target.parent}. "
                f"Available: {[p.name for p in target.parent.rglob('*.pth')]}"
            )

        import shutil
        shutil.copy2(str(candidates[0]), str(target))

    _fetch.__name__ = f"gdrive_folder:{folder_id}/{filename}"
    return _fetch


def _gdrive_file(file_id: str) -> Callable:
    """Download a single file from Google Drive by file ID.

    Uses ``gdown`` for reliable large-file downloads with confirmation
    token handling (handles the "virus scan" interstitial page that
    ``torch.hub.load_state_dict_from_url`` cannot handle).
    """
    def _fetch(target: Path) -> None:
        try:
            import gdown
        except ImportError:
            raise RuntimeError(
                "gdown is required for Google Drive downloads. "
                "Install with: pip install gdown"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(target), quiet=False)
        if not target.exists():
            raise RuntimeError(
                f"gdown reported success but file not found at {target}"
            )

    _fetch.__name__ = f"gdrive_file:{file_id}"
    return _fetch


# ----------------------------------------------------------------------
# 注册表 / Registry
# ----------------------------------------------------------------------

WEIGHT_REGISTRY: dict[str, WeightSource] = {}


def register(src: WeightSource) -> WeightSource:
    """注册 a 权重 来源 ( idempotent on repeated import )。
        Register a weight source (idempotent on repeated import)."""
    WEIGHT_REGISTRY[src.name] = src
    return src


# --- MedSAM (Ma et al., Nature Communications 2024) ----------------------
register(WeightSource(
    name="medsam_vit_b",
    filename="medsam_vit_b.pth",
    sources=[
        _hf_file("SansuiHan/medical_models", "medsam_vit_b.pth"),
    ],
    manual_url=(
        "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN "
        "or https://zenodo.org/records/10689643"
    ),
    manual_instructions=(
        "Download medsam_vit_b.pth from the official MedSAM release and "
        "place it at the printed cache path. The HF SamModel format at "
        "flaviagiammarino/medsam-vit-base uses different key names and is "
        "not bit-equivalent — do not substitute it."
    ),
    size_mb=375,
))


# --- GroundingDINO (Liu et al., ECCV 2024) -------------------------------
register(WeightSource(
    name="groundingdino_swint_ogc",
    filename="groundingdino_swint_ogc.pth",
    sources=[
        _hf_file("ShilongLiu/GroundingDINO", "groundingdino_swint_ogc.pth"),
    ],
    manual_url=(
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ),
    manual_instructions=(
        "Swin-T variant of GroundingDINO (open-set object detector). "
        "Used as the prompt source for MedSAM/SAM2 in detector→segmenter "
        "pipelines (configs/training_paradigms/text_guided/synapse_grounding_dino_*.yaml)."
    ),
    size_mb=694,
))

register(WeightSource(
    name="groundingdino_swinb_cogcoor",
    filename="groundingdino_swinb_cogcoor.pth",
    sources=[
        _hf_file("ShilongLiu/GroundingDINO", "groundingdino_swinb_cogcoor.pth"),
    ],
    manual_url=(
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    ),
    manual_instructions="Swin-B variant of GroundingDINO.",
    size_mb=938,
))


# --- SAM ViT-B (Kirillov et al., ICCV 2023) ------------------------------
register(WeightSource(
    name="sam_vit_b",
    filename="sam_vit_b_01ec64.pth",
    sources=[
        _http("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    ],
    manual_url="https://github.com/facebookresearch/segment-anything#model-checkpoints",
    manual_instructions="Vanilla SAM ViT-B; required by SaLIP when sam_variant='vit_b'.",
    size_mb=375,
))

register(WeightSource(
    name="sam_vit_l",
    filename="sam_vit_l_0b3195.pth",
    sources=[
        _http("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    ],
    manual_url="https://github.com/facebookresearch/segment-anything#model-checkpoints",
    manual_instructions="Vanilla SAM ViT-L.",
    size_mb=1250,
))

register(WeightSource(
    name="sam_vit_h",
    filename="sam_vit_h_4b8939.pth",
    sources=[
        _http("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
    ],
    manual_url="https://github.com/facebookresearch/segment-anything#model-checkpoints",
    manual_instructions="Vanilla SAM ViT-H (default backbone for many med-seg papers).",
    size_mb=2560,
))


# --- SAM-Med2D ViT-B (Cheng et al., 2024) --------------------------------
# Official OpenGVLab fine-tune on 4. 6M 医学的 image-mask pairs. The released / Official OpenGVLab fine-tune on 4.6M medical image-mask pairs. The released
# checkpoint adds per-block bottleneck adapters and a custom prompt 编码器 / checkpoint adds per-block bottleneck adapters and a custom prompt encoder
# tuned for 医学的 clicks / boxes; it is NOT bit-equivalent to 基础版 SAM / tuned for medical clicks/boxes; it is NOT bit-equivalent to vanilla SAM.
register(WeightSource(
    name="sam_med2d_vit_b",
    filename="sam-med2d_b.pth",
    sources=[
        _hf_file("schengal1/SAM-Med2D_model", "sam-med2d_b.pth"),
        _hf_file("OpenGVLab/SAM-Med2D", "sam-med2d_b.pth"),
    ],
    manual_url=(
        "https://huggingface.co/OpenGVLab/SAM-Med2D/resolve/main/sam-med2d_b.pth "
        "or https://github.com/OpenGVLab/SAM-Med2D#-weights"
    ),
    manual_instructions=(
        "Cheng et al., 'SAM-Med2D' (arXiv 2308.16184, 2024). The official "
        "ViT-B checkpoint is fine-tuned on 4.6M medical image-mask pairs and "
        "supports point/box/mask prompts. Drop the file at the printed cache "
        "path. The vanilla SAM ViT-B weight cannot be substituted."
    ),
    size_mb=2467,
))


# --- GloVe 6B 300d (Pennington et al., EMNLP 2014) -----------------------
register(WeightSource(
    name="glove_6b_300d",
    filename="glove.6B.300d.txt",
    sources=[
        _hf_file("stanfordnlp/glove", "glove.6B.300d.txt", repo_type="dataset"),
        _http_zip_extract(
            "https://nlp.stanford.edu/data/glove.6B.zip",
            "glove.6B.300d.txt",
        ),
    ],
    manual_url="https://nlp.stanford.edu/data/glove.6B.zip",
    manual_instructions=(
        "Stanford GloVe 6B 300-dim word embeddings, used by TGANet's "
        "EmbeddingFeatureFusion. Download the zip, extract "
        "glove.6B.300d.txt, and drop it at the printed path. The zip is "
        "822 MB; just the 300-d file is ~990 MB uncompressed."
    ),
    size_mb=1000,
))


# --- MediSee composite (LLaVA-Med + CLIP + MediSee fine-tune) ------------
# The 4 components are handled separately by medseg / 推理 / mllm / medisee / / The 4 components are handled separately by medseg/inference/mllm/medisee/
# 权重 _ loader. py ( snapshot _ download for HF dirs ). Registered here only / weights_loader.py (snapshot_download for HF dirs). Registered here only
# so ` python - m medseg. utils. 权重 _ downloader - - list ` enumerates them / so `python -m medseg.utils.weight_downloader --list` enumerates them.
register(WeightSource(
    name="medisee_llava_med",
    filename="(snapshot)",
    sources=[],
    manual_url="https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b",
    manual_instructions=(
        "LLaVA-Med backbone for MediSee. Auto-downloaded via "
        "huggingface_hub.snapshot_download by the MediSee wrapper. "
        "~14 GB; requires HF account."
    ),
    size_mb=14000,
))

register(WeightSource(
    name="medisee_finetune",
    filename="(snapshot)",
    sources=[],
    manual_url="https://huggingface.co/Carryyy/MediSee",
    manual_instructions=(
        "MediSee fine-tune weights (ACM MM 2025). "
        "Auto-downloaded by the MediSee wrapper."
    ),
    size_mb=3000,
))


# --- Swin-Tiny ImageNet (SwinUNet / TransNuSeg) ---------------------------
register(WeightSource(
    name="swin_tiny_patch4_window7_224",
    filename="swin_tiny_patch4_window7_224.pth",
    sources=[
        _http("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"),
    ],
    manual_url="https://github.com/SwinTransformer/storage/releases/tag/v1.0.0",
    manual_instructions=(
        "Swin-Tiny ImageNet pretrained weights (used by SwinUNet, TransNuSeg). "
        "Download swin_tiny_patch4_window7_224.pth from the SwinTransformer "
        "storage releases page and place at the printed cache path."
    ),
    size_mb=110,
))


# --- TransUNet R50+ViT-B/16 (JAX .npz) ------------------------------------
register(WeightSource(
    name="transunet_r50_vit_b16",
    filename="R50+ViT-B_16.npz",
    sources=[
        _http("https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"),
    ],
    manual_url="https://console.cloud.google.com/storage/browser/vit_models/imagenet21k/",
    manual_instructions=(
        "TransUNet R50+ViT-B/16 JAX checkpoint (~346 MB). "
        "Download R50+ViT-B_16.npz and place at the printed cache path."
    ),
    size_mb=346,
))


# --- VMamba Tiny (VMUNet) -------------------------------------------------
register(WeightSource(
    name="vmunet_vmamba_tiny",
    filename="vmunet_vmamba_tiny.pth",
    sources=[
        _http("https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"),
    ],
    manual_url="https://github.com/MzeroMiko/VMamba/releases",
    manual_instructions=(
        "VMamba-Tiny ImageNet pretrained weights for VMUNet. "
        "Download vssmtiny_dp01_ckpt_epoch_292.pth from "
        "https://github.com/MzeroMiko/VMamba/releases/tag/%23v0cls "
        "and place at the printed cache path."
    ),
    size_mb=120,
))


# --- RWKV-UNet 编码器 / --- RWKV-UNet encoder pretrained (T / S / B variants) -------------------
_RWKV_GDRIVE_FOLDER = "1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e"

register(WeightSource(
    name="rwkv_unet_encoder_b",
    filename="rwkv_unet_encoder_b.pth",
    sources=[
        _gdrive_folder_file(_RWKV_GDRIVE_FOLDER, "netB.pth"),
    ],
    manual_url="https://drive.google.com/drive/folders/1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e",
    manual_instructions=(
        "RWKV-UNet Base encoder pretrained weights (ImageNet). "
        "Download netB.pth from the Google Drive folder and "
        "rename to rwkv_unet_encoder_b.pth, then place at the printed cache path."
    ),
    size_mb=120,
))

register(WeightSource(
    name="rwkv_unet_encoder_s",
    filename="rwkv_unet_encoder_s.pth",
    sources=[
        _gdrive_folder_file(_RWKV_GDRIVE_FOLDER, "netS.pth"),
    ],
    manual_url="https://drive.google.com/drive/folders/1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e",
    manual_instructions=(
        "RWKV-UNet Small encoder pretrained weights (ImageNet). "
        "Download netS.pth from the Google Drive folder and "
        "rename to rwkv_unet_encoder_s.pth, then place at the printed cache path."
    ),
    size_mb=80,
))

register(WeightSource(
    name="rwkv_unet_encoder_t",
    filename="rwkv_unet_encoder_t.pth",
    sources=[
        _gdrive_folder_file(_RWKV_GDRIVE_FOLDER, "netT.pth"),
    ],
    manual_url="https://drive.google.com/drive/folders/1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e",
    manual_instructions=(
        "RWKV-UNet Tiny encoder pretrained weights (ImageNet). "
        "Download netT.pth from the Google Drive folder and "
        "rename to rwkv_unet_encoder_t.pth, then place at the printed cache path."
    ),
    size_mb=50,
))

# 遗留 alias / Legacy alias
register(WeightSource(
    name="rwkv_unet_encoder",
    filename="rwkv_unet_encoder.pth",
    sources=[],
    manual_url="https://drive.google.com/drive/folders/1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e",
    manual_instructions=(
        "RWKV-UNet encoder pretrained weights (legacy key). "
        "Use rwkv_unet_encoder_b / _s / _t for auto-download. "
        "Download from the Google Drive link and place at the printed cache path."
    ),
    size_mb=100,
))


# --- PVTv2-B3 ImageNet (FCBFormer) ----------------------------------------
register(WeightSource(
    name="pvtv2_b3",
    filename="pvt_v2_b3.pth",
    sources=[
        _hf_file("timm/pvt_v2_b3.in1k", "model.safetensors"),
    ],
    manual_url="https://huggingface.co/timm/pvt_v2_b3.in1k",
    manual_instructions=(
        "PVTv2-B3 ImageNet pretrained weights (used by FCBFormer TB branch). "
        "Download from the HuggingFace timm model hub "
        "(https://huggingface.co/timm/pvt_v2_b3.in1k) "
        "and place at the printed cache path."
    ),
    size_mb=180,
))


# - - - HoverNet-Lite 预训练 ( CoNIC challenge ) - - - - - - - - - - - - - - - - - - - - - - - - - - - - / --- HoverNet-Lite pretrained (CoNIC challenge) ----------------------------
register(WeightSource(
    name="hovernet_lite_pretrained",
    filename="hovernet_lite_pretrained.pth",
    sources=[],
    manual_url="https://github.com/vqdang/hover_net",
    manual_instructions=(
        "HoverNet-Lite pretrained weights for nuclei segmentation. "
        "Download from the HoVerNet repo (https://github.com/vqdang/hover_net) "
        "or the CoNIC challenge (https://github.com/TissueImageAnalytics/CoNIC). "
        "Available formats: .npz (pannuke/monusac). "
        "Convert to .pth and place at the printed cache path."
    ),
    size_mb=50,
))


# --- CSWin-Tiny ImageNet (CSWin-UNet) ------------------------------------
register(WeightSource(
    name="cswin_tiny_224",
    filename="cswin_tiny_224.pth",
    sources=[
        _http("https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth"),
    ],
    manual_url="https://github.com/microsoft/CSWin-Transformer",
    manual_instructions=(
        "CSWin-Tiny ImageNet pretrained weights for CSWin-UNet. "
        "Download cswin_tiny_224.pth from the Microsoft CSWin-Transformer "
        "releases page and place at the printed cache path."
    ),
    size_mb=85,
))


# - - - Mamba-UNet 预训练 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - / --- Mamba-UNet pretrained ------------------------------------------------
register(WeightSource(
    name="mamba_unet_pretrained",
    filename="mamba_unet_pretrained.pth",
    sources=[
        _http("https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"),
    ],
    manual_url="https://github.com/MzeroMiko/VMamba/releases",
    manual_instructions=(
        "Mamba-UNet pretrained encoder weights (VMamba-Tiny ImageNet). "
        "Download vssmtiny_dp01_ckpt_epoch_292.pth from "
        "https://github.com/MzeroMiko/VMamba/releases/tag/%23v0cls "
        "and place at the printed cache path."
    ),
    size_mb=120,
))


# - - - DA-TransUNet R50-ViT 预训练 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - / --- DA-TransUNet R50-ViT pretrained --------------------------------------
register(WeightSource(
    name="da_transunet_r50_vit",
    filename="da_transunet_r50_vit.npz",
    sources=[
        _http("https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"),
    ],
    manual_url="https://console.cloud.google.com/storage/browser/vit_models/imagenet21k/",
    manual_instructions=(
        "DA-TransUNet R50-ViT pretrained weights. "
        "Uses the same R50+ViT-B_16 JAX checkpoint as TransUNet (~346 MB). "
        "Download R50+ViT-B_16.npz from the GCS link above "
        "and place at the printed cache path."
    ),
    size_mb=346,
))


# --- Res2Net-50 (CFANet) --------------------------------------------------
register(WeightSource(
    name="res2net50_v1b_26w_4s",
    filename="res2net50_v1b_26w_4s-3cf99910.pth",
    sources=[],
    manual_url="https://github.com/Res2Net/Res2Net-PretrainedModels",
    manual_instructions=(
        "Res2Net-50 v1b (baseWidth=26, scale=4) ImageNet pretrained weights "
        "for CFANet. Download res2net50_v1b_26w_4s-3cf99910.pth from the "
        "Res2Net-PretrainedModels repo and place at the printed cache path."
    ),
    size_mb=97,
))


# --- Foundation model encoders -------------------------------------------
# Foundation encoders (in medseg/models/encoders/foundation/) auto-download
# through HuggingFace ``transformers``/``open_clip``, ``timm``, or Google
# Drive at runtime.  Registered here so ``python -m medseg.utils.weight_downloader
# list`` enumerates *every* weight the project needs.  ``sources=[]`` means
# the download is delegated to a third-party library (HF/timm); non-empty
# ``sources`` are fetched by ``ensure_weight()`` directly.

# ── General Medical ──────────────────────────────────────────────────────
register(WeightSource(
    name="biomedclip",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    manual_instructions="BiomedCLIP (Microsoft). Auto-downloaded via open_clip at runtime.",
    size_mb=420,
))

register(WeightSource(
    name="medsiglip",
    filename="(HF auto-download, gated)",
    sources=[],
    manual_url="https://huggingface.co/google/medsiglip-448",
    manual_instructions="MedSigLIP (Google). Auto-downloaded via transformers. Requires HF token (gated repo).",
    size_mb=870,
))

register(WeightSource(
    name="medclip",
    filename="(GCS auto-download)",
    sources=[],
    manual_url="https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip",
    manual_instructions="MedCLIP (Wang et al.). Auto-downloaded from GCS at runtime; extracts pytorch_model.bin from zip.",
    size_mb=350,
))

# ── Pathology ────────────────────────────────────────────────────────────
register(WeightSource(
    name="phikon",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/owkin/phikon",
    manual_instructions="Phikon (Owkin, pathology ViT-B/16). Auto-downloaded via transformers at runtime.",
    size_mb=340,
))

register(WeightSource(
    name="phikon_v2",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/owkin/phikon-v2",
    manual_instructions="Phikon-v2 (Owkin). Auto-downloaded via transformers at runtime.",
    size_mb=600,
))

register(WeightSource(
    name="plip",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/vinid/plip",
    manual_instructions="PLIP (Stanford, pathology CLIP ViT-B/16). Auto-downloaded via transformers at runtime.",
    size_mb=600,
))

register(WeightSource(
    name="musk",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/xiangjx/musk",
    manual_instructions="MUSK (pathology multimodal). Auto-downloaded via transformers at runtime.",
    size_mb=1200,
))

register(WeightSource(
    name="uni",
    filename="(HF auto-download, gated)",
    sources=[],
    manual_url="https://huggingface.co/MahmoodLab/UNI",
    manual_instructions="UNI (Mahmood Lab, pathology ViT-B/16). Auto-downloaded via transformers. Requires HF token (gated repo).",
    size_mb=340,
))

register(WeightSource(
    name="keep",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/Astaxanthin/KEEP",
    manual_instructions="KEEP (pathology ViT). Auto-downloaded via hf_hub_download at runtime.",
    size_mb=1660,
))

# ── Radiology ────────────────────────────────────────────────────────────
register(WeightSource(
    name="raddino",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/microsoft/rad-dino",
    manual_instructions="RadDINO (Microsoft, chest X-ray ViT-B/16). Auto-downloaded via transformers at runtime.",
    size_mb=340,
))

register(WeightSource(
    name="omnirad",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/Snarcy/OmniRad-base",
    manual_instructions="OmniRad (radiology ViT). Auto-downloaded via transformers at runtime.",
    size_mb=340,
))

register(WeightSource(
    name="chexzero",
    filename="(timm auto-download)",
    sources=[],
    manual_url="https://github.com/rajpurkarlab/CheXzero",
    manual_instructions="CheXzero (chest X-ray CLIP ViT-B/32). pretrained=True loads timm CLIP weights; for the actual CheXzero checkpoint download from GitHub and pass via pretrained_path.",
    size_mb=350,
))

register(WeightSource(
    name="biovil",
    filename="(timm auto-download)",
    sources=[],
    manual_url="https://github.com/microsoft/hi-ml",
    manual_instructions="BioViL (chest X-ray ResNet-50). pretrained=True loads timm ImageNet weights; for the actual BioViL checkpoint download from GitHub and pass via pretrained_path.",
    size_mb=100,
))

# ── Ophthalmology ────────────────────────────────────────────────────────
register(WeightSource(
    name="retfound",
    filename="(HF auto-download, gated)",
    sources=[],
    manual_url="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP",
    manual_instructions="RETFound (retinal ViT-L). Auto-downloaded via transformers. Requires HF token (gated repo).",
    size_mb=1200,
))

register(WeightSource(
    name="retfound_dinov2",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai",
    manual_instructions="RETFound-DINOv2 (retinal ViT-L). Auto-downloaded via transformers at runtime.",
    size_mb=1200,
))

register(WeightSource(
    name="flair",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/jusiro2/FLAIR",
    manual_instructions="FLAIR (retinal ViT). Auto-downloaded via hf_hub_download at runtime.",
    size_mb=340,
))

register(WeightSource(
    name="ophmae",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/OphMAE/OphMAE_ckpt",
    manual_instructions="OphMAE (ophthalmology ViT). Auto-downloaded via transformers at runtime.",
    size_mb=340,
))

# ── Dermatology ──────────────────────────────────────────────────────────
register(WeightSource(
    name="dermclip",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/redlessone/DermLIP_ViT-B-16",
    manual_instructions="DermCLIP / DermLIP (dermatology CLIP ViT-B/16). Auto-downloaded via open_clip at runtime.",
    size_mb=340,
))

register(WeightSource(
    name="monet",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/suinleelab/monet",
    manual_instructions="MoNET (dermatology CLIP ViT-L/14). Auto-downloaded via transformers CLIPModel at runtime.",
    size_mb=1700,
))

register(WeightSource(
    name="panderm",
    filename="panderm.pth",
    sources=[],
    manual_url="https://github.com/SiyuanYan1/PanDerm",
    manual_instructions="PanDerm (dermatology ViT-L/16). No HF artifact — download from Google Drive (see GitHub README) and pass via pretrained_path.",
    size_mb=1200,
))

# ── Endoscopy ────────────────────────────────────────────────────────────
register(WeightSource(
    name="endovit",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/egeozsoy/EndoViT",
    manual_instructions="EndoViT (endoscopy ViT-B/16). Auto-downloaded via transformers at runtime.",
    size_mb=340,
))

register(WeightSource(
    name="endo_fm",
    filename="(timm auto-download)",
    sources=[],
    manual_url="https://github.com/med-air/Endo-FM",
    manual_instructions="Endo-FM (endoscopy ViT-B/16). pretrained=True loads timm ImageNet weights; for the actual Endo-FM checkpoint download from GitHub and pass via pretrained_path.",
    size_mb=340,
))

register(WeightSource(
    name="surgical_sam",
    filename="(timm auto-download)",
    sources=[],
    manual_url="https://github.com/wenxi-yue/SurgicalSAM",
    manual_instructions="SurgicalSAM (SAM ViT-H/14). pretrained=True loads timm weights; for SAM weights use sam_vit_h (already in WEIGHT_REGISTRY).",
    size_mb=2560,
))

# ── Ultrasound ───────────────────────────────────────────────────────────
register(WeightSource(
    name="usfmae",
    filename="usf_mae_vitb16_100ep.pth",
    sources=[
        _gdrive_file("1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr"),
    ],
    manual_url="https://drive.google.com/file/d/1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr/view",
    manual_instructions=(
        "USF-MAE (ultrasound foundation model, 100-ep ViT-B/16). "
        "Auto-downloaded via gdown from Google Drive. "
        "Requires 'pip install gdown'."
    ),
    size_mb=340,
))

register(WeightSource(
    name="ultrafedfm",
    filename="ultrafedfm.pth",
    sources=[],
    manual_url="https://github.com/yuncheng97/UltraFedFM",
    manual_instructions="UltraFedFM (ultrasound federated model). Download from Baidu NetDisk (code: v74x) or GitHub and pass via pretrained_path.",
    size_mb=340,
))

# ── MLLM Vision Towers ──────────────────────────────────────────────────
# CLIP ViT-L/14-336 is shared by llava_med_vision, huatuogpt_vision,
# and healthgpt_vision towers.
register(WeightSource(
    name="clip_vit_l_336",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/openai/clip-vit-large-patch14-336",
    manual_instructions="OpenAI CLIP ViT-L/14-336 — vision tower shared by LLaVA-Med, HuatuoGPT-Vision, HealthGPT. Auto-downloaded via transformers at runtime.",
    size_mb=1700,
))

register(WeightSource(
    name="medgemma_vision",
    filename="(HF auto-download, gated)",
    sources=[],
    manual_url="https://huggingface.co/google/medgemma-4b-pt",
    manual_instructions="MedGemma-4B (Google, medical MLLM). Auto-downloaded via transformers. Requires HF token (gated repo).",
    size_mb=8000,
))

register(WeightSource(
    name="hulumed_vision",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/ZJU-AI4H/Hulu-Med-7B",
    manual_instructions="HuLuMed (ZJU AI4H, medical MLLM). Auto-downloads full MLLM via transformers AutoModel and extracts vision tower.",
    size_mb=14000,
))

register(WeightSource(
    name="lingshu_vision",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/lingshu-medical-mllm/Lingshu-7B",
    manual_instructions="LingShu (medical MLLM). Auto-downloads via transformers AutoModel and extracts vision tower.",
    size_mb=14000,
))

register(WeightSource(
    name="qwen25_vl_vision",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    manual_instructions="Qwen2.5-VL-7B (Alibaba, general MLLM). Auto-downloads via transformers AutoModel and extracts vision tower.",
    size_mb=15000,
))

register(WeightSource(
    name="qwen3_vl_vision",
    filename="(HF auto-download)",
    sources=[],
    manual_url="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
    manual_instructions="Qwen3-VL-8B (Alibaba, general MLLM). Auto-downloads via transformers AutoModel and extracts vision tower.",
    size_mb=17000,
))


# ----------------------------------------------------------------------
# 预训练 requirement 注册表 / Pretrained requirement registry
# ----------------------------------------------------------------------

# 架构 names whose papers REQUIRE 预训练 权重 / Architecture names whose papers REQUIRE pretrained weights.
# SAM family excluded — SAM models handle 预训练 = False gracefully / SAM family excluded — SAM models handle pretrained=False gracefully.
# LeViT-UNet excluded — paper 消融实验 shows small variants work / LeViT-UNet excluded — paper ablation shows small variants work
# better WITHOUT ImageNet pretraining.
REQUIRES_PRETRAINED: set = {
    # ── Transformer (ViT / Swin / PVT / DeiT backbones) ────────────
    "swinunet",               # Swin-T ImageNet
    "transunet",              # R50+ViT-B/16 JAX
    "h2former",               # Swin-T ImageNet
    "hiformer",               # Swin-T ImageNet
    "cswin_unet",             # CSWin-Tiny ImageNet
    "da_transunet",           # R50+ViT
    "fcbformer",              # PVTv2-B3
    "segformer_b0", "segformer_b1", "segformer_b2",
    "segformer_b3", "segformer_b4", "segformer_b5",  # MiT backbone
    "esfpnet",                # PVTv2-B2 (timm)
    "ssformer",               # PVTv2
    "hsnet",                  # PVT backbone
    "polyp_pvt",              # PVT (v1)
    "pvtb2_cascade",          # PVTv2-B2 (WACV 2023)
    "pvtb2_emcad",            # PVTv2-B2 (CVPR 2024)
    "resnet34_deit_fatnet",   # DeiT/ResNet
    "transfuse",              # ResNet+ViT
    "transnuseg",             # Swin
    "ldnet",                  # ResNet
    "mist",                   # PVT
    "nulite",                 # PVT
    # ── Mamba / SSM (VMamba / Mamba backbones) ─────────────────
    "vm_unet",                # VMamba Tiny
    "mamba_unet",             # Mamba encoder
    # ── RWKV ───────────────────────────────────────────────────────
    "rwkv_unet",              # RWKV encoder (B/S/T)
    # ─ ─ CNN ( 预训练 ResNet / Res2Net / MobileNet backbones ) ─ ─ ─ ─ / ── CNN (pretrained ResNet / Res2Net / MobileNet backbones) ────
    "cfanet",                 # Res2Net-50
    "dconnnet",               # ResNet
    "lv_unet",                # MobileNetV3
    "polyper",                # ResNet
    # ─ ─ SAM family ( Segment Anything 预训练 权重 ) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ / ── SAM family (Segment Anything pretrained weights) ───────────
    "sam_b",                  # SAM ViT-Base
    "sam_l",                  # SAM ViT-Large
    "mobile_sam",             # MobileSAM (distilled)
    "sam2",                   # SAM 2
    "sam3",                   # SAM 3
    "medsam2",                # MedSAM2 (2D variant)
    "medsam3",                # MedSAM3 (2D variant)
    "sam_med2d",              # SAM-Med2D
    "samed",                  # SAMed (LoRA fine-tune)
    "sammed2d_wrapper",       # SAM-Med2D wrapper variant
    "samus",                  # SAMUS
    "auto_sam",               # AutoSAM
    "lite_medsam",            # LiteMedSAM (distilled)
    "medical_sam_adapter",    # Medical SAM Adapter
}

# Human-readable reason for each entry ( for 警告 message ) / Human-readable reason for each entry (for warning message)
_PRETRAINED_REASON = {
    "_default": "原论文中使用了预训练权重",
}


def warn_pretrained_false(
    model_name: str,
    *,
    delay: int = 10,
) -> None:
    """Emit a prominent warning when ``pretrained=False`` for a model
        Emit a prominent 警告 when ` ` 预训练 = False ` ` for a 模型。
    whose paper requires pretrained weights."""
    if model_name not in REQUIRES_PRETRAINED:
        return

    reason = _PRETRAINED_REASON.get(
        model_name, _PRETRAINED_REASON["_default"]
    )

    border = "=" * 70
    msg = (
        f"\n{border}\n"
        f"  ⚠️  警告: {model_name} 的论文要求使用预训练权重!\n"
        f"\n"
        f"  {model_name} 原论文中使用了预训练权重进行实验。\n"
        f"  原因: {reason}\n"
        f"\n"
        f"  ⚠️  不使用预训练权重可能导致:\n"
        f"    1. 复现结果不正确 (与原论文数值不一致)\n"
        f"    2. 与其他方法的比较不公平\n"
        f"    3. 模型性能指标偏低\n"
        f"\n"
        f"  如果你确定不需要预训练权重，将在 {delay} 秒后继续...\n"
        f"  (设置 pretrained: true 或使用 pretrained_path 指定本地权重路径)\n"
        f"{border}\n"
    )

    print(msg, flush=True)
    import time
    for remaining in range(delay, 0, -1):
        print(f"  ⏳ {remaining}...", end="", flush=True)
        time.sleep(1)
    print("\n  继续执行 (pretrained=False)\n", flush=True)



def ensure_weight(
    name: str,
    cache_dir: Optional[str | Path] = None,
    verify: bool = True,
) -> Path:
    """Return a local path to the registered weight, downloading if needed.
        返回 a 局部的 path to the registered 权重, downloading if needed。

    Parameters
    ----------
    name
        Registry key (e.g. ``"medsam_vit_b"``).
    cache_dir
        Override for the cache root. If ``None``, uses
        ``$MEDSEG_WEIGHT_CACHE`` or ``~/.cache/medseg/weights``.
    verify
        When ``True`` and the registered :attr:`WeightSource.sha256` is
        set, the downloaded file is verified after the fetch.

    Raises
    ------
    KeyError
        If ``name`` is not in :data:`WEIGHT_REGISTRY`.
    WeightDownloadError
        If every registered source fails.
    """
    if name not in WEIGHT_REGISTRY:
        raise KeyError(
            f"Unknown weight name '{name}'. "
            f"Known: {sorted(WEIGHT_REGISTRY.keys())}"
        )

    src = WEIGHT_REGISTRY[name]
    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_root()
    target = root / src.filename
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 0:
        if verify and src.sha256 and _sha256(target) != src.sha256:
            logger.warning(
                f"sha256 mismatch for {target}; re-downloading"
            )
            target.unlink()
        else:
            return target

    if not src.sources:
        raise WeightDownloadError(_manual_message(src, target))

    errors = []
    for fetch in src.sources:
        try:
            logger.info(f"Downloading {name} via {fetch.__name__} -> {target}")
            fetch(target)
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(
                    f"fetcher {fetch.__name__} returned without writing the target"
                )
            if verify and src.sha256 and _sha256(target) != src.sha256:
                target.unlink(missing_ok=True)
                raise RuntimeError(
                    f"sha256 mismatch after download via {fetch.__name__}"
                )
            logger.info(f"OK: {name} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
            return target
        except Exception as e:
            errors.append(f"  - {fetch.__name__}: {type(e).__name__}: {e}")
            target.unlink(missing_ok=True)

    raise WeightDownloadError(_manual_message(src, target, errors))


def _manual_message(
    src: WeightSource,
    target: Path,
    errors: Optional[List[str]] = None,
) -> str:
    """Compose the actionable failure message handed to the user."""
    parts = [
        "",
        f"Failed to auto-download weight '{src.name}'.",
        f"  expected file: {target}",
    ]
    if src.size_mb:
        parts.append(f"  approximate size: {src.size_mb} MB")
    if errors:
        parts.append("  attempted sources:")
        parts.extend(errors)
    if src.manual_url:
        parts.append(f"  manual download: {src.manual_url}")
    if src.manual_instructions:
        parts.append("  instructions:")
        parts.extend(f"    {line}" for line in src.manual_instructions.splitlines())
    parts.append(f"  → place the file at: {target}")
    parts.append("")
    return "\n".join(parts)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_path(name: str, cache_dir: Optional[str | Path] = None) -> Path:
    """返回 the path a 权重 would live at, without triggering a download。
        Return the path a weight would live at, without triggering a download."""
    if name not in WEIGHT_REGISTRY:
        raise KeyError(f"Unknown weight name '{name}'")
    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_root()
    return root / WEIGHT_REGISTRY[name].filename


# ----------------------------------------------------------------------
# Standalone 模型 预训练 loading helper / Standalone model pretrained loading helper
# ----------------------------------------------------------------------

def load_pretrained_standalone(
    model: "torch.nn.Module",
    pretrained_path: Optional[str] = None,
    registry_key: Optional[str] = None,
    model_name: str = "Model",
    filter_prefixes: Optional[List[str]] = None,
    strict: bool = False,
) -> bool:
    """Load pretrained weights into a standalone segmentation model.
        加载 预训练 权重 into a standalone 分割 模型。

    Tries ``pretrained_path`` first, then ``registry_key`` via auto-download.
    Returns ``True`` on success, ``False`` on failure (with a warning logged).
    """
    import torch

    weight_path = pretrained_path
    if weight_path is None and registry_key:
        weight_path = str(ensure_weight(registry_key))

    if weight_path is None:
        raise WeightDownloadError(
            f"\n{model_name}: pretrained=True but no pretrained_path or "
            f"auto-download source available.\n"
            f"  → Set pretrained_path=<local_file> in your YAML, "
            f"or set pretrained: false to skip.\n"
        )

    try:
        if weight_path.endswith(".npz"):
            # JAX / NumPy 检查点 / JAX / NumPy checkpoint
            import numpy as np
            arrays = np.load(weight_path)
            state = {}
            for key in arrays.files:
                arr = arrays[key]
                if arr.ndim == 4 and ("kernel" in key or "conv" in key):
                    arr = arr.transpose(3, 2, 0, 1)  # JAX → PyTorch
                state[key] = torch.from_numpy(arr.copy())
        elif weight_path.endswith(".safetensors"):
            # HuggingFace safetensors format
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise WeightDownloadError(
                    f"\n{model_name}: checkpoint is in safetensors format "
                    f"({weight_path}) but the `safetensors` package is not "
                    f"installed.\n  → Run: pip install safetensors"
                )
            state = load_file(weight_path)
        else:
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("model", "state_dict"):
                    if key in state:
                        state = state[key]
                        break
        if filter_prefixes:
            state = {k: v for k, v in state.items()
                     if not k.startswith(tuple(filter_prefixes))}
        msg = model.load_state_dict(state, strict=strict)
        logger.info("%s: loaded pretrained weights from %s: %s",
                    model_name, weight_path, msg)
        return True
    except Exception as e:
        import warnings
        warnings.warn(
            f"{model_name}: failed to load pretrained weights from "
            f"{weight_path}: {e}. Model initialized from scratch."
        )
        return False


# ----------------------------------------------------------------------
# HF AutoModel/AutoTokenizer wrappers — clearer error on download failure
# ----------------------------------------------------------------------


def hf_from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *args,
    **kwargs,
):
    """Wrap ``cls.from_pretrained`` so HF download failures surface with
        Wrap ` ` cls. from _ 预训练 ` ` so HF download failures surface with。
    an actionable manual URL.

    Drop-in replacement for e.g. ``AutoModel.from_pretrained(...)``::

        from medseg.utils.weight_downloader import hf_from_pretrained
        model = hf_from_pretrained(AutoModel, "microsoft/BiomedVLP-CXR-BERT-specialized")

    If torch.load version check fails (transformers >= 4.48 + torch < 2.6),
    automatically retry with ``use_safetensors=True``.
    """
    try:
        kwargs.setdefault("trust_remote_code", True)
        return cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    except ValueError as e:
        err_str = str(e).lower()
        if ("upgrade torch" in err_str or "torch.load" in err_str or "safetensors" in err_str):
            # transformers/torch require newer torch for torch.load; retry with safetensors
            if not kwargs.get("use_safetensors", False):
                kwargs["use_safetensors"] = True
                try:
                    return cls.from_pretrained(
                        pretrained_model_name_or_path, *args, **kwargs
                    )
                except Exception:
                    pass  # Fall through to original error
        raise WeightDownloadError(
            f"\nFailed to load HF model '{pretrained_model_name_or_path}' "
            f"via {cls.__name__}.from_pretrained.\n"
            f"  underlying error: {type(e).__name__}: {e}\n"
            f"  manual download: https://huggingface.co/{pretrained_model_name_or_path}\n"
            f"  instructions:\n"
            f"    1. If the repo is gated (e.g. Llama / CogVLM), "
            f"`huggingface-cli login` with an account that has access.\n"
            f"    2. Downloads retry automatically via hf-mirror.com when the official "
            f"Hub is unreachable. Pin an endpoint with MEDSEG_HF_MIRROR=1 or "
            f"HF_ENDPOINT if needed.\n"
            f"    3. To use a local copy, pass the local directory path "
            f"instead of the repo id.\n"
        ) from e
    except Exception as e:
        raise WeightDownloadError(
            f"\nFailed to load HF model '{pretrained_model_name_or_path}' "
            f"via {cls.__name__}.from_pretrained.\n"
            f"  underlying error: {type(e).__name__}: {e}\n"
            f"  manual download: https://huggingface.co/{pretrained_model_name_or_path}\n"
            f"  instructions:\n"
            f"    1. If the repo is gated (e.g. Llama / CogVLM), "
            f"`huggingface-cli login` with an account that has access.\n"
            f"    2. Downloads retry automatically via hf-mirror.com when the official "
            f"Hub is unreachable. Pin an endpoint with MEDSEG_HF_MIRROR=1 or "
            f"HF_ENDPOINT if needed.\n"
            f"    3. To use a local copy, pass the local directory path "
            f"instead of the repo id.\n"
        ) from e


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="medseg weight downloader / inspector"
    )
    parser.add_argument("--cache", type=str, default=None,
                        help="override cache root (defaults to $MEDSEG_WEIGHT_CACHE "
                             "or ~/.cache/medseg/weights)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list registered weights")

    p_dl = sub.add_parser("download", help="download a registered weight")
    p_dl.add_argument("name", help="registry key (use `list` to see them)")

    sub.add_parser("check", help="check which weights are present in the cache")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "list" or args.cmd is None:
        root = Path(args.cache).expanduser() if args.cache else default_cache_root()
        print(f"cache root: {root}")
        for name in sorted(WEIGHT_REGISTRY.keys()):
            src = WEIGHT_REGISTRY[name]
            size = f"~{src.size_mb} MB" if src.size_mb else ""
            here = (root / src.filename).exists()
            print(f"  [{'OK' if here else '  '}] {name:35s} {size:10s} -> {src.filename}")
        return 0

    if args.cmd == "download":
        try:
            path = ensure_weight(args.name, cache_dir=args.cache)
            print(f"OK: {path}")
            return 0
        except WeightDownloadError as e:
            print(str(e))
            return 1

    if args.cmd == "check":
        root = Path(args.cache).expanduser() if args.cache else default_cache_root()
        missing = [n for n, s in WEIGHT_REGISTRY.items()
                   if s.sources and not (root / s.filename).exists()]
        if missing:
            print("missing weights (run `download <name>` to fetch):")
            for n in missing:
                print(f"  - {n}")
            return 1
        print("all auto-downloadable weights are present")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
