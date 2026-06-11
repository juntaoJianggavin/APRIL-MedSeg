"""Central Hugging Face Hub configuration and download helpers.

All HF downloads in medseg should go through this module so endpoint
selection and mirror fallback apply consistently.

Default behaviour (no env vars set):

1. Try the official Hugging Face Hub (``huggingface.co``).
2. On retryable network / connectivity errors, automatically retry via
   ``https://hf-mirror.com``.

Explicit overrides (skip auto-fallback — use only the configured endpoint):

1. ``HF_ENDPOINT`` — standard ``huggingface_hub`` variable (highest priority)
2. ``MEDSEG_HF_ENDPOINT`` — project alias when ``HF_ENDPOINT`` is unset
3. ``MEDSEG_HF_MIRROR=1`` — shorthand for ``https://hf-mirror.com``

Optional auth for gated repos: ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Sequence, TypeVar, Union

logger = logging.getLogger(__name__)

HF_OFFICIAL = "https://huggingface.co"
HF_MIRROR_DEFAULT = "https://hf-mirror.com"
_CONFIGURED = False

T = TypeVar("T")


def resolve_hf_endpoint() -> Optional[str]:
    """Return an explicitly configured HF endpoint, or ``None`` for auto mode."""
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if endpoint:
        return endpoint.rstrip("/")
    endpoint = os.environ.get("MEDSEG_HF_ENDPOINT", "").strip()
    if endpoint:
        return endpoint.rstrip("/")
    mirror_flag = os.environ.get("MEDSEG_HF_MIRROR", "").strip().lower()
    if mirror_flag in ("1", "true", "yes", "on"):
        return HF_MIRROR_DEFAULT
    return None


def user_pinned_hf_endpoint() -> bool:
    """True when the user explicitly chose an endpoint (no auto-fallback)."""
    return resolve_hf_endpoint() is not None


def configure_hf_hub(*, log: bool = True) -> Optional[str]:
    """Apply resolved endpoint to ``HF_ENDPOINT`` once per process."""
    global _CONFIGURED
    endpoint = resolve_hf_endpoint()
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
    if log and endpoint and not _CONFIGURED:
        logger.info("Hugging Face Hub endpoint (pinned): %s", endpoint)
    _CONFIGURED = True
    return endpoint or os.environ.get("HF_ENDPOINT")


@contextmanager
def hf_endpoint(endpoint: Optional[str]):
    """Temporarily set ``HF_ENDPOINT`` for huggingface_hub / transformers."""
    previous = os.environ.get("HF_ENDPOINT")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    else:
        os.environ.pop("HF_ENDPOINT", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous


def is_retryable_hf_error(exc: BaseException) -> bool:
    """Return True when a download failure may succeed on the mirror endpoint."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        if isinstance(exc, OSError) and getattr(exc, "errno", None) not in (
            None, 101, 110, 111, 113, 104, 105,
        ):
            msg = str(exc).lower()
            if "network" not in msg and "unreachable" not in msg and "timed out" not in msg:
                return False
        return True

    name = type(exc).__name__
    if name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "NetworkError",
        "RemoteProtocolError",
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "HfHubHTTPError",
        "RepositoryNotFoundError",
    }:
        return True

    msg = str(exc).lower()
    retry_markers = (
        "network is unreachable",
        "connection refused",
        "connection reset",
        "timed out",
        "timeout",
        "cannot send a request",
        "client has been closed",
        "failed to establish a new connection",
        "name or service not known",
        "temporary failure in name resolution",
        "ssl:",
        "connection error",
        "connecterror",
        "max retries exceeded",
        "couldn't connect",
        "unable to resolve",
        "locate the files on the hub",
    )
    return any(marker in msg for marker in retry_markers)


def call_with_hf_fallback(func: Callable[..., T], *args, **kwargs) -> T:
    """Call ``func`` using official HF first, then mirror on network failure."""
    pinned = resolve_hf_endpoint()
    if pinned:
        configure_hf_hub(log=False)
        return func(*args, **kwargs)

    with hf_endpoint(HF_OFFICIAL):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not is_retryable_hf_error(exc):
                raise
            logger.warning(
                "Hugging Face download via %s failed (%s: %s); retrying via mirror %s",
                HF_OFFICIAL,
                type(exc).__name__,
                exc,
                HF_MIRROR_DEFAULT,
            )

    with hf_endpoint(HF_MIRROR_DEFAULT):
        return func(*args, **kwargs)


def hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token.strip() if token else None


def hf_hub_download_file(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    local_dir: Optional[Union[str, os.PathLike]] = None,
    local_dir_use_symlinks: Union[bool, str] = False,
    token: Optional[str] = None,
):
    """Download a single file from Hugging Face Hub (official first, mirror fallback)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": repo_type,
        "token": token if token is not None else hf_token(),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
        kwargs["local_dir_use_symlinks"] = local_dir_use_symlinks

    return call_with_hf_fallback(hf_hub_download, **kwargs)


def hf_snapshot_download(
    repo_id: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    local_dir: Optional[Union[str, os.PathLike]] = None,
    allow_patterns: Optional[Union[str, Sequence[str]]] = None,
    ignore_patterns: Optional[Union[str, Sequence[str]]] = None,
    token: Optional[str] = None,
):
    """Download a repository snapshot from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "token": token if token is not None else hf_token(),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = ignore_patterns

    return call_with_hf_fallback(snapshot_download, **kwargs)


def download_repo_files(
    repo_id: str,
    filenames: Iterable[str],
    *,
    repo_type: str = "model",
    local_dir: Union[str, os.PathLike],
    revision: Optional[str] = None,
) -> list:
    """Download specific files from a repo into ``local_dir``."""
    from pathlib import Path

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in filenames:
        path = hf_hub_download_file(
            repo_id,
            filename,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
        )
        paths.append(Path(path))
    return paths


# Known HF dataset repos useful for medseg (not all are TransUNet npz format).
HF_DATASET_CATALOG = {
    "medseg7d": {
        "repo_id": "MaybeRichard/MedSeg-7D",
        "repo_type": "dataset",
        "description": "7 public 2D med-seg datasets incl. ACDC (PNG, not TransUNet npz)",
        "includes": ["ACDC", "BraTS2020", "CVC-ClinicDB", "..."],
    },
    "acdc_nifti": {
        "repo_id": "viennh2012/cardiac_cine_acdc",
        "repo_type": "dataset",
        "description": "ACDC cardiac cine-MRI (processed NIfTI)",
        "includes": ["ACDC"],
    },
    "m3d_seg": {
        "repo_id": "GoodBaiBai88/M3D-Seg",
        "repo_type": "dataset",
        "description": "25 3D CT seg datasets incl. BTCV (NIfTI zips, not TransUNet npz)",
        "includes": ["BTCV", "..."],
    },
}
