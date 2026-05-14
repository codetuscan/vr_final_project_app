"""Centralized configuration for the demo app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_path(name: str, default: Path, base: Path | None = None) -> Path:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    p = Path(value).expanduser()
    # Resolve relative paths against base directory
    if not p.is_absolute() and base is not None:
        p = base / p
    return p


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    data_root: Path
    gallery_root: Path
    query_root: Path
    catalog_path: Path
    artifacts_dir: Path
    cache_dir: Path
    index_backend: str
    top_k: int
    alpha: float
    embedding_dim: int
    use_mock: bool
    allow_missing_images: bool
    yolo_weights: str
    clip_weights: str
    blip2_weights: str
    reranker_weights: str
    clip_model: str
    clip_pretrain: str
    device: str
    crop_pad: int
    hnsw_m: int
    hnsw_ef: int


def get_config() -> AppConfig:
    root_dir = Path(__file__).resolve().parents[2]
    data_root = _env_path("VPS_DATA_ROOT", root_dir / "archive", base=root_dir)
    gallery_root = _env_path("VPS_GALLERY_ROOT", data_root / "gallery" / "img", base=root_dir)
    query_root = _env_path("VPS_QUERY_ROOT", data_root / "query" / "img", base=root_dir)

    # Default catalog path: look in root_dir first
    default_catalog = root_dir / "catalog.json"
    catalog_path = _env_path("VPS_CATALOG_PATH", default_catalog, base=root_dir)

    artifacts_dir = _env_path("VPS_ARTIFACTS_DIR", root_dir / "artifacts", base=root_dir)
    cache_dir = _env_path("VPS_CACHE_DIR", artifacts_dir / "cache", base=root_dir)

    # Default model weight paths: look in root_dir
    default_yolo = root_dir / "yolo.pt"
    default_clip = root_dir / "clip_best.pt"

    # Resolve model weight paths (env values are relative to root_dir)
    yolo_raw = os.getenv("VPS_YOLO_WEIGHTS", "")
    if yolo_raw:
        yolo_path = Path(yolo_raw)
        if not yolo_path.is_absolute():
            yolo_path = root_dir / yolo_path
        yolo_weights = str(yolo_path)
    else:
        yolo_weights = str(default_yolo)

    clip_raw = os.getenv("VPS_CLIP_WEIGHTS", "")
    if clip_raw:
        clip_path = Path(clip_raw)
        if not clip_path.is_absolute():
            clip_path = root_dir / clip_path
        clip_weights = str(clip_path)
    else:
        clip_weights = str(default_clip)

    return AppConfig(
        root_dir=root_dir,
        data_root=data_root,
        gallery_root=gallery_root,
        query_root=query_root,
        catalog_path=catalog_path,
        artifacts_dir=artifacts_dir,
        cache_dir=cache_dir,
        index_backend=os.getenv("VPS_INDEX_BACKEND", "hnsw").lower(),
        top_k=_env_int("VPS_TOP_K", 15),
        alpha=_env_float("VPS_ALPHA", 0.7),
        embedding_dim=_env_int("VPS_EMBED_DIM", 512),
        use_mock=_env_bool("VPS_USE_MOCK_MODELS", False),
        allow_missing_images=_env_bool("VPS_ALLOW_MISSING_IMAGES", True),
        yolo_weights=yolo_weights,
        clip_weights=clip_weights,
        blip2_weights=os.getenv("VPS_BLIP2_WEIGHTS", ""),
        reranker_weights=os.getenv("VPS_RERANKER_WEIGHTS", ""),
        clip_model=os.getenv("VPS_CLIP_MODEL", "ViT-B-32"),
        clip_pretrain=os.getenv("VPS_CLIP_PRETRAIN", "openai"),
        device=os.getenv("VPS_DEVICE", "cpu"),
        crop_pad=_env_int("VPS_CROP_PAD", 10),
        hnsw_m=_env_int("VPS_HNSW_M", 32),
        hnsw_ef=_env_int("VPS_HNSW_EF", 64),
    )

