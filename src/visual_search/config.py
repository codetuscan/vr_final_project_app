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


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return Path(value).expanduser()


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
    data_root = _env_path("VPS_DATA_ROOT", root_dir / "archive")
    gallery_root = _env_path("VPS_GALLERY_ROOT", data_root / "gallery" / "img")
    query_root = _env_path("VPS_QUERY_ROOT", data_root / "query" / "img")
    catalog_path = _env_path("VPS_CATALOG_PATH", root_dir / "catalog.json")
    artifacts_dir = _env_path("VPS_ARTIFACTS_DIR", root_dir / "artifacts")
    cache_dir = _env_path("VPS_CACHE_DIR", artifacts_dir / "cache")

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
        alpha=_env_float("VPS_ALPHA", 0.5),
        embedding_dim=_env_int("VPS_EMBED_DIM", 128),
        use_mock=_env_bool("VPS_USE_MOCK_MODELS", True),
        allow_missing_images=_env_bool("VPS_ALLOW_MISSING_IMAGES", False),
        yolo_weights=os.getenv("VPS_YOLO_WEIGHTS", ""),
        clip_weights=os.getenv("VPS_CLIP_WEIGHTS", ""),
        blip2_weights=os.getenv("VPS_BLIP2_WEIGHTS", ""),
        reranker_weights=os.getenv("VPS_RERANKER_WEIGHTS", ""),
        clip_model=os.getenv("VPS_CLIP_MODEL", "ViT-B-32"),
        clip_pretrain=os.getenv("VPS_CLIP_PRETRAIN", "openai"),
        device=os.getenv("VPS_DEVICE", "cpu"),
        crop_pad=_env_int("VPS_CROP_PAD", 10),
        hnsw_m=_env_int("VPS_HNSW_M", 32),
        hnsw_ef=_env_int("VPS_HNSW_EF", 64),
    )
