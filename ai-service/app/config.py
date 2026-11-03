"""Environment-backed service settings."""

from __future__ import annotations

import os
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            raise ValueError
        return value
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc


class Settings:
    DATASET_ROOT = Path(os.getenv("DEFECTZERO_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "dataset")))
    INSPECTION_STORAGE_ROOT = Path(
        os.getenv(
            "DEFECTZERO_INSPECTION_STORAGE",
            str(Path(__file__).resolve().parents[1] / "inspection_storage"),
        ),
    )
    SUPPORTED_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper",
    ]
    CORS_ORIGINS = [
        origin.strip().rstrip("/")
        for origin in os.getenv("DEFECTZERO_CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    try:
        DEFECT_THRESHOLD = float(os.getenv("DEFECTZERO_DEFECT_THRESHOLD", "0.5"))
    except ValueError as exc:
        raise RuntimeError("DEFECTZERO_DEFECT_THRESHOLD must be a number in [0, 1]") from exc
    if not 0.0 <= DEFECT_THRESHOLD <= 1.0:
        raise RuntimeError("DEFECTZERO_DEFECT_THRESHOLD must be in [0, 1]")
    MAX_UPLOAD_BYTES = _positive_int("DEFECTZERO_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    MAX_IMAGE_PIXELS = _positive_int("DEFECTZERO_MAX_IMAGE_PIXELS", 25_000_000)


settings = Settings()
