"""Exact pixel-content matching for local MVTec AD ground-truth masks."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class GroundTruthMatch(NamedTuple):
    defect_type: str
    mask_path: Path


def pixel_content_hash(image: Image.Image) -> str:
    """Hash dimensions and canonical RGB pixels, ignoring filename/metadata."""
    canonical = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(canonical.width.to_bytes(4, "big"))
    digest.update(canonical.height.to_bytes(4, "big"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def find_ground_truth_match(
    category: str,
    image: Image.Image,
    dataset_root: Path,
) -> GroundTruthMatch | None:
    """Find a unique content match within a category's MVTec test set.

    Filename matching is deliberately not used. If duplicate image content maps
    to different masks (or to both a normal and defective sample), the match is
    ambiguous and no mask is returned.
    """
    image_hash = pixel_content_hash(image)
    candidates = _category_index(str(dataset_root.resolve()), category).get(image_hash, ())
    if not candidates:
        return None

    # A repeated identical image is safe only when every candidate resolves to
    # the same defective class and identical mask bytes. A normal/defect content
    # collision is ambiguous and must stay unavailable.
    defect_types = {defect_type for defect_type, _ in candidates}
    if len(defect_types) != 1 or "good" in defect_types:
        if len(defect_types) > 1:
            logger.warning("Ambiguous MVTec ground-truth content match for category %s; omitting mask", category)
        return None

    unique: dict[str, GroundTruthMatch] = {}
    for defect_type, mask_path in candidates:
        if not mask_path.is_file():
            return None
        try:
            mask_digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        except OSError:
            return None
        unique[mask_digest] = GroundTruthMatch(defect_type, mask_path)

    if len(unique) != 1:
        logger.warning("Ambiguous MVTec ground-truth content match for category %s; omitting mask", category)
        return None
    return next(iter(unique.values()))


@lru_cache(maxsize=15)
def _category_index(dataset_root: str, category: str) -> dict[str, tuple[tuple[str, Path], ...]]:
    """Build a small per-category index once; cache hashes, never image data."""
    test_root = Path(dataset_root) / category / "test"
    if not test_root.is_dir():
        logger.warning("MVTec test directory is unavailable: %s", test_root)
        return {}

    index: dict[str, list[tuple[str, Path]]] = {}
    for image_path in sorted(test_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        defect_type = image_path.parent.name
        mask_path = Path(dataset_root) / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        try:
            with Image.open(image_path) as source:
                image_hash = pixel_content_hash(source)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            logger.warning("Skipping unreadable MVTec test image %s: %s", image_path, exc)
            continue
        index.setdefault(image_hash, []).append((defect_type, mask_path))

    return {key: tuple(value) for key, value in index.items()}
