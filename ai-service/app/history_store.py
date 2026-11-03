"""SQLite metadata and safe filesystem storage for completed inspections."""

from __future__ import annotations

import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class InvalidInspectionID(ValueError):
    """Raised when an inspection ID is not a canonical UUID."""


class UnsafeAssetPath(ValueError):
    """Raised when a stored relative asset path escapes the storage root."""


class InspectionHistoryStore:
    """Persist inspection metadata in SQLite and associated assets as files."""

    IMAGE_SUFFIXES = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root.resolve()
        self.database_path = self.storage_root / "inspections.sqlite3"

    def initialize(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('normal', 'defective')),
                    anomaly_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    heatmap_path TEXT NOT NULL,
                    ground_truth_mask_path TEXT
                )
                """,
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_timestamp "
                "ON inspections(timestamp_utc DESC, inspection_id DESC)",
            )
            connection.commit()

    def save_inspection(
        self,
        *,
        category: str,
        filename: str,
        predicted_label: str,
        anomaly_score: float,
        threshold: float,
        inference_time_ms: float,
        uploaded_image: bytes,
        image_format: str,
        heatmap_png: bytes,
        ground_truth_mask_png: bytes | None,
    ) -> dict[str, Any]:
        """Atomically write generated asset paths and insert their metadata."""
        inspection_id = str(uuid.uuid4())
        directory = self._safe_path(inspection_id)
        directory.mkdir(parents=True, exist_ok=False)

        suffix = self.IMAGE_SUFFIXES.get(image_format.upper(), ".img")
        image_relative = f"{inspection_id}/input{suffix}"
        heatmap_relative = f"{inspection_id}/heatmap.png"
        mask_relative = f"{inspection_id}/ground_truth.png" if ground_truth_mask_png is not None else None
        timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        try:
            self._write_asset(image_relative, uploaded_image)
            self._write_asset(heatmap_relative, heatmap_png)
            if mask_relative is not None:
                self._write_asset(mask_relative, ground_truth_mask_png or b"")

            record = {
                "inspection_id": inspection_id,
                "category": category,
                "filename": filename,
                "predicted_label": predicted_label,
                "anomaly_score": anomaly_score,
                "threshold": threshold,
                "inference_time_ms": inference_time_ms,
                "timestamp_utc": timestamp_utc,
                "image_path": image_relative,
                "heatmap_path": heatmap_relative,
                "ground_truth_mask_path": mask_relative,
            }
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(
                        """INSERT INTO inspections (
                            inspection_id, category, filename, predicted_label,
                            anomaly_score, threshold, inference_time_ms, timestamp_utc,
                            image_path, heatmap_path, ground_truth_mask_path
                        ) VALUES (
                            :inspection_id, :category, :filename, :predicted_label,
                            :anomaly_score, :threshold, :inference_time_ms, :timestamp_utc,
                            :image_path, :heatmap_path, :ground_truth_mask_path
                        )""",
                        record,
                    )
            return record
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def list_inspections(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM inspections
                   ORDER BY timestamp_utc DESC, inspection_id DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get_inspection(self, inspection_id: str) -> dict[str, Any] | None:
        canonical_id = self.validate_inspection_id(inspection_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM inspections WHERE inspection_id = ?",
                (canonical_id,),
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    @staticmethod
    def validate_inspection_id(inspection_id: str) -> str:
        try:
            parsed = uuid.UUID(inspection_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise InvalidInspectionID("Inspection ID must be a UUID") from exc
        canonical = str(parsed)
        if inspection_id != canonical:
            raise InvalidInspectionID("Inspection ID must be a canonical lowercase UUID")
        return canonical

    def resolve_asset(self, relative_path: str) -> Path:
        """Resolve a stored POSIX relative path without allowing root escape."""
        if not relative_path or "\\" in relative_path:
            raise UnsafeAssetPath("Asset path must be a non-empty POSIX relative path")
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise UnsafeAssetPath("Asset path is not safe")
        candidate = self.storage_root.joinpath(*relative.parts).resolve()
        if not candidate.is_relative_to(self.storage_root):
            raise UnsafeAssetPath("Asset path escapes the inspection storage directory")
        return candidate

    def _safe_path(self, relative_path: str) -> Path:
        return self.resolve_asset(relative_path)

    def _write_asset(self, relative_path: str, content: bytes) -> None:
        if not content:
            raise ValueError("Inspection assets must not be empty")
        path = self.resolve_asset(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(content)

    def _record_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        self.validate_inspection_id(record["inspection_id"])
        allowed_files = {
            "image_path": {f"input{suffix}" for suffix in (*self.IMAGE_SUFFIXES.values(), ".img")},
            "heatmap_path": {"heatmap.png"},
            "ground_truth_mask_path": {"ground_truth.png"},
        }
        for field, allowed_names in allowed_files.items():
            relative_path = record[field]
            if relative_path is None and field == "ground_truth_mask_path":
                continue
            if not isinstance(relative_path, str):
                raise UnsafeAssetPath(f"Inspection record has no safe {field}")
            path = self.resolve_asset(relative_path)
            parts = PurePosixPath(relative_path).parts
            if len(parts) != 2 or parts[0] != record["inspection_id"] or parts[1] not in allowed_names:
                raise UnsafeAssetPath(f"Inspection record has an invalid {field}")
            if not path.is_file():
                raise UnsafeAssetPath(f"Inspection record references a missing {field}")
        return record

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection
