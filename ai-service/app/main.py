"""FastAPI application for zero-shot WinCLIP inference."""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import mimetypes
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.ground_truth import find_ground_truth_match
from app.history_store import InspectionHistoryStore, InvalidInspectionID
from app.reports import build_batch_summary_pdf, build_inspection_pdf
from app.schemas import BatchReportRequest, HealthResponse, InspectionHistoryRecord, PredictionResponse
from app.winclip_runtime import WinClipRuntime

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("defectzero.ai")

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create the storage directory and database before serving requests."""
    store = InspectionHistoryStore(settings.INSPECTION_STORAGE_ROOT)
    try:
        store.initialize()
    except Exception:
        logger.exception("Could not initialize persistent inspection history")
        raise
    application.state.history_store = store
    yield


app = FastAPI(
    title="DefectZero AI — Zero-Shot WinCLIP API",
    version="1.0.0",
    description=(
        "Zero-shot industrial anomaly detection using pretrained Anomalib WinCLIP. "
        "The model is not trained or fine-tuned by this service."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

runtime = WinClipRuntime(categories=settings.SUPPORTED_CATEGORIES)
EVALUATION_RESULTS_ROOT = settings.DATASET_ROOT.parent / "experiments" / "evaluation_results"


def get_history_store(request: Request) -> InspectionHistoryStore:
    return request.app.state.history_store


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Return process liveness and whether a category model is currently loaded."""
    loaded, category, device, error = runtime.status()
    return HealthResponse(
        status="ok",
        model_ready=loaded,
        loaded_category=category,
        device=device,
        last_model_error=error,
    )


@app.get("/categories", response_model=list[str], tags=["service"])
def categories() -> list[str]:
    """List the supported MVTec AD categories."""
    return settings.SUPPORTED_CATEGORIES


@app.get("/settings", tags=["service"])
def get_public_settings() -> dict[str, object]:
    """Expose only non-sensitive runtime settings used by the dashboard."""
    return {
        "inference_mode": "zero-shot",
        "k_shot": 0,
        "defect_threshold": settings.DEFECT_THRESHOLD,
        "max_upload_bytes": settings.MAX_UPLOAD_BYTES,
        "max_image_pixels": settings.MAX_IMAGE_PIXELS,
        "supported_categories": settings.SUPPORTED_CATEGORIES,
        "persistence": "SQLite metadata with inspection-scoped local asset files",
        "model_cache_policy": "one category model cached at a time; inference serialized",
    }


@app.get("/evaluation", tags=["evaluation"])
def get_evaluation_results() -> dict[str, object]:
    """Read the saved pilot summary and per-category metrics without running evaluation."""
    summary_path = EVALUATION_RESULTS_ROOT / "summary_metrics.csv"
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail={"error": "evaluation_not_available", "message": "Saved evaluation summary is not available."})
    try:
        with summary_path.open("r", encoding="utf-8-sig", newline="") as summary_file:
            rows = list(csv.DictReader(summary_file))
        results: list[dict[str, object]] = []
        for row in rows:
            category = row.get("category", "")
            if category not in settings.SUPPORTED_CATEGORIES:
                continue
            metrics_path = EVALUATION_RESULTS_ROOT / f"{category}_metrics.json"
            with metrics_path.open("r", encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
            results.append({
                "category": category,
                "image_auroc": float(row["image_auroc"]),
                "pixel_auroc": float(row["pixel_auroc"]),
                "image_count": int(row["image_count"]),
                "normal_count": int(row["normal_count"]),
                "defective_count": int(row["defective_count"]),
                "mean_inference_time_sec": float(row["mean_inference_time_sec"]),
                "std_inference_time_sec": metrics.get("std_inference_time_sec"),
                "min_inference_time_sec": metrics.get("min_inference_time_sec"),
                "max_inference_time_sec": metrics.get("max_inference_time_sec"),
                "total_evaluation_time_sec": metrics.get("total_evaluation_time_sec"),
                "class_distribution": metrics.get("class_distribution", {}),
                "model_name": metrics.get("model_name"),
                "inference_mode": metrics.get("inference_mode"),
                "timestamp_utc": metrics.get("timestamp_utc"),
            })
        return {"results": results, "source": "experiments/evaluation_results/summary_metrics.csv"}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.exception("Could not read saved evaluation metrics")
        raise HTTPException(status_code=500, detail={"error": "evaluation_unavailable", "message": "Saved evaluation metrics are malformed or unavailable."}) from exc


@app.get("/evaluation/report", tags=["evaluation"])
def get_evaluation_report() -> FileResponse:
    """Serve the saved Markdown evaluation summary from the fixed results folder."""
    path = EVALUATION_RESULTS_ROOT / "summary_report.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"error": "evaluation_report_not_available", "message": "The saved evaluation report is unavailable."})
    return FileResponse(path, media_type="text/markdown", headers={"X-Content-Type-Options": "nosniff"})


@app.get("/evaluation/assets/{asset_name}", tags=["evaluation"])
def get_evaluation_chart(asset_name: Literal["image_auroc", "pixel_auroc"]) -> FileResponse:
    """Serve one allowlisted saved AUROC chart."""
    path = EVALUATION_RESULTS_ROOT / f"{asset_name}_comparison.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"error": "evaluation_chart_not_available", "message": "The saved evaluation chart is unavailable."})
    return FileResponse(path, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"})


@app.get("/dataset", tags=["dataset"])
def get_dataset_summary() -> dict[str, object]:
    """Count available MVTec image files under supported local categories."""
    root = settings.DATASET_ROOT.resolve()
    categories_data = []
    for category in settings.SUPPORTED_CATEGORIES:
        category_root = root / category
        if not category_root.is_dir() or not category_root.resolve().is_relative_to(root):
            continue
        train_count = _count_images(category_root / "train" / "good")
        normal_count = _count_images(category_root / "test" / "good")
        test_root = category_root / "test"
        defective_count = sum(_count_images(directory) for directory in test_root.iterdir() if directory.is_dir() and directory.name.lower() != "good") if test_root.is_dir() else 0
        mask_count = _count_images(category_root / "ground_truth")
        categories_data.append({
            "category": category,
            "training_count": train_count,
            "normal_test_count": normal_count,
            "defective_test_count": defective_count,
            "test_count": normal_count + defective_count,
            "ground_truth_mask_count": mask_count,
            "training_count_verified": (category_root / "train" / "good").is_dir(),
            "normal_test_count_verified": (category_root / "test" / "good").is_dir(),
            "defective_test_count_verified": (category_root / "test").is_dir(),
            "ground_truth_mask_count_verified": (category_root / "ground_truth").is_dir(),
        })
    return {"categories": categories_data, "category_count": len(categories_data)}


@app.get("/dataset/{category}/samples", tags=["dataset"])
def get_dataset_samples(
    category: str,
    limit_per_group: int = Query(default=3, ge=1, le=12),
) -> dict[str, object]:
    """Return a bounded set of sample references without exposing local paths."""
    category_root = _dataset_category_root(category)
    result: list[dict[str, object]] = []
    for group in ("normal", "defective"):
        files = _dataset_sample_files(category_root, group)
        for index, path in enumerate(files[:limit_per_group]):
            defect_type = path.parent.name if group == "defective" else "good"
            mask_path = category_root / "ground_truth" / defect_type / f"{path.stem}_mask.png" if group == "defective" else None
            mask_available = mask_path is not None and _safe_dataset_path(mask_path).is_file()
            base_url = f"/dataset/{category}/samples/{group}/{index}"
            result.append({
                "sample_id": f"{group}-{index}",
                "filename": path.name,
                "group": group,
                "defect_type": defect_type,
                "mask_available": mask_available,
                "image_url": f"{base_url}/image",
                "mask_url": f"{base_url}/mask" if mask_available else None,
            })
    return {"category": category, "samples": result}


@app.get("/dataset/{category}/samples/{group}/{sample_index}/{asset_name}", tags=["dataset"])
def get_dataset_sample_asset(
    category: str,
    group: Literal["normal", "defective"],
    sample_index: int,
    asset_name: Literal["image", "mask"],
) -> FileResponse:
    """Serve an enumerated sample image or its matching mask, never a caller path."""
    category_root = _dataset_category_root(category)
    files = _dataset_sample_files(category_root, group)
    if sample_index < 0 or sample_index >= min(len(files), 12):
        raise HTTPException(status_code=404, detail={"error": "dataset_sample_not_found", "message": "The requested sample is unavailable."})
    image_path = _safe_dataset_path(files[sample_index])
    path = image_path
    if asset_name == "mask":
        if group != "defective":
            raise HTTPException(status_code=404, detail={"error": "dataset_mask_not_available", "message": "Normal images do not have defect masks."})
        path = _safe_dataset_path(category_root / "ground_truth" / image_path.parent.name / f"{image_path.stem}_mask.png")
        if not path.is_file():
            raise HTTPException(status_code=404, detail={"error": "dataset_mask_not_available", "message": "No ground-truth mask exists for this sample."})
    media_type = "image/png" if asset_name == "mask" else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    return FileResponse(path, media_type=media_type, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})


def _dataset_category_root(category: str) -> Path:
    if category not in settings.SUPPORTED_CATEGORIES:
        raise HTTPException(status_code=404, detail={"error": "dataset_category_not_found", "message": "The requested MVTec category is not supported."})
    root = settings.DATASET_ROOT.resolve()
    category_root = (root / category).resolve()
    if not category_root.is_relative_to(root) or not category_root.is_dir():
        raise HTTPException(status_code=404, detail={"error": "dataset_category_not_found", "message": "The requested MVTec category is unavailable."})
    return category_root


def _dataset_sample_files(category_root: Path, group: str) -> list[Path]:
    test_root = category_root / "test"
    if group == "normal":
        sample_root = test_root / "good"
        candidates = sorted(sample_root.iterdir()) if sample_root.is_dir() else []
    else:
        candidates = sorted(path for defect_dir in test_root.iterdir() if defect_dir.is_dir() and defect_dir.name.lower() != "good" for path in defect_dir.iterdir()) if test_root.is_dir() else []
    valid = []
    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            try:
                safe_path = _safe_dataset_path(path)
            except ValueError:
                continue
            valid.append(safe_path)
    return valid


def _count_images(directory: Path) -> int:
    root = settings.DATASET_ROOT.resolve()
    try:
        safe_root = _safe_dataset_path(directory)
    except ValueError:
        return 0
    if not safe_root.is_dir():
        return 0
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(1 for path in safe_root.rglob("*") if path.is_file() and path.suffix.lower() in extensions and path.resolve().is_relative_to(root))


def _safe_dataset_path(path: Path) -> Path:
    root = settings.DATASET_ROOT.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Dataset path escapes configured dataset root")
    return resolved


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    category: str = Form(..., description="MVTec AD category, e.g. bottle"),
    image: UploadFile = File(..., description="Image file to inspect"),
    store: InspectionHistoryStore = Depends(get_history_store),
) -> PredictionResponse:
    """Run pretrained zero-shot WinCLIP and return image score and pixel map."""
    if category not in settings.SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail={"error": "unsupported_category", "message": "Use one of GET /categories."},
        )

    started = time.perf_counter()
    image_format = ""
    try:
        raw = image.file.read(settings.MAX_UPLOAD_BYTES + 1)
        if not raw:
            raise HTTPException(status_code=400, detail={"error": "empty_upload", "message": "The uploaded file is empty."})
        if len(raw) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"error": "image_too_large", "message": f"Maximum image size is {settings.MAX_UPLOAD_BYTES} bytes."},
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                source.verify()
            with Image.open(io.BytesIO(raw)) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise HTTPException(status_code=415, detail={"error": "unsupported_image", "message": "Animated or multi-frame images are not supported."})
                if source.width * source.height > settings.MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail={"error": "image_dimensions_too_large", "message": "Image dimensions exceed the configured pixel limit."})
                image_format = source.format or ""
                pil_image = ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=415, detail={"error": "invalid_image", "message": "Upload must be a valid, non-corrupt image."}) from exc
    except Exception as exc:
        logger.exception("Image decoding failed")
        raise HTTPException(status_code=400, detail={"error": "image_read_error", "message": "Could not read the uploaded image."}) from exc

    try:
        score, anomaly_map, device = runtime.predict(category, pil_image)
    except Exception as exc:
        logger.exception("WinCLIP inference failed for category %s", category)
        raise HTTPException(
            status_code=503,
            detail={"error": "inference_unavailable", "message": "WinCLIP could not initialize or complete inference. Check service logs."},
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    map_image = _render_map(anomaly_map)
    image_buffer = io.BytesIO()
    map_image.save(image_buffer, format="PNG", optimize=True)
    encoded_map = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    ground_truth = _ground_truth_payload(category, pil_image)
    ground_truth_mask_base64 = ground_truth["ground_truth_mask_base64"]
    try:
        record = store.save_inspection(
            category=category,
            filename=image.filename or "uploaded-image",
            predicted_label="defective" if score >= settings.DEFECT_THRESHOLD else "normal",
            anomaly_score=score,
            threshold=settings.DEFECT_THRESHOLD,
            inference_time_ms=round(elapsed_ms, 3),
            uploaded_image=raw,
            image_format=image_format,
            heatmap_png=image_buffer.getvalue(),
            ground_truth_mask_png=(
                base64.b64decode(ground_truth_mask_base64, validate=True)
                if isinstance(ground_truth_mask_base64, str)
                else None
            ),
        )
    except Exception as exc:
        logger.exception("Could not persist completed inspection")
        raise HTTPException(
            status_code=500,
            detail={"error": "inspection_persistence_failed", "message": "Inference completed, but the result could not be saved."},
        ) from exc
    return PredictionResponse(
        inspection_id=record["inspection_id"],
        category=category,
        anomaly_score=score,
        predicted_label="defective" if score >= settings.DEFECT_THRESHOLD else "normal",
        threshold=settings.DEFECT_THRESHOLD,
        anomaly_map_width=len(anomaly_map[0]),
        anomaly_map_height=len(anomaly_map),
        anomaly_map_visualization_base64=encoded_map,
        anomaly_map_visualization_mime_type="image/png",
        **ground_truth,
        inference_time_ms=round(elapsed_ms, 3),
        device=device,
        inference_mode="zero-shot",
        k_shot=0,
    )


@app.get("/inspections", response_model=list[InspectionHistoryRecord], tags=["history"])
def list_inspections(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    store: InspectionHistoryStore = Depends(get_history_store),
) -> list[InspectionHistoryRecord]:
    """List saved inspection metadata, newest first."""
    try:
        return store.list_inspections(limit=limit, offset=offset)
    except Exception as exc:
        logger.exception("Could not list inspection history")
        raise HTTPException(status_code=500, detail={"error": "history_unavailable", "message": "Inspection history could not be read."}) from exc


@app.get("/inspections/{inspection_id}", response_model=InspectionHistoryRecord, tags=["history"])
def get_inspection(
    inspection_id: str,
    store: InspectionHistoryStore = Depends(get_history_store),
) -> InspectionHistoryRecord:
    """Fetch one saved inspection by canonical UUID."""
    try:
        record = store.get_inspection(inspection_id)
    except InvalidInspectionID as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_inspection_id", "message": str(exc)}) from exc
    except Exception as exc:
        logger.exception("Could not fetch inspection %s", inspection_id)
        raise HTTPException(status_code=500, detail={"error": "history_unavailable", "message": "Inspection history could not be read."}) from exc
    if record is None:
        raise HTTPException(status_code=404, detail={"error": "inspection_not_found", "message": "No saved inspection has that ID."})
    return record


@app.get("/inspections/{inspection_id}/assets/{asset_name}", tags=["history"])
def get_inspection_asset(
    inspection_id: str,
    asset_name: Literal["image", "heatmap", "ground_truth"],
    store: InspectionHistoryStore = Depends(get_history_store),
) -> FileResponse:
    """Serve only assets referenced by the selected validated inspection."""
    record = _require_inspection(inspection_id, store)
    asset_field = {
        "image": "image_path",
        "heatmap": "heatmap_path",
        "ground_truth": "ground_truth_mask_path",
    }[asset_name]
    relative_path = record[asset_field]
    if relative_path is None:
        raise HTTPException(status_code=404, detail={"error": "asset_not_available", "message": "This inspection has no ground-truth mask."})
    try:
        path = store.resolve_asset(relative_path)
    except Exception as exc:
        logger.exception("Rejected unsafe inspection asset path")
        raise HTTPException(status_code=500, detail={"error": "asset_unavailable", "message": "The requested inspection asset is unavailable."}) from exc
    media_type = "image/png" if asset_name in {"heatmap", "ground_truth"} else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/inspections/{inspection_id}/report.pdf", tags=["history"])
def download_inspection_report(
    inspection_id: str,
    inline: bool = Query(default=False, description="Display the generated PDF in the browser instead of forcing download"),
    store: InspectionHistoryStore = Depends(get_history_store),
) -> Response:
    """Build and download a PDF report for one persisted inspection."""
    record = _require_inspection(inspection_id, store)
    try:
        assets = {
            "image": store.resolve_asset(record["image_path"]),
            "heatmap": store.resolve_asset(record["heatmap_path"]),
            "ground_truth": (
                store.resolve_asset(record["ground_truth_mask_path"])
                if record["ground_truth_mask_path"] is not None
                else None
            ),
        }
        pdf_bytes = build_inspection_pdf(record, assets)
    except Exception as exc:
        logger.exception("Could not generate PDF report for inspection %s", inspection_id)
        raise HTTPException(status_code=500, detail={"error": "report_generation_failed", "message": "The inspection report could not be generated."}) from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{"inline" if inline else "attachment"}; filename="defectzero-inspection-{inspection_id}.pdf"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/inspections/batch-report.pdf", tags=["history"])
def download_batch_report(
    request: BatchReportRequest,
    store: InspectionHistoryStore = Depends(get_history_store),
) -> Response:
    """Create a batch summary only from existing persisted inspection records."""
    if len(set(request.inspection_ids)) != len(request.inspection_ids):
        raise HTTPException(status_code=422, detail={"error": "duplicate_inspection_id", "message": "Inspection IDs must be unique."})
    records = []
    for inspection_id in request.inspection_ids:
        record = _require_inspection(inspection_id, store)
        records.append(record)
    try:
        pdf_bytes = build_batch_summary_pdf(records)
    except Exception as exc:
        logger.exception("Could not generate batch report")
        raise HTTPException(status_code=500, detail={"error": "report_generation_failed", "message": "The batch summary could not be generated."}) from exc
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="defectzero-batch-summary.pdf"', "Cache-Control": "no-store"})


def _require_inspection(inspection_id: str, store: InspectionHistoryStore) -> dict[str, object]:
    try:
        record = store.get_inspection(inspection_id)
    except InvalidInspectionID as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_inspection_id", "message": str(exc)}) from exc
    except Exception as exc:
        logger.exception("Could not fetch inspection %s", inspection_id)
        raise HTTPException(status_code=500, detail={"error": "history_unavailable", "message": "Inspection history could not be read."}) from exc
    if record is None:
        raise HTTPException(status_code=404, detail={"error": "inspection_not_found", "message": "No saved inspection has that ID."})
    return record


def _ground_truth_payload(category: str, image: Image.Image) -> dict[str, object]:
    """Return a mask only for an unambiguous pixel-content match in MVTec."""
    empty: dict[str, object] = {
        "ground_truth_available": False,
        "ground_truth_defect_type": None,
        "ground_truth_mask_base64": None,
        "ground_truth_mask_mime_type": None,
        "ground_truth_mask_width": None,
        "ground_truth_mask_height": None,
    }
    try:
        match = find_ground_truth_match(category, image, settings.DATASET_ROOT)
        if match is None:
            return empty
        mask_bytes = match.mask_path.read_bytes()
        with Image.open(io.BytesIO(mask_bytes)) as mask:
            if mask.format != "PNG":
                logger.warning("Ground-truth mask is not PNG: %s", match.mask_path)
                return empty
            width, height = mask.size
            mask.verify()
        return {
            "ground_truth_available": True,
            "ground_truth_defect_type": match.defect_type,
            "ground_truth_mask_base64": base64.b64encode(mask_bytes).decode("ascii"),
            "ground_truth_mask_mime_type": "image/png",
            "ground_truth_mask_width": width,
            "ground_truth_mask_height": height,
        }
    except Exception:
        logger.exception("Could not resolve ground-truth mask for category %s", category)
        return empty


def _render_map(values: list[list[float]]) -> Image.Image:
    """Render a per-image normalized blue-yellow-red PNG for map visualization."""
    height, width = len(values), len(values[0])
    flat = [value for row in values for value in row]
    low, high = min(flat), max(flat)
    scale = high - low
    rgb = bytearray()
    for row in values:
        for value in row:
            level = 0.0 if scale <= 1e-12 else min(1.0, max(0.0, (value - low) / scale))
            # Compact blue -> cyan -> yellow -> red palette.
            if level < 1 / 3:
                t = level * 3
                color = (0, int(255 * t), 255)
            elif level < 2 / 3:
                t = (level - 1 / 3) * 3
                color = (int(255 * t), 255, int(255 * (1 - t)))
            else:
                t = (level - 2 / 3) * 3
                color = (255, int(255 * (1 - t)), 0)
            rgb.extend(color)
    return Image.frombytes("RGB", (width, height), bytes(rgb))
