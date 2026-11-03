"""Documented HTTP response models."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_ready: bool = Field(description="True after the selected category model has loaded")
    loaded_category: str | None = Field(description="Currently cached category model, if loaded")
    device: str | None = Field(description="Inference device of the cached model")
    last_model_error: str | None = Field(description="Most recent initialization/inference error, if any")


class PredictionResponse(BaseModel):
    inspection_id: str
    category: str
    anomaly_score: float = Field(description="WinCLIP image-level anomaly score; interpret with the configured threshold")
    predicted_label: Literal["normal", "defective"]
    threshold: float = Field(description="Decision threshold applied to anomaly_score")
    anomaly_map_width: int
    anomaly_map_height: int
    anomaly_map_visualization_base64: str = Field(description="Base64 PNG heatmap, normalized for visualization only")
    anomaly_map_visualization_mime_type: Literal["image/png"] = "image/png"
    ground_truth_available: bool = Field(description="True only when uploaded pixels uniquely match a defective MVTec test image and its mask exists")
    ground_truth_defect_type: str | None = Field(description="Matched MVTec defect type, if a mask is available")
    ground_truth_mask_base64: str | None = Field(description="Base64 PNG ground-truth mask, or null when unavailable")
    ground_truth_mask_mime_type: Literal["image/png"] | None = "image/png"
    ground_truth_mask_width: int | None = None
    ground_truth_mask_height: int | None = None
    inference_time_ms: float
    device: str
    inference_mode: Literal["zero-shot"] = "zero-shot"
    k_shot: Literal[0] = 0


class InspectionHistoryRecord(BaseModel):
    inspection_id: str
    category: str
    filename: str
    predicted_label: Literal["normal", "defective"]
    anomaly_score: float
    threshold: float
    inference_time_ms: float
    timestamp_utc: str
    image_path: str
    heatmap_path: str
    ground_truth_mask_path: str | None


class BatchReportRequest(BaseModel):
    inspection_ids: list[str] = Field(min_length=1, max_length=30, description="Saved inspections to include in the batch summary")
