"""
Pydantic schemas and data contracts for DefectZero AI inference requests and responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health state")
    service: str = Field(default="DefectZero AI Microservice")
    version: str = Field(default="0.1.0")


class SystemStatusResponse(BaseModel):
    device: str = Field(description="Compute execution device (cuda or cpu)")
    model_name: str = Field(description="Visual foundation model checkpoint")
    supported_categories: List[str]
    window_scales: List[int]


class InspectionResponse(BaseModel):
    category: str = Field(description="Target industrial category")
    anomaly_score: float = Field(description="Global image anomaly score in range [0, 1]")
    is_defective: bool = Field(description="Binary classification verdict against threshold")
    threshold_applied: float = Field(description="Anomaly score decision threshold")
    heatmap_base64: Optional[str] = Field(default=None, description="Base64 encoded PNG anomaly heatmap")
    binary_mask_base64: Optional[str] = Field(default=None, description="Base64 encoded binary segmentation mask")
    latency_ms: float = Field(description="Inference pipeline latency in milliseconds")
