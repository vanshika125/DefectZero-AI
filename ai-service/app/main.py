"""
DefectZero AI — FastAPI Inference Microservice Entrypoint.
Provides REST API endpoints for zero-shot industrial defect inspection.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import time

from app.config import settings
from app.schemas import HealthResponse, SystemStatusResponse, InspectionResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="Zero-Shot Industrial Defect Detection & Pixel-Level Localization using Vision-Language Models",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-Origin Resource Sharing (CORS) for Node.js backend and React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Service liveness probe."""
    return HealthResponse(status="healthy")


@app.get(f"{settings.API_V1_STR}/status", response_model=SystemStatusResponse, tags=["Diagnostics"])
async def system_status():
    """Retrieve runtime compute and model specifications."""
    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return SystemStatusResponse(
        device=device,
        model_name=settings.DEFAULT_MODEL_NAME,
        supported_categories=settings.SUPPORTED_CATEGORIES,
        window_scales=settings.WINDOW_SIZES,
    )


@app.post(
    f"{settings.API_V1_STR}/inspect",
    response_model=InspectionResponse,
    tags=["Inference"],
    summary="Zero-Shot Defect Inspection",
)
async def inspect_image(
    image: UploadFile = File(..., description="High-resolution industrial image to inspect"),
    category: str = Form(..., description="Target MVTec AD category or custom industrial SKU"),
    threshold: float = Form(default=settings.DEFAULT_THRESHOLD, description="Anomaly decision threshold"),
):
    """
    Execute zero-shot defect classification and pixel-level anomaly localization.
    Note: Full PyTorch WinCLIP pipeline will be invoked during active inference phase.
    """
    start_time = time.time()

    # Validate category
    if category.lower() not in settings.SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Category '{category}' is unsupported. Must be one of: {settings.SUPPORTED_CATEGORIES}",
        )

    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image format (PNG, JPEG).",
        )

    latency_ms = (time.time() - start_time) * 1000.0

    # Architectural contract placeholder for initial setup
    return InspectionResponse(
        category=category.lower(),
        anomaly_score=0.0,
        is_defective=False,
        threshold_applied=threshold,
        heatmap_base64=None,
        binary_mask_base64=None,
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
