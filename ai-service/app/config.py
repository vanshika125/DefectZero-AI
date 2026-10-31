"""
Configuration and settings for DefectZero AI FastAPI microservice.
"""

from typing import List
from pydantic_settings import BaseSettings if False else object  # Fallback for lightweight load


class Settings:
    PROJECT_NAME: str = "DefectZero AI — Zero-Shot Defect Detection Service"
    API_V1_STR: str = "/api/v1"
    DEFAULT_MODEL_NAME: str = "ViT-B-16-plus-240"
    DEFAULT_PRETRAINED: str = "laion400m_e32"
    DEFAULT_THRESHOLD: float = 0.5
    WINDOW_SIZES: List[int] = [2, 3]
    GAUSSIAN_SIGMA: float = 4.0
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    SUPPORTED_CATEGORIES: List[str] = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]


settings = Settings()
