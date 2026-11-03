"""Lazy, single-category cache for the verified Anomalib WinCLIP API."""

from __future__ import annotations

import threading
from typing import Any

from PIL import Image


class WinClipRuntime:
    """Keep one category model alive and serialize access to its inference state.

    WinClipModel is initialized with the selected class name and zero reference
    images, matching experiments/run_winclip_demo.py. Keeping one model limits
    memory growth; alternating categories may reload the model and its weights.
    """

    def __init__(self, categories: list[str]) -> None:
        self._categories = frozenset(categories)
        self._lock = threading.RLock()
        # Health checks must never wait behind model initialization or inference.
        self._state_lock = threading.Lock()
        self._model: Any = None
        self._category: str | None = None
        self._device: Any = None
        self._last_error: str | None = None

    def status(self) -> tuple[bool, str | None, str | None, str | None]:
        with self._state_lock:
            return (
                self._model is not None,
                self._category,
                str(self._device) if self._device is not None else None,
                self._last_error,
            )

    def predict(self, category: str, image: Image.Image) -> tuple[float, list[list[float]], str]:
        if category not in self._categories:
            raise ValueError(f"Unsupported category: {category}")

        with self._lock:
            try:
                model, device = self._get_model(category)
                import torch
                from torchvision.transforms import functional as TF

                # Same preprocessing as the verified demo: RGB PIL -> tensor ->
                # WinCLIP's own transform -> single-item batch.
                image_tensor = TF.to_tensor(image.convert("RGB"))
                batch = model.transform(image_tensor).unsqueeze(0).to(device)
                with torch.inference_mode():
                    output = model(batch)

                if hasattr(output, "pred_score") and hasattr(output, "anomaly_map"):
                    score_tensor, map_tensor = output.pred_score, output.anomaly_map
                elif isinstance(output, (tuple, list)) and len(output) >= 2:
                    score_tensor, map_tensor = output[0], output[1]
                else:
                    raise TypeError(f"Unexpected WinClipModel output type: {type(output).__name__}")

                score = float(score_tensor.squeeze().detach().cpu().item())
                anomaly_map = map_tensor.squeeze().detach().cpu().float()
                if anomaly_map.ndim != 2 or anomaly_map.numel() == 0:
                    raise ValueError(f"Expected a non-empty 2D anomaly map, got {tuple(anomaly_map.shape)}")
                if not torch.isfinite(anomaly_map).all() or not torch.isfinite(torch.tensor(score)):
                    raise ValueError("WinCLIP produced a non-finite score or anomaly map")
                values = anomaly_map.tolist()
                with self._state_lock:
                    self._last_error = None
                return score, values, str(device)
            except Exception as exc:
                with self._state_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                raise

    def _get_model(self, category: str) -> tuple[Any, Any]:
        if self._model is not None and self._category == category:
            return self._model, self._device

        import torch
        from anomalib.models.image.winclip.torch_model import WinClipModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WinClipModel(class_name=category, reference_images=None)
        model.eval()
        model.to(device)
        with self._state_lock:
            self._model = model
            self._category = category
            self._device = device
            self._last_error = None
        return model, device
