"""
DefectZero AI - Zero-Shot WinCLIP Inference Demo
=================================================
First zero-shot anomaly detection experiment using WinCLIP (Anomalib 2.7.0).

This script:
  1. Loads the WinCLIP model in zero-shot mode (no reference images).
  2. Automatically selects one defective bottle image from the MVTec dataset.
  3. Preprocesses the image using the model's transform pipeline.
  4. Runs inference to produce an image-level anomaly score and a pixel-level
     anomaly map.
  5. Saves a three-panel visualization (original | heatmap | overlay) to
     experiments/eda_outputs/winclip_demo.png.

Usage:
    cd DefectZero-AI
    .venv-winclip\\Scripts\\python.exe experiments/run_winclip_demo.py

Environment:
    Python 3.11  |  PyTorch 2.15.0+cpu  |  Anomalib 2.7.0
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Set stdout/stderr encoding to utf-8 if possible to prevent cp1252 errors on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 0. Path Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../experiments
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../DefectZero-AI
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR = SCRIPT_DIR / "eda_outputs"
OUTPUT_PATH = OUTPUT_DIR / "winclip_demo.png"

# Ensure output directory exists inside experiments/
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as exc:
    sys.exit(f"[ERROR] PyTorch not found: {exc}\n"
             "        Please activate .venv-winclip.")

try:
    from PIL import Image
except ImportError as exc:
    sys.exit(f"[ERROR] Pillow not found: {exc}")

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend safe for all environments
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError as exc:
    sys.exit(f"[ERROR] Matplotlib / NumPy not found: {exc}")

try:
    from torchvision.transforms import functional as TF
except ImportError as exc:
    sys.exit(f"[ERROR] torchvision not found: {exc}")

try:
    from anomalib.models.image.winclip.torch_model import WinClipModel
    from anomalib.data import InferenceBatch
except ImportError as exc:
    sys.exit(f"[ERROR] Anomalib not found: {exc}\n"
             "        Install with: pip install anomalib[vlm,clip]")


# ---------------------------------------------------------------------------
# 2. Automatically select ONE defective test image
# ---------------------------------------------------------------------------
def find_defective_image(dataset_root: Path, category: str = "bottle") -> Path:
    """Find and return the path to a defective image from dataset/<category>/test/.

    Excludes the 'good' subdirectory and selects the first available image
    from one of the defect subdirectories (e.g., broken_large, broken_small, etc.).
    """
    test_dir = dataset_root / category / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found at: {test_dir}\n"
            f"Expected dataset layout: dataset/{category}/test/<defect_type>/*.png"
        )

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    # Sort subdirectories for deterministic selection
    for subfolder in sorted(test_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        # Exclude normal/good images
        if subfolder.name.lower() == "good":
            continue

        for img_path in sorted(subfolder.iterdir()):
            if img_path.suffix.lower() in valid_extensions:
                return img_path

    raise FileNotFoundError(
        f"No defective image files found under {test_dir}.\n"
        "Please check that the dataset contains defective subdirectories."
    )


# ---------------------------------------------------------------------------
# 3. Visualization Helper
# ---------------------------------------------------------------------------
def save_visualization(original_pil: Image.Image,
                       anomaly_map: np.ndarray,
                       save_path: Path) -> None:
    """Save a 3-panel visualization: Original Image | Anomaly Heatmap | Overlay."""
    orig_w, orig_h = original_pil.size
    original_np = np.array(original_pil)

    # Resize anomaly map to match original image dimensions for accurate spatial overlay
    anomaly_pil = Image.fromarray(anomaly_map)
    anomaly_resized = anomaly_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
    anomaly_resized_np = np.array(anomaly_resized)

    # Normalize heatmap to [0, 1]
    norm = Normalize(vmin=float(anomaly_resized_np.min()), vmax=float(anomaly_resized_np.max()))
    heatmap_norm = norm(anomaly_resized_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original Image
    axes[0].imshow(original_np)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Predicted Anomaly Heatmap
    im = axes[1].imshow(heatmap_norm, cmap="jet")
    axes[1].set_title("Predicted Anomaly Heatmap", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Heatmap Overlay on Original Image
    axes[2].imshow(original_np)
    axes[2].imshow(heatmap_norm, cmap="jet", alpha=0.45)
    axes[2].set_title("Heatmap Overlay", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle("DefectZero AI - WinCLIP Zero-Shot Anomaly Detection",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualization saved to: {save_path}")


# ---------------------------------------------------------------------------
# 4. Main Inference Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("  DefectZero AI - WinCLIP Zero-Shot Inference Experiment")
    print("=" * 65)

    # Step 1: Select a defective test image
    image_path = find_defective_image(DATASET_ROOT, category="bottle")
    defect_type = image_path.parent.name
    print(f"\n[+] Selected Image : {image_path}")
    print(f"[+] Defect Category: {defect_type}")

    # Step 2: Load image with PIL and convert to RGB
    pil_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    print(f"[+] Original Size  : {orig_w}x{orig_h} (RGB)")

    # Step 3: Initialize WinCLIP model for the "bottle" category (zero-shot)
    print("\n[+] Initializing WinCLIP model (zero-shot, class_name='bottle')...")
    init_start = time.perf_counter()

    model = WinClipModel(
        class_name="bottle",
        reference_images=None,
    )

    init_elapsed = time.perf_counter() - init_start
    print(f"[OK] WinCLIP model initialized in {init_elapsed:.2f}s (k_shot={model.k_shot})")

    # Step 4: Preprocess image using model's transform pipeline
    # model.transform starts with ToPILImage(), so pass tensor input
    image_tensor = TF.to_tensor(pil_image)
    preprocessed = model.transform(image_tensor)
    batch = preprocessed.unsqueeze(0)  # Shape: (1, C, H, W)
    print(f"[+] Preprocessed batch shape: {tuple(batch.shape)}")

    # Step 5: Run inference without training or fine-tuning
    model.eval()
    print("\n[+] Running zero-shot inference (torch.inference_mode)...")
    infer_start = time.perf_counter()
    with torch.inference_mode():
        output = model(batch)
    infer_elapsed = time.perf_counter() - infer_start

    # Step 6: Robustly handle model output format (InferenceBatch or tuple)
    if isinstance(output, InferenceBatch):
        pred_score_tensor = output.pred_score
        anomaly_map_tensor = output.anomaly_map
    elif isinstance(output, tuple) and len(output) >= 2:
        pred_score_tensor = output[0]
        anomaly_map_tensor = output[1]
    elif hasattr(output, "pred_score") and hasattr(output, "anomaly_map"):
        pred_score_tensor = getattr(output, "pred_score")
        anomaly_map_tensor = getattr(output, "anomaly_map")
    else:
        raise TypeError(f"Unexpected output type from WinClipModel: {type(output)}")

    anomaly_score = float(pred_score_tensor.squeeze().item())
    anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()

    # Step 7: Print required experiment metrics
    print("\n" + "=" * 65)
    print("  EXPERIMENT RESULTS")
    print("=" * 65)
    print(f"  Selected Image Path : {image_path}")
    print(f"  Model Name          : WinCLIP (ViT-B-16-plus-240, LAION-400M)")
    print(f"  Inference Mode      : Zero-Shot (k_shot={model.k_shot})")
    print(f"  Anomaly Score       : {anomaly_score:.6f}")
    print(f"  Anomaly Map Shape   : {anomaly_map.shape}")
    print(f"  Inference Time      : {infer_elapsed:.4f} seconds")
    print("=" * 65)

    # Step 8: Save visualization
    save_visualization(pil_image, anomaly_map, OUTPUT_PATH)
    print(f"\n[OK] Experiment completed successfully.")


if __name__ == "__main__":
    main()
