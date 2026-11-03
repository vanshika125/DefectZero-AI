"""
DefectZero AI - WinCLIP Zero-Shot Evaluation Script
===================================================
Quantitative evaluation pipeline for zero-shot WinCLIP on MVTec AD.

Phase 1: Pilot Evaluation (Bottle Category)
- Supports command-line options: --category <category> --limit <N>
- Balanced sampling across normal and defective categories
- Computes:
    1. Image-level AUROC
    2. Pixel-level AUROC
    3. Mean inference time per image
- Aligns ground truth masks with predicted anomaly maps
- Outputs:
    - experiments/evaluation_results/<category>_predictions.csv
    - experiments/evaluation_results/<category>_metrics.json
    - experiments/evaluation_results/<category>_evaluation_samples.png

Usage:
    cd DefectZero-AI
    .venv-winclip\\Scripts\\python.exe experiments/evaluate_winclip.py --category bottle --limit 20

Environment:
    Python 3.11  |  PyTorch 2.15.0+cpu  |  Anomalib 2.7.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure Windows standard output encoding to avoid cp1252 charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent           # .../experiments
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../DefectZero-AI
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"

# ---------------------------------------------------------------------------
# Dependency Imports
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as exc:
    sys.exit(f"[ERROR] PyTorch not found: {exc}\n        Please activate .venv-winclip.")

try:
    from PIL import Image
except ImportError as exc:
    sys.exit(f"[ERROR] Pillow not found: {exc}")

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Headless backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError as exc:
    sys.exit(f"[ERROR] Matplotlib / NumPy not found: {exc}")

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    sys.exit(f"[ERROR] scikit-learn not found: {exc}")

try:
    from anomalib.models.image.winclip.torch_model import WinClipModel
    from anomalib.data import InferenceBatch
except ImportError as exc:
    sys.exit(f"[ERROR] Anomalib not found: {exc}\n        Install with: pip install anomalib[vlm,clip]")


# ---------------------------------------------------------------------------
# Sample Discovery & Balanced Selection
# ---------------------------------------------------------------------------
def discover_and_sample_images(
    dataset_root: Path,
    category: str,
    limit: Optional[int] = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Select a balanced set of normal and defective images for evaluation.

    Parameters
    ----------
    dataset_root : Path
        Path to the dataset directory containing <category>/test/.
    category : str
        Object class name (e.g. 'bottle').
    limit : int or None
        Maximum number of images to evaluate. If limit is None or <= 0,
        evaluates all available test images.

    Returns
    -------
    selected_items : list of dict
        List of dictionaries with keys:
          - 'image_path': Path to test image
          - 'defect_type': Subfolder name ('good' or defect name)
          - 'label': 0 for normal, 1 for defective
          - 'mask_path': Path to ground truth mask or None if normal
    class_counts : dict
        Counts of selected images per subfolder.
    """
    test_dir = dataset_root / category / "test"
    gt_dir = dataset_root / category / "ground_truth"

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            f"Expected dataset layout: {dataset_root}/{category}/test/"
        )

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    # 1. Collect normal (good) images
    good_dir = test_dir / "good"
    good_images: List[Path] = []
    if good_dir.is_dir():
        good_images = sorted([
            p for p in good_dir.iterdir()
            if p.suffix.lower() in valid_extensions and p.is_file()
        ])

    # 2. Collect defective images grouped by defect type
    defect_dirs = sorted([
        d for d in test_dir.iterdir()
        if d.is_dir() and d.name.lower() != "good"
    ])

    defect_images_by_type: Dict[str, List[Path]] = {}
    for d_dir in defect_dirs:
        imgs = sorted([
            p for p in d_dir.iterdir()
            if p.suffix.lower() in valid_extensions and p.is_file()
        ])
        if imgs:
            defect_images_by_type[d_dir.name] = imgs

    # 3. Determine counts for normal and defective samples
    total_good = len(good_images)
    total_defective = sum(len(imgs) for imgs in defect_images_by_type.values())

    if limit is None or limit <= 0 or limit >= (total_good + total_defective):
        target_good = total_good
        target_defective = total_defective
    else:
        # Balanced sampling: half normal, half defective
        target_good = min(limit // 2, total_good)
        target_defective = min(limit - target_good, total_defective)
        # If defective images were limited, try filling remainder with good images
        if target_defective + target_good < limit and total_good > target_good:
            target_good = min(limit - target_defective, total_good)

    # 4. Select normal images
    selected_good = good_images[:target_good]

    # 5. Select defective images distributed across defect types
    selected_defect: List[Tuple[str, Path]] = []
    if target_defective > 0 and defect_images_by_type:
        defect_types = list(defect_images_by_type.keys())
        # Round-robin selection across defect types for maximum diversity
        type_indices = {t: 0 for t in defect_types}
        while len(selected_defect) < target_defective:
            added_any = False
            for t in defect_types:
                if len(selected_defect) >= target_defective:
                    break
                idx = type_indices[t]
                if idx < len(defect_images_by_type[t]):
                    selected_defect.append((t, defect_images_by_type[t][idx]))
                    type_indices[t] += 1
                    added_any = True
            if not added_any:
                break

    # 6. Build unified item list
    selected_items: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}

    for p in selected_good:
        selected_items.append({
            "image_path": p,
            "defect_type": "good",
            "label": 0,
            "mask_path": None,
        })
        class_counts["good"] = class_counts.get("good", 0) + 1

    for defect_type, p in selected_defect:
        mask_path = gt_dir / defect_type / f"{p.stem}_mask.png"
        selected_items.append({
            "image_path": p,
            "defect_type": defect_type,
            "label": 1,
            "mask_path": mask_path if mask_path.exists() else None,
        })
        class_counts[defect_type] = class_counts.get(defect_type, 0) + 1

    return selected_items, class_counts


# ---------------------------------------------------------------------------
# Ground Truth Mask Loading & Alignment
# ---------------------------------------------------------------------------
def load_aligned_mask(
    item: Dict[str, Any],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Load and align ground truth mask to target anomaly map dimensions.

    Parameters
    ----------
    item : dict
        Sample item dictionary containing 'label' and 'mask_path'.
    target_shape : tuple of (int, int)
        (height, width) of the predicted anomaly map.

    Returns
    -------
    aligned_mask : np.ndarray
        Binary mask of shape target_shape with values in {0, 1}.
    """
    target_h, target_w = target_shape

    # Normal images: all-zero mask
    if item["label"] == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    mask_path = item.get("mask_path")
    if mask_path is None or not Path(mask_path).is_file():
        # Fallback if mask file is missing (warn and assume all-ones or zeros)
        print(f"[WARN] Mask file missing for defective sample: {item['image_path']}")
        return np.ones((target_h, target_w), dtype=np.uint8)

    # Load grayscale mask and resize to anomaly map dimensions using Nearest-Neighbor
    mask_pil = Image.open(mask_path).convert("L")
    mask_resized = mask_pil.resize((target_w, target_h), resample=Image.NEAREST)
    binary_mask = (np.array(mask_resized) > 127).astype(np.uint8)
    return binary_mask


# ---------------------------------------------------------------------------
# Safe AUROC Calculation
# ---------------------------------------------------------------------------
def compute_auroc_safe(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute AUROC safely, returning None if score cannot be calculated."""
    try:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception as exc:
        print(f"[WARN] Could not compute AUROC: {exc}")
        return None


# ---------------------------------------------------------------------------
# Representative Visualization Helper
# ---------------------------------------------------------------------------
def save_evaluation_visualization(
    samples_to_plot: List[Dict[str, Any]],
    output_path: Path,
    category: str,
) -> None:
    """Save multi-row visualization of representative evaluation samples.

    Shows: Original Image | Ground Truth Mask | Anomaly Heatmap | Overlay.
    """
    n_samples = len(samples_to_plot)
    if n_samples == 0:
        return

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, s in enumerate(samples_to_plot):
        orig_np = np.array(s["pil_image"])
        orig_w, orig_h = s["pil_image"].size
        gt_mask = s["gt_mask"]
        anomaly_map = s["anomaly_map"]

        # Resize heatmap and mask to original image size for display overlay
        heat_pil = Image.fromarray(anomaly_map).resize((orig_w, orig_h), resample=Image.BILINEAR)
        heat_np = np.array(heat_pil)
        norm = Normalize(vmin=float(heat_np.min()), vmax=float(heat_np.max()))
        heat_normed = norm(heat_np)

        mask_disp_pil = Image.fromarray((gt_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)
        mask_disp_np = np.array(mask_disp_pil)

        score_str = f"Score: {s['anomaly_score']:.4f}"
        type_str = f"[{'NORMAL' if s['label'] == 0 else 'DEFECT'}] {s['defect_type']}"

        # Col 0: Original
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title(f"Image: {s['filename']}\n{type_str}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 1: Ground Truth Mask
        axes[i, 1].imshow(mask_disp_np, cmap="gray")
        axes[i, 1].set_title(f"Ground Truth Mask\nLabel={s['label']}", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # Col 2: Predicted Heatmap
        im = axes[i, 2].imshow(heat_normed, cmap="jet")
        axes[i, 2].set_title(f"WinCLIP Anomaly Map\n{score_str}", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Col 3: Overlay
        axes[i, 3].imshow(orig_np)
        axes[i, 3].imshow(heat_normed, cmap="jet", alpha=0.45)
        axes[i, 3].set_title(f"Overlay\nInfer: {s['inference_time']:.2f}s", fontsize=11, fontweight="bold")
        axes[i, 3].axis("off")

    fig.suptitle(
        f"DefectZero AI - WinCLIP Zero-Shot Evaluation Samples ({category})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Representative visualization saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main Evaluation Runner
# ---------------------------------------------------------------------------
def run_evaluation(
    category: str = "bottle",
    limit: Optional[int] = 20,
    dataset_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute zero-shot WinCLIP quantitative evaluation."""
    if dataset_root is None:
        dataset_root = DEFAULT_DATASET_ROOT
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{category}_predictions.csv"
    json_path = output_dir / f"{category}_metrics.json"
    plot_path = output_dir / f"{category}_evaluation_samples.png"

    print("=" * 70)
    print(f"  DefectZero AI - Zero-Shot WinCLIP Evaluation ({category})")
    print("=" * 70)
    print(f"[+] Category     : {category}")
    print(f"[+] Limit        : {limit if limit and limit > 0 else 'All'}")
    print(f"[+] Dataset root : {dataset_root}")
    print(f"[+] Output dir   : {output_dir}")

    # Step 1: Discover and sample images
    selected_items, class_counts = discover_and_sample_images(
        dataset_root=dataset_root,
        category=category,
        limit=limit,
    )

    total_selected = len(selected_items)
    n_normal = class_counts.get("good", 0)
    n_defective = total_selected - n_normal

    print(f"\n[+] Selected {total_selected} test images ({n_normal} normal, {n_defective} defective):")
    for cls_name, count in sorted(class_counts.items()):
        print(f"    - {cls_name:15s}: {count} image(s)")

    if total_selected == 0:
        raise RuntimeError(f"No test images found for category '{category}' at {dataset_root}")

    # Step 2: Initialize WinCLIP model for zero-shot inference
    print(f"\n[+] Initializing WinCLIP model for '{category}' (zero-shot, k_shot=0)...")
    init_start = time.perf_counter()
    model = WinClipModel(
        class_name=category,
        reference_images=None,
    )
    init_time = time.perf_counter() - init_start
    print(f"[OK] WinCLIP model initialized in {init_time:.2f}s (k_shot={model.k_shot})")

    model.eval()

    # Step 3: Run inference loop
    predictions_rows: List[Dict[str, Any]] = []
    image_labels: List[int] = []
    image_scores: List[float] = []
    pixel_gt_list: List[np.ndarray] = []
    pixel_pred_list: List[np.ndarray] = []
    inference_times: List[float] = []

    samples_for_visualization: List[Dict[str, Any]] = []
    seen_viz_types = set()

    print(f"\n[+] Running inference across {total_selected} images...")
    total_eval_start = time.perf_counter()

    for idx, item in enumerate(selected_items, start=1):
        img_path = item["image_path"]
        defect_type = item["defect_type"]
        label = item["label"]

        # Load PIL RGB image
        pil_img = Image.open(img_path).convert("RGB")

        # Preprocess using model's transform pipeline (model.transform begins with ToPILImage, which accepts ndarray)
        img_array = np.array(pil_img)
        preprocessed = model.transform(img_array)
        batch = preprocessed.unsqueeze(0)

        # Run inference
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model(batch)
        t_infer = time.perf_counter() - t0
        inference_times.append(t_infer)

        # Extract output robustly
        if isinstance(output, InferenceBatch):
            pred_score_t = output.pred_score
            anomaly_map_t = output.anomaly_map
        elif isinstance(output, tuple) and len(output) >= 2:
            pred_score_t, anomaly_map_t = output[0], output[1]
        elif hasattr(output, "pred_score") and hasattr(output, "anomaly_map"):
            pred_score_t = getattr(output, "pred_score")
            anomaly_map_t = getattr(output, "anomaly_map")
        else:
            raise TypeError(f"Unexpected output type from WinClipModel: {type(output)}")

        anomaly_score = float(pred_score_t.squeeze().item())
        anomaly_map = anomaly_map_t.squeeze().cpu().numpy()  # 2D float array (H, W)

        # Load and align ground truth mask to anomaly map dimensions
        aligned_mask = load_aligned_mask(item, anomaly_map.shape)

        # Accumulate metrics
        image_labels.append(label)
        image_scores.append(anomaly_score)
        pixel_gt_list.append(aligned_mask.ravel())
        pixel_pred_list.append(anomaly_map.ravel())

        # Collect prediction row for CSV
        rel_path = img_path.relative_to(PROJECT_ROOT) if img_path.is_relative_to(PROJECT_ROOT) else img_path
        predictions_rows.append({
            "image_path": str(rel_path).replace("\\", "/"),
            "filename": img_path.name,
            "category": category,
            "defect_type": defect_type,
            "ground_truth_label": label,
            "anomaly_score": round(anomaly_score, 6),
            "inference_time_sec": round(t_infer, 4),
            "anomaly_map_min": round(float(anomaly_map.min()), 6),
            "anomaly_map_max": round(float(anomaly_map.max()), 6),
            "anomaly_map_mean": round(float(anomaly_map.mean()), 6),
            "has_ground_truth_mask": item["mask_path"] is not None and Path(item["mask_path"]).is_file(),
        })

        # Progress print
        print(
            f"  [{idx:2d}/{total_selected:2d}] {defect_type:15s} | {img_path.name:8s} | "
            f"Label: {label} | Score: {anomaly_score:.4f} | Time: {t_infer:.2f}s"
        )

        # Pick representative samples for visualization (up to 2 good, and 1 of each defect type)
        should_add_viz = False
        if defect_type == "good" and sum(1 for s in samples_for_visualization if s["defect_type"] == "good") < 2:
            should_add_viz = True
        elif defect_type != "good" and defect_type not in seen_viz_types:
            should_add_viz = True
            seen_viz_types.add(defect_type)

        if should_add_viz and len(samples_for_visualization) < 6:
            samples_for_visualization.append({
                "pil_image": pil_img,
                "filename": img_path.name,
                "defect_type": defect_type,
                "label": label,
                "anomaly_score": anomaly_score,
                "anomaly_map": anomaly_map,
                "gt_mask": aligned_mask,
                "inference_time": t_infer,
            })

    total_eval_time = time.perf_counter() - total_eval_start

    # Step 4: Compute Metrics
    y_true_img = np.array(image_labels, dtype=int)
    y_score_img = np.array(image_scores, dtype=float)
    image_auroc = compute_auroc_safe(y_true_img, y_score_img)

    all_gt_pixels = np.concatenate(pixel_gt_list)
    all_pred_pixels = np.concatenate(pixel_pred_list)
    pixel_auroc = compute_auroc_safe(all_gt_pixels, all_pred_pixels)

    mean_infer_time = float(np.mean(inference_times))
    std_infer_time = float(np.std(inference_times))

    # Step 5: Save Predictions CSV
    csv_fieldnames = [
        "image_path",
        "filename",
        "category",
        "defect_type",
        "ground_truth_label",
        "anomaly_score",
        "inference_time_sec",
        "anomaly_map_min",
        "anomaly_map_max",
        "anomaly_map_mean",
        "has_ground_truth_mask",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(predictions_rows)
    print(f"\n[OK] Predictions CSV saved to: {csv_path}")

    # Step 6: Save Metrics JSON
    metrics_summary = {
        "category": category,
        "model_name": "WinCLIP (ViT-B-16-plus-240, LAION-400M)",
        "inference_mode": "Zero-Shot (k_shot=0)",
        "total_samples_evaluated": total_selected,
        "class_distribution": class_counts,
        "normal_samples": n_normal,
        "defective_samples": n_defective,
        "image_auroc": round(image_auroc, 6) if image_auroc is not None else None,
        "image_auroc_display": f"{image_auroc:.4f}" if image_auroc is not None else "N/A",
        "pixel_auroc": round(pixel_auroc, 6) if pixel_auroc is not None else None,
        "pixel_auroc_display": f"{pixel_auroc:.4f}" if pixel_auroc is not None else "N/A",
        "mean_inference_time_sec": round(mean_infer_time, 4),
        "std_inference_time_sec": round(std_infer_time, 4),
        "min_inference_time_sec": round(float(np.min(inference_times)), 4),
        "max_inference_time_sec": round(float(np.max(inference_times)), 4),
        "total_evaluation_time_sec": round(total_eval_time, 2),
        "anomaly_map_shape": list(aligned_mask.shape),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[OK] Summary metrics JSON saved to: {json_path}")

    # Step 7: Save Representative Visualization
    save_evaluation_visualization(samples_for_visualization, plot_path, category)

    # Step 8: Print Final Report
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION REPORT")
    print("=" * 70)
    print(f"  Category               : {category}")
    print(f"  Samples Evaluated      : {total_selected} ({n_normal} normal, {n_defective} defective)")
    print(f"  Image-Level AUROC      : {metrics_summary['image_auroc_display']}")
    print(f"  Pixel-Level AUROC      : {metrics_summary['pixel_auroc_display']}")
    print(f"  Mean Inference Time    : {mean_infer_time:.4f}s (+/- {std_infer_time:.4f}s) per image")
    print(f"  Total Evaluation Time  : {total_eval_time:.2f}s")
    print("=" * 70)

    return metrics_summary


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot WinCLIP quantitative evaluation on MVTec AD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="MVTec category to evaluate (e.g., bottle, cable, capsule).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of test images to evaluate (balanced across classes). Set <=0 for all.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to root dataset directory containing category subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to directory where CSV, JSON, and PNG outputs will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        category=args.category,
        limit=args.limit,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )
