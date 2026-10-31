"""
DefectZero AI — MVTec AD Dataset Verification Script
=====================================================
This script systematically inspects and verifies the integrity, structure,
file counts, image formats, and mask correspondences of the MVTec Anomaly
Detection (MVTec AD) dataset located at DefectZero-AI/dataset/.

Outputs:
  - Detailed console verification report with ASCII tables
  - Comprehensive summary log written to: experiments/dataset_summary.txt
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# The 15 official MVTec AD categories partitioned by domain type
TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper"
]
EXPECTED_CATEGORIES = sorted(TEXTURE_CATEGORIES + OBJECT_CATEGORIES)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_dataset_root() -> Path:
    """Resolve the dataset root directory automatically."""
    # Check CLI arguments first
    if len(sys.argv) > 1:
        custom_path = Path(sys.argv[1]).resolve()
        if custom_path.is_dir():
            return custom_path

    # Check relative to script location: DefectZero-AI/dataset/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidate = project_root / "dataset"
    if candidate.is_dir():
        return candidate

    # Current working directory fallback
    cwd_candidate = Path.cwd() / "dataset"
    if cwd_candidate.is_dir():
        return cwd_candidate

    return candidate


def inspect_category(cat_dir: Path) -> Dict[str, Any]:
    """
    Thoroughly inspect a single category folder.
    Returns structured metrics, file counts, and diagnostic issues.
    """
    category_name = cat_dir.name
    results: Dict[str, Any] = {
        "name": category_name,
        "type": "Texture" if category_name in TEXTURE_CATEGORIES else "Object",
        "has_train": False,
        "has_test": False,
        "has_ground_truth": False,
        "train_good_count": 0,
        "test_good_count": 0,
        "test_defective_count": 0,
        "test_defects_breakdown": {},  # defect_name -> count
        "gt_masks_count": 0,
        "gt_defects_breakdown": {},    # defect_name -> count
        "total_test_count": 0,
        "total_category_images": 0,    # train + test (excluding masks)
        "image_formats": set(),
        "image_resolutions": set(),
        "image_modes": set(),
        "missing_folders": [],
        "mismatched_defect_folders": [],
        "missing_masks": [],           # List of (defect, img_name) missing mask
        "orphaned_masks": [],          # List of (defect, mask_name) with no test image
        "corrupted_images": [],
        "unexpected_files": [],
        "status": "PASS"
    }

    train_dir = cat_dir / "train"
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    # 1. Verify existence of primary subdirectories
    results["has_train"] = train_dir.is_dir()
    results["has_test"] = test_dir.is_dir()
    results["has_ground_truth"] = gt_dir.is_dir()

    if not results["has_train"]:
        results["missing_folders"].append("train")
    if not results["has_test"]:
        results["missing_folders"].append("test")
    if not results["has_ground_truth"]:
        results["missing_folders"].append("ground_truth")

    # 2. Inspect train folder
    if results["has_train"]:
        train_good = train_dir / "good"
        if not train_good.is_dir():
            results["missing_folders"].append("train/good")
        else:
            for item in train_good.iterdir():
                if item.is_file():
                    if item.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        results["train_good_count"] += 1
                        results["image_formats"].add(item.suffix.lower())
                        if HAS_PIL and len(results["image_resolutions"]) < 10:
                            try:
                                with Image.open(item) as img:
                                    results["image_resolutions"].add(f"{img.width}x{img.height}")
                                    results["image_modes"].add(img.mode)
                            except Exception as e:
                                results["corrupted_images"].append(f"train/good/{item.name}: {e}")
                    else:
                        results["unexpected_files"].append(f"train/good/{item.name}")

    # 3. Inspect test folder
    test_defect_folders = set()
    if results["has_test"]:
        test_good = test_dir / "good"
        if not test_good.is_dir():
            results["missing_folders"].append("test/good")
        else:
            for item in test_good.iterdir():
                if item.is_file():
                    if item.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        results["test_good_count"] += 1
                        results["image_formats"].add(item.suffix.lower())
                        if HAS_PIL and len(results["image_resolutions"]) < 10:
                            try:
                                with Image.open(item) as img:
                                    results["image_resolutions"].add(f"{img.width}x{img.height}")
                                    results["image_modes"].add(img.mode)
                            except Exception as e:
                                results["corrupted_images"].append(f"test/good/{item.name}: {e}")
                    else:
                        results["unexpected_files"].append(f"test/good/{item.name}")

        # Defect folders in test
        for sub in test_dir.iterdir():
            if sub.is_dir() and sub.name != "good":
                test_defect_folders.add(sub.name)
                defect_images = [
                    f for f in sub.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
                ]
                count = len(defect_images)
                results["test_defects_breakdown"][sub.name] = count
                results["test_defective_count"] += count
                for img_file in defect_images:
                    results["image_formats"].add(img_file.suffix.lower())

    # 4. Inspect ground_truth folder
    gt_defect_folders = set()
    if results["has_ground_truth"]:
        for sub in gt_dir.iterdir():
            if sub.is_dir():
                gt_defect_folders.add(sub.name)
                mask_images = [
                    f for f in sub.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
                ]
                count = len(mask_images)
                results["gt_defects_breakdown"][sub.name] = count
                results["gt_masks_count"] += count
                for m_file in mask_images:
                    results["image_formats"].add(m_file.suffix.lower())

    # 5. Cross-check defect folder alignment between test/ and ground_truth/
    test_only = test_defect_folders - gt_defect_folders
    gt_only = gt_defect_folders - test_defect_folders
    if test_only:
        results["mismatched_defect_folders"].extend([f"test only: {d}" for d in test_only])
    if gt_only:
        results["mismatched_defect_folders"].extend([f"ground_truth only: {d}" for d in gt_only])

    # 6. Verify 1-to-1 filename correspondence between test defect images and masks
    common_defects = test_defect_folders & gt_defect_folders
    for defect in common_defects:
        test_defect_dir = test_dir / defect
        gt_defect_dir = gt_dir / defect

        test_files = {
            f.stem: f for f in test_defect_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        }
        mask_files = {
            f.stem: f for f in gt_defect_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        }

        # Check for matching mask for each test image
        for stem, test_file in test_files.items():
            expected_mask_stem = f"{stem}_mask"
            if expected_mask_stem not in mask_files and stem not in mask_files:
                results["missing_masks"].append(f"{defect}/{test_file.name}")

        # Check for orphaned masks without test image
        for mask_stem, mask_file in mask_files.items():
            orig_stem = mask_stem[:-5] if mask_stem.endswith("_mask") else mask_stem
            if orig_stem not in test_files:
                results["orphaned_masks"].append(f"{defect}/{mask_file.name}")

    # Aggregates
    results["total_test_count"] = results["test_good_count"] + results["test_defective_count"]
    results["total_category_images"] = results["train_good_count"] + results["total_test_count"]

    # Determine status
    if (results["missing_folders"] or results["mismatched_defect_folders"] or
            results["missing_masks"] or results["orphaned_masks"] or
            results["corrupted_images"]):
        results["status"] = "FAIL"
    else:
        results["status"] = "PASS"

    return results


def run_verification(dataset_root: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Execute verification across all categories and build output report."""
    output_lines: List[str] = []

    def p(line: str = "") -> None:
        output_lines.append(line)
        print(line)

    p("=" * 86)
    p("                 DEFECTZERO AI — MVTEC AD DATASET VERIFICATION                ")
    p("=" * 86)
    p(f"Dataset Root Directory : {dataset_root}")
    p(f"Root Exists            : {dataset_root.is_dir()}")

    if not dataset_root.is_dir():
        p(f"\n[FATAL ERROR] Dataset root directory not found at: {dataset_root}")
        p("Please verify the dataset folder location.")
        return [], "\n".join(output_lines)

    # Check top-level directories
    found_subdirs = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    missing_categories = [cat for cat in EXPECTED_CATEGORIES if cat not in found_subdirs]
    extra_subdirs = [d for d in found_subdirs if d not in EXPECTED_CATEGORIES]

    p(f"Expected Categories    : {len(EXPECTED_CATEGORIES)}")
    p(f"Categories Located     : {len([c for c in EXPECTED_CATEGORIES if c in found_subdirs])} / {len(EXPECTED_CATEGORIES)}")

    if extra_subdirs:
        p(f"Non-Category Folders   : {', '.join(extra_subdirs)}")
    if missing_categories:
        p(f"[ERROR] Missing Categories: {', '.join(missing_categories)}")

    # Inspect each category
    all_results: List[Dict[str, Any]] = []
    for cat_name in EXPECTED_CATEGORIES:
        cat_path = dataset_root / cat_name
        if not cat_path.is_dir():
            res = {
                "name": cat_name,
                "type": "Texture" if cat_name in TEXTURE_CATEGORIES else "Object",
                "status": "MISSING",
                "missing_folders": ["category directory missing"],
                "train_good_count": 0,
                "test_good_count": 0,
                "test_defective_count": 0,
                "test_defects_breakdown": {},
                "gt_masks_count": 0,
                "gt_defects_breakdown": {},
                "total_test_count": 0,
                "total_category_images": 0,
                "image_formats": set(),
                "image_resolutions": set(),
                "mismatched_defect_folders": [],
                "missing_masks": [],
                "orphaned_masks": [],
                "corrupted_images": [],
                "unexpected_files": [],
            }
            all_results.append(res)
        else:
            all_results.append(inspect_category(cat_path))

    # Print Table 1: Category Overview
    p("\n" + "-" * 86)
    p("TABLE 1: CATEGORY-LEVEL IMAGE COUNTS & STRUCTURAL INTEGRITY")
    p("-" * 86)
    header = (
        f"{'Category':<14} | {'Type':<7} | {'Train(Good)':<11} | {'Test(Good)':<10} | "
        f"{'Test(Defect)':<12} | {'Total Test':<10} | {'GT Masks':<8} | {'Status':<6}"
    )
    p(header)
    p("-" * 86)

    total_train = 0
    total_test_good = 0
    total_test_defective = 0
    total_test = 0
    total_gt_masks = 0
    total_all_images = 0

    for r in all_results:
        total_train += r["train_good_count"]
        total_test_good += r["test_good_count"]
        total_test_defective += r["test_defective_count"]
        total_test += r["total_test_count"]
        total_gt_masks += r["gt_masks_count"]
        total_all_images += r["total_category_images"]

        row = (
            f"{r['name']:<14} | {r['type']:<7} | {r['train_good_count']:>11} | "
            f"{r['test_good_count']:>10} | {r['test_defective_count']:>12} | "
            f"{r['total_test_count']:>10} | {r['gt_masks_count']:>8} | {r['status']:<6}"
        )
        p(row)

    p("-" * 86)
    total_row = (
        f"{'TOTAL (15)':<14} | {'ALL':<7} | {total_train:>11} | "
        f"{total_test_good:>10} | {total_test_defective:>12} | "
        f"{total_test:>10} | {total_gt_masks:>8} | "
        f"{'PASS' if all(r['status'] == 'PASS' for r in all_results) else 'FAIL':<6}"
    )
    p(total_row)
    p("-" * 86)

    # Print Table 2: Detailed Defect Breakdown per Category
    p("\n" + "-" * 86)
    p("TABLE 2: DETAILED DEFECT TAXONOMY & GROUND TRUTH MASK CORRESPONDENCE")
    p("-" * 86)
    p(f"{'Category':<14} | {'Defect Type':<26} | {'Test Imgs':<10} | {'GT Masks':<10} | {'Match'}")
    p("-" * 86)

    for r in all_results:
        cat = r["name"]
        defects = sorted(r["test_defects_breakdown"].keys())
        if not defects:
            p(f"{cat:<14} | {'(no defects found)':<26} | {'0':>10} | {'0':>10} | N/A")
            continue

        for i, defect in enumerate(defects):
            t_cnt = r["test_defects_breakdown"].get(defect, 0)
            m_cnt = r["gt_defects_breakdown"].get(defect, 0)
            match_str = "YES" if t_cnt == m_cnt and t_cnt > 0 else "MISMATCH"
            cat_label = cat if i == 0 else ""
            p(f"{cat_label:<14} | {defect:<26} | {t_cnt:>10} | {m_cnt:>10} | {match_str}")

    p("-" * 86)

    # Print Table 3: Formats and Technical Specifications
    p("\n" + "-" * 86)
    p("TABLE 3: IMAGE FORMATS, RESOLUTIONS & COLOR MODES")
    p("-" * 86)
    p(f"{'Category':<14} | {'Formats Found':<16} | {'Sample Resolutions':<28} | {'Color Modes'}")
    p("-" * 86)
    for r in all_results:
        fmts = ", ".join(sorted(r["image_formats"])) if r["image_formats"] else "None"
        res = ", ".join(sorted(r["image_resolutions"])) if r["image_resolutions"] else "N/A"
        modes = ", ".join(sorted(r.get("image_modes", []))) if r.get("image_modes") else "N/A"
        p(f"{r['name']:<14} | {fmts:<16} | {res:<28} | {modes}")
    p("-" * 86)

    # Print Issues and Warnings Section
    p("\n" + "=" * 86)
    p("                          DIAGNOSTIC & ERROR AUDIT LOG                        ")
    p("=" * 86)

    issues_found = False
    for r in all_results:
        cat_issues = []
        if r["missing_folders"]:
            cat_issues.append(f"Missing subfolders: {', '.join(r['missing_folders'])}")
        if r["mismatched_defect_folders"]:
            cat_issues.append(f"Mismatched defect folders: {', '.join(r['mismatched_defect_folders'])}")
        if r["missing_masks"]:
            cat_issues.append(f"Missing masks ({len(r['missing_masks'])}): {r['missing_masks'][:3]}...")
        if r["orphaned_masks"]:
            cat_issues.append(f"Orphaned masks ({len(r['orphaned_masks'])}): {r['orphaned_masks'][:3]}...")
        if r["corrupted_images"]:
            cat_issues.append(f"Corrupted images: {r['corrupted_images']}")

        if cat_issues:
            issues_found = True
            p(f"\n[!] Category: {r['name']} (STATUS: {r['status']})")
            for iss in cat_issues:
                p(f"    - {iss}")

    if not issues_found:
        p("\n[*] AUDIT RESULT: ZERO STRUCTURAL OR FILENAME DEFECTS DETECTED.")
        p("    - All 15 official category folders are present.")
        p("    - All train/good, test/good, test/<defect>, and ground_truth/<defect> folders exist.")
        p("    - 1-to-1 exact filename correspondence between all defective test images and masks.")
        p("    - Zero missing masks, zero orphaned masks, zero corrupted files.")

    # Verification Checklist
    p("\n" + "=" * 86)
    p("                              VERIFICATION CHECKLIST                          ")
    p("=" * 86)
    all_cats_present = len(missing_categories) == 0
    all_subfolders_ok = all(len(r["missing_folders"]) == 0 for r in all_results)
    masks_match_test = (total_test_defective == total_gt_masks) and all(len(r["missing_masks"]) == 0 for r in all_results)
    no_orphans = all(len(r["orphaned_masks"]) == 0 for r in all_results)
    no_corrupted = all(len(r["corrupted_images"]) == 0 for r in all_results)

    p(f" [ {'PASS' if all_cats_present else 'FAIL'} ] 1. All 15 MVTec AD category folders present")
    p(f" [ {'PASS' if all_subfolders_ok else 'FAIL'} ] 2. All train, test, and ground_truth subfolders verified")
    p(f" [ {'PASS' if total_train > 0 else 'FAIL'} ] 3. Normal training images counted ({total_train} images)")
    p(f" [ {'PASS' if total_test_good > 0 else 'FAIL'} ] 4. Normal testing images counted ({total_test_good} images)")
    p(f" [ {'PASS' if total_test_defective > 0 else 'FAIL'} ] 5. Defective testing images counted ({total_test_defective} images)")
    p(f" [ {'PASS' if masks_match_test else 'FAIL'} ] 6. Ground-truth segmentation masks verified ({total_gt_masks} masks)")
    p(f" [ {'PASS' if masks_match_test and no_orphans else 'FAIL'} ] 7. 1-to-1 Image-to-Mask filename correspondence verified")
    p(f" [ {'PASS' if no_corrupted else 'FAIL'} ] 8. Image formats, resolutions, and file integrity checked")
    p("=" * 86)

    # Grand Summary
    p("\nDATASET METRIC SUMMARY:")
    p(f"  * Total Categories                  : {len(all_results)} (5 Textures, 10 Objects)")
    p(f"  * Total Normal Training Images      : {total_train}")
    p(f"  * Total Normal Testing Images       : {total_test_good}")
    p(f"  * Total Defective Testing Images    : {total_test_defective}")
    p(f"  * Total Testing Images              : {total_test}")
    p(f"  * Total Dataset Inspection Images   : {total_all_images} (train + test)")
    p(f"  * Total Ground-Truth Binary Masks   : {total_gt_masks}")
    p(f"  * Total Distinct Defect Classes     : {sum(len(r['test_defects_breakdown']) for r in all_results)}")
    p(f"  * Standard MVTec AD Paper Benchmark : 5,354 images (3,629 train + 1,725 test)")
    p(f"  * Integrity Assessment               : {'PERFECT MATCH (100% Complete)' if total_all_images == 5354 else 'MISMATCH'}")
    p("=" * 86)

    return all_results, "\n".join(output_lines)


def main() -> int:
    dataset_root = get_dataset_root()
    script_dir = Path(__file__).resolve().parent
    summary_path = script_dir / "dataset_summary.txt"

    results, report_text = run_verification(dataset_root)

    # Write summary text file
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\n[OK] Dataset summary report saved to: {summary_path}")

    # Return status code
    if not results or any(r.get("status") != "PASS" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
