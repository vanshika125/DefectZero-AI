# DefectZero AI — Comprehensive Exploratory Data Analysis (EDA) Report

**Document Code:** `docs/09_eda_report.md`  
**Project Phase:** Phase 3 — Exploratory Data Analysis & Preprocessing Pipeline  
**Dataset:** MVTec Anomaly Detection (MVTec AD)  
**Total Categories:** 15 (5 Textures, 10 Rigid Structural Objects)  
**Total Verified Images:** 5,354 (excluding 1,258 ground-truth binary masks)  

---

## 1. Executive Summary & Verification Highlights

A complete, non-destructive audit and exploratory analysis of the MVTec AD dataset was conducted at `DefectZero-AI/dataset/`. All image counts, defect classes, image resolutions, and ground-truth segmentation masks were programmatically verified against official benchmark standards.

| Metric | Calculated Count | Notes & Proportions |
| :--- | :---: | :--- |
| **Total Categories** | 15 | 5 Textures, 10 Objects |
| **Normal Training Images (`train/good`)** | 3,629 | 67.78% of overall dataset |
| **Normal Testing Images (`test/good`)** | 467 | 8.72% of dataset (27.07% of test split) |
| **Defective Testing Images (`test/*`)** | 1,258 | 23.50% of dataset (72.93% of test split) |
| **Total Testing Images** | 1,725 | 32.22% of overall dataset |
| **Total Inspection Images** | 5,354 | 100.0% (`train` + `test`) |
| **Ground-Truth Binary Masks** | 1,258 | Exact 1:1 match with defective test images |
| **Total Distinct Defect Typologies** | 73 | 73 defect classes across 15 categories |
| **Missing or Corrupted Images** | 0 | 100% readable PNG files |

---

## 2. Category-Level Statistical Breakdown

The table below details image distribution, defect classes, and native resolutions across all 15 categories:

| Category | Domain Type | Train (Good) | Test (Good) | Test (Defective) | Total Test | Masks | Defect Types | Resolution | Color Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bottle** | Object | 209 | 20 | 63 | 83 | 63 | 3 | 900x900 | RGB |
| **cable** | Object | 224 | 58 | 92 | 150 | 92 | 8 | 1024x1024 | RGB |
| **capsule** | Object | 219 | 23 | 109 | 132 | 109 | 5 | 1000x1000 | RGB |
| **carpet** | Texture | 280 | 28 | 89 | 117 | 89 | 5 | 1024x1024 | RGB |
| **grid** | Texture | 264 | 21 | 57 | 78 | 57 | 5 | 1024x1024 | L |
| **hazelnut** | Object | 391 | 40 | 70 | 110 | 70 | 4 | 1024x1024 | RGB |
| **leather** | Texture | 245 | 32 | 92 | 124 | 92 | 5 | 1024x1024 | RGB |
| **metal_nut** | Object | 220 | 22 | 93 | 115 | 93 | 4 | 700x700 | RGB |
| **pill** | Object | 267 | 26 | 141 | 167 | 141 | 7 | 800x800 | RGB |
| **screw** | Object | 320 | 41 | 119 | 160 | 119 | 5 | 1024x1024 | L |
| **tile** | Texture | 230 | 33 | 84 | 117 | 84 | 5 | 840x840 | RGB |
| **toothbrush** | Object | 60 | 12 | 30 | 42 | 30 | 1 | 1024x1024 | RGB |
| **transistor** | Object | 213 | 60 | 40 | 100 | 40 | 4 | 1024x1024 | RGB |
| **wood** | Texture | 247 | 19 | 60 | 79 | 60 | 5 | 1024x1024 | RGB |
| **zipper** | Object | 240 | 32 | 119 | 151 | 119 | 7 | 1024x1024 | L |
| **TOTAL (15)** | **ALL** | **3,629** | **467** | **1,258** | **1,725** | **1,258** | **73** | **700-1024** | **RGB / L** |

---

## 3. Defect Taxonomy & Typology Breakdown

The 73 defect types span superficial surface abrasions, mechanical structural damages, and foreign contaminants:

* **Bottle (3 types):** `broken_large` (20), `broken_small` (22), `contamination` (21)
* **Cable (8 types):** `bent_wire` (13), `cable_swap` (12), `combined` (11), `cut_inner_insulation` (14), `cut_outer_insulation` (10), `missing_cable` (12), `missing_wire` (10), `poke_insulation` (10)
* **Capsule (5 types):** `crack` (23), `faulty_imprint` (22), `poke` (21), `scratch` (23), `squeeze` (20)
* **Carpet (5 types):** `color` (19), `cut` (17), `hole` (17), `metal_contamination` (17), `thread` (19)
* **Grid (5 types):** `bent` (12), `broken` (12), `glue` (11), `metal_contamination` (11), `thread` (11)
* **Hazelnut (4 types):** `crack` (18), `cut` (17), `hole` (18), `print` (17)
* **Leather (5 types):** `color` (19), `cut` (19), `fold` (17), `glue` (19), `poke` (18)
* **Metal_nut (4 types):** `bent` (25), `color` (22), `flip` (23), `scratch` (23)
* **Pill (7 types):** `color` (25), `combined` (17), `contamination` (21), `crack` (26), `faulty_imprint` (19), `pill_type` (9), `scratch` (24)
* **Screw (5 types):** `manipulated_front` (24), `scratch_head` (24), `scratch_neck` (25), `thread_side` (23), `thread_top` (23)
* **Tile (5 types):** `crack` (17), `glue_strip` (18), `gray_stroke` (16), `oil` (18), `rough` (15)
* **Toothbrush (1 types):** `defective` (30)
* **Transistor (4 types):** `bent_lead` (10), `cut_lead` (10), `damaged_case` (10), `misplaced` (10)
* **Wood (5 types):** `color` (8), `combined` (11), `hole` (10), `liquid` (10), `scratch` (21)
* **Zipper (7 types):** `broken_teeth` (19), `combined` (16), `fabric_border` (17), `fabric_interior` (16), `rough` (17), `split_teeth` (18), `squeezed_teeth` (16)

---

## 4. Ground-Truth Segmentation Mask Alignment & Verification

1. **Exact 1-to-1 Correspondence:** Every test image `test/<defect>/<stem>.png` corresponds strictly to a binary mask `ground_truth/<defect>/<stem>_mask.png`.
2. **Dimension Integrity:** For all 1,258 pairs, image dimensions and mask dimensions match identically (e.g. $900 \times 900$ for bottle, $1024 \times 1024$ for cable).
3. **Resizing Protocol:** Any downstream resizing for Vision Transformer inputs (e.g. to $224 \times 224$ or $240 \times 240$) MUST employ **nearest-neighbor interpolation** on the mask to prevent gray interpolation artifacts and preserve strict binarization $\{0, 1\}$.

---

## 5. Architectural Implications for WinCLIP Zero-Shot Detection

The empirical findings from this EDA establish three critical architectural principles for our zero-shot model:
1. **The Micro-Defect Dilemma:** Defect areas frequently occupy $<1.5\%$ of total image pixels (e.g. fine scratches on screws and tiles). A standard global `[CLS]` token averages over the entire image and is blinded by the dominant nominal background. Multi-scale sliding window pooling ($2 \times 2, 3 \times 3$) is mathematically mandatory.
2. **Spatial Priors:** Rigid objects (e.g., bottle, pill, capsule) have defects concentrated around mechanical structures, whereas texture categories have anomalies distributed uniformly across the entire spatial surface. Uniform sliding-window coverage ensures no boundary defects are missed.
3. **Color Space Harmonization:** While 12 categories are 3-channel RGB, 3 categories (`grid`, `screw`, `zipper`) are naturally single-channel grayscale (`L`). Preprocessing must reliably map all inputs to 3-channel RGB (`convert_to_rgb()`) to ensure compatibility with pretrained CLIP visual encoders.

---

## 6. Generated EDA Artifacts Directory

The visual and statistical outputs generated by `experiments/eda.py` are saved in `experiments/eda_outputs/`:
* `01_normal_images_grid.png`: Representative normal training samples across all 15 categories.
* `02_defective_images_grid.png`: Representative defective test samples labeled by category and defect type.
* `03_ground_truth_mask_overlays.png`: Side-by-side original, binary mask, and translucent defect overlay verification.
* `04_category_image_counts.png`: Category-wise image count distribution bar chart.
* `05_normal_vs_defective_distribution.png`: Complete dataset and test-partition balance donut charts.
* `06_image_dimensions_distribution.png`: Image resolution distribution across categories.
* `07_defect_types_summary.png`: Defect typology diversity chart per category.

---
*Report generated automatically during Phase 3 EDA by `experiments/eda.py`.*
