# DefectZero AI — Exploratory Data Analysis (EDA) Report

**Dataset:** MVTec Anomaly Detection (MVTec AD)  
**Source:** Bergmann et al., CVPR 2019  
**Analysis Scope:** Complete 15 Categories (5 Textures, 10 Objects)  
**Total Sample Count:** 5,354 images (excluding 1,258 ground truth masks)  

---

## 1. Executive Dataset Characteristics

The MVTec Anomaly Detection dataset serves as the primary benchmark for the **DefectZero AI** project. This exploratory data analysis systematically evaluates class balances, defect scale distributions, and spatial defect priors to inform model architecture selection (specifically **WinCLIP** window scaling and patch token aggregation).

| Metric | Empirical Count | Percentage / Notes |
| :--- | :---: | :--- |
| **Total Categories** | 15 | 5 Textures (33.3%), 10 Objects (66.7%) |
| **Normal Training Images** | 3,629 | 67.8% of total dataset |
| **Normal Testing Images** | 467 | 8.7% of total dataset (27.1% of test set) |
| **Defective Testing Images** | 1,258 | 23.5% of total dataset (72.9% of test set) |
| **Total Testing Images** | 1,725 | 32.2% of total dataset |
| **Total Inspection Images** | 5,354 | 100.0% |
| **Ground-Truth Binary Masks** | 1,258 | Exact 1:1 match with defective test images |
| **Unique Defect Topologies** | 73 | Average ~4.87 defect types per category |

---

## 2. Category-by-Category Quantitative Distribution

| Category | Domain | Train (Good) | Test (Good) | Test (Defective) | Total Test | Masks | Defect Types | Mean Defect Area (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bottle** | Object | 209 | 20 | 63 | 83 | 63 | 3 | 7.62% |
| **cable** | Object | 224 | 58 | 92 | 150 | 92 | 8 | 4.68% |
| **capsule** | Object | 219 | 23 | 109 | 132 | 109 | 5 | 1.11% |
| **carpet** | Texture | 280 | 28 | 89 | 117 | 89 | 5 | 2.11% |
| **grid** | Texture | 264 | 21 | 57 | 78 | 57 | 5 | 0.95% |
| **hazelnut** | Object | 391 | 40 | 70 | 110 | 70 | 4 | 3.35% |
| **leather** | Texture | 245 | 32 | 92 | 124 | 92 | 5 | 0.87% |
| **metal_nut** | Object | 220 | 22 | 93 | 115 | 93 | 4 | 14.49% |
| **pill** | Object | 267 | 26 | 141 | 167 | 141 | 7 | 3.98% |
| **screw** | Object | 320 | 41 | 119 | 160 | 119 | 5 | 0.34% |
| **tile** | Texture | 230 | 33 | 84 | 117 | 84 | 5 | 9.80% |
| **toothbrush** | Object | 60 | 12 | 30 | 42 | 30 | 1 | 2.14% |
| **transistor** | Object | 213 | 60 | 40 | 100 | 40 | 4 | 11.98% |
| **wood** | Texture | 247 | 19 | 60 | 79 | 60 | 5 | 5.09% |
| **zipper** | Object | 240 | 32 | 119 | 151 | 119 | 7 | 2.62% |

---

## 3. Defect Geometry & Spatial Area Distribution

### 3.1 Defect Scale Categorization
A vital engineering consideration for Vision Transformer (ViT) patch token pooling is the physical area occupied by defects. When defects are microscopic (<1% of image area), standard global `[CLS]` token pooling completely drowns the defect signal in background features.

* **Micro-Defects (<1.0% Area):** 499 instances (**39.7%**)
* **Small Defects (1.0%–5.0% Area):** 516 instances (**41.0%**)
* **Medium Defects (5.0%–15.0% Area):** 167 instances (**13.3%**)
* **Macro Defects (>15.0% Area):** 76 instances (**6.0%**)

**Overall Defect Statistics:**  
- Minimum Defect Area: `0.0371%`  
- Maximum Defect Area: `49.32%`  
- Mean Defect Area: `4.39%`  
- Median Defect Area: `1.54%`  
- Standard Deviation: `8.74%`  

### 3.2 Critical Architectural Implication for WinCLIP
> [!IMPORTANT]
> **Why Sliding Windows are Mandatory:**  
> Over **39.7%** of defects occupy less than 1.0% of the image, with a dataset median defect area of only **1.54%**. Standard CLIP processes $224 \times 224$ images partitioned into $16 \times 16$ patches (a $14 \times 14$ token grid). A micro-defect occupies as few as 1–2 patch tokens. Consequently, multi-scale sliding window aggregation ($2 \times 2$ and $3 \times 3$ window pooling) is mathematically essential to isolate defective tokens from overwhelming normal background tokens.

---

## 4. Spatial Defect Density & Priors

Accumulated spatial heatmaps across all 15 categories reveal two distinct spatial distributions:
1. **Rigid Object Categories (e.g., bottle, pill, capsule, screw):** Defects strongly cluster around the component center and geometric boundaries (threads, caps, printed letters). Center-crop normalization or object-centered windowing preserves defect fidelity.
2. **Texture Categories (e.g., carpet, grid, leather, tile, wood):** Defects are uniformly distributed across the entire 2D surface manifold (scratches, holes, color stains). Dense sliding-window coverage with uniform stride is mandatory.

---

## 5. Visual Artifacts Generated

The following visual charts were programmatically generated and exported to `experiments/eda_outputs/`:
1. **`01_category_distributions.png`:** Grouped bar chart comparing Train Normal, Test Normal, and Test Defect counts.
2. **`02_defect_area_distribution.png`:** Category-wise boxplot and dataset-wide histogram of relative defect area percentages.
3. **`03_defect_size_categories.png`:** Donut chart illustrating the breakdown between Micro (<1%), Small (1-5%), Medium (5-15%), and Macro (>15%) defects.
4. **`04_spatial_defect_heatmaps.png`:** 15-panel grid visualizing cumulative defect occurrence heatmaps across all categories.
5. **`eda_metrics.json`:** Structured machine-readable JSON metrics.

---
*Report generated automatically by `experiments/eda.py` for the DefectZero AI research repository.*
