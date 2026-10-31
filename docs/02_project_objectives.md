# DefectZero AI — Project Objectives, Scope & Expected Outcomes

**Project Title:** DefectZero AI — Zero-Shot Industrial Defect Detection  
**Course Context:** Deep Learning for Computer Vision  
**Status:** Prospective Research Specification (Pre-Implementation Phase)  

---

## 1. Primary Research & Engineering Objective

The central objective of **DefectZero AI** is to design, formulate, and systematically investigate an automated visual inspection framework capable of identifying and localizing previously unseen industrial defects in a **zero-shot** manner. By leveraging pretrained Vision-Language Models (VLMs)—specifically **WinCLIP** (Window-based CLIP)—the system eliminates the conventional requirement of training dedicated neural network classifiers or fitting per-category nominal feature memory banks for every distinct industrial product type.

---

## 2. Specific Secondary Objectives

To achieve this primary goal, the project is structured around five focused research and engineering milestones:

```
+-------------------------------------------------------------------------------------------------+
|                                    Core Project Objectives                                      |
+-------------------------------------------------------------------------------------------------+
|  Objective 1: Vision-Language Representation Research                                           |
|  Investigate contrastive cross-modal alignment between visual representations and text prompts  |
|  in the context of subtle, localized industrial surface irregularities.                         |
+-------------------------------------------------------------------------------------------------+
|  Objective 2: WinCLIP Pipeline Formulation                                                      |
|  Formulate and implement the Window-based CLIP (WinCLIP) architecture for dense, multi-scale    |
|  patch feature extraction and window-level aggregation for zero-shot and few-shot detection.    |
+-------------------------------------------------------------------------------------------------+
|  Objective 3: Industrial Prompt Engineering & Compositional Ensembling                          |
|  Formulate structured prompt ensembles defining nominal states ("flawless", "unblemished") and  |
|  generic/specific defect semantics ("scratches", "cracks", "contamination", "deformities").     |
+-------------------------------------------------------------------------------------------------+
|  Objective 4: Quantitative Benchmarking Protocol on MVTec AD                                    |
|  Establish a rigorous evaluation suite covering all 15 MVTec AD categories across standard      |
|  metrics: Image-Level AUROC, Pixel-Level AUROC, and Per-Region Overlap (PRO).                   |
+-------------------------------------------------------------------------------------------------+
|  Objective 5: Full-Stack Industrial Dashboard & Serving Architecture                            |
|  Design an interactive, decoupled software architecture consisting of a FastAPI inference engine,|
|  Node.js application backend, SQLite database, and a React/TypeScript/Tailwind CSS interface.   |
+-------------------------------------------------------------------------------------------------+
```

### Objective 1: Vision-Language Cross-Modal Representation Analysis
* Analyze the behavioral differences between standard web-scale image classification (where global object semantics dominate) and industrial defect detection (where microscopic local deviations define anomalies).
* Assess the cross-modal transfer capabilities of Vision Transformers (ViT) pretrained via contrastive objectives (OpenCLIP / CLIP).

### Objective 2: WinCLIP Architecture Design
* Adapt the standard CLIP image encoder into a multi-scale, window-based visual feature extractor capable of computing aligned text-image similarity scores across overlapping spatial windows.
* Incorporate both zero-shot text-driven prediction and few-shot ($k$-shot) visual reference aggregation using harmonic mean and associative distance metrics.

### Objective 3: Domain-Specific Text Prompt Engineering
* Formulate domain-specific prompt engineering strategies:
  * **Nominal Prompt Ensembles ($\mathcal{T}_{norm}$):** Expressing clean, intact, defect-free states across varied grammatical templates.
  * **Anomalous Prompt Ensembles ($\mathcal{T}_{anom}$):** Expressing both generic defect states ("damaged", "defective") and fine-grained visual defect tokens ("scratched surface", "metal burr", "discoloration").
* Study the sensitivity of zero-shot detection performance to linguistic prompt variations and vocabulary nuances.

### Objective 4: Comprehensive Benchmarking Methodology
* Establish an automated benchmarking pipeline to evaluate the model across all 15 categories (5 textures, 10 structural objects) of the standard **MVTec Anomaly Detection (MVTec AD)** benchmark.
* Measure both detection performance (Image-level AUROC) and spatial localization accuracy (Pixel-level AUROC and Per-Region Overlap [PRO]).

### Objective 5: Full-Stack Proof-of-Concept Prototype
* Formulate an end-to-end system architecture allowing industrial QA technicians to upload inspection imagery, configure inspection prompts, visualize anomaly heatmaps overlaid on high-resolution parts, and record inspection audit trails.

---

## 3. Expected Outcomes & Key Deliverables

Upon completion of the proposed engineering and experimental phases, DefectZero AI will deliver:

1. **A Modular Python/PyTorch Deep Learning Core:**
   * A clean, modular implementation of the WinCLIP inference pipeline.
   * Multi-scale window feature aggregation and spatial interpolation modules.
   * Automated prompt synthesizer generating nominal and defect prompt ensembles.

2. **Empirical Benchmarking Suite & Analysis Report:**
   * A reproducible testing harness executed in Google Colab utilizing standard MVTec AD ground truth annotations.
   * Comparative performance logs tracking Image AUROC, Pixel AUROC, and PRO across both texture categories (e.g., carpet, grid, leather, tile, wood) and rigid object categories (e.g., bottle, cable, capsule, hazelnut, metal nut, pill, screw, toothbrush, transistor, zipper).

3. **Production-Ready AI Microservice (FastAPI):**
   * High-throughput REST API serving the PyTorch inference pipeline.
   * Support for multipart image upload, dynamic prompt configuration, and binary mask generation.

4. **Industrial Visual Inspection Operator Dashboard:**
   * A modern, responsive web application built with React, TypeScript, and Tailwind CSS.
   * Interactive heatmap opacity slider, anomaly threshold tuning, and defect region bounding overlays.
   * Inspection history log and defect telemetry powered by Node.js and SQLite.

5. **Academic Project Documentation & Research Artifacts:**
   * Comprehensive codebase, experimental notebooks, design documentation, and final university dissertation report.

---

## 4. Technical Limitations & Project Constraints

In the spirit of scientific transparency, several intrinsic limitations and operational constraints of this prospective project must be explicitly acknowledged:

```
+---------------------------------------------------------------------------------------------------+
|                                 Technical Constraints & Bottlenecks                               |
+---------------------------------------------------------------------------------------------------+
|  1. High-Resolution Compute Overhead: Extracting dense multi-scale window tokens using ViT       |
|     backbones is memory- and compute-intensive, requiring GPU acceleration (e.g. Colab T4/A100).   |
|  2. Semantic Vocabulary Gap: Natural language prompts may struggle to characterize highly         |
|     abstract or microscopic structural defects that lack common linguistic terminology.            |
|  3. Inference Latency Trade-offs: Dense sliding-window inference is intrinsically slower than     |
|     lightweight single-pass CNNs, creating challenges for ultra-high-speed lines (60+ FPS).        |
|  4. Inherent Domain Shift: Standard CLIP checkpoints were pretrained on natural internet imagery,  |
|     meaning industrial surfaces (e.g., matte silicon, anisotropic metal finishes) present a domain |
|     discrepancy.                                                                                  |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Prospective Status Note
> [!IMPORTANT]
> **Declaration of Academic Rigor:**  
> This documentation represents the *pre-implementation architectural and theoretical specification* of DefectZero AI. At this initial project phase, no models have been deployed, no neural network weights have been trained or fine-tuned, and no benchmark figures have been empirically established. All quantitative evaluations described herein represent planned experimental targets to be executed during the project lifecycle.
