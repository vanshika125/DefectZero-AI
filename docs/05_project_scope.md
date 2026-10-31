# DefectZero AI — Project Scope, Boundaries & User Personas

**Project Title:** DefectZero AI — Zero-Shot Industrial Defect Detection  
**Course Context:** Deep Learning for Computer Vision  
**Document Purpose:** Definition of Technical Scope, System Boundaries, and Target Stakeholders  

---

## 1. Project Scope & Technical Inclusions

The scope of **DefectZero AI** encompasses both rigorous academic research in zero-shot vision-language modeling and the engineering of a functional full-stack prototype for industrial visual inspection:

```
+-------------------------------------------------------------------------------------------------+
|                                     DefectZero AI Project Scope                                 |
+-------------------------------------------------------------------------------------------------+
| [Research & Deep Learning Core]                                                                 |
| * PyTorch implementation of the WinCLIP architecture (multi-scale sliding window visual tokens).|
| * Prompt engineering pipeline: Compositional nominal & anomalous prompt ensembles.             |
| * Zero-shot and few-shot (1, 2, 4-shot) anomaly scoring algorithms.                             |
| * Quantitative benchmarking on all 15 MVTec AD categories (Image AUROC, Pixel AUROC, PRO).     |
+-------------------------------------------------------------------------------------------------+
| [Serving & Application Architecture]                                                            |
| * FastAPI inference microservice exposing REST endpoints for high-resolution defect scoring.    |
| * Node.js application backend managing inspection records, image uploads, and business logic.   |
| * SQLite database storing inspection session logs, anomaly scores, and audit trails.            |
| * React / TypeScript / Tailwind CSS dashboard featuring interactive heatmap overlays & sliders. |
+-------------------------------------------------------------------------------------------------+
```

### 1.1 Deep Learning & Algorithmic Modules (In-Scope)
* **Pretrained VLM Integration:** Incorporating frozen Vision Transformer backbones (e.g., ViT-B/16+ from OpenCLIP / OpenAI) as dense visual feature extractors.
* **Window-Based Multi-Scale Aggregation:** Implementing sliding window feature pooling over spatial patch tokens to detect micro-scale and macro-scale defects.
* **Linguistic Prompt Ensembling:** Creating structured prompt banks containing:
  * Broad nominal templates (e.g., `"a flawless [c]"`, `"an undamaged [c]"`).
  * Domain-specific defect templates (e.g., `"a [c] with scratches"`, `"a [c] with cracks"`, `"a [c] with surface contamination"`).
* **Dual Inference Modes:**
  * **Strict Zero-Shot:** Evaluating novel categories without any nominal reference images.
  * **Few-Shot (WinCLIP+):** Incorporating $k \in \{1, 2, 4\}$ normal reference images to refine anomaly boundaries via associative distance matching.
* **Benchmarking Suite:** Formulating reproducible Jupyter notebooks in Google Colab executing standard evaluation protocols on the **MVTec AD** dataset.

### 1.2 Full-Stack Software Prototype (In-Scope)
* **AI Serving Layer (FastAPI / Python):** Exposing high-performance REST APIs for synchronous image inspection, returning scalar anomaly scores, classification decisions, and base64-encoded/NumPy-serialized localization heatmaps.
* **Application Backend (Node.js):** Managing file upload pipelines (Multer), orchestrating requests to the AI engine, formatting API responses, and persisting inspection metadata.
* **Persistence Layer (SQLite):** Storing inspection records, image filenames, anomaly scores, classification outcomes, timestamps, and operator verification tags.
* **User Interface (React, TypeScript, Tailwind CSS):**
  * Modern, dark-mode-first industrial QA dashboard.
  * Real-time image upload and inspection triggering.
  * Interactive heatmap visualization with threshold tuning and opacity blending.
  * Inspection history table with filtering and statistical summaries.

---

## 2. Project Boundaries & Explicit Scope Exclusions

To ensure focus and feasibility within the timeframe of a university Deep Learning project, the following areas are strictly **out of scope**:

```
+-------------------------------------------------------------------------------------------------+
|                                     Explicit Scope Exclusions                                   |
+-------------------------------------------------------------------------------------------------+
| [X] High-Speed Edge Deployment (>60 FPS on embedded ASICs, FPGAs, or TensorRT edge devices).    |
| [X] 3D Point Cloud Anomaly Detection (e.g., MVTec 3D-AD, LiDAR point clouds, depth maps).       |
| [X] Physical Hardware & Industrial Automation Integration (PLCs, Modbus, robotic actuators).    |
| [X] Full-Scale Multi-Tenant Enterprise Cloud SaaS (OAuth2 SSO, Stripe billing, Kubernetes).     |
| [X] Full Fine-Tuning of Transformer Parameters (backbones remain frozen to preserve zero-shot). |
+-------------------------------------------------------------------------------------------------+
```

* **Embedded Hardware Acceleration:** While runtime optimization will be addressed, deploying to resource-constrained industrial edge microcontrollers or specialized FPGAs is excluded.
* **Multimodal 3D / Sensor Fusion:** The project focuses strictly on 2D high-resolution RGB visual inspection. 3D point cloud surfaces, thermal imaging, and X-ray modalities are excluded.
* **Robotic Pick-and-Place Actuation:** Physical mechanical sorting mechanisms (pneumatic ejectors, robotic arms, conveyor belt motor controls) are outside the computational focus.
* **Backbone Re-training:** The visual and textual encoder weights will remain frozen. We will not perform full backpropagation into the foundation model, preserving zero-shot generalization and avoiding catastrophic forgetting.

---

## 3. Target Stakeholders & User Personas

DefectZero AI is designed to serve three distinct user archetypes:

```
+-------------------------------------------------------------------------------------------------+
|                                     Target User Personas                                        |
+-------------------------------------------------------------------------------------------------+
| Persona 1: Manufacturing QA / Quality Control Technician                                        |
| * Needs: Fast visual confirmation, clear heatmap overlays, adjustable sensitivity threshold.   |
| * Pain Point: Retraining visual models takes hours; human inspection causes fatigue.            |
+-------------------------------------------------------------------------------------------------+
| Persona 2: Industrial Vision & Deep Learning Engineer                                          |
| * Needs: Prompt-driven flexibility, zero-shot adaptation to new SKUs, API-driven integration.   |
| * Pain Point: Collecting hundreds of defective training images per SKU is impossible.           |
+-------------------------------------------------------------------------------------------------+
| Persona 3: University Academic Researchers & Evaluators                                         |
| * Needs: Rigorous mathematical formulation, reproducible Colab notebooks, standard MVTec metrics|
| * Pain Point: Black-box claims without verifiable benchmark code and clear metric breakdowns.   |
+-------------------------------------------------------------------------------------------------+
```

---

## 4. System Assumptions & Dependencies

The design and eventual implementation of DefectZero AI rely upon the following baseline assumptions:

1. **Hardware & Compute Infrastructure:** Access to an NVIDIA GPU (e.g., NVIDIA T4, V100, or A100 via Google Colab / local workstation) with at least 12–16 GB VRAM to host the Vision Transformer backbones during dense feature extraction.
2. **Dataset Availability & Standard Splits:** Access to the full, uncorrupted MVTec AD dataset, organized into nominal training sets and annotated testing sets across all 15 official categories.
3. **Software Library Stability:** Ongoing compatibility with standard open-source libraries: PyTorch 2.x, Torchvision, OpenCLIP / Hugging Face Transformers, FastAPI, Node.js LTS, and React 18+.

---

## 5. Prospective Status Declaration

> [!IMPORTANT]
> This document specifies the functional boundaries and target objectives of DefectZero AI. At this pre-implementation stage, all modules represent prospective designs. Implementation of the core algorithms, serving APIs, and frontend dashboards will follow the specifications detailed herein.
