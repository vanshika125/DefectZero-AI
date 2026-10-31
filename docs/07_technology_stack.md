# DefectZero AI — Technology Stack & Infrastructure Rationale

**Project Title:** DefectZero AI — Zero-Shot Industrial Defect Detection  
**Course Context:** Deep Learning for Computer Vision  
**Document Purpose:** Architectural Justification of Software & Hardware Technologies  

---

## 1. Overview of Technology Stack

The **DefectZero AI** platform is engineered using a decoupled, modern technology stack tailored for high-compute deep learning workloads combined with an intuitive industrial operator experience:

```
+--------------------------------------------------------------------------------------------------+
|                                  DefectZero AI Technology Matrix                                 |
+--------------------------------------------------------------------------------------------------+
| Layer                  | Technology                       | Primary Responsibility                |
+------------------------+----------------------------------+---------------------------------------+
| AI-Assisted Dev        | Google Antigravity               | Architectural planning, agentic code  |
|                        |                                  | structuring & documentation management|
+------------------------+----------------------------------+---------------------------------------+
| Deep Learning Core     | Python 3.10+, PyTorch 2.x,       | Tensor calculations, ViT windowing,   |
|                        | OpenCLIP / Hugging Face, WinCLIP | prompt cross-modal cosine alignment   |
+------------------------+----------------------------------+---------------------------------------+
| Experimentation & GPU  | Google Colab (T4 / A100 GPUs)    | High-VRAM benchmark execution and     |
|                        |                                  | full-suite MVTec AD evaluation        |
+------------------------+----------------------------------+---------------------------------------+
| Benchmark Dataset      | MVTec Anomaly Detection (MVTec AD)| 15 industrial categories, 5354 images, |
|                        |                                  | pixel-level ground truth masks        |
+------------------------+----------------------------------+---------------------------------------+
| AI Serving Microservice| FastAPI, Uvicorn, Pydantic,      | Asynchronous REST API, tensor parsing,|
|                        | OpenCV, Pillow, NumPy            | base64 heatmap serialization          |
+------------------------+----------------------------------+---------------------------------------+
| Application Backend    | Node.js (LTS), Express / Fastify,| Multipart upload handling, request    |
|                        | Multer                           | routing, database access orchestration|
+------------------------+----------------------------------+---------------------------------------+
| Database & Persistence | SQLite 3 (better-sqlite3 / Prisma)| Zero-configuration relational storage  |
|                        |                                  | of inspection records and audit logs  |
+------------------------+----------------------------------+---------------------------------------+
| Operator Frontend      | React 18+, TypeScript,           | Industrial inspection UI, real-time   |
|                        | Tailwind CSS, Lucide Icons       | heatmap viewer, sensitivity controls  |
+------------------------+----------------------------------+---------------------------------------+
```

---

## 2. In-Depth Component Justification & Architecture

### 2.1 AI-Assisted Development: Google Antigravity
* **Role:** Advanced agentic coding assistant supporting iterative system design, rigorous documentation authoring, unit test synthesis, and code quality maintenance.
* **Rationale:** Provides structured, context-aware assistance across both Python deep learning routines and TypeScript full-stack components while maintaining architectural consistency.

### 2.2 Deep Learning Framework: Python, PyTorch & WinCLIP
* **Python 3.10+ & PyTorch 2.x:** The standard ecosystem for modern computer vision research, offering native support for CUDA GPU acceleration, dynamic computational graphs, and automated mixed precision (`torch.cuda.amp`).
* **OpenCLIP / Hugging Face Transformers:** Provides production-grade pretrained Vision Transformer checkpoints (e.g., `ViT-B/16`, `ViT-L/14`) pretrained on LAION-400M and LAION-2B, offering superior cross-modal alignment compared to early OpenAI releases.
* **WinCLIP Core:** Formulated natively in PyTorch to enable dense feature extraction, multi-scale sliding window token pooling, and compositional prompt ensembling without modifying or fine-tuning underlying model weights.

### 2.3 Experimentation & Compute Environment: Google Colab
* **Role:** Primary experimental and benchmarking compute environment.
* **Rationale:** Evaluating dense multi-scale Vision Transformer windows across 5,354 high-resolution images in the MVTec AD benchmark is computationally demanding. Google Colab provides on-demand access to high-performance NVIDIA GPUs (T4 with 16GB VRAM, or A100 with 40GB VRAM), ensuring reproducible, accelerator-backed benchmark runs without exhausting local student hardware.

### 2.4 Benchmark Dataset: MVTec AD
* **Role:** Standard real-world dataset for industrial anomaly detection.
* **Rationale:** Contains 5,354 high-resolution industrial images across 15 categories (5 textures, 10 objects) with over 70 real-world defect topologies and pixel-precise ground truth binary masks. It enables rigorous, universally recognized benchmarking on Image AUROC, Pixel AUROC, and PRO metrics.

### 2.5 AI Serving Microservice: FastAPI & Uvicorn
* **Role:** High-performance, low-latency asynchronous microservice hosting the PyTorch inference model.
* **Rationale:**
  * **Asynchronous Concurrency:** Built on Starlette and `asyncio`, FastAPI handles concurrent requests efficiently while offloading tensor processing to worker threads or CUDA streams.
  * **Pydantic Validation:** Strictly enforces data contracts for request schemas (e.g., category selection, threshold values, prompt overrides) and response payloads.
  * **Direct NumPy/OpenCV Integration:** Seamlessly processes raw image buffers, performs Gaussian smoothing on heatmap arrays, and encodes binary masks for transmission.

### 2.6 Application Backend: Node.js & Express
* **Role:** Central application gateway orchestrating client requests, managing persistent inspection records, and handling file storage.
* **Rationale:**
  * **I/O Efficiency:** Node.js excels at non-blocking I/O operations, making it ideal for streaming high-resolution image uploads, managing disk caching, and serving client sessions.
  * **Ecosystem Maturity:** Robust libraries for multipart data processing (`multer`), schema validation (`zod`), and database interactions.

### 2.7 Database: SQLite
* **Role:** Relational persistence engine for inspection history, anomaly scores, and operator notes.
* **Rationale:**
  * **Zero Configuration & Portability:** Stored as a single self-contained disk file, eliminating the administrative overhead of hosting external database daemons (e.g., PostgreSQL or MySQL) during university project development.
  * **ACID Compliance & Performance:** Provides robust transactional safety and lightning-fast local read/write performance suitable for thousands of inspection log entries.
  * **Future-Proofing:** Can be seamlessly migrated to PostgreSQL in production via an ORM (e.g., Prisma or Drizzle) without altering application schemas.

### 2.8 Frontend Dashboard: React, TypeScript & Tailwind CSS
* **Role:** Interactive industrial QA inspection interface.
* **Rationale:**
  * **React & TypeScript:** Component-based modularity paired with strict static typing prevents runtime UI errors and ensures robust state management during image upload, processing, and visualization.
  * **Tailwind CSS:** Modern utility-first CSS framework enabling rapid construction of sleek, dark-mode-first industrial dashboards with glassmorphism accents, crisp tabular layouts, and responsive controls.
  * **Canvas & Heatmap Overlays:** Enables interactive alpha blending, threshold toggling ($\tau_{thresh}$), and real-time defect mask inspection on high-resolution parts.

---

## 3. Decoupled Architecture Rationale: Node.js vs. FastAPI

A crucial architectural decision in DefectZero AI is the complete separation between the **Node.js Application Gateway** and the **FastAPI AI Microservice**:

```
+---------------------------------------------------------------------------------------------------+
|                            Why a Decoupled Full-Stack Architecture?                               |
+---------------------------------------------------------------------------------------------------+
|  1. GPU Memory Protection & Stability                                                             |
|     Heavy PyTorch model initialization and CUDA memory spikes are isolated within the Python     |
|     runtime. A fatal CUDA out-of-memory (OOM) error will NOT crash the user session or web server.|
+---------------------------------------------------------------------------------------------------+
|  2. Separation of Concerns                                                                        |
|     * Node.js handles: HTTP request routing, user auth, file uploads, SQLite CRUD, logging.       |
|     * FastAPI handles: Heavy matrix multiplication, Vision Transformer token pooling, inference.  |
+---------------------------------------------------------------------------------------------------+
|  3. Independent Scalability & Deployment Flexibility                                              |
|     * The Node.js application gateway can run on lightweight, low-cost CPU instances.             |
|     * The FastAPI AI microservice can be independently deployed to dedicated GPU compute nodes.   |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Prospective Implementation Status

> [!IMPORTANT]
> **Implementation Note:**  
> In accordance with the initial planning phase of this university project, no software packages have been installed, and no application services have been launched. This document defines the formal technical stack and architectural design that will guide implementation in subsequent development phases.
