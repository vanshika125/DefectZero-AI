# DefectZero AI — AI Serving Microservice (`ai-service`)

**Role:** High-Performance Zero-Shot Inference Engine  
**Framework:** FastAPI, Uvicorn, PyTorch, OpenCLIP  
**Status:** Architecture & Contract Specification (Pre-Implementation Phase)  

---

## 1. Overview & Architecture

The `ai-service` is an asynchronous microservice dedicated to hosting and serving the **WinCLIP** deep learning model. By isolating the PyTorch inference environment from the Node.js application gateway, we ensure:
* **GPU Memory Isolation:** Heavy CUDA allocations and VRAM state are contained within Python worker processes.
* **Non-blocking Concurrency:** High-resolution image tensor operations do not block the application's I/O event loop.
* **Independent Scalability:** The AI microservice can be containerized and scaled horizontally on GPU-enabled instances (e.g., AWS EC2 G4dn/G5, Google Cloud Vertex AI, or local workstations).

---

## 2. Directory Structure

```
ai-service/
├── README.md               # Microservice documentation & API contracts
├── requirements.txt        # Python dependency manifest
└── app/
    ├── __init__.py         # Module initializer
    ├── main.py             # FastAPI entrypoint & router registration
    ├── config.py           # Application settings & environment variables
    ├── schemas.py          # Pydantic request/response data contracts
    ├── model/              # WinCLIP architecture & visual encoder (planned)
    │   └── winclip.py
    └── utils/              # Heatmap rendering, post-processing & I/O
        └── visualizer.py
```

---

## 3. Core API Endpoints Specification

### 3.1 Health & Diagnostics
* `GET /health`: Microservice liveness and GPU readiness check.
* `GET /api/v1/status`: Returns active device (`cuda:0` / `cpu`), model checkpoint name, and supported categories.

### 3.2 Zero-Shot Anomaly Inspection
* `POST /api/v1/inspect`:
  * **Input:** Multipart form upload (`image` file, `category` string, optional `k_shot` reference images, optional `threshold` float).
  * **Processing:**
    1. Resizes and normalizes input image to CLIP preprocessing statistics.
    2. Synthesizes compositional nominal and anomalous prompt banks.
    3. Computes dense multi-scale window tokens and cosine similarity matrices.
    4. Upsamples spatial window scores and applies 2D Gaussian smoothing.
  * **Response:**
    * `category`: Target industrial category.
    * `anomaly_score`: Global scalar score $\in [0, 1]$.
    * `is_defective`: Boolean pass/fail verdict.
    * `heatmap_base64`: Base64-encoded PNG of the Jet/Inferno anomaly localization heatmap.
    * `binary_mask_base64`: Binary mask thresholded at $\tau$.
    * `latency_ms`: End-to-end inference execution time.

---

## 4. Dependencies (`requirements.txt`)

* `fastapi>=0.110.0`
* `uvicorn[standard]>=0.28.0`
* `pydantic>=2.6.0`
* `torch>=2.2.0`
* `torchvision>=0.17.0`
* `open-clip-torch>=2.24.0`
* `pillow>=10.2.0`
* `numpy>=1.26.0`
* `scipy>=1.12.0`
