# DefectZero AI FastAPI service

FastAPI serving for the project's **pretrained, zero-shot WinCLIP** inference. The API uses Anomalib 2.7.0's `WinClipModel` with `reference_images=None` (`k_shot=0`). It does not train or fine-tune weights. Inference follows the flow already used in `experiments/run_winclip_demo.py`: convert input to RGB, `torchvision.transforms.functional.to_tensor`, apply `model.transform`, run under `torch.inference_mode()`, then read the image score and pixel anomaly map from Anomalib's result.

## Install and run on Windows PowerShell

From the repository root, install the service and model requirements into the existing WinCLIP environment:

```powershell
.\.venv-winclip\Scripts\python.exe -m pip install -r .\ai-service\requirements.txt
```

Start the API from the repository root:

```powershell
.\.venv-winclip\Scripts\python.exe -m uvicorn app.main:app --app-dir .\ai-service --host 127.0.0.1 --port 8000
```

Open interactive API documentation at <http://127.0.0.1:8000/docs>. The service imports the model lazily. The first prediction initializes WinCLIP and may download pretrained backbone files if they are not already cached. Later calls for the same category reuse the loaded model. The service retains only one category model at a time to bound memory; switching categories replaces it, so switching back may incur initialization time. A lock serializes model loading and inference within this process. For multiple Uvicorn workers, each worker has its own model cache and memory use.

The default device is CUDA when available, otherwise CPU. The service readiness flag stays false until the first category model has loaded successfully; `/health` remains available before then.

## Endpoints

### `GET /health`

Returns process status and model state:

```json
{
  "status": "ok",
  "model_ready": false,
  "loaded_category": null,
  "device": null,
  "last_model_error": null
}
```

`model_ready` becomes true after the first successful model initialization. If initialization or inference fails, the latest error is included for diagnosis.

### `GET /categories`

Returns the supported MVTec AD category names: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, and `zipper`.

### `GET /evaluation`

Reads the existing `experiments/evaluation_results/summary_metrics.csv` and per-category metrics JSON files. It does not run inference or recompute metrics. The Model Evaluation page uses these saved pilot results; the endpoint returns `404` when the summary is absent and `500` if its data is malformed.

### Evaluation report files

- `GET /evaluation/report` serves the saved `summary_report.md`.
- `GET /evaluation/assets/image_auroc` and `GET /evaluation/assets/pixel_auroc` serve the corresponding saved comparison charts. These are allowlisted filenames; callers cannot request arbitrary files.

### `GET /dataset`

Returns only supported category folders present under the configured dataset root, with counts of supported image files in `train/good`, `test/good`, test defect-type folders, and `ground_truth`. Counts are computed from available files. Per-count verification flags distinguish missing folders from verified zero counts.

### Dataset sample routes

- `GET /dataset/{category}/samples?limit_per_group=3` returns up to the requested number of normal and defective test sample references, with matching ground-truth masks when available.
- `GET /dataset/{category}/samples/{group}/{sample_index}/{asset_name}` serves an enumerated `image` or `mask`. The route accepts only supported categories, the `normal`/`defective` groups, numeric sample indexes, and the two asset names. It never accepts a filesystem path, and resolved files must remain under the configured dataset root.

### `POST /predict`

Send `multipart/form-data` with:

| Field | Type | Required | Description |
|---|---|---:|---|
| `category` | text | Yes | Exact category from `/categories` |
| `image` | file | Yes | Single-frame image; decoded as RGB before the model transform |

Example from PowerShell (replace the image path with a local image):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "category=bottle" `
  -F "image=@C:\path\to\image.png"
```

Successful response (`200`) schema:

```json
{
  "inspection_id": "7f26d4a1-4372-4c99-9a22-2cc410a73b17",
  "category": "bottle",
  "anomaly_score": 0.42,
  "predicted_label": "normal",
  "threshold": 0.5,
  "anomaly_map_width": 240,
  "anomaly_map_height": 240,
  "anomaly_map_visualization_base64": "<base64-encoded PNG>",
  "anomaly_map_visualization_mime_type": "image/png",
  "ground_truth_available": true,
  "ground_truth_defect_type": "broken_small",
  "ground_truth_mask_base64": "<base64-encoded mask PNG>",
  "ground_truth_mask_mime_type": "image/png",
  "ground_truth_mask_width": 900,
  "ground_truth_mask_height": 900,
  "inference_time_ms": 1234.5,
  "device": "cpu",
  "inference_mode": "zero-shot",
  "k_shot": 0
}
```

On successful inference, the service writes the uploaded image and visualizations to the inspection storage directory, inserts metadata into SQLite, and returns the generated `inspection_id`. If persistence fails, `/predict` returns `500` rather than reporting an unsaved inspection as complete.

`anomaly_score` is the scalar image-level output from WinCLIP. `predicted_label` is `defective` when the score is greater than or equal to the configured threshold, and `normal` otherwise. The default threshold is `0.5`; this is a service decision setting, not a threshold calibrated by the pilot evaluation. The PNG is a per-image normalized visualization of the raw pixel anomaly map; it is not a numeric export and should not be used to compare absolute map values across images.

Ground truth is resolved independently of the prediction. The service hashes the uploaded image's EXIF-corrected RGB pixel content and compares it with images in `dataset/<category>/test/`. For a unique defective-image match with an existing `dataset/<category>/ground_truth/<defect_type>/<filename>_mask.png`, the response includes the exact mask bytes and dimensions. Normal images, arbitrary images, missing masks, and ambiguous duplicate-content matches return `ground_truth_available: false` and null mask fields. The category index is cached after its first lookup. Set `DEFECTZERO_DATASET_ROOT` if the dataset is stored outside the repository's `dataset/` folder.

Invalid categories return `422`, empty uploads return `400`, unsupported/corrupt images return `415`, images exceeding upload or pixel limits return `413`, and model errors return `503` with a stable error code and a short message. Detailed exception information remains in the server log.

### `GET /inspections`

Lists saved metadata newest first. Optional `limit` (1–200, default 50) and `offset` (default 0) query parameters support pagination. The response includes category, original filename, predicted label, anomaly score, threshold, inference time, UTC timestamp, inspection ID, and safe relative references to stored assets.

### `GET /inspections/{inspection_id}`

Returns one record by its canonical UUID. Malformed IDs return `422`; valid IDs with no matching record return `404`. Stored asset references are validated as relative paths contained by the inspection storage root.

### `GET /inspections/{inspection_id}/assets/{asset_name}`

Serves one persisted asset for an existing inspection. `asset_name` must be `image`, `heatmap`, or `ground_truth`; the ground-truth asset returns `404` when no mask was recorded. The service resolves the stored reference only after validating the inspection UUID and its fixed, inspection-scoped asset filename.

### `GET /inspections/{inspection_id}/report.pdf`

Generates and downloads a PDF containing the saved inspection metadata, input image, heatmap, and ground-truth mask when present. Reports are generated from the persisted files on request and are not stored as additional inspection blobs.
Add `?inline=true` to display the PDF in the browser for the Reports page's **Open** action.

### `POST /inspections/batch-report.pdf`

Builds a batch summary PDF from already saved inspection IDs. Request JSON is `{"inspection_ids": ["<canonical-uuid>", ...]}`; between 1 and 30 unique IDs are accepted. It does not trigger inference. Every ID must refer to an existing history record.

The React Batch Inspection page submits selected images sequentially through the same `POST /predict` endpoint used by single-image inspection. Each successful response is therefore saved as an ordinary inspection and gets an individual report at `/inspections/{inspection_id}/report.pdf`. Invalid files and individual request failures are marked on their own cards so the remaining queue continues.

### Persistent storage

At FastAPI startup, SQLite schema and the storage directory are initialized automatically. By default, the directory is `ai-service/inspection_storage/`, containing `inspections.sqlite3` and one generated UUID directory per inspection. Each inspection directory contains generated filenames for the uploaded image and heatmap, plus the ground-truth mask when available. SQLite stores metadata and relative paths only; image and visualization bytes remain in files. The generated storage directory is excluded from Git.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `DEFECTZERO_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated exact browser origins allowed to call the API. Set this to the actual React dev origin if it differs. |
| `DEFECTZERO_DEFECT_THRESHOLD` | `0.5` | Image-score cutoff for the predicted label. |
| `DEFECTZERO_MAX_UPLOAD_BYTES` | `10485760` | Maximum uploaded file size (10 MiB). |
| `DEFECTZERO_MAX_IMAGE_PIXELS` | `25000000` | Maximum decoded width × height. |
| `DEFECTZERO_DATASET_ROOT` | `<repository>/dataset` | Local MVTec AD root used for exact content-to-mask lookup. |
| `DEFECTZERO_INSPECTION_STORAGE` | `ai-service/inspection_storage` | Directory for the SQLite history database and inspection asset files. |

The React dashboard uses the Vite development origin by default. If it runs on another origin (for example `http://localhost:3000`), set its exact origin before launching:

```powershell
$env:DEFECTZERO_CORS_ORIGINS = "http://localhost:3000"
.\.venv-winclip\Scripts\python.exe -m uvicorn app.main:app --app-dir .\ai-service --host 127.0.0.1 --port 8000
```

No wildcard origin is enabled.
