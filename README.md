# DefectZero AI

**Zero-shot industrial defect inspection with pretrained WinCLIP**

DefectZero AI is a university project that demonstrates image-level anomaly detection and pixel-level defect localization for the 15 categories in the MVTec Anomaly Detection (MVTec AD) dataset. An operator selects a category and submits an image to pretrained WinCLIP through a FastAPI service. The service returns an anomaly score and heatmap; completed inspections can be reviewed in history and exported as PDF reports.

WinCLIP is used in **zero-shot mode** (`k_shot=0`) with pretrained weights. This project does not train or fine-tune the model.

## Project objectives

- Evaluate pretrained WinCLIP on industrial object and texture categories without category-specific training.
- Return an image-level anomaly score and pixel-level anomaly map for an uploaded image.
- Provide a dashboard for single and sequential batch inspection.
- Keep completed inspection metadata in SQLite and image assets in local file storage.
- Present the saved evaluation results with their sample counts and limitations.

## Features

- **Single-image inspection:** choose one of the supported MVTec AD categories and submit an image.
- **Batch inspection:** add up to 30 images and process them sequentially. Each successful image is saved as a standard history record. A failed or invalid file does not stop the remaining queue.
- **Visual results:** display the input, WinCLIP heatmap, and a matching ground-truth mask when one is available. The heatmap is a visualization, not a numeric map export.
- **Inspection history:** browse saved results, filter by category or prediction, view assets and metadata, and download an individual PDF report.
- **Batch PDF summary:** download a summary for completed inspections in the current batch.
- **Model evaluation:** review image AUROC, pixel AUROC, sample counts, and saved timing metrics across the 15 evaluation categories.
- **Persistent storage:** store inspection metadata and relative asset references in SQLite; store uploaded images, heatmaps, and available masks as files.
- **Exact ground-truth lookup:** match uploaded pixel content to local MVTec AD test images. A mask is shown only when the defective image match is unique and its mask exists.

## Technology and architecture

| Layer | Implementation |
|---|---|
| Dashboard | React 19, TypeScript, Vite, custom CSS |
| API | FastAPI and Uvicorn |
| Inference | Anomalib 2.7.0 WinCLIP, PyTorch, Pillow |
| Persistence | SQLite metadata database and inspection-scoped local asset files |
| Reports | ReportLab-generated PDF files |
| Evaluation | Python evaluation script, scikit-learn AUROC, CSV/JSON outputs |
| Dataset | MVTec AD, 15 object and texture categories |

```text
Browser (React/Vite)
  ├─ GET /categories, GET /health
  ├─ POST /predict (also used sequentially by Batch Inspection)
  ├─ GET /inspections and saved asset/report routes
  └─ GET /evaluation
          │
          ▼
FastAPI service
  ├─ Lazy, one-category-at-a-time WinCLIP model cache
  ├─ SQLite: inspection metadata and safe relative asset paths
  └─ ai-service/inspection_storage/<inspection UUID>/: image, heatmap, optional mask
```

The backend caches one category model at a time and serializes inference within the process. Selecting another category can replace the cached model and require initialization again. Health checks are available before the model loads.

## Dataset and evaluation methodology

The evaluation artifacts in `experiments/evaluation_results/` report **20 test images per category**, balanced as 10 normal and 10 defective images. Defective samples are selected round-robin across defect types by `experiments/evaluate_winclip.py`; the 20-image pilot is not the full MVTec AD test split. The script uses pretrained WinCLIP with no reference images (`reference_images=None`, `k_shot=0`). It records image-level AUROC, pixel-level AUROC from flattened aligned masks and anomaly maps, per-image inference times, predictions, and representative visualizations.

The checked-in summary contains results for all 15 categories (**300 image evaluations total**). The unweighted means across categories are **0.8940 image AUROC** and **0.8628 pixel AUROC**. These are descriptive pilot summaries, not full-benchmark estimates. With only 20 samples per category, AUROC can be unstable and the sample may not represent the full range of defect types, severities, or production conditions. The results do not establish statistical significance, generalization, or production readiness.

Inference timing depends on hardware, initialization, and system load. The saved results include a transistor mean of **4030.4256 seconds per image**, an extreme recorded outlier that should be checked against its source measurement before drawing timing conclusions. Batch inference in the application is sequential; approximate batch duration therefore grows with the number of images.

The evaluation dashboard reads the existing CSV and JSON files. It does not run evaluation or recompute metrics. Review the full [evaluation report](experiments/evaluation_results/summary_report.md), [summary CSV](experiments/evaluation_results/summary_metrics.csv), and per-category files in that folder.

## Requirements

- Windows 10/11 and PowerShell
- Python environment compatible with the pinned Anomalib dependency
- Node.js and npm
- The MVTec AD dataset at `dataset/` for local ground-truth matching and evaluation. The dataset is not included in this repository.
- A CUDA-capable GPU is optional. The service selects CUDA when PyTorch reports it available; otherwise it uses CPU.

## Windows PowerShell setup

Run from the repository root. To create a new environment, activate it, and install backend dependencies:

```powershell
py -3 -m venv .venv-winclip
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv-winclip\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\ai-service\requirements.txt
```

For an existing project environment, use the same interpreter directly:

```powershell
.\.venv-winclip\Scripts\python.exe -m pip install -r .\ai-service\requirements.txt
```

Install frontend packages:

```powershell
Set-Location .\frontend
npm install
```

The dataset should be arranged in the MVTec AD layout, for example:

```text
dataset/
  bottle/
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png
```

Set `DEFECTZERO_DATASET_ROOT` if the dataset is stored elsewhere. Inference itself accepts user-uploaded images and does not require the dataset; local data is used for exact ground-truth matching and evaluation.

## Start the application

Start the FastAPI backend in one PowerShell window from the repository root and leave it running:

```powershell
.\.venv-winclip\Scripts\python.exe -m uvicorn app.main:app --app-dir .\ai-service --host 127.0.0.1 --port 8000
```

The first inference initializes pretrained WinCLIP and can take longer than subsequent requests. Check service status at <http://127.0.0.1:8000/health> and interactive API documentation at <http://127.0.0.1:8000/docs>.

In a second PowerShell window, start the React dashboard:

```powershell
Set-Location .\frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev -- --host localhost
```

Open <http://localhost:5173/>. The backend allows the default development origin `http://localhost:5173`; set `DEFECTZERO_CORS_ORIGINS` to the exact frontend origin if it differs. Do not use a wildcard origin for a deployed service.

## API endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Service and lazy model readiness status. |
| `GET /categories` | Supported MVTec AD category names. |
| `POST /predict` | Multipart fields `category` and `image`; runs zero-shot inference and saves the successful inspection. |
| `GET /dataset` | Return categories present under the configured dataset root and verifiable train/test/mask counts. |
| `GET /dataset/{category}/samples?limit_per_group=3` | Return bounded representative normal/defective sample references without local filesystem paths. |
| `GET /dataset/{category}/samples/{group}/{sample_index}/{asset_name}` | Serve an enumerated sample `image` or available `mask`; no path input is accepted. |
| `GET /evaluation` | Reads existing evaluation CSV/JSON files; does not rerun evaluation. |
| `GET /evaluation/report` | Serve the saved evaluation Markdown report. |
| `GET /evaluation/assets/{asset_name}` | Serve one allowlisted saved AUROC comparison chart. |
| `GET /inspections?limit=50&offset=0` | List saved inspection metadata, newest first. `limit` is 1–200. |
| `GET /inspections/{inspection_id}` | Read one saved inspection record. |
| `GET /inspections/{inspection_id}/assets/{asset_name}` | Serve only `image`, `heatmap`, or `ground_truth` for the selected inspection. |
| `GET /inspections/{inspection_id}/report.pdf` | Generate an individual PDF report. Add `?inline=true` to open it in a browser tab. |
| `POST /inspections/batch-report.pdf` | Generate a PDF summary from 1–30 unique, already-saved inspection IDs. Request JSON: `{"inspection_ids":["<uuid>"]}`. Does not run inference. |

The `/predict` request is `multipart/form-data`, with a category from `/categories` and a single-frame image. Uploads are limited by default to 10 MiB and 25 million pixels. Invalid categories, malformed images, unsupported images, oversized inputs, and inference failures return HTTP errors with a message.

## Persistence and generated files

FastAPI initializes the history database at startup. By default:

- SQLite database: `ai-service/inspection_storage/inspections.sqlite3`
- Asset files: `ai-service/inspection_storage/<inspection UUID>/`
- Each record stores category, original filename, prediction, anomaly score, threshold, inference duration, UTC timestamp, and safe relative asset paths.
- Asset directories contain a generated input image and heatmap, plus a ground-truth mask only when an exact match is available.

Generated inspection storage and the local `dataset/` are excluded from Git. Do not commit private or production inspection images.

## Project structure

```text
DefectZero-AI/
├── ai-service/
│   ├── app/
│   │   ├── main.py                 # FastAPI endpoints and shared predict workflow
│   │   ├── config.py               # Dataset, CORS, threshold, and upload settings
│   │   ├── winclip_runtime.py      # Lazy pretrained WinCLIP runtime/cache
│   │   ├── history_store.py        # SQLite metadata and safe asset storage
│   │   ├── ground_truth.py         # Content-based MVTec mask matching
│   │   ├── reports.py              # Individual and batch PDF generation
│   │   └── schemas.py              # Request and response schemas
│   ├── requirements.txt
│   └── README.md                   # Backend setup and endpoint details
├── frontend/
│   ├── src/App.tsx                 # Dashboard shell, single-image workflow, navigation
│   ├── src/components/
│   │   ├── BatchInspection.tsx
│   │   ├── InspectionHistory.tsx
│   │   └── ModelEvaluation.tsx
│   ├── src/api.ts
│   ├── src/styles.css
│   └── package.json
├── experiments/
│   ├── evaluate_winclip.py         # MVTec zero-shot evaluation script
│   ├── run_winclip_demo.py         # Verified single-image WinCLIP demo
│   ├── verify_dataset.py
│   ├── eda.py
│   └── evaluation_results/         # Saved per-category CSV/JSON, charts, summary
├── docs/                           # Project problem, objectives, methodology, literature
├── dataset/                        # Local MVTec AD data; not tracked in Git
└── README.md
```

## Known limitations

- The evaluation is a 20-image-per-category pilot, not a full test-set benchmark.
- The service uses one cached category model at a time and serializes inference. Changing categories may cause model reinitialization.
- CPU inference can be slow, and the first request may include model initialization. Batch inspection processes files sequentially, so total time grows with batch size.
- Ground truth is unavailable for arbitrary uploads, normal samples, ambiguous content matches, or missing local masks.
- Inspection records and uploaded assets are stored locally and are not synchronized across machines.
- The prediction threshold is a configurable service decision setting; the pilot AUROC metrics do not calibrate it.
- A timing outlier is present in the saved transistor evaluation metrics; consult the evaluation report before interpreting timing comparisons.

## References

1. Jeong, J. et al. (CVPR 2023). *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.*
2. Bergmann, P. et al. (CVPR 2019). *MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*

For the project’s broader motivation and methodology, see the files in [`docs/`](docs/).
