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

- **Overview dashboard:** show supported and on-disk categories, saved pilot evaluation averages, recent inspections, and shortcuts to project workspaces. Values come from the current API; unavailable sources are identified in the UI.
- **Single-image inspection:** choose one of the supported MVTec AD categories and submit an image.
- **Batch inspection:** add up to 30 images and process them sequentially. Each successful image is saved as a standard history record. A failed or invalid file does not stop the remaining queue.
- **Visual results:** display the input, WinCLIP heatmap, and a matching ground-truth mask when one is available. The heatmap is a visualization, not a numeric map export.
- **Dataset Explorer:** browse category folders, verified training and test counts, representative test images, and matching masks when present. Dataset files are served through allowlisted API routes; filesystem paths are not exposed.
- **Inspection history:** browse saved results, filter by category or prediction, view assets and metadata, and download an individual PDF report.
- **Batch PDF summary:** download a summary for completed inspections in the current batch.
- **Model evaluation:** review image AUROC, pixel AUROC, sample counts, and saved timing metrics across the 15 evaluation categories.
- **Dashboard search:** search page names and supported categories locally, and find recent saved inspections returned by the history API. Search does not run inference.
- **Persistent storage:** store inspection metadata and relative asset references in SQLite; store uploaded images, heatmaps, and available masks as files.
- **Exact ground-truth lookup:** match uploaded pixel content to local MVTec AD test images. A mask is shown only when the defective image match is unique and its mask exists.

## Technology and architecture

| Layer | Implementation |
|---|---|
| Dashboard | React 19, TypeScript, Vite, custom CSS (no Tailwind dependency) |
| API | FastAPI and Uvicorn |
| Inference | Anomalib 2.7.0 WinCLIP, PyTorch, Pillow |
| Persistence | SQLite metadata database and inspection-scoped local asset files |
| Reports | ReportLab-generated PDF files |
| Evaluation | Python evaluation script, scikit-learn AUROC, CSV/JSON outputs |
| Dataset | MVTec AD, 15 object and texture categories |

```text
Browser (React/Vite)
  ├─ Overview: GET /health, /categories, /dataset, /evaluation, /inspections
  ├─ Inspect and Batch Inspection: POST /predict
  ├─ Dataset Explorer: /dataset and bounded sample asset routes
  ├─ History and Reports: /inspections, saved asset, and PDF routes
  ├─ Model Evaluation: saved /evaluation data and report/chart routes
  └─ Global search: dashboard page labels, categories, recent inspections
          │
          ▼
FastAPI service
  ├─ Lazy, one-category-at-a-time WinCLIP model cache
  ├─ SQLite: inspection metadata and safe relative asset paths
  └─ ai-service/inspection_storage/<inspection UUID>/: image, heatmap, optional mask
```

The backend caches one category model at a time and serializes inference within the process. Selecting another category can replace the cached model and require initialization again. Health checks are available before the model loads. Overview and search are frontend pages/features; they do not add model endpoints or change inference.

## Dataset and evaluation methodology

The evaluation artifacts in `experiments/evaluation_results/` report **20 test images per category**, balanced as 10 normal and 10 defective images. Defective samples are selected round-robin across defect types by `experiments/evaluate_winclip.py`; the 20-image pilot is not the full MVTec AD test split. The script uses pretrained WinCLIP with no reference images (`reference_images=None`, `k_shot=0`). It records image-level AUROC, pixel-level AUROC from flattened aligned masks and anomaly maps, per-image inference times, predictions, and representative visualizations.

The saved summary CSV contains valid results for all 15 categories (**300 image evaluations total**). Every category has 20 evaluated images (10 normal and 10 defective). The unweighted category means are **0.8940 image AUROC** and **0.8628 pixel AUROC**. The table below reproduces the saved CSV values; timings are mean inference seconds per image as recorded.

| Category | Image AUROC | Pixel AUROC | Images | Normal / defective | Mean inference (s/image) |
|---|---:|---:|---:|---:|---:|
| bottle | 1.0000 | 0.934743 | 20 | 10 / 10 | 10.8466 |
| cable | 0.7000 | 0.517988 | 20 | 10 / 10 | 9.9821 |
| capsule | 0.6400 | 0.932119 | 20 | 10 / 10 | 10.6402 |
| carpet | 0.9700 | 0.969399 | 20 | 10 / 10 | 10.4546 |
| grid | 1.0000 | 0.892450 | 20 | 10 / 10 | 9.5603 |
| hazelnut | 0.8800 | 0.960235 | 20 | 10 / 10 | 9.8298 |
| leather | 1.0000 | 0.973786 | 20 | 10 / 10 | 28.6000 |
| metal_nut | 0.9100 | 0.468040 | 20 | 10 / 10 | 21.5856 |
| pill | 0.8700 | 0.793884 | 20 | 10 / 10 | 11.1983 |
| screw | 0.6800 | 0.901443 | 20 | 10 / 10 | 11.1195 |
| tile | 1.0000 | 0.945541 | 20 | 10 / 10 | 18.4168 |
| toothbrush | 1.0000 | 0.894067 | 20 | 10 / 10 | 31.7377 |
| transistor | 0.7900 | 0.882760 | 20 | 10 / 10 | 4030.4256 |
| wood | 1.0000 | 0.937139 | 20 | 10 / 10 | 6.8529 |
| zipper | 0.9700 | 0.937692 | 20 | 10 / 10 | 7.1936 |

These are descriptive pilot results, not full-benchmark estimates. With only 20 samples per category, AUROC can be unstable and the sample may not represent the full range of defect types, severities, or production conditions. They do not establish statistical significance, generalization, or production readiness.

The evaluation script header records Python 3.11, PyTorch 2.15.0+cpu, and Anomalib 2.7.0 for this run. Recorded per-category means range from **6.8529 seconds/image** to the transistor value of **4030.4256 seconds/image**. Fourteen category means are between 6.8529 and 31.7377 seconds/image; the transistor mean is an extreme saved outlier (its JSON also records a 67523.012-second maximum). It is preserved as recorded and should be checked against the source measurement before interpreting timing comparisons. CPU inference is slow, model initialization can add delay, and the dashboard processes batches sequentially, so total batch time grows with the queue size.

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
py -3.11 -m venv .venv-winclip
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

Open <http://localhost:5173/>. The backend defaults to the exact origins `http://localhost:5173`, `http://127.0.0.1:5173`, and `https://defect-zero-ai.vercel.app`. Set `DEFECTZERO_CORS_ORIGINS` to a comma-separated list of exact origins in Render, for example `http://localhost:5173,http://127.0.0.1:5173,https://defect-zero-ai.vercel.app`. Do not use a wildcard origin.

## API endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Service and lazy model readiness status. |
| `GET /settings` | Public, non-sensitive inference, input-limit, category, and persistence settings. |
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

Dashboard search uses existing frontend navigation, `/categories`, and `/inspections`; there is no separate search API endpoint. Overview uses the existing health, category, dataset, evaluation, and history endpoints.

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
│   │   ├── Overview.tsx            # API-backed dashboard summary and recent inspections
│   │   ├── GlobalSearch.tsx        # Page, category, and recent-inspection search
│   │   ├── DatasetExplorer.tsx     # Verified counts, samples, and masks
│   │   ├── BatchInspection.tsx     # Sequential batch workflow
│   │   ├── InspectionHistory.tsx   # Persistent results and individual PDF reports
│   │   ├── ModelEvaluation.tsx     # Saved pilot metrics and AUROC charts
│   │   ├── Reports.tsx             # Inspection and evaluation report links
│   │   └── Settings.tsx            # Read-only service and inference configuration
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
- The pilot evaluation ran with the recorded CPU environment and is slow; the transistor timing is an extreme source-recorded outlier, not a representative estimate.
- The service uses one cached category model at a time and serializes inference. Changing categories may cause model reinitialization.
- The first request for a category may include model initialization. Batch inspection processes files sequentially, so total time grows with batch size.
- Ground truth is unavailable for arbitrary uploads, normal samples, ambiguous content matches, or missing local masks.
- Inspection records and uploaded assets are stored locally and are not synchronized across machines.
- The prediction threshold is a configurable service decision setting; the pilot AUROC metrics do not calibrate it.
- The saved transistor evaluation mean is 4030.4256 seconds per image; consult the evaluation report before interpreting timing comparisons.

## References

1. Jeong, J. et al. (CVPR 2023). *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.*
2. Bergmann, P. et al. (CVPR 2019). *MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*

For the project’s broader motivation and methodology, see the files in [`docs/`](docs/).
