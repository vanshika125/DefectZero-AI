# DefectZero AI — Final Project Audit

**Audit date:** 2026-11-12
**Scope:** Read-only review of the frontend, FastAPI service, inference runtime, dataset/evaluation artifacts, persistence, reports, configuration, security, and deployment setup. No application source or ML workflow was changed. This document is the only requested deliverable created by the audit.

## Executive summary

The repository contains a substantial local university-project implementation: a React dashboard, FastAPI endpoints, lazy pretrained WinCLIP inference, local MVTec AD integration, SQLite history, file-backed visuals, and PDF report generation. Frontend typecheck/build, Python import/syntax checks, dependency consistency, evaluation artifact validation, known-mask matching, and in-memory PDF generation passed.

This is **not a fully verified end-to-end release**. There is no backend test suite in the repository and `pytest` is not installed in `.venv-winclip`. The running API could not be reached at `127.0.0.1:8000` from the audit session, so its HTTP routes and browser-to-API behavior could not be smoke-tested. No `/predict` request was sent, so live model initialization, single/batch inference, response rendering, and inference-time persistence were not exercised in this audit. Public deployment is blocked by the development-oriented setup and lack of authentication/production hosting configuration.

## Feature-wise status

| Area | Status | Evidence and limits |
|---|---|---|
| Frontend startup and app shell | **PASS** | React entry point, Vite config, app shell, component imports, and production compilation reviewed. `npm run typecheck` and `npm run build` passed. |
| Pages and navigation | **PARTIAL** | Overview, Inspect, Batch Inspection, Datasets, Inspection History, Model Evaluation, Reports, and Settings are represented in the app and navigation. Source wiring was reviewed; not every visible control was clicked through this audit. |
| Overview | **PARTIAL** | Reads health, categories, dataset, evaluation, and history through existing API routes; code reviewed, live page/API loading not tested in this audit. |
| Single-image inspection | **PARTIAL** | Multipart form request to `/predict` and result UI are wired. No request was run because that would initialize/run WinCLIP and add history data. |
| Batch inspection | **PARTIAL** | Frontend queues up to 30 images sequentially through `/predict`, gives each file an outcome, and uses saved IDs for individual and batch PDFs. No batch was run; performance and persistence are therefore not end-to-end verified. |
| FastAPI routes and request/response handling | **PARTIAL** | Route definitions, schemas, validation, and error handling were inspected; module import passed. HTTP smoke checks could not connect because the local service refused the connection. |
| WinCLIP load and inference | **NOT TESTED** | Runtime source lazily imports Anomalib `WinClipModel`, initializes with `class_name=category` and `reference_images=None`, applies the model transform, and uses `torch.inference_mode()`. A single category is cached and inference is serialized. No model was loaded or prediction run. |
| MVTec dataset integration | **PASS** | Local scan found 15 category folders, 3,629 train images, 467 normal test images, 1,258 defective test images, and 1,258 masks (5,354 images total). Dataset API uses supported categories, bounded samples, and guarded asset routes. HTTP route behavior was not tested. |
| Ground-truth lookup | **PASS** | Content-hash lookup was executed on `dataset/bottle/test/broken_small/018.png`; it uniquely resolved to `broken_small/018_mask.png`. Arbitrary/normal or ambiguous content yields no mask by implementation. |
| Image scores and heatmap visualization | **PARTIAL** | Inference response contains score, threshold, prediction, and generated PNG heatmap; UI renders these fields. Live output correctness was not tested because inference was not run. |
| SQLite history and persistence | **PARTIAL** | SQLite file and six inspection asset directories exist; store code validates UUIDs and relative asset paths and reads saved records. Restart persistence was not tested during this audit, and no new record was written. |
| Individual and batch PDF generation | **PASS** | Both PDF builders were run in memory using existing saved records/assets and returned `%PDF` output. HTTP download and browser-download routes were not reachable for smoke testing. |
| Dataset Explorer | **PARTIAL** | Counts, sample list, image/mask allowlists, category filter, and sample-type filter are present and backed by local dataset APIs. Dataset counts were independently verified; browser/API interaction was not exercised. |
| Model Evaluation dashboard | **PARTIAL** | Dashboard reads `/evaluation` and renders per-category AUROC and counts. All 15 summary rows and 15 JSON/CSV pairs were validated locally (300 prediction rows, 20 per category; 10 normal and 10 defective). API/UI refresh behavior was not live-tested. |
| Evaluation methodology and claims | **PASS** | Saved artifacts and script describe pretrained WinCLIP zero-shot (`k_shot=0`), 20 samples/category (10/10), and pilot limitations. No training claim is present in the reviewed workflow/report. |
| Reports page | **PARTIAL** | Reads inspection history and evaluation summary; provides report and evaluation artifact links. UI/API actions were code-reviewed, not network-tested. |
| Global search | **PARTIAL** | Page/category navigation and recent-history lookup exist. A retry bug is recorded below; search behavior was not exercised in the browser during this audit. |
| Settings | **PARTIAL** | Displays API health, model readiness, threshold, upload limits, categories, persistence, and capabilities. It is explicitly read-only; it does not edit runtime settings. |
| CORS and environment configuration | **PASS for local development; PARTIAL for deployment** | CORS defaults to exact origin `http://localhost:5173`, allows only GET/POST and Content-Type, and disables credentials. API URL and backend settings are environment-configurable. Production origins/URLs still need deployment configuration. |
| Upload validation and asset-path security | **PASS by code review** | Enforces category allowlist, byte/pixel limits, Pillow image decoding, single-frame images, UUID validation, allowlisted asset names, root-contained relative paths, and safe dataset sample routes. No penetration/security test suite exists. |
| Error handling | **PARTIAL** | API provides structured errors for malformed/oversize input, unsupported categories, inference, persistence, missing records/assets, and report failures. Not every failure branch was induced. |
| Dependency health | **PASS in current environment; PARTIAL for reproducible evaluation setup** | `pip check` passed; currently installed NumPy, scikit-learn, and Matplotlib imports passed. `pytest` is missing, and evaluation-only dependencies are not listed directly in the backend requirements file. |
| Production deployment readiness | **FAIL for public production deployment** | No Docker/compose or production deployment manifest, authentication, TLS/reverse-proxy configuration, production frontend API origin, or operational storage/backup plan was found. Current setup is suitable as a local development/demo architecture, not an internet-exposed service. |

## Bugs and missing implementations

### Confirmed functional issue

- **Global search recent-inspection lookup does not retry after its first failure.** In `frontend/src/components/GlobalSearch.tsx`, the lookup marks `recordsLoaded` as true in both the success and failure cases. If the API is offline or transiently fails when a user first searches, page/category matches continue to work, but recent-inspection results remain unavailable until a full page reload. Add a retry action or reset the loaded/request state after failure.

### Verification gaps and deployment risks

- **No automated backend tests are present.** The project has no discovered `test_*.py`, `*_test.py`, or `conftest.py`; the project venv also lacks `pytest`.
- **The audit session could not reach the running API.** Its local health smoke request was refused. Consequently, CORS preflight, live response schemas, restart persistence through HTTP, secure asset downloads, and PDF HTTP downloads remain unverified here.
- **Inference is slow on CPU and unmeasured in this run.** The saved evaluation files record 20 samples/category and an extreme transistor mean of 4,030.4256 seconds/image; the saved report calls out this timing as an outlier requiring source-measurement review. The service keeps only one category model loaded and serializes inference, so alternating categories can trigger model replacement and concurrent requests queue behind inference.
- **Evaluation dependencies are not directly declared for the standalone evaluation script.** The current environment has NumPy, scikit-learn, and Matplotlib, but a clean evaluation environment depends on dependency resolution through other packages unless these are declared in a dedicated evaluation requirements file.
- **Evaluation path follows the configured dataset parent.** `EVALUATION_RESULTS_ROOT` is derived from `settings.DATASET_ROOT.parent`. If `DEFECTZERO_DATASET_ROOT` points outside the repository, the API will look for `experiments/evaluation_results` beside that external dataset location. Keep the default layout or configure/separate an explicit evaluation-results root before relocating the dataset.
- **No authentication or rate limiting is implemented.** This is acceptable only for a trusted local demo. Do not expose the service publicly without an authenticated gateway and deployment security controls.
- **Settings are informational only.** Threshold, upload limits, CORS, and dataset location must be set in the backend environment and service restarted; the dashboard Settings page does not modify them.

## Tests and audit commands

Commands below are PowerShell commands. Run repository commands from the project root and frontend commands from `frontend/`.

| Command/check | Result |
|---|---|
| `npm run typecheck` (in `frontend/`) | **PASS** |
| `npm run build` (in `frontend/`) | **PASS** |
| `..\.venv-winclip\Scripts\python.exe -m pytest -q` (from `frontend/`, or `.\.venv-winclip\Scripts\python.exe -m pytest -q` from root) | **NOT RUN:** failed because `pytest` is not installed (`No module named pytest`). No backend test suite was found. |
| `$env:PYTHONPATH = '.\ai-service'; .\.venv-winclip\Scripts\python.exe -B -c "import app.main; print('BACKEND_IMPORT PASS')"` | **PASS** |
| Read-only AST parse of all Python files in `ai-service/` and `experiments/` | **PASS:** 13 files parsed. |
| `\.venv-winclip\Scripts\python.exe -m pip check` | **PASS:** no broken requirements found. |
| `\.venv-winclip\Scripts\python.exe -B -c "import numpy, sklearn, matplotlib; print(numpy.__version__, sklearn.__version__, matplotlib.__version__)"` | **PASS:** imports available (NumPy 2.4.6, scikit-learn 1.9.1, Matplotlib 3.11.2). |
| Read-only local evaluation-artifact validation (Python stdlib `csv`/`json`) | **PASS:** 15 summary rows, all 15 metrics JSON and prediction CSV files, 20 rows/category, 10 normal and 10 defective/category. |
| Read-only dataset file count | **PASS:** 15 categories; 3,629 train, 467 normal test, 1,258 defective test, 1,258 masks. |
| Ground-truth match: `bottle/test/broken_small/018.png` | **PASS:** matched the `broken_small` defect type and `018_mask.png`. |
| In-memory single and batch PDF generation from existing history records | **PASS:** both outputs had a valid `%PDF` header. |
| Read-only API smoke requests to `http://127.0.0.1:8000/health`, `/categories`, `/dataset`, `/evaluation`, `/inspections?limit=1`, and `/settings` | **BLOCKED:** connection actively refused at `127.0.0.1:8000` from this audit session. `/predict` was intentionally not called. |

The build command creates/updates generated frontend build output under `frontend/dist/`; it does not change application source. Existing user changes in the repository were preserved.

## Deployment blockers

1. The audit session could not connect to a live API on port 8000; confirm that Uvicorn is running in the intended environment before presenting a live demo.
2. The current service has no authentication or rate limit and must not be directly exposed to an untrusted network.
3. Production CORS origins, frontend `VITE_API_BASE_URL`, HTTPS termination, static frontend hosting/fallback routing, and persistent storage/backup must be configured for the actual deployment.
4. CPU inference and single-category serialized model caching limit throughput; production capacity cannot be inferred from this pilot.
5. No automated backend or browser end-to-end suite exists, so regression coverage is insufficient for a production release.

## Prioritized fixes

| Priority | Fix | Reason |
|---|---|---|
| P0 | Keep deployment local/trusted until authentication, HTTPS/reverse proxy, and production CORS are configured. | The API accepts uploads and serves persisted images/reports without user authentication. |
| P1 | Fix Global Search recent-record retry after API failure. | Current failure is sticky for the lifetime of the app. |
| P1 | Add backend tests for `/predict` validation, history persistence/path safety, ground-truth match, asset routes, and individual/batch PDFs; add a small mocked inference path. | Current automated backend test coverage is absent; these tests can verify behavior without slow model inference. |
| P1 | Add a browser/API integration smoke test using a temporary storage root and mocked inference. | The API was unreachable in this audit and page code alone does not prove CORS or route compatibility. |
| P2 | Add explicit evaluation dependencies or a separate `experiments/requirements.txt`; pin tested versions. | Makes evaluation reproducible in a clean environment rather than relying on transitive packages. |
| P2 | Decouple evaluation artifact root from the configurable dataset root. | Keeps `/evaluation` correct if the MVTec dataset is moved. |
| P2 | Recheck the saved transistor inference-time outlier against its source CSV/JSON and report any correction through the normal evaluation provenance process. | The value is extreme and currently dominates timing interpretation. Do not silently replace it. |
| P2 | Document and validate production model resource requirements and use an explicit worker/concurrency policy. | Multiple workers may each load model weights; current per-process model caching is memory-intensive. |

## Conclusion

The codebase is feature-rich and its frontend build, local data artifacts, ground-truth content matching, persistence reads, and PDF builders passed the checks described above. The core live inference workflow and HTTP/browser integration remain **not verified by this audit**, and public deployment is not ready. The appropriate project status is **working local demo with important test and deployment gaps**, not “fully complete and production-ready.”
