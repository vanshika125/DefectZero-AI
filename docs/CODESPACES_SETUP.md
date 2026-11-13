# Run the DefectZero AI backend in GitHub Codespaces

This setup runs the existing FastAPI service and its original pretrained, zero-shot Anomalib WinCLIP model on CPU. It does not change the model, inference code, threshold, frontend, or evaluation outputs. The devcontainer installs the previously validated Python 3.11 Linux x86_64 CPU dependency lock copied into `.devcontainer/requirements-linux.lock`. Keeping the lock beside the configuration makes Codespaces independent of the untracked experimental benchmark folder.

## What the configuration provides

- Python 3.11 on Linux x86_64, requesting a Codespaces machine with 2 cores, 8 GB RAM, and 32 GB storage.
- CPU-only PyTorch and torchvision from the official PyTorch CPU wheel index. The image build checks Python/platform and fails if CUDA is available.
- Exact versions from the existing complete Linux lock; `pip check` runs during image build.
- Port 8000 forwarded as **private by default**.
- The backend's existing Vercel and localhost CORS origins, the unchanged 0.5 threshold, and explicit dataset/storage locations.
- WinCLIP's model cache under the ignored workspace path `.cache/torch`. It persists with the same Codespace workspace across stop/start and container rebuild, avoids repeat downloads while that workspace is retained, and is lost if the Codespace is deleted/recreated.

The backend keeps model loading lazy. Its existing `WinClipRuntime._get_model()` creates Anomalib `WinClipModel(class_name=category, reference_images=None)` on the first request for a category; this is the project's zero-shot WinCLIP / OpenCLIP ViT-B-16-plus-240 pretrained LAION-400M path. OpenCLIP downloads missing pretrained weights on that first prediction and uses the Torch cache under `TORCH_HOME`; the container build itself does not fetch model weights. The runtime keeps one category model in memory and reuses the model for the same category until another category replaces it.

The SQLite database and generated inspection assets are stored in `ai-service/inspection_storage/`, which is already ignored by Git. The dataset is expected under the ignored `dataset/` folder. Those files remain in the Codespace workspace across stop/start, but they are not included in Git and are lost if the Codespace is deleted. Do not rely on a deleted Codespace as backup storage.

## Create the Codespace

1. Push this repository and the `.devcontainer/` files to GitHub. Keep `dataset/`, inspection storage, checkpoints, and other generated files out of the commit.
2. On GitHub, open the repository, select **Code → Codespaces → Create codespace on the branch**.
3. Select the 2-core / 8-GB machine if it is available in your included usage. The devcontainer requests that minimum and may prevent creation if no matching machine is available.
4. Let the Codespace build the devcontainer. It installs dependencies, checks `pip`, verifies Python 3.11/x86_64 and asserts CUDA is unavailable. It does not download model weights or run inference.

Do not add a payment method, increase a spending limit, or choose a paid machine. If the included allowance is exhausted, stop and wait for its reset instead of enabling paid usage.

## Check available storage before downloading data

The dataset currently present in the local project workspace was measured at **5,271,247,124 bytes (about 4.91 GiB across 6,644 files)**. This is an observed local copy, not a promise that every archive has identical bytes. The Codespaces machine configuration requests 32 GB of storage, while the GitHub Free Codespaces plan also applies a separate monthly storage allowance. Check available disk and monthly allowance before copying data or downloading checkpoints:

```bash
df -h /workspaces
du -sh dataset 2>/dev/null || true
du -sh "$TORCH_HOME" 2>/dev/null || true
```

Download MVTec AD through the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and follow its current access and license terms. The dataset is released under CC BY-NC-SA 4.0 and is not for commercial use. The dataset is not included in the repository. Upload the archive to the Codespace, inspect its top-level layout, and extract it so that these paths exist:

```text
dataset/bottle/train/good/
dataset/bottle/test/good/
dataset/bottle/test/broken_small/018.png
dataset/bottle/ground_truth/broken_small/018_mask.png
```

The archive may contain an enclosing directory. Inspect it before extraction and remove exactly one wrapper component only if needed to put the 15 category folders directly under `dataset/`. Then verify the tree:

```bash
set -e
expected='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'
for category in $expected; do
  test -d "dataset/$category/train/good" || { echo "Missing train/good: $category"; exit 1; }
  test -d "dataset/$category/test/good" || { echo "Missing test/good: $category"; exit 1; }
  test -d "dataset/$category/test" || { echo "Missing test folder: $category"; exit 1; }
  test -d "dataset/$category/ground_truth" || { echo "Missing ground_truth: $category"; exit 1; }
done
test "$(find dataset -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 15 || { echo 'Expected exactly 15 category folders'; exit 1; }
test -f dataset/bottle/test/good/000.png
test -f dataset/bottle/test/broken_small/018.png
test -f dataset/bottle/ground_truth/broken_small/018_mask.png
echo 'Dataset root and requested bottle samples are present.'
```

Expected category folders are `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, and `zipper`. If the `018.png` example is not in the downloaded version, choose an existing defective bottle sample and use its matching mask path. Do not commit the dataset archive or extracted data.

## Start and check the backend

From the repository root in the Codespace terminal, run one Uvicorn process so only one category model is resident:

```bash
python -m uvicorn app.main:app --app-dir ai-service --host 0.0.0.0 --port 8000 --workers 1
```

Port 8000 is private by default. For the local checks below, use a second Codespace terminal. Health is a liveness endpoint and reports `model_ready: false` until the first category model has loaded:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/categories
```

### One normal bottle inspection

Use the real local dataset sample; this call loads pretrained weights on first use and runs one inference:

```bash
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -F 'category=bottle' \
  -F 'image=@dataset/bottle/test/good/000.png' \
  -o /tmp/defectzero-bottle-normal.json
python - <<'PY'
import json
with open('/tmp/defectzero-bottle-normal.json', encoding='utf-8') as f:
    result = json.load(f)
for key in ('inspection_id', 'category', 'predicted_label', 'anomaly_score', 'threshold',
            'anomaly_map_width', 'anomaly_map_height', 'anomaly_map_visualization_base64',
            'inference_time_ms', 'device', 'inference_mode', 'k_shot'):
    value = result.get(key)
    if key == 'anomaly_map_visualization_base64' and value:
        value = f"present ({len(value)} base64 characters)"
    print(f'{key}: {value}')
print('ground_truth_available:', result.get('ground_truth_available'))
PY
```

### One defective bottle inspection

```bash
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -F 'category=bottle' \
  -F 'image=@dataset/bottle/test/broken_small/018.png' \
  -o /tmp/defectzero-bottle-defective.json
python - <<'PY'
import json
with open('/tmp/defectzero-bottle-defective.json', encoding='utf-8') as f:
    result = json.load(f)
for key in ('inspection_id', 'category', 'predicted_label', 'anomaly_score', 'threshold',
            'anomaly_map_width', 'anomaly_map_height', 'inference_time_ms', 'device',
            'inference_mode', 'k_shot', 'ground_truth_available', 'ground_truth_defect_type'):
    print(f'{key}: {result.get(key)}')
assert result.get('category') == 'bottle'
assert result.get('predicted_label') in ('normal', 'defective')
assert result.get('anomaly_map_visualization_base64')
assert result.get('device') == 'cpu'
PY
```

`ground_truth_available` should be true if this upload exactly matches the local dataset image and its mask exists. The application is zero-shot, so this check reports the actual model label; it does not force a label for a known defective sample.

After either prediction, check history and model readiness:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS 'http://127.0.0.1:8000/inspections?limit=5&offset=0'
```

### Record process memory and inference time

The JSON response includes `inference_time_ms`; `curl` can also measure end-to-end time. With the server running as one process:

```bash
curl -sS -o /dev/null -w 'HTTP %{http_code}; total_seconds=%{time_total}\n' \
  -X POST http://127.0.0.1:8000/predict \
  -F 'category=bottle' \
  -F 'image=@dataset/bottle/test/broken_small/018.png'
server_pid=$(pgrep -fo 'uvicorn app.main:app')
grep -E '^(VmRSS|VmHWM):' "/proc/$server_pid/status"
for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak; do
  if [ -r "$f" ]; then printf '%s=' "$f"; cat "$f"; fi
done
```

Measure immediately after the first request (which includes model loading) and again after the next request. `VmHWM` and `memory.peak` are high-water readings since process/container start. Save the output and the JSON responses as the Codespaces test record. This is a measurement instruction, not a measurement already performed in this local workspace.

## Connect the Vercel frontend

1. In Codespaces, open the **Ports** panel. Port 8000 is private by default. To allow the deployed Vercel browser app to call it, change port 8000 visibility to **Public** and copy the forwarded HTTPS URL. Making the port public allows anyone who knows the URL to call the API; CORS restricts browser origins but is not authentication. Use this only for a demo, do not upload sensitive images, and stop the Codespace when done.
2. In the Vercel project, open **Settings → Environment Variables** and set `VITE_API_BASE_URL` to the copied base URL, with no trailing slash, for the environment you use.
3. Redeploy the frontend because Vite embeds `VITE_API_BASE_URL` at build time.
4. If a new Codespace is created or the forwarded address changes, update that Vercel value and redeploy again. The React source already reads `VITE_API_BASE_URL`; no source URL is hardcoded for Codespaces.

The backend's existing CORS defaults already include `https://defect-zero-ai.vercel.app` and localhost development origins. If the Vercel deployment uses a different hostname, set `DEFECTZERO_CORS_ORIGINS` in the Codespace to a comma-separated list of exact origins, preserving the Vercel site and any required local origin. Do not use `*`.

## What remains manual and what is not promised

- The repository must be pushed to GitHub and the Codespace created from the GitHub UI or CLI.
- The user must acquire and upload/extract the MVTec AD dataset under its license.
- The first real prediction downloads the pretrained OpenCLIP checkpoint if it is not cached; later calls in the same surviving Codespace reuse it.
- The user must make port 8000 public for Vercel integration and update/redeploy the Vercel environment variable.
- Codespaces itself was not launched in the current workspace, so container build, model download/load, normal/defective predictions, memory, and latency remain **untested here**. Do not treat the sample commands as passed results.
- GitHub Free personal accounts currently include 120 Codespaces core-hours and 15 GB-month storage. A 2-core Codespace therefore uses two core-hours per active wall-clock hour (up to 60 active hours/month before included compute is used). It normally stops after 30 minutes idle and has a 12-hour maximum lifetime. It is not an always-on hosting service. Do not configure paid overage.
- The MVTec AD license is CC BY-NC-SA 4.0, including a non-commercial-use restriction. See the current [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) before downloading or sharing data.

Official references:

- [GitHub Codespaces machine sizes](https://docs.github.com/en/codespaces/about-codespaces/what-are-codespaces)
- [GitHub Free included usage](https://docs.github.com/en/billing/reference/product-usage-included)
- [Codespaces stop and maximum lifetime](https://docs.github.com/en/codespaces/about-codespaces/understanding-the-codespace-lifecycle)
- [Forwarding ports and public visibility](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace)

