# SAM 2 Segmentation + Sam-3d-objects 3D Generation API 🔧✨

A small FastAPI service that:

- Runs Meta's Segment Anything Model 2 (SAM 2) to produce segmentation masks from point clicks
- Invokes the `sam-3d-objects` pipeline to generate a 3D Gaussian splat and export a PLY/GIF

This repo contains a single HTTP API (`api.py`) and a subprocess wrapper (`generate_3d_subprocess.py`) which runs the heavier Sam-3d-objects inference in a separate process to avoid GPU/spconv state issues.

> 🚧 **Note**
>
> This project is meant to work in conjunction with the mobile app - [Sam3D Mobile](https://github.com/andrisgauracs/sam3d-mobile)

---

## Features ✅

- POST `/segment` — single-point segmentation (returns one or multiple masks)
- POST `/segment-binary` — multi-point segmentation that returns a masked image (PNG, base64)
- POST `/generate-3d` — async 3D generation from image+mask (returns a task_id to poll)
- GET `/generate-3d-status/{task_id}` — poll for PLY/GIF results or error
- GET `/assets-list` — list saved PLY/GIF assets
- Health check: GET `/health`

---

## Requirements & Ops ⚙️

- Python 3.10+ recommended
- GPU recommended for speed (CUDA supported); MPS fallback is used on macOS where available
- Optional: `open3d` for mesh simplification (not required)

Dependencies are in `requirements.txt` and the repo includes `setup.sh` to bootstrap `sam-3d-objects` and a Conda environment.

Key packages include: `fastapi`, `uvicorn`, `torch`, `transformers`, `opencv-python`, `trimesh`, etc.

---

## Quick Setup (summary) 🛠️

1. Install the Hugging Face CLI and authenticate:

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
```

2. Run the repo setup (clones `sam-3d-objects`, creates conda env, installs deps, and downloads checkpoints):

```bash
source setup.sh
```

3. Ensure the `sam-3d-objects` repository and checkpoints are present under the repository root (the setup script places them at `./sam-3d-objects`).

> Note: The subprocess currently uses fixed paths and expects:
>
> - `./sam-3d-objects/notebook`
> - `./sam-3d-objects/checkpoints/hf/pipeline.yaml`
>   Do not rely on changing these paths via environment variables unless you update the code.

---

## Environment variables and notes ❗

Note: The subprocess expects the following fixed paths (relative to the repo root):

- `./sam-3d-objects/notebook` — the `sam-3d-objects` notebook folder required by the subprocess (fixed path).
- `./sam-3d-objects/checkpoints/hf/pipeline.yaml` — the `sam-3d-objects` pipeline config (fixed path used by the subprocess).

Important runtime environment requirements (these are already set in `api.py` and `generate_3d_subprocess.py` but are useful to know):

- Several env vars are set before importing `torch` / `spconv` to avoid tuning issues (e.g., `SPCONV_TUNE_DEVICE`, `SPCONV_ALGO_TIME_LIMIT`).
- For macOS, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set as a fallback.

> ⚠️ The 3D generation is executed in a subprocess (`generate_3d_subprocess.py`) to avoid state conflicts with spconv / Sam-3d-objects. The subprocess expects the Sam-3d-objects repo and the checkpoints to be available.

---

## Running the API (development) ▶️

You can run the app directly with Python, or use **Uvicorn** (recommended) for a cleaner server and easy configuration.

### Launch with Uvicorn (development)

Auto-reload (recommended for development):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

Simple run (no reload):

```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --log-level info
```

### Launch with Uvicorn/Gunicorn (production)

Run with multiple worker processes (recommended in production when you want process-level parallelism):

Using Gunicorn + Uvicorn worker class:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 api:app --log-level info
```

Or using Uvicorn's `--workers` flag directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

Notes & tips:

- Use `--reload` only in development (it restarts the process on file changes).
- Tune `--workers` (or Gunicorn `-w`) based on CPU and memory. If your workload is GPU-bound, avoid starting multiple processes that compete for the same GPU unless appropriately isolated.
- Ensure `CUDA_VISIBLE_DEVICES` (or equivalent GPU pinning) is set for your production service manager (systemd, container, or supervisor). Also ensure the required `sam-3d-objects` folders and checkpoint file exist at `./sam-3d-objects/notebook` and `./sam-3d-objects/checkpoints/hf/pipeline.yaml`.
- For long-running/production deployments, consider a process manager (systemd, docker-compose, k8s) and a reverse proxy (NGINX) for TLS, buffering, and routing.

Visit the interactive docs: http://localhost:8000/docs

---

## Endpoints & Examples 📡

All requests that take images or masks expect base64-encoded PNG/JPEG payloads.

### Health

- GET `/health`

Example:

```bash
curl http://localhost:8000/health
```

---

### Segment (single point)

- POST `/segment`

Body (JSON):

```json
{
  "image": "<base64 PNG/JPEG>",
  "x": 200,
  "y": 150,
  "multimask_output": true,
  "mask_threshold": 0.0
}
```

Response: JSON with `masks` (base64 PNGs), `scores`, and `image_shape`.

cURL example (using jq for compact output):

```bash
curl -s -X POST http://localhost:8000/segment \
  -H 'Content-Type: application/json' \
  -d '{"image":"<BASE64>","x":200,"y":150}' | jq .
```

---

### Segment Binary (multi-point, returns masked PNG)

- POST `/segment-binary`

Body:

```json
{
  "image": "<base64 image>",
  "points": [
    { "x": 200, "y": 150 },
    { "x": 220, "y": 170 }
  ],
  "previous_mask": "<optional base64 mask PNG>",
  "mask_threshold": 0.0
}
```

Response: JSON containing `mask` (base64 PNG) and `score`.

---

### Generate 3D (async)

- POST `/generate-3d`

Body:

```json
{
  "image": "<base64 image>",
  "mask": "<base64 binary mask PNG>",
  "seed": 42
}
```

Response: `{ "task_id": "<uuid>", "status": "queued" }` — poll `/generate-3d-status/{task_id}` for updates.

Poll example:

```bash
curl http://localhost:8000/generate-3d-status/<task_id> | jq .
```

When completed, the status contains `output_b64` (PLY or GIF), `output_type` (`"ply"`/`"gif"`), `ply_url` (public `/assets/...` path), and `mesh_url` if a mesh or GLB was generated.

### GLB export & mesh outputs

The subprocess attempts to export a textured GLB (native or via `to_glb`) as the primary mesh output when available. Notes:

- If GLB export succeeds, the `/generate-3d-status/{task_id}` response will include `mesh_url` (e.g. `/assets/mesh_<id>.glb`) and the API will also return `mesh_b64` and `mesh_size_bytes` when you poll the task status.
- The GLB/mesh is saved in the `assets/` folder and is accessible at the `mesh_url` path exposed by the API.

Example: download and save the mesh (server returns `mesh_b64`):

```bash
curl -s http://localhost:8000/generate-3d-status/<task_id> | jq -r '.mesh_b64' | base64 --decode > result.glb
```

Troubleshooting & tips for GLB/mesh export:

- The subprocess prints detailed debug lines; check the subprocess stdout logs for markers such as `MESH_URL_START` / `MESH_URL_END`, `PLY_URL_START` / `PLY_URL_END`, or warnings about `to_glb()`.
- If the pipeline returns unexpected structures (for example, `mesh` as a `list`), the subprocess will try to select a mesh-like element. If none is suitable, `to_glb()` will be skipped and a warning will be printed — the PLY or GIF output may still be available.
- If `to_glb()` raises an AttributeError (for example, because an object in the list is not a mesh with `.vertices`), the subprocess now catches the error and continues; inspect the logs and the pipeline output to find and fix the root cause.
- Native GLB export may require additional sam-3d-objects dependencies (texture baking, etc.) and can be GPU/CPU intensive.

---

---

### Assets

- GET `/assets-list` — lists files saved to the `assets/` folder with metadata.

---

## Example Python client snippet 🧪

```python
import base64, requests

# Read image and encode
with open('input.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

resp = requests.post('http://localhost:8000/segment', json={
    'image': img_b64,
    'x': 200, 'y': 150
})
print(resp.json())
```

---

## Troubleshooting & Tips 💡

- If models fail to load, ensure you authenticated with the Hugging Face CLI and downloaded checkpoints via `setup.sh`.
- The 3D generation may require a GPU and substantial memory — the subprocess prints memory and timing info to stdout for debugging.
- Install `open3d` if you want full mesh simplification (note: CPU intensive).
- If you run into `spconv` tuning/float64 issues, ensure the env vars are set before importing `torch` (the code already sets them early).

> ⚠️ Large PLY files may be written in ASCII/UTF-8 format by the post-processing step; validate that clients can handle large base64 payloads when polling for results.

---

## Development Notes & Contribution 🔭

- The heavy Sam-3d-objects logic is executed in `generate_3d_subprocess.py`; the API enqueues a background task which spawns that subprocess.
- Keep subprocess isolation when experimenting with `spconv` and GPU settings.

Contributions welcome — open issues or PRs with improvements, examples, and CI tests.

---

## License

MIT
