"""Quick test for Fast-SAM3D single sample inference."""
import sys, os, json, time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '/usr/local/cuda')
os.environ['LIDRA_SKIP_INIT'] = 'true'
os.environ['SPCONV_TUNE_DEVICE'] = '0'
os.environ['SPCONV_ALGO_TIME_LIMIT'] = '100'
os.environ['WARP_QUIET'] = '1'

import torch, numpy as np
torch.set_default_dtype(torch.float32)

_BENCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FASTSAM3D_DIR = os.path.abspath(os.path.join(_BENCH_DIR, '..', '..', 'fast-sam3d'))
sys.path.insert(0, FASTSAM3D_DIR)
sys.path.insert(0, os.path.join(FASTSAM3D_DIR, 'notebook'))

from types import SimpleNamespace
from omegaconf import OmegaConf
from inference import Inference
from fft.fft2d import calculate_hfer_robust
from PIL import Image

defaults = SimpleNamespace(
    ss_cache_stride=3, ss_warmup=2, ss_order=1, ss_momentum_beta=0.5,
    slat_thresh=1.5, slat_warmup=3, slat_carving_ratio=0.1,
    mesh_spectral_threshold_low=0.5, mesh_spectral_threshold_high=0.7,
    enable_ss_cache=True, enable_slat_carving=True,
    enable_mesh_aggregation=True, enable_acceleration=True,
)

config_path = os.path.join(FASTSAM3D_DIR, 'checkpoints/hf/pipeline.yaml')
config = OmegaConf.load(config_path)
config.rendering_engine = 'pytorch3d'
config.compile_model = False
config.workspace_dir = os.path.dirname(config_path)
config['ss_generator_config_path'] = 'ss_generator_faster.yaml'
config['slat_generator_config_path'] = 'slat_generator_faster.yaml'

print('Loading model...', flush=True)
t0 = time.time()
inference = Inference(config, compile=False, args=defaults)
inference.get_params(defaults)
print(f'Model loaded in {time.time()-t0:.1f}s', flush=True)
print(f'VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f}GB', flush=True)

# Test sample
with open(os.path.join(_BENCH_DIR, 'data', 'benchmark_samples.json')) as f:
    samples = json.load(f)
s = samples[0]

PIX3D = os.path.abspath(os.path.join(_BENCH_DIR, '..', 'seed_variance', 'data', 'pix3d'))
img = np.array(Image.open(os.path.join(PIX3D, s['img_path'])).convert('RGB'))
mask_arr = np.array(Image.open(os.path.join(PIX3D, s['mask_path'])).convert('L'))
mask_bool = mask_arr > 0

hfer = calculate_hfer_robust(os.path.join(PIX3D, s['mask_path']))
inference.get_hfer(hfer)
print(f'HFER: {hfer:.4f}', flush=True)

# Synthetic pointmap
H, W = img.shape[:2]
f_val = 0.9 * max(H, W)
u, v = np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
uu, vv = np.meshgrid(u, v)
cx, cy = (W-1)*0.5, (H-1)*0.5
Z = np.full((H,W), 1.0, dtype=np.float32)
pointmap = torch.from_numpy(np.stack([(uu-cx)/f_val*Z, (vv-cy)/f_val*Z, Z], axis=-1))

torch.set_grad_enabled(False)
torch.manual_seed(42)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
print('Running inference...', flush=True)
t_start = time.perf_counter()

output = inference(img, mask_bool, seed=42, pointmap=pointmap)

torch.cuda.synchronize()
latency = time.perf_counter() - t_start
vram_peak = torch.cuda.max_memory_allocated() / (1024**2)

gs = output['gaussian'][0]
pts = gs._xyz.detach().cpu().numpy()

# OBB
centered = pts - pts.mean(axis=0)
cov = np.cov(centered.T)
_, eigvec = np.linalg.eigh(cov)
rotated = centered @ eigvec
dims = rotated.max(axis=0) - rotated.min(axis=0)
norm = sorted((dims / dims.max()).tolist())

print(f'SUCCESS!', flush=True)
print(f'  Latency: {latency:.2f}s', flush=True)
print(f'  VRAM peak: {vram_peak:.0f}MB ({vram_peak/1024:.2f}GB)', flush=True)
print(f'  Gaussian points: {pts.shape[0]}', flush=True)
print(f'  Dims (normalized sorted): {norm}', flush=True)
