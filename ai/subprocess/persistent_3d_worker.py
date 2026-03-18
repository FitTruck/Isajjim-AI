"""
Persistent 3D Worker for SAM-3D Generation

모델을 한 번만 로드하고 여러 작업을 순차적으로 처리하는 워커 프로세스.
stdin으로 JSON 작업 요청을 받고, stdout으로 JSON 결과를 반환.

Usage:
    python persistent_3d_worker.py <worker_id> <gpu_id>
"""

import sys
import os
import base64
import tempfile
import time

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# ============================================================================
# GPU ID는 커맨드라인에서 받음
if len(sys.argv) >= 3:
    gpu_id = int(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX") or "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPCONV_TUNE_DEVICE"] = "0"  # Always 0 due to CUDA_VISIBLE_DEVICES remap
os.environ["SPCONV_ALGO_TIME_LIMIT"] = os.environ.get("SPCONV_ALGO_TIME_LIMIT", "100")
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"
os.environ["WARP_QUIET"] = "1"  # Suppress Warp stdout output (interferes with JSON protocol)

# Phase C: Persistent AUTOTUNE cache (survives worker restarts)
# torch.compile(mode="max-autotune") benchmarks CUDA kernels on first run.
# Without persistent cache, every worker restart re-benchmarks (~5 min).
_autotune_cache_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "torch_compile"
)
os.makedirs(_autotune_cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _autotune_cache_dir
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

# Prevent thread explosion
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import torch

# Set PyTorch threading limits
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from PIL import Image

# ============================================================================
# Performance Optimization Configuration
# ============================================================================
# Phase 1: Image downsampling - DISABLED for volume accuracy
# 테스트 결과: 다운샘플링이 부피 정확도에 91.7% 영향 (vs Steps 3.8%)
# 특히 작은 객체(Pillow, Lamp)에서 최대 576% 부피 차이 발생
MAX_IMAGE_SIZE = None  # None = 다운샘플링 비활성화 (부피 정확도 유지)

# Phase 2: Inference steps optimization
# Stage1 (Sparse Structure): 3D 구조 생성 품질
# Stage2 (SLAT): 세부 디테일 품질
# 상대 길이 정확도 향상을 위해 기본값 이상으로 설정
STAGE1_INFERENCE_STEPS = 14  # 속도/정확도 균형 (12~16 사이 최적값)
STAGE2_INFERENCE_STEPS = 4   # 테스트 결과: 4로 줄여도 치수 오차 0.5% 이내, 30% 속도 향상

# Phase 3: PLY output format
# Binary is ~70% smaller and ~50% faster to write
USE_BINARY_PLY = True

# Phase 5: Gaussian-only decode (GLB/Mesh 생성 스킵)
# 테스트 결과: 37.4% 속도 향상, 부피 오차 0.005% (무시 가능)
# GLB/Mesh가 필요 없고 부피 계산만 필요한 경우 활성화
GAUSSIAN_ONLY_MODE = True  # True = ["gaussian"], False = ["gaussian", "glb", "mesh"]

# Phase A: SS Generator Step Caching (Fast-SAM3D, Section 4.1)
# cache_stride=3: 매 3번째 step만 full backbone 실행, 나머지는 캐시 재사용
# warmup_steps=2: 처음 2 step은 항상 full (궤적 안정화)
# 14 steps 기준: full=6, cached=8 → backbone 호출 18/42 = 57% 감소
ENABLE_SS_STEP_CACHING = True
SS_CACHE_STRIDE = 3
SS_CACHE_WARMUP_STEPS = 2

# Phase B: SLaT Generator Step Caching
# cache_stride=2: 4 steps에서 보수적 캐싱 (step 0,1=warmup, 2=cached, 3=full)
ENABLE_SLAT_STEP_CACHING = False
SLAT_CACHE_STRIDE = 2
SLAT_CACHE_WARMUP_STEPS = 1

# ============================================================================

# Import protocol after environment setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_protocol import (
    MessageType, TaskMessage, ResultMessage, InitMessage,
    HeartbeatMessage, parse_message
)


def log(msg: str):
    """stderr로 로그 출력 (stdout은 JSON 통신용)"""
    print(f"[Worker] {msg}", file=sys.stderr, flush=True)


def send_message(msg_obj):
    """stdout으로 JSON 메시지 전송 (stdout 오염 방지)"""
    # Restore real stdout in case it was suppressed
    if hasattr(send_message, '_real_stdout'):
        sys.stdout = send_message._real_stdout
    print(msg_obj.to_json(), flush=True)


def suppress_stdout():
    """stdout을 /dev/null로 리다이렉트 (Warp 등 라이브러리 stdout 오염 방지)"""
    send_message._real_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')


def restore_stdout():
    """stdout 복원"""
    if hasattr(send_message, '_real_stdout'):
        sys.stdout = send_message._real_stdout


def downsample_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    max_size: int = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Downsample image and mask if larger than max_size.
    SAM-3D internally resizes to 518x518, so pre-downsampling saves preprocessing time.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        max_size: Maximum dimension size (None = no downsampling)

    Returns:
        (downsampled_image, downsampled_mask, scale_factor)
    """
    H, W = image.shape[:2]

    # None means no downsampling
    if max_size is None or max(H, W) <= max_size:
        return image, mask, 1.0

    scale = max_size / max(H, W)
    new_W = int(W * scale)
    new_H = int(H * scale)

    # Use PIL for high-quality downsampling
    image_pil = Image.fromarray(image)
    image_resized = image_pil.resize((new_W, new_H), Image.Resampling.LANCZOS)
    image_downsampled = np.array(image_resized)

    # Use nearest neighbor for mask to preserve binary values
    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize((new_W, new_H), Image.Resampling.NEAREST)
    mask_downsampled = np.array(mask_resized)

    return image_downsampled, mask_downsampled, scale


def make_synthetic_pointmap(image: np.ndarray, z: float = 1.0, f: float = None) -> torch.Tensor:
    """Create a simple pinhole-camera pointmap"""
    H, W = image.shape[:2]
    if f is None:
        f = 0.9 * max(H, W)

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    Z = np.full((H, W), z, dtype=np.float32)
    X = (uu - cx) / f * Z
    Y = (vv - cy) / f * Z

    pm = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return torch.from_numpy(pm)


def add_rgb_to_ply(ply_path: str, use_binary: bool = True):
    """
    Post-process PLY file to add RGB colors from SH coefficients.

    Phase 3 Optimization: Uses binary format by default for ~70% smaller files
    and ~50% faster post-processing.

    Args:
        ply_path: Path to PLY file
        use_binary: Use binary format (True) or ASCII format (False)
    """
    with open(ply_path, "rb") as f:
        data = f.read()

    text_data = data.decode("utf-8", errors="ignore")
    header_end = text_data.find("end_header")
    if header_end == -1:
        raise ValueError("Invalid PLY: no end_header found")

    header_text = text_data[: header_end + len("end_header")]
    header_lines = header_text.split("\n")

    vertex_count = 0
    properties = []
    property_types = {}

    for line in header_lines:
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            prop_type = parts[1]
            prop_name = parts[2]
            properties.append(prop_name)
            property_types[prop_name] = prop_type

    numpy_dtype = []
    for prop_name in properties:
        if property_types[prop_name] == "float":
            numpy_dtype.append((prop_name, "<f4"))
        elif property_types[prop_name] in ["uchar", "uint8"]:
            numpy_dtype.append((prop_name, "u1"))

    binary_start = len(header_text.encode("utf-8")) + 1
    binary_data = data[binary_start:]

    vertices = np.frombuffer(binary_data, dtype=np.dtype(numpy_dtype), count=vertex_count)

    SH0 = 0.282095
    f_dc = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    ).astype(np.float32)

    f_dc_tensor = torch.from_numpy(f_dc).cuda()
    rgb_linear = torch.clamp(f_dc_tensor * SH0 + 0.5, 0, 1)
    rgb_gamma = rgb_linear ** (1.0 / 2.2)
    rgb_255 = torch.clamp(rgb_gamma * 255, 0, 255).byte()
    rgb_cpu = rgb_255.cpu().numpy()

    x_vals = vertices["x"].astype(np.float32)
    y_vals = vertices["y"].astype(np.float32)
    z_vals = vertices["z"].astype(np.float32)

    if use_binary:
        # Binary format: ~70% smaller, ~50% faster to write
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        # Create structured array for binary output
        vertex_dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex_data = np.zeros(vertex_count, dtype=vertex_dtype)
        vertex_data['x'] = x_vals
        vertex_data['y'] = y_vals
        vertex_data['z'] = z_vals
        vertex_data['red'] = rgb_cpu[:, 0]
        vertex_data['green'] = rgb_cpu[:, 1]
        vertex_data['blue'] = rgb_cpu[:, 2]

        with open(ply_path, "wb") as out:
            out.write(header.encode('ascii'))
            out.write(vertex_data.tobytes())
    else:
        # ASCII format (legacy, slower but more compatible)
        ply_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header"
        ]

        for i in range(vertex_count):
            x, y, z = x_vals[i], y_vals[i], z_vals[i]
            r, g, b = rgb_cpu[i]
            ply_lines.append(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}")

        ply_content = "\n".join(ply_lines) + "\n"

        with open(ply_path, "w") as out:
            out.write(ply_content)


class PersistentWorker:
    """Persistent 3D Worker - 모델을 미리 로드하고 작업 요청 대기"""

    def __init__(self, worker_id: int, gpu_id: int):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.sam3d_inference = None
        self.pipe = None
        self.running = True

    def initialize(self) -> bool:
        """SAM-3D 모델 로드"""
        try:
            log(f"Initializing worker {self.worker_id} on GPU {self.gpu_id}")
            log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

            if torch.cuda.is_available():
                log(f"torch.cuda.device_count()={torch.cuda.device_count()}")
                log(f"torch.cuda.current_device()={torch.cuda.current_device()}")
                log(f"Using GPU: {torch.cuda.get_device_name(0)}")

            # Suppress stdout during model loading (Warp/kaolin pollute stdout)
            suppress_stdout()

            # Import SAM-3D
            sam3d_notebook_path = "./sam-3d-objects/notebook"
            if not os.path.exists(sam3d_notebook_path):
                raise ImportError(f"Sam-3d-objects notebook path not found: {sam3d_notebook_path}")

            sys.path.insert(0, sam3d_notebook_path)
            from inference import Inference, make_scene, ready_gaussian_for_video_rendering

            self.make_scene = make_scene
            self.ready_gaussian_for_video_rendering = ready_gaussian_for_video_rendering

            config_path = "./sam-3d-objects/checkpoints/hf/pipeline.yaml"
            if not os.path.exists(config_path):
                raise ImportError(f"Sam-3d-objects config not found at {config_path}")

            log(f"Loading SAM-3D from {config_path}...")
            load_start = time.time()

            # compile은 항상 False로 로드 (SAM3D 내부 _warmup()에 run_layout_model 버그 존재)
            # 대신 Phase C에서 수동으로 torch.compile 적용 + 자체 warmup 수행
            try:
                self.sam3d_inference = Inference(config_path, compile=False, device="cuda")
            except TypeError:
                self.sam3d_inference = Inference(config_path, compile=False)

            self.pipe = getattr(self.sam3d_inference, "_pipeline", None)
            if self.pipe is None:
                raise RuntimeError("Inference object has no _pipeline")

            # ============================================================
            # VRAM Optimization: Gaussian-only 모드에서 불필요한 모델 제거
            # slat_decoder_mesh (~3-4GB), slat_decoder_gs_4 (~2-3GB),
            # depth_model/MoGe (~1GB) 제거 → 총 ~6-8GB VRAM 절약
            # ============================================================
            if GAUSSIAN_ONLY_MODE and hasattr(self.pipe, "models"):
                models_to_remove = []

                # mesh decoder: Gaussian-only에서 절대 사용 안 함
                if "slat_decoder_mesh" in self.pipe.models:
                    models_to_remove.append("slat_decoder_mesh")

                # gs_4 decoder: stage2=4에서는 기본 gs decoder 사용
                if "slat_decoder_gs_4" in self.pipe.models and self.pipe.models["slat_decoder_gs_4"] is not None:
                    models_to_remove.append("slat_decoder_gs_4")

                for name in models_to_remove:
                    model = self.pipe.models[name]
                    if model is not None:
                        # CPU로 이동 후 삭제 (ModuleDict에서 직접 삭제 불가하므로 CPU 이동)
                        model.cpu()
                        del model
                    log(f"Unloaded {name} from GPU (not needed in Gaussian-only mode)")

                # depth_model (MoGe): synthetic pointmap 사용 시 불필요
                if hasattr(self.pipe, "depth_model") and self.pipe.depth_model is not None:
                    depth = self.pipe.depth_model
                    # MoGe wraps an nn.Module internally
                    if hasattr(depth, "model") and hasattr(depth.model, "cpu"):
                        depth.model.cpu()
                    elif hasattr(depth, "cpu"):
                        depth.cpu()
                    del depth
                    self.pipe.depth_model = None
                    log("Unloaded depth_model (MoGe) from GPU (using synthetic pointmap)")

                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # VRAM 사용량 확인
                vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                log(f"VRAM after cleanup: {vram_used:.2f}GB allocated, {vram_reserved:.2f}GB reserved")

            # Move remaining models to GPU and set eval mode
            moved_count = 0
            if hasattr(self.pipe, "models"):
                for name, model in self.pipe.models.items():
                    if model is not None and hasattr(model, "cuda"):
                        model.cuda()
                        moved_count += 1
                    if model is not None and hasattr(model, "eval"):
                        model.eval()

            log(f"Moved {moved_count} models to GPU")

            # ============================================================
            # Phase A: SS Generator Step Caching
            # ============================================================
            if ENABLE_SS_STEP_CACHING:
                try:
                    from sam3d_objects.model.backbone.generator.flow_matching.cached_solver import CachedEuler
                except ImportError:
                    from cached_solver import CachedEuler
                ss_gen = self.pipe.models["ss_generator"]
                ss_gen._solver = CachedEuler(
                    cache_stride=SS_CACHE_STRIDE,
                    warmup_steps=SS_CACHE_WARMUP_STEPS,
                )
                ss_gen._solver_method = "cached_euler"
                log(f"Phase A: SS step caching (stride={SS_CACHE_STRIDE}, warmup={SS_CACHE_WARMUP_STEPS})")

            # ============================================================
            # Phase B: SLaT Generator Step Caching
            # ============================================================
            if ENABLE_SLAT_STEP_CACHING:
                try:
                    from sam3d_objects.model.backbone.generator.flow_matching.cached_solver import CachedEuler as CachedEulerB
                except ImportError:
                    from cached_solver import CachedEuler as CachedEulerB
                slat_gen = self.pipe.models["slat_generator"]
                slat_gen._solver = CachedEulerB(
                    cache_stride=SLAT_CACHE_STRIDE,
                    warmup_steps=SLAT_CACHE_WARMUP_STEPS,
                )
                slat_gen._solver_method = "cached_euler"
                log(f"Phase B: SLaT step caching (stride={SLAT_CACHE_STRIDE}, warmup={SLAT_CACHE_WARMUP_STEPS})")

            torch.set_grad_enabled(False)

            load_time = time.time() - load_start
            log(f"SAM-3D loaded in {load_time:.2f}s")

            # ============================================================
            # Phase C: Manual torch.compile + AUTOTUNE warmup
            # SAM3D 내부 compile=True는 _warmup()에서 run_layout_model 버그가 있어
            # compile=False로 로드한 뒤, 핵심 모듈만 수동 compile하고 자체 warmup 수행
            # ============================================================
            ENABLE_COMPILE = True  # True = 추론 10-20% 빠름, False = 빠른 시작 (테스트용)
            if ENABLE_COMPILE:
                log("Phase C: Manual torch.compile on critical modules...")
                compile_start = time.time()
                compile_mode = "reduce-overhead"  # max-autotune은 첫 실행 10분+, reduce-overhead는 ~2분

                try:
                    _dynamo = torch._dynamo
                    _dynamo.config.cache_size_limit = 64
                    _dynamo.config.accumulated_cache_size_limit = 2048
                    _dynamo.config.capture_scalar_outputs = True

                    # SS Generator backbone (가장 호출 빈도 높음: 14 steps × 3 calls = 42회)
                    ss_gen = self.pipe.models["ss_generator"]
                    if hasattr(ss_gen, "reverse_fn") and hasattr(ss_gen.reverse_fn, "inner_forward"):
                        ss_gen.reverse_fn.inner_forward = torch.compile(
                            ss_gen.reverse_fn.inner_forward,
                            mode=compile_mode,
                            fullgraph=True,
                        )
                        log("Phase C: Compiled SS generator backbone")

                    # SS Decoder
                    ss_dec = self.pipe.models["ss_decoder"] if "ss_decoder" in self.pipe.models else None
                    if ss_dec is not None:
                        ss_dec.forward = torch.compile(
                            ss_dec.forward,
                            mode=compile_mode,
                            fullgraph=True,
                        )
                        log("Phase C: Compiled SS decoder")

                    # Condition embedding (fullgraph=False: PointPatchEmbed 호환성)
                    if hasattr(self.pipe, "embed_condition"):
                        self.pipe.embed_condition = torch.compile(
                            self.pipe.embed_condition,
                            mode=compile_mode,
                            fullgraph=False,
                        )
                        log("Phase C: Compiled condition embedding")

                    log(f"Phase C: Compilation setup in {time.time() - compile_start:.1f}s")

                    # Warmup: 첫 실행으로 AUTOTUNE 캐시 생성
                    log("Phase C: Running AUTOTUNE warmup (first run may take ~2-5min)...")
                    warmup_start = time.time()
                    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                    dummy_mask = np.ones((512, 512), dtype=np.uint8) * 255
                    dummy_pointmap = make_synthetic_pointmap(dummy_image, z=1.0)

                    with torch.no_grad():
                        _ = self.pipe.run(
                            image=dummy_image,
                            mask=dummy_mask,
                            seed=42,
                            pointmap=dummy_pointmap,
                            decode_formats=["gaussian"] if GAUSSIAN_ONLY_MODE else ["gaussian", "mesh"],
                            stage1_inference_steps=STAGE1_INFERENCE_STEPS,
                            stage2_inference_steps=STAGE2_INFERENCE_STEPS,
                            with_mesh_postprocess=False,
                            with_texture_baking=False,
                            with_layout_postprocess=False,
                            use_vertex_color=True,
                        )
                    torch.cuda.empty_cache()
                    log(f"Phase C: AUTOTUNE warmup completed in {time.time() - warmup_start:.1f}s")
                    log(f"Phase C: Cache dir = {_autotune_cache_dir}")

                except Exception as e:
                    log(f"Phase C: Compile/warmup failed (non-fatal, continuing without compile): {e}")
                    import traceback
                    traceback.print_exc(file=sys.stderr)

            # Send init success message
            send_message(InitMessage(
                worker_id=self.worker_id,
                gpu_id=self.gpu_id,
                model_loaded=True
            ))

            return True

        except Exception as e:
            log(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)

            send_message(InitMessage(
                worker_id=self.worker_id,
                gpu_id=self.gpu_id,
                model_loaded=False,
                error=str(e)
            ))
            return False

    def process_task(self, task: TaskMessage) -> ResultMessage:
        """3D 생성 작업 처리"""
        suppress_stdout()  # Protect stdout during 3D generation
        start_time = time.time()
        log(f"Processing task {task.task_id}")

        try:
            # Decode image and mask
            image_bytes = base64.b64decode(task.image_b64)
            mask_bytes = base64.b64decode(task.mask_b64)

            # Save to temp files
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(mask_bytes)
                mask_path = tmp.name

            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                output_ply_path = tmp.name

            try:
                # Load image and mask
                image_pil = Image.open(image_path).convert("RGB")
                image = np.array(image_pil)

                mask_pil = Image.open(mask_path).convert("L")
                mask = np.array(mask_pil)

                # Validate dtypes
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)

                mask_u8 = (mask > 0).astype(np.uint8) * 255
                mask_pixel_count = np.sum(mask_u8 > 0)

                if mask_u8.sum() == 0:
                    raise ValueError("Mask is empty")

                log(f"Original Image: {image.shape}, Mask pixels: {mask_pixel_count}")

                # Phase 1: Downsample large images for faster processing
                # SAM-3D resizes to 518x518 internally, so pre-downsampling saves time
                image, mask_u8, scale = downsample_image_and_mask(image, mask_u8, max_size=MAX_IMAGE_SIZE)
                if scale < 1.0:
                    log(f"Downsampled to {image.shape} (scale={scale:.3f})")

                # Set seed
                torch.manual_seed(task.seed)
                np.random.seed(task.seed)

                # Clear GPU cache
                torch.cuda.empty_cache()

                # Create pointmap
                pointmap = make_synthetic_pointmap(image, z=1.0)

                # Run inference
                # Phase 5: Gaussian-only mode for volume calculation (skip GLB/mesh)
                if GAUSSIAN_ONLY_MODE:
                    decode_formats = ["gaussian"]
                    log("Phase 5: Gaussian-only mode (37.4% faster, 0.005% volume diff)")
                else:
                    decode_formats = ["gaussian", "glb", "mesh"]

                # Phase 2: stage2_inference_steps configured at module level

                with torch.no_grad():
                    output = self.pipe.run(
                        image=image,
                        mask=mask_u8,
                        seed=task.seed,
                        pointmap=pointmap,
                        decode_formats=decode_formats,
                        stage1_inference_steps=STAGE1_INFERENCE_STEPS,
                        stage2_inference_steps=STAGE2_INFERENCE_STEPS,
                        with_mesh_postprocess=False,
                        with_texture_baking=False,
                        with_layout_postprocess=False,
                        use_vertex_color=True,
                    )

                torch.cuda.synchronize()

                # Prepare scene (in_place=True로 deepcopy 제거하여 메모리/속도 최적화)
                scene_gs = self.make_scene(output, in_place=True)
                scene_gs = self.ready_gaussian_for_video_rendering(
                    scene_gs, in_place=True, fix_alignment=False
                )

                # Save PLY
                scene_gs.save_ply(output_ply_path)

                # Post-process PLY (Phase 3: Binary format for faster I/O)
                try:
                    add_rgb_to_ply(output_ply_path, use_binary=USE_BINARY_PLY)
                except Exception as e:
                    log(f"Warning: Could not add RGB to PLY: {e}")

                # Read PLY
                with open(output_ply_path, "rb") as f:
                    ply_bytes = f.read()

                ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
                ply_size_bytes = len(ply_bytes)

                processing_time = time.time() - start_time
                log(f"Task {task.task_id} completed in {processing_time:.2f}s, PLY: {ply_size_bytes} bytes")

                return ResultMessage(
                    task_id=task.task_id,
                    success=True,
                    ply_b64=ply_b64,
                    ply_size_bytes=ply_size_bytes,
                    processing_time_seconds=processing_time
                )

            finally:
                # Cleanup temp files
                for path in [image_path, mask_path, output_ply_path]:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except Exception:
                            pass

        except Exception as e:
            log(f"Task {task.task_id} failed: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)

            return ResultMessage(
                task_id=task.task_id,
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time
            )

    def run(self):
        """메인 루프 - stdin에서 작업 요청 대기"""
        log("Worker starting main loop")

        while self.running:
            try:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    log("stdin closed, shutting down")
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse message
                msg = parse_message(line)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type is None:
                    log(f"Invalid message: {data.get('error', 'unknown')}")
                    continue

                if msg_type == MessageType.TASK:
                    task = TaskMessage.from_dict(data)
                    result = self.process_task(task)
                    send_message(result)

                elif msg_type == MessageType.HEARTBEAT:
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                    send_message(HeartbeatMessage(
                        worker_id=self.worker_id,
                        status="alive",
                        gpu_memory_used_mb=gpu_mem
                    ))

                elif msg_type == MessageType.SHUTDOWN:
                    log(f"Shutdown requested: {data.get('reason', 'unknown')}")
                    self.running = False

            except KeyboardInterrupt:
                log("Interrupted, shutting down")
                break
            except Exception as e:
                log(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)

        log("Worker exiting")


def main():
    if len(sys.argv) < 3:
        print("Usage: python persistent_3d_worker.py <worker_id> <gpu_id>", file=sys.stderr)
        sys.exit(1)

    worker_id = int(sys.argv[1])
    gpu_id = int(sys.argv[2])

    worker = PersistentWorker(worker_id, gpu_id)

    if not worker.initialize():
        sys.exit(1)

    worker.run()


if __name__ == "__main__":
    main()
