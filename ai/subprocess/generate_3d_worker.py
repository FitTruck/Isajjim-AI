"""
Subprocess wrapper for 3D generation.
This runs in a fresh Python process to avoid spconv state issues.

Usage:
    python generate_3d_subprocess.py <image_path> <mask_path> <seed> <output_ply_path>
"""

import sys
import os
import base64

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# ============================================================================
# Use existing CUDA_HOME, or fallback to CONDA_PREFIX (local conda), or default path
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX") or "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

# Multi-GPU 지원:
# CUDA_VISIBLE_DEVICES가 부모 프로세스에서 설정되면 그 값을 사용
# CUDA_VISIBLE_DEVICES로 GPU를 제한하면 내부적으로 항상 device 0이 됨
# 따라서 SPCONV_TUNE_DEVICE는 항상 "0"으로 설정
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
print(f"[Subprocess] CUDA_VISIBLE_DEVICES={cuda_visible}")

# SPCONV_TUNE_DEVICE는 항상 0 (CUDA_VISIBLE_DEVICES로 remap되므로)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = os.environ.get("SPCONV_ALGO_TIME_LIMIT", "100")
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"

# Prevent thread explosion - limit OpenMP threads
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

# Multi-GPU: GPU 상태 로깅
if torch.cuda.is_available():
    print(f"[Subprocess] torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"[Subprocess] torch.cuda.current_device()={torch.cuda.current_device()}")
    print(f"[Subprocess] Using GPU: {torch.cuda.get_device_name(0)} (remapped as cuda:0)")

from PIL import Image

# Optional: import open3d for mesh simplification (if available)
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARNING] open3d not available - mesh simplification disabled")

# Optional: import to_glb for native texture-baked GLB export
HAS_TO_GLB = False
to_glb = None
try:
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb

    HAS_TO_GLB = True
    print("[INFO] to_glb imported successfully for native GLB export")
except ImportError as e:
    print(f"[WARNING] Could not import to_glb: {e}")


def gaussian_to_simplified_mesh(
    gaussian_splat, output_mesh_path: str, depth: int = 6, reduction: float = 0.95
):
    """
    Convert Gaussian point cloud to a simplified mesh using Poisson reconstruction.
    Uses _features_dc (DC SH component) for vertex colors.

    Args:
        gaussian_splat: The Gaussian Splat object
        output_mesh_path: Where to save the mesh PLY
        depth: Poisson reconstruction depth (higher = more detail)
        reduction: Mesh reduction factor after generation
    """
    if not HAS_OPEN3D:
        print("[Subprocess] WARNING: open3d not available, skipping mesh generation")
        return None

    try:
        import time
        import trimesh

        start_time = time.time()
        print(f"[Subprocess] Converting Gaussian to mesh...")

        # Extract points
        print(f"[Subprocess] Extracting points...")
        points_gpu = gaussian_splat.get_xyz
        print(f"[Subprocess] Points: {points_gpu.shape}")

        # Extract colors from _features_dc
        colors_gpu = None
        print(f"[Subprocess] Extracting vertex colors...")
        try:
            if hasattr(gaussian_splat, "_features_dc"):
                colors_gpu = gaussian_splat._features_dc.clone()  # [N, 3, 1]
                print(f"[Subprocess] Found _features_dc: {colors_gpu.shape}")

                # Reshape [N, 3, 1] to [N, 3]
                if colors_gpu.dim() == 3:
                    colors_gpu = colors_gpu.squeeze(-1)
                if len(colors_gpu) != len(points_gpu):
                    print(
                        f"[Subprocess] Color mismatch: {len(colors_gpu)} vs {len(points_gpu)}, using white"
                    )
                    colors_gpu = None
                else:
                    # SH DC in log space, use sigmoid to recover RGB
                    colors_gpu = torch.sigmoid(colors_gpu)
                    colors_gpu = torch.clamp(colors_gpu, 0.0, 1.0)
                    print(f"[Subprocess] Colors ready: {colors_gpu.shape}")
            else:
                print(f"[Subprocess] _features_dc not found")
                colors_gpu = None
        except Exception as e:
            print(f"[Subprocess] Color extraction failed: {e}")
            colors_gpu = None

        # Fallback
        if colors_gpu is None:
            colors_gpu = torch.ones(len(points_gpu), 3, device=points_gpu.device)
            print(f"[Subprocess] Using white fallback")

        # Convert to numpy - use float32 for trimesh
        points_np = points_gpu.detach().cpu().numpy().astype(np.float32)
        colors_np = (colors_gpu.detach().cpu().numpy() * 255).astype(
            np.uint8
        )  # trimesh uses 0-255

        print(f"[Subprocess] Creating point cloud with {len(points_np)} points...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))

        # Estimate normals (fast KNN)
        print(f"[Subprocess] Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))

        # Poisson reconstruction (depth=8 is good balance, depth=6 is fast)
        print(f"[Subprocess] Poisson mesh reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        print(f"[Subprocess] Poisson reconstruction complete")

        # Get triangle count
        try:
            initial_triangles = mesh.get_triangle_count()
        except AttributeError:
            initial_triangles = (
                len(np.asarray(mesh.triangles)) if hasattr(mesh, "triangles") else 0
            )

        print(f"[Subprocess] Initial mesh: {initial_triangles} triangles")

        # Convert to trimesh to add vertex colors
        print(f"[Subprocess] Converting to trimesh with vertex colors...")
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.uint32)

        # Create trimesh with vertex colors
        trimesh_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors_np,  # trimesh supports per-vertex colors
            process=False,
        )

        print(
            f"[Subprocess] Trimesh created with {len(trimesh_mesh.vertices)} vertices and {len(trimesh_mesh.faces)} faces"
        )

        # Simplify if too many triangles
        if reduction > 0 and initial_triangles > 5000:
            target_count = max(1000, int(initial_triangles * (1 - reduction)))
            print(f"[Subprocess] Simplifying mesh to {target_count} triangles...")
            trimesh_mesh = trimesh_mesh.simplify(target_count=target_count)
            final_triangles = len(trimesh_mesh.faces)
            print(f"[Subprocess] Simplified to {final_triangles} triangles")

        # Save mesh with colors using trimesh
        print(f"[Subprocess] Saving mesh to {output_mesh_path}...")
        trimesh_mesh.export(output_mesh_path, file_type="ply")
        mesh_size = os.path.getsize(output_mesh_path)
        total_time = time.time() - start_time
        print(
            f"[Subprocess] ✓ Mesh saved: {mesh_size} bytes (total time: {total_time:.2f}s)"
        )

        return mesh_size

    except Exception as e:
        print(f"[Subprocess] ERROR converting to mesh: {e}")
        import traceback

        traceback.print_exc()
        return None


def make_synthetic_pointmap(
    image: np.ndarray, z: float = 1.0, f: float = None
) -> torch.Tensor:
    """
    Create a simple pinhole-camera pointmap:
      X = (u - cx) / f * Z
      Y = (v - cy) / f * Z
      Z = constant depth

    This is non-degenerate (unlike all-zeros XY) and stays finite.
    Used to avoid intrinsics recovery failures from MoGe or dummy pointmaps.

    Args:
        image: numpy array [H,W,3]
        z: constant depth value (default 1.0)
        f: focal length in pixels (if None, estimated as 0.9 * max(H,W))

    Returns:
        torch tensor [H,W,3] float32 pinhole pointmap
    """
    H, W = image.shape[:2]
    if f is None:
        # Reasonable focal guess in pixel units
        f = 0.9 * max(H, W)

    # Pixel grid
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


def sanitize_pointmap(pointmap: torch.Tensor, mask_u8: np.ndarray) -> torch.Tensor:
    """
    Sanitize pointmap by removing NaNs/Infs and clamping depth to reasonable range.

    Args:
        pointmap: torch [H,W,3] float32 (x,y,z)
        mask_u8: numpy [H,W] uint8 (0/255)

    Returns:
        Sanitized pointmap tensor
    """
    pm = pointmap.float()
    m = torch.from_numpy(mask_u8 > 0)

    # Replace non-finite anywhere with 0
    finite = torch.isfinite(pm).all(dim=-1)
    pm[~finite] = 0.0

    # Enforce positive depth in masked region
    z = pm[..., 2]
    z_mask = z[m & finite]
    if z_mask.numel() > 0:
        # Clamp z to a reasonable range based on percentiles
        z_sorted = torch.sort(z_mask.flatten()).values
        lo = z_sorted[int(0.02 * (z_sorted.numel() - 1))]
        hi = z_sorted[int(0.98 * (z_sorted.numel() - 1))]
        z = torch.clamp(z, min=float(lo), max=float(hi))
        pm[..., 2] = z
    else:
        # Nothing valid in mask
        raise ValueError("Pointmap has no valid points under mask")

    return pm


def add_rgb_to_ply(ply_path: str):
    """
    Post-process PLY file to add RGB colors from SH coefficients using GPU acceleration.
    Vectorized binary parsing and writing for speed.
    """
    with open(ply_path, "rb") as f:
        data = f.read()

    # Parse header
    text_data = data.decode("utf-8", errors="ignore")
    header_end = text_data.find("end_header")
    if header_end == -1:
        raise ValueError("Invalid PLY: no end_header found")

    header_text = text_data[: header_end + len("end_header")]
    header_lines = header_text.split("\n")

    # Extract vertex count and properties
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

    print(f"[PLY Post-process] Found {vertex_count} vertices")

    # Build dtype for numpy structured array (vectorized binary reading)
    numpy_dtype = []
    for prop_name in properties:
        if property_types[prop_name] == "float":
            numpy_dtype.append((prop_name, "<f4"))
        elif property_types[prop_name] in ["uchar", "uint8"]:
            numpy_dtype.append((prop_name, "u1"))

    # Extract binary data
    binary_start = len(header_text.encode("utf-8")) + 1
    binary_data = data[binary_start:]

    # Read all vertices at once using numpy (vectorized!)
    print(f"[PLY Post-process] Vectorized binary parsing...")
    vertices = np.frombuffer(
        binary_data, dtype=np.dtype(numpy_dtype), count=vertex_count
    )

    # Extract SH coefficients into GPU tensors (vectorized)
    print(f"[PLY Post-process] Converting SH to RGB on GPU...")
    SH0 = 0.282095

    f_dc = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    ).astype(np.float32)

    # GPU processing
    f_dc_tensor = torch.from_numpy(f_dc).cuda()
    rgb_linear = torch.clamp(f_dc_tensor * SH0 + 0.5, 0, 1)
    rgb_gamma = rgb_linear ** (1.0 / 2.2)
    rgb_255 = torch.clamp(rgb_gamma * 255, 0, 255).byte()
    rgb_cpu = rgb_255.cpu().numpy()

    # Write ASCII PLY (much faster than binary rewriting)
    print(f"[PLY Post-process] Writing ASCII PLY with colors...")

    # Safely extract coordinates with validation
    try:
        x_vals = vertices["x"]
        y_vals = vertices["y"]
        z_vals = vertices["z"]
        print(f"[PLY Post-process] Extracted coordinates successfully")
    except Exception as e:
        print(f"[PLY Post-process] ERROR: Failed to extract coordinates: {e}")
        print(f"[PLY Post-process] Available properties: {vertices.dtype.names}")
        raise

    # Write ASCII PLY file
    ply_lines = []
    ply_lines.append("ply")
    ply_lines.append("format ascii 1.0")
    ply_lines.append(f"element vertex {vertex_count}")
    ply_lines.append("property float x")
    ply_lines.append("property float y")
    ply_lines.append("property float z")
    ply_lines.append("property uchar red")
    ply_lines.append("property uchar green")
    ply_lines.append("property uchar blue")
    ply_lines.append("end_header")

    # Write vertex data
    print(f"[PLY Post-process] Writing {vertex_count} vertices...")
    for i in range(vertex_count):
        x, y, z = x_vals[i], y_vals[i], z_vals[i]
        r, g, b = rgb_cpu[i]
        ply_lines.append(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}")

    # Write all at once for safety
    ply_content = "\n".join(ply_lines) + "\n"

    with open(ply_path, "w") as out:
        out.write(ply_content)

    # Validate file was written correctly
    with open(ply_path, "r") as f:
        content = f.read()
        if "end_header" not in content:
            raise ValueError("PLY file missing 'end_header' - write may have failed")
        lines = content.split("\n")
        if len(lines) < vertex_count + 10:
            raise ValueError(
                f"PLY file incomplete - expected {vertex_count} vertices, got {len(lines) - 10}"
            )

    print(
        f"[PLY Post-process] ✓ PLY conversion complete: {vertex_count} vertices with RGB colors"
    )


def main():
    if len(sys.argv) < 6:
        print(
            "Usage: python generate_3d_subprocess.py <image_path> <mask_path> <seed> <output_ply_path> <assets_dir> [--skip-gif]"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    seed = int(sys.argv[3])
    output_ply_path = sys.argv[4]
    assets_dir = sys.argv[5]

    # GIF 렌더링 스킵 옵션 (부피 계산 최적화)
    skip_gif = "--skip-gif" in sys.argv or os.environ.get("SAM3D_SKIP_GIF", "").lower() in ("1", "true", "yes")

    try:
        # Import sam3d_inference here (fresh import in subprocess)
        sam3d_notebook_path = "./sam-3d-objects/notebook"

        if not sam3d_notebook_path or not os.path.exists(sam3d_notebook_path):
            # Try common locations (parent directory first since API is inside sam-3d-objects)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            possible_paths = [
                os.path.join(
                    parent_dir, "notebook"
                ),  # Parent dir (sam-3d-objects/notebook)
                "../notebook",
                "/workspace/sam-3d-objects/notebook",
                "/workspace/notebook",
                os.path.expanduser("~/sam-3d-objects/notebook"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    sam3d_notebook_path = path
                    break

            if not sam3d_notebook_path or not os.path.exists(sam3d_notebook_path):
                error_msg = (
                    f"Sam-3d-objects notebook path not found. Tried:\n"
                    + "\n".join(f"  - {p}" for p in possible_paths)
                    + f"\n\nPlease set SAM3D_NOTEBOOK_PATH environment variable to the correct path."
                )
                raise ImportError(error_msg)

        sys.path.insert(0, sam3d_notebook_path)
        from inference import (
            Inference,
            make_scene,
            ready_gaussian_for_video_rendering,
            render_video,
        )

        # Use fixed pipeline config path (no search or environment variable)
        config_path = "./sam-3d-objects/checkpoints/hf/pipeline.yaml"

        if not os.path.exists(config_path):
            raise ImportError(f"Sam-3d-objects config not found at {config_path}")

        print(f"[Subprocess] Loading Sam-3d-objects from {config_path}...")

        # Try to pass device to Inference if it supports it
        try:
            sam3d_inference = Inference(config_path, compile=False, device="cuda")
            print(f"[Subprocess] Inference initialized with device='cuda'")
        except TypeError:
            # If device parameter not supported, initialize normally
            sam3d_inference = Inference(config_path, compile=False)
            print(
                f"[Subprocess] Inference initialized (device parameter not supported)"
            )

        # Force all models to GPU and verify
        moved_count = 0
        if hasattr(sam3d_inference, "_pipeline") and hasattr(
            sam3d_inference._pipeline, "models"
        ):
            for model_name, model in sam3d_inference._pipeline.models.items():
                if hasattr(model, "cuda"):
                    model.cuda()
                    moved_count += 1
                    # Verify it's on GPU
                    if hasattr(model, "parameters"):
                        for param in model.parameters():
                            if param.is_cuda:
                                print(f"[Subprocess] ✓ {model_name} on GPU")
                            else:
                                print(
                                    f"[Subprocess] ⚠ {model_name} NOT on GPU (failed to move)"
                                )
                            break
                if hasattr(model, "eval"):
                    model.eval()

        print(f"[Subprocess] Moved {moved_count} models to GPU")

        # Also try direct attributes
        if hasattr(sam3d_inference, "model"):
            sam3d_inference.model.cuda()
            print(f"[Subprocess] Moved sam3d_inference.model to GPU")
        if hasattr(sam3d_inference, "device"):
            print(f"[Subprocess] Device attribute: {sam3d_inference.device}")

        torch.set_grad_enabled(False)

        print(f"[Subprocess] ✓ Sam3d-objects loaded successfully")

        # Get access to the pipeline for direct run() call with GLB export
        pipe = getattr(sam3d_inference, "_pipeline", None)
        if pipe is None:
            raise RuntimeError(
                "Inference object has no _pipeline; cannot call pipeline.run()"
            )
        print(f"[Subprocess] ✓ Pipeline object accessed for GLB export")

        # Load image and mask from files
        print(f"[Subprocess] Loading image from {image_path}")
        image_pil = Image.open(image_path).convert("RGB")
        image = np.array(image_pil)
        print(f"[Subprocess] Image shape: {image.shape}, dtype: {image.dtype}")

        print(f"[Subprocess] Loading mask from {mask_path}")
        mask_pil = Image.open(mask_path).convert("L")
        mask = np.array(mask_pil)
        print(f"[Subprocess] Mask shape: {mask.shape}, dtype: {mask.dtype}")

        # Validate and convert dtypes
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert mask to uint8 (0 or 255) - critical for pipeline compatibility
        # Boolean masks can get mangled by PIL conversions and resizing
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        # Check if mask has any valid pixels
        mask_pixel_count = np.sum(mask_u8 > 0)
        print(f"[Subprocess] Mask stats:")
        print(f"  - Shape: {mask.shape}")
        print(f"  - Dtype: uint8 (0/255)")
        print(f"  - Min: {mask_u8.min()}, Max: {mask_u8.max()}")
        print(f"  - Valid pixels (255): {mask_pixel_count}")
        print(f"  - Percentage masked: {100.0 * mask_pixel_count / mask_u8.size:.2f}%")

        if mask_u8.sum() == 0:
            raise ValueError(
                "Mask is empty (no valid pixels). The mask must contain at least some "
                "pixels with value > 0 to indicate the object region."
            )

        if mask_pixel_count < 100:
            print(
                f"[Subprocess] ⚠ WARNING: Mask has very few pixels ({mask_pixel_count}). "
                "This may result in poor 3D generation quality."
            )

        mask = mask_u8

        # Final validation before inference
        print(f"[Subprocess] Final input validation:")
        print(f"  - Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  - Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  - Shapes match: {image.shape[:2] == mask.shape}")

        # Ensure mask and image dimensions match
        if image.shape[:2] != mask.shape:
            raise ValueError("Image/mask dimensions mismatch")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"[Subprocess] Running inference on GPU...")
        print(f"[Subprocess] GPU available: {torch.cuda.is_available()}")
        print(f"[Subprocess] Current GPU: {torch.cuda.current_device()}")

        # Clear GPU cache before inference
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Ensure GPU is synchronized before inference
        torch.cuda.synchronize()

        import time

        start_time = time.time()

        print(
            f"[Subprocess] Before inference - GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

        decode_formats = ["gaussian", "glb"]

        print(f"[Subprocess] Calling pipe.run() with (OPTIMIZED for volume calculation):")
        print(f"  - decode_formats: {decode_formats}")
        print(f"  - with_texture_baking: False (disabled for speed)")
        print(f"  - with_mesh_postprocess: False (disabled for speed)")
        print(f"  - with_layout_postprocess: False (disabled for speed)")
        print(f"  - pointmap: synthetic pinhole (avoid intrinsics recovery failures)")

        # Use synthetic pinhole pointmap to avoid intrinsics recovery failures
        # This is non-degenerate (unlike dummy x=y=0) and avoids MoGe NaN/Inf issues
        pointmap = make_synthetic_pointmap(image, z=1.0)
        print(f"[Subprocess] Generated synthetic pointmap: {pointmap.shape}")

        # Request mesh data for to_glb() if available
        # Note: "mesh" decode format may trigger mesh extraction from pipeline
        decode_formats_with_mesh = (
            decode_formats + ["mesh"]
            if "mesh" not in decode_formats
            else decode_formats
        )
        print(f"[Subprocess] Using decode_formats: {decode_formats_with_mesh}")

        # Use pipeline.run() directly
        # OPTIMIZATION: Disabled texture baking, mesh postprocess, layout postprocess
        # for faster volume calculation (70-90 seconds savings)
        with torch.no_grad():
            output = pipe.run(
                image=image,
                mask=mask,
                seed=seed,
                pointmap=pointmap,  # Use synthetic pinhole pointmap
                decode_formats=decode_formats_with_mesh,
                with_mesh_postprocess=False,     # Disabled for speed (20-40s saving)
                with_texture_baking=False,       # Disabled for speed (30-60s saving)
                with_layout_postprocess=False,   # Disabled for speed (2-5s saving)
                use_vertex_color=True,
            )

        # Wait for GPU to finish
        torch.cuda.synchronize()
        end_time = time.time()

        # DEBUG: Print output dict structure
        print("[Subprocess] DEBUG: Output dict keys and types:")
        if isinstance(output, dict):
            for key, value in output.items():
                value_type = type(value).__name__
                if hasattr(value, "shape"):
                    print(f"  - {key}: {value_type} (shape: {value.shape})")
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    try:
                        print(f"  - {key}: {value_type} (len: {len(value)})")
                    except Exception:
                        print(f"  - {key}: {value_type}")
                else:
                    print(f"  - {key}: {value_type}")

            # Check for "gaussian" key
            if "gaussian" in output:
                gaussian_val = output["gaussian"]
                print(f"\n[Subprocess] DEBUG: 'gaussian' details:")
                print(f"  - Type: {type(gaussian_val).__name__}")
                if hasattr(gaussian_val, "save_ply"):
                    print(f"  - Has save_ply: YES")
                if hasattr(gaussian_val, "scene_attributes"):
                    print(f"  - Has scene_attributes: YES")

            # Check for "glb" key
            if "glb" in output:
                glb_val = output["glb"]
                print(f"\n[Subprocess] DEBUG: 'glb' found!")
                print(f"  - Type: {type(glb_val).__name__}")
                if glb_val is not None and hasattr(glb_val, "export"):
                    print(f"  - Has export method: YES")
            else:
                print(f"\n[Subprocess] DEBUG: 'glb' key NOT found in output")

            # Check for mesh-related keys
            mesh_keys = [
                k for k in output.keys() if "mesh" in k.lower() or "slat" in k.lower()
            ]
            if mesh_keys:
                print(f"\n[Subprocess] DEBUG: Found mesh-related keys: {mesh_keys}")

            # Check 'gs' key (might be the mesh)
            if "gs" in output:
                gs_val = output["gs"]
                print(f"\n[Subprocess] DEBUG: 'gs' key found!")
                print(f"  - Type: {type(gs_val).__name__}")
                if hasattr(gs_val, "__len__"):
                    print(
                        f"  - Length/Count: {len(gs_val) if not isinstance(gs_val, str) else 'N/A'}"
                    )
                # Check for mesh export methods
                for method in ["export", "save", "to_glb", "to_mesh", "to_ply"]:
                    if hasattr(gs_val, method):
                        print(f"  - Has {method}(): YES")
        else:
            print(f"  - Output is not a dict: {type(output).__name__}")

        print(
            f"[Subprocess] After inference - GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

        inference_time = end_time - start_time
        gpu_mem_used = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB

        print(f"[Subprocess] ✓ Inference completed in {inference_time:.2f}s")
        print(f"[Subprocess] Peak GPU memory used: {gpu_mem_used:.2f} GB")

        # Check what's in output
        if isinstance(output, dict):
            print(f"[Subprocess] Output keys: {list(output.keys())}")
            if "gaussian" in output:
                print(f"[Subprocess] Gaussian type: {type(output['gaussian'])}")
            if "glb" in output:
                print(f"[Subprocess] GLB type: {type(output['glb'])}")
        else:
            print(f"[Subprocess] Output type: {type(output)}")
        print(
            f"[Subprocess] Output keys: {output.keys() if isinstance(output, dict) else 'Not a dict'}"
        )

        # Check if output has gaussian with features
        if isinstance(output, dict) and "gaussian" in output:
            gs = output["gaussian"][0]
            print(f"[Subprocess] Gaussian properties:")
            print(f"  - Number of points: {gs.get_xyz.shape[0]}")
            print(f"  - Has _features_dc: {hasattr(gs, '_features_dc')}")
            if hasattr(gs, "_features_dc") and gs._features_dc is not None:
                print(f"    Shape: {gs._features_dc.shape}")
            print(f"  - Has _features_rest: {hasattr(gs, '_features_rest')}")
            print(f"  - Has SH features: {hasattr(gs, 'get_features')}")

        # Prepare scene using make_scene and ready_gaussian_for_video_rendering
        print(f"[Subprocess] Preparing Gaussian scene for export...")
        scene_gs = make_scene(output, in_place=False)
        scene_gs = ready_gaussian_for_video_rendering(
            scene_gs, in_place=False, fix_alignment=False
        )

        print(f"[Subprocess] Scene Gaussian after prep:")
        print(f"  - Number of points: {scene_gs.get_xyz.shape[0]}")
        print(f"  - Has _features_dc: {hasattr(scene_gs, '_features_dc')}")
        if hasattr(scene_gs, "_features_dc") and scene_gs._features_dc is not None:
            print(f"    Shape: {scene_gs._features_dc.shape}")

        # GIF 렌더링 (skip_gif=False일 때만)
        if skip_gif:
            print(f"[Subprocess] ⚡ GIF rendering SKIPPED (--skip-gif or SAM3D_SKIP_GIF=1)")
            print(f"[Subprocess] Exporting PLY only for volume calculation...")
            scene_gs.save_ply(output_ply_path)
        else:
            print(f"[Subprocess] Rendering rotating GIF with proper colors...")
            try:
                # Render video frames (rotating 360° view)
                # OPTIMIZATION: Reduced frames and resolution for faster volume calculation
                # Note: render_video returns a dict with 'color', 'depth', etc. keys
                render_output = render_video(
                    scene_gs,
                    resolution=256,   # 512 → 256 (15-25s saving)
                    bg_color=(0, 0, 0),
                    num_frames=12,    # 60 → 12 (25-35s saving)
                    r=2.0,
                    fov=40,
                    pitch_deg=0,
                    yaw_start_deg=-90,
                )

                # Extract the color frames from the output dict
                if isinstance(render_output, dict):
                    print(
                        f"[Subprocess] render_video returned dict with keys: {list(render_output.keys())}"
                    )
                    video_frames = render_output.get("color", render_output)
                else:
                    video_frames = render_output

                print(
                    f"[Subprocess] Video frames shape/type: {type(video_frames)}, len: {len(video_frames) if hasattr(video_frames, '__len__') else 'N/A'}"
                )

                # Use imageio to save directly (mimics demo_single_object.ipynb approach)
                import imageio

                gif_path = output_ply_path.replace(".ply", ".gif")
                print(f"[Subprocess] Saving GIF with imageio to {gif_path}...")

                # imageio.mimsave handles numpy array video sequences directly
                imageio.mimsave(
                    gif_path,
                    video_frames,
                    format="GIF",
                    duration=50,  # milliseconds per frame
                    loop=0,  # loop indefinitely
                )

                gif_size = os.path.getsize(gif_path)
                print(f"[Subprocess] ✓ GIF saved: {gif_size} bytes")

                # Read GIF and encode as base64 for export
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()

                print(f"[Subprocess] ✓ GIF encoded for base64 transport")

                # Encode GIF as base64 and output for API to capture
                gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
                print(f"[Subprocess] ✓ GIF encoded: {len(gif_b64)} chars (base64)")

                # Also save as PLY for compatibility (even if white)
                print(f"[Subprocess] Exporting PLY as well (for compatibility)...")
                scene_gs.save_ply(output_ply_path)

                # Output GIF data between markers for API to extract
                print("GIF_DATA_START")
                print(gif_b64)
                print("GIF_DATA_END")

            except Exception as e:
                print(
                    f"[Subprocess] ⚠ Warning: GIF rendering failed, falling back to PLY: {e}"
                )
                import traceback

                traceback.print_exc()

                # Fallback to PLY export
                print(f"[Subprocess] Exporting PLY to {output_ply_path}...")
                if hasattr(scene_gs, "save_ply"):
                    scene_gs.save_ply(output_ply_path)
                else:
                    raise ValueError("Could not find save_ply method on Gaussian scene")

        file_size = os.path.getsize(output_ply_path)
        print(f"[Subprocess] ✓ PLY saved: {file_size} bytes")

        # Generate low-poly mesh (DISABLED - causes 100% CPU spike)
        # TODO: Re-enable with optimized implementation if needed
        mesh_filename = None
        mesh_url = None
        print(f"[Subprocess] ⚠ Mesh generation skipped (CPU intensive)")

        # Old mesh generation code kept commented out for reference:
        # try:
        #     print(f"[Subprocess] Generating low-poly mesh from Gaussian...")
        #     mesh_start = time.time()
        #     mesh_size = gaussian_to_simplified_mesh(
        #         scene_gs, mesh_ply_path, depth=5, reduction=0.95
        #     )
        #     ...

        # Post-process PLY to add RGB colors from SH coefficients
        try:
            print(f"[Subprocess] Post-processing PLY to add RGB colors...")
            add_rgb_to_ply(output_ply_path)
            file_size_after = os.path.getsize(output_ply_path)
            print(f"[Subprocess] ✓ PLY post-processed: {file_size_after} bytes")
        except Exception as e:
            print(f"[Subprocess] ⚠ Warning: Could not add RGB colors to PLY: {e}")
            import traceback

            traceback.print_exc()

        # Save textured GLB mesh to assets folder (primary output)
        mesh_url = None
        glb_obj = None

        print(f"[Subprocess] Attempting to generate GLB from SAM3D pipeline...")

        # First, try to use SAM3D's native to_glb() if available
        if HAS_TO_GLB:
            try:
                print(f"[Subprocess] Using SAM3D's to_glb() for native GLB export...")

                # Extract Gaussian representation from output
                if isinstance(output, dict) and "gaussian" in output:
                    gs_list = output["gaussian"]
                    gs = gs_list[0] if isinstance(gs_list, (list, tuple)) else gs_list
                    print(
                        f"[Subprocess] DEBUG: Gaussian from output has {gs.get_xyz.shape[0]} points"
                    )

                    # Check if output has mesh data (MeshExtractResult)
                    mesh_data = None
                    if "mesh" in output:
                        mesh_data = output["mesh"]
                        print(f"[Subprocess] DEBUG: Found mesh data in output")
                    elif "slat_mesh" in output:
                        mesh_data = output["slat_mesh"]
                        print(f"[Subprocess] DEBUG: Found slat_mesh data in output")

                    if mesh_data is not None:
                        # If mesh_data is a list/tuple, pick the first mesh-like element
                        if isinstance(mesh_data, (list, tuple)):
                            selected_mesh = None
                            for i, m in enumerate(mesh_data):
                                if hasattr(m, "vertices") or hasattr(m, "triangles"):
                                    selected_mesh = m
                                    print(
                                        f"[Subprocess] DEBUG: selecting mesh_data[{i}] (has vertices/triangles) for to_glb()"
                                    )
                                    break
                                if isinstance(m, dict) and (
                                    "vertices" in m or "faces" in m or "triangles" in m
                                ):
                                    selected_mesh = m
                                    print(
                                        f"[Subprocess] DEBUG: selecting mesh_data[{i}] (dict with vertices/faces) for to_glb()"
                                    )
                                    break
                            if selected_mesh is None:
                                print(
                                    "[Subprocess] WARNING: mesh_data is a list but no mesh-like element was found; skipping to_glb()"
                                )
                                mesh_data = None
                            else:
                                mesh_data = selected_mesh

                        if mesh_data is not None:
                            print(
                                f"[Subprocess] Calling to_glb() with Gaussian and mesh..."
                            )
                            try:
                                glb_obj = to_glb(
                                    app_rep=gs,
                                    mesh=mesh_data,
                                    simplify=0.95,
                                    fill_holes=True,
                                    fill_holes_max_size=0.04,
                                    texture_size=1024,
                                    debug=False,
                                    verbose=True,
                                    with_mesh_postprocess=True,
                                    with_texture_baking=True,
                                    use_vertex_color=False,
                                    rendering_engine="nvdiffrast",
                                )
                                print(
                                    f"[Subprocess] ✓ GLB generated from to_glb(): {type(glb_obj).__name__}"
                                )
                            except AttributeError as e:
                                # Avoid crashing when mesh is not the expected type
                                print(
                                    f"[Subprocess] WARNING: to_glb() failed with AttributeError: {e}"
                                )
                                import traceback

                                traceback.print_exc()
                                glb_obj = None
                        else:
                            print(
                                f"[Subprocess] DEBUG: No mesh data found in output, cannot use to_glb()"
                            )
                        print(
                            f"[Subprocess] DEBUG: No mesh data found in output, cannot use to_glb()"
                        )
                        print(
                            f"[Subprocess] DEBUG: Available output keys: {list(output.keys())}"
                        )
                else:
                    print(f"[Subprocess] DEBUG: No gaussian in output for to_glb()")

            except Exception as e:
                print(f"[Subprocess] ⚠ to_glb() failed: {e}")
                import traceback

                traceback.print_exc()
                glb_obj = None

        # Check for native GLB from pipeline
        if glb_obj is None:
            print(f"[Subprocess] DEBUG: Checking for GLB in output dict...")
            glb_obj = None
            if isinstance(output, dict):
                glb_obj = output.get("glb", None)
                print(f"[Subprocess] DEBUG: 'glb' key present: {'glb' in output}")
                print(f"[Subprocess] DEBUG: 'glb' value type: {type(glb_obj).__name__}")
                print(f"[Subprocess] DEBUG: 'glb' is None: {glb_obj is None}")
                print(f"[Subprocess] DEBUG: All output keys: {list(output.keys())}")

                # GLB might be wrapped in a list
                if isinstance(glb_obj, (list, tuple)):
                    print(
                        f"[Subprocess] DEBUG: 'glb' is a {type(glb_obj).__name__} with {len(glb_obj)} items"
                    )
                    if len(glb_obj) > 0:
                        glb_obj = glb_obj[0]
                        print(
                            f"[Subprocess] DEBUG: Extracted first item from list: {type(glb_obj).__name__}"
                        )
                    else:
                        print(f"[Subprocess] DEBUG: 'glb' list is empty")
                        glb_obj = None
            else:
                print(f"[Subprocess] DEBUG: output is not a dict, type={type(output)}")

        if glb_obj is not None:
            try:
                import uuid
                import json
                from datetime import datetime

                mesh_filename = f"mesh_{uuid.uuid4().hex[:8]}.glb"
                mesh_assets_path = os.path.join(assets_dir, mesh_filename)

                os.makedirs(assets_dir, exist_ok=True)

                # Export GLB using native to_glb() result
                if hasattr(glb_obj, "export"):
                    print(
                        f"[Subprocess] Exporting native texture-baked GLB to {mesh_assets_path}..."
                    )
                    glb_obj.export(mesh_assets_path)
                else:
                    raise ValueError(
                        f"GLB object has no export method: {type(glb_obj).__name__}"
                    )

                # Check if file exists and save metadata
                if os.path.exists(mesh_assets_path):
                    file_size = os.path.getsize(mesh_assets_path)
                    print(f"[Subprocess] ✓ GLB mesh saved: {file_size} bytes")

                    # Save metadata
                    metadata_path = os.path.join(
                        assets_dir, f"{mesh_filename}.metadata.json"
                    )
                    with open(metadata_path, "w") as f:
                        json.dump(
                            {
                                "filename": mesh_filename,
                                "created_at": datetime.now().isoformat(),
                                "size_bytes": file_size,
                                "format": "glb",
                                "has_textures": True,
                                "method": "SAM3D_to_glb_native",
                            },
                            f,
                        )

                    mesh_url = f"/assets/{mesh_filename}"
                    print("MESH_URL_START")
                    print(mesh_url)
                    print("MESH_URL_END")
                else:
                    raise ValueError("Export reported success but file not found")

            except Exception as e:
                print(f"[Subprocess] ERROR: Could not generate GLB: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Save high-poly PLY to assets folder (secondary output for debugging)
        try:
            import uuid
            import json
            from datetime import datetime

            ply_filename = f"ply_{uuid.uuid4().hex[:8]}.ply"
            ply_assets_path = os.path.join(assets_dir, ply_filename)
            os.makedirs(assets_dir, exist_ok=True)
            import shutil

            shutil.copy(output_ply_path, ply_assets_path)

            # Save metadata file with timestamp
            metadata_filename = f"{ply_filename}.metadata.json"
            metadata_path = os.path.join(assets_dir, metadata_filename)
            metadata = {
                "filename": ply_filename,
                "created_at": datetime.now().isoformat(),
                "size_bytes": os.path.getsize(ply_assets_path),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            ply_url = f"/assets/{ply_filename}"
            print(f"[Subprocess] ✓ PLY saved to assets: {ply_url}")
            print("PLY_URL_START")
            print(ply_url)
            print("PLY_URL_END")
        except Exception as e:
            print(f"[Subprocess] ⚠ Warning: Could not save PLY to assets: {e}")

        print("SUCCESS")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
