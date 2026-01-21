"""
Segmentation Routes

/segment, /segment-binary endpoints
"""

import io
import base64
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import device
from api.models import SegmentRequest, SegmentBinaryRequest
from api.services.sam2 import get_model, get_processor

router = APIRouter()


@router.post("/segment")
async def segment_image(request: SegmentRequest):
    """
    Segment an object in an image based on a point coordinate.

    Returns:
        - masks: The segmentation masks as arrays
        - scores: Quality scores for each mask
        - input_point: The input point coordinate
        - image_shape: Dimensions of the input image
    """
    model = get_model()
    processor = get_processor()

    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=500, content={"error": "Model not initialized"}
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare input points and labels
        input_points = [[[[request.x, request.y]]]]
        input_labels = [[[1]]]

        # Process inputs
        inputs = processor(
            images=image_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process masks
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        # Convert masks to list and get scores
        mask_list = []
        scores = (
            outputs.iou_preds[0].cpu().numpy().tolist()
            if hasattr(outputs, "iou_preds")
            else [0.95] * masks.shape[0]
        )

        for i in range(masks.shape[0]):
            mask = masks[i].numpy()
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                mask = mask[0] if mask.ndim > 2 else mask

            # Threshold mask
            mask = (mask > request.mask_threshold).astype(np.uint8) * 255

            # Apply morphological smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = (mask > 127).astype(np.uint8) * 255

            # Invert if requested
            if request.invert_mask:
                mask = 255 - mask

            mask_image = Image.fromarray(mask, mode="L")
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            buffer.seek(0)
            mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mask_list.append({
                "mask": mask_base64,
                "mask_shape": mask.shape,
                "score": float(scores[i]) if i < len(scores) else 0.95,
            })

        return JSONResponse({
            "success": True,
            "masks": mask_list,
            "input_point": [request.x, request.y],
            "image_shape": [image_pil.height, image_pil.width],
        })

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/segment-binary")
async def segment_image_binary(request: SegmentBinaryRequest):
    """
    Segment an image and return the mask as base64 encoded PNG.
    """
    model = get_model()
    processor = get_processor()

    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=500, content={"error": "Model not initialized"}
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Validate points
        if not request.points or len(request.points) == 0:
            return JSONResponse(
                status_code=400, content={"error": "At least one point required"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_pil_array = np.array(image_pil)

        # Collect masks from each point
        all_masks = []
        scores = None
        best_mask_idx = 0

        for point in request.points:
            input_points = [[[[point["x"], point["y"]]]]]
            input_labels = [[[1]]]

            inputs = processor(
                images=image_pil,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]

            scores = (
                outputs.iou_preds[0].cpu().numpy()
                if hasattr(outputs, "iou_preds")
                else np.array([0.95] * masks.shape[0])
            )

            best_mask_idx = np.argmax(scores)
            point_mask = masks[best_mask_idx].numpy()
            point_mask = np.squeeze(point_mask)
            if point_mask.ndim != 2:
                point_mask = point_mask[0] if point_mask.ndim > 2 else point_mask

            point_mask = (point_mask > request.mask_threshold).astype(np.uint8) * 255
            all_masks.append(point_mask)

        # Union all masks
        mask = all_masks[0].copy()
        for i in range(1, len(all_masks)):
            mask = np.maximum(mask, all_masks[i])

        # Add previous mask
        if request.previous_mask:
            try:
                mask_data = base64.b64decode(request.previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_array = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_array)
            except Exception:
                pass

        mask = (mask > request.mask_threshold).astype(np.uint8) * 255

        # Apply morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = (mask > 127).astype(np.uint8) * 255

        # Check if mask is inverted
        if mask.mean() > 127:
            mask = 255 - mask

        # Verify dimensions match
        if image_pil_array.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image_pil_array.shape[1], image_pil_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply mask
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        masked_image = (image_pil_array.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Convert to PNG
        masked_image_pil = Image.fromarray(masked_image, mode="RGB")
        buffer = io.BytesIO()
        masked_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        score = float(scores[best_mask_idx]) if scores is not None else 0.95

        return JSONResponse({
            "success": True,
            "mask": mask_base64,
            "score": score,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})
