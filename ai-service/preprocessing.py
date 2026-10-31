"""
DefectZero AI — Preprocessing Module (`ai-service/preprocessing.py`)
===================================================================
Phase 3: Data Preprocessing Pipeline for Zero-Shot Industrial Defect Detection

This module implements deterministic, reproducible, and configurable image
and segmentation mask preprocessing routines tailored for Vision-Language Models
(specifically WinCLIP / CLIP architectures).

Key Design Principles:
1. Deterministic Execution: Zero random augmentations to ensure evaluation reproducibility.
2. Alignment Integrity: Exact spatial synchronization between inspection images and masks.
3. Nearest-Neighbor Mask Resizing: Masks are strictly interpolated via NEAREST to prevent
   gray boundary artifacts and guarantee binary integrity {0, 1}.
4. Non-Destructive: Never modifies, overwrites, or creates redundant copies of dataset files.
5. Configurable Normalization: Supports custom, CLIP, ImageNet, or unnormalized [0, 1] tensors.

Functions:
- load_image()
- convert_to_rgb()
- preprocess_image()
- preprocess_mask()
- load_ground_truth_mask()
"""

import os
from pathlib import Path
from typing import Union, Tuple, Optional, Sequence

import numpy as np
from PIL import Image
import torch

# Standard Normalization Constants
# CLIP default normalization parameters (LAION / OpenAI checkpoints)
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# Standard ImageNet normalization parameters
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ============================================================================
# 1. LOAD IMAGE
# ============================================================================

def load_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    keep_aspect_ratio: bool = False,
) -> Image.Image:
    """
    Loads an image from the local filesystem and returns a PIL Image.

    Args:
        image_path: Path to the target image file.
        target_size: Optional (width, height) tuple to resize the image.
                     Uses BICUBIC resampling suitable for continuous natural images.
        keep_aspect_ratio: If True and target_size is provided, pads or crops
                           while maintaining original proportions. Defaults to False.

    Returns:
        PIL.Image.Image: Loaded (and optionally resized) image in its native mode.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If file is not a valid or readable image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"[Error] Image not found at path: {path}")

    try:
        # Load image via PIL
        img = Image.open(path)
        img.load()  # Force reading byte buffer from disk
    except Exception as e:
        raise ValueError(f"[Error] Failed to read image '{path}': {e}")

    # Optional resizing
    if target_size is not None:
        if keep_aspect_ratio:
            # Thumbnail / letterbox approach
            img.thumbnail(target_size, resample=Image.Resampling.BICUBIC)
        else:
            img = img.resize(target_size, resample=Image.Resampling.BICUBIC)

    return img


# ============================================================================
# 2. CONVERT TO RGB
# ============================================================================

def convert_to_rgb(
    image: Union[Image.Image, np.ndarray, torch.Tensor]
) -> Image.Image:
    """
    Ensures an input image has exactly 3 color channels (RGB).
    
    This is critical for MVTec AD where categories such as 'grid', 'screw',
    and 'zipper' are stored as single-channel grayscale ('L') images, whereas
    pretrained Vision Transformers (ViT) mandate 3-channel RGB inputs.

    Args:
        image: Input image as a PIL Image, NumPy array, or PyTorch tensor.

    Returns:
        PIL.Image.Image: 3-channel RGB image.

    Raises:
        TypeError: If input type is not supported.
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    elif isinstance(image, np.ndarray):
        # Handle numpy arrays [H, W], [H, W, 1], [H, W, 3], or [H, W, 4]
        if image.ndim == 2:
            # Grayscale 2D -> replicate across 3 channels
            rgb_arr = np.stack([image] * 3, axis=-1)
            return Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB")
        elif image.ndim == 3:
            if image.shape[2] == 1:
                rgb_arr = np.repeat(image, 3, axis=2)
                return Image.fromarray(rgb_arr.squeeze(-1).astype(np.uint8), mode="RGB")
            elif image.shape[2] == 3:
                return Image.fromarray(image.astype(np.uint8), mode="RGB")
            elif image.shape[2] == 4:
                # RGBA -> RGB
                pil_rgba = Image.fromarray(image.astype(np.uint8), mode="RGBA")
                return pil_rgba.convert("RGB")
        raise ValueError(f"[Error] Unsupported numpy array shape for RGB conversion: {image.shape}")

    elif isinstance(image, torch.Tensor):
        # Handle tensor [C, H, W] or [H, W]
        t = image.detach().cpu()
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] == 4:
            t = t[:3]

        # Convert to numpy uint8
        arr = t.permute(1, 2, 0).numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    else:
        raise TypeError(f"[Error] Unsupported input type: {type(image)}")


# ============================================================================
# 3. PREPROCESS IMAGE
# ============================================================================

def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    target_size: Optional[Tuple[int, int]] = (224, 224),
    normalize: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    to_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Executes the standard vision-language preprocessing pipeline on an image:
    1. Loads from disk if path is provided.
    2. Converts to 3-channel RGB.
    3. Resizes to target resolution (default: 224x224 for ViT-B/16).
    4. Scales pixel values to floating-point range [0.0, 1.0].
    5. Optionally normalizes channels using mean & std (defaults to CLIP constants).
    6. Formats output as a (3, H, W) PyTorch Tensor or NumPy array.

    Args:
        image: Path string, Path object, PIL Image, or NumPy array.
        target_size: Desired (width, height) resolution. If None, preserves original dimensions.
        normalize: If True, applies channel-wise z-score normalization.
        mean: Normalization mean (3 floats). Defaults to CLIP_MEAN if normalize=True.
        std: Normalization std (3 floats). Defaults to CLIP_STD if normalize=True.
        to_tensor: If True, returns torch.FloatTensor of shape (3, H, W).
                   If False, returns float32 np.ndarray of shape (H, W, 3).

    Returns:
        torch.Tensor or np.ndarray: Formatted and preprocessed image.
    """
    # 1. Load image if path
    if isinstance(image, (str, Path)):
        img = load_image(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))
    else:
        raise TypeError(f"[Error] Invalid image input type: {type(image)}")

    # 2. Harmonize to 3-channel RGB
    img = convert_to_rgb(img)

    # 3. Resize with bicubic interpolation
    if target_size is not None:
        img = img.resize(target_size, resample=Image.Resampling.BICUBIC)

    # 4. Convert to float array in [0.0, 1.0]
    arr = np.array(img, dtype=np.float32) / 255.0

    # 5. Channel-wise normalization (if configured)
    if normalize:
        norm_mean = np.array(mean if mean is not None else CLIP_MEAN, dtype=np.float32)
        norm_std = np.array(std if std is not None else CLIP_STD, dtype=np.float32)
        arr = (arr - norm_mean) / norm_std

    # 6. Format output
    if to_tensor:
        # Reorder to (C, H, W) for PyTorch
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor

    return arr


# ============================================================================
# 4. PREPROCESS MASK
# ============================================================================

def preprocess_mask(
    mask: Union[str, Path, Image.Image, np.ndarray],
    target_size: Optional[Tuple[int, int]] = None,
    threshold: float = 0.5,
    to_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Preprocesses a ground-truth segmentation mask:
    1. Loads from disk if path is provided (in single-channel 'L' grayscale mode).
    2. Resizes to target resolution using strictly NEAREST-NEIGHBOR interpolation.
       (CRITICAL: Prevents boundary blur and maintains binary integrity).
    3. Binarizes values strictly into {0, 1} based on threshold.
    4. Formats output as a (1, H, W) PyTorch Tensor or (H, W) NumPy array.

    Args:
        mask: Path string, Path object, PIL Image, or NumPy array.
        target_size: Optional (width, height) to resize. Uses NEAREST resampling.
        threshold: Binarization cutoff in range [0, 1]. Defaults to 0.5 (or pixel value 128).
        to_tensor: If True, returns torch.FloatTensor of shape (1, H, W).
                   If False, returns uint8 / float32 np.ndarray of shape (H, W).

    Returns:
        torch.Tensor or np.ndarray: Strict binary mask {0, 1}.
    """
    # 1. Load mask if path
    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        if not mask_path.is_file():
            raise FileNotFoundError(f"[Error] Mask file not found: {mask_path}")
        mask_img = Image.open(mask_path).convert("L")
    elif isinstance(mask, Image.Image):
        mask_img = mask.convert("L")
    elif isinstance(mask, np.ndarray):
        mask_img = Image.fromarray(mask.astype(np.uint8)).convert("L")
    else:
        raise TypeError(f"[Error] Invalid mask input type: {type(mask)}")

    # 2. Resample using NEAREST NEIGHBOR ONLY
    if target_size is not None:
        mask_img = mask_img.resize(target_size, resample=Image.Resampling.NEAREST)

    # 3. Binarize to {0, 1}
    raw_arr = np.array(mask_img, dtype=np.float32)
    cutoff = threshold * 255.0 if raw_arr.max() > 1.0 else threshold
    binary_arr = (raw_arr > cutoff).astype(np.float32)

    # 4. Format output
    if to_tensor:
        # Shape: (1, H, W)
        tensor = torch.from_numpy(binary_arr).unsqueeze(0).float()
        return tensor

    return binary_arr


# ============================================================================
# 5. LOAD GROUND TRUTH MASK
# ============================================================================

def load_ground_truth_mask(
    mask_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    to_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Loads and binarizes an official MVTec AD ground-truth segmentation mask.

    Verifies file existence, ensures correct single-channel interpretation,
    and applies nearest-neighbor spatial resizing when a target_size is specified.

    Args:
        mask_path: Local filesystem path to the ground-truth mask PNG.
        target_size: Optional (width, height) tuple to resize the mask.
        to_tensor: If True, returns torch.FloatTensor of shape (1, H, W).
                   If False, returns binary NumPy array of shape (H, W).

    Returns:
        torch.Tensor or np.ndarray: Binarized ground-truth mask with values in {0.0, 1.0}.
    """
    return preprocess_mask(
        mask=mask_path,
        target_size=target_size,
        threshold=0.5,
        to_tensor=to_tensor,
    )
