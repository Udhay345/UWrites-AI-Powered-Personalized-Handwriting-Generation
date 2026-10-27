"""
UWrites - Handwriting Preprocessing and Segmentation Module
Provides computer vision routines for deskewing, binarization, ruled-line suppression,
and projection-profile-based line and word segmentation.
"""

import os
import cv2
import numpy as np
from PIL import Image


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Detects page orientation slant and deskews grayscale page scan using minAreaRect.
    
    Args:
        img: Grayscale image numpy array (H, W).
    Returns:
        Deskewed grayscale image array.
    """
    gray = cv2.bitwise_not(img)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # OpenCV minAreaRect angle adjustment convention
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
        
    # Ignore negligible angles (< 0.2 deg) to avoid unnecessary resampling blur
    if abs(angle) < 0.2 or abs(angle) > 45:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def binarize(img: np.ndarray, block_size: int = 25, c_offset: int = 12) -> np.ndarray:
    """
    Applies Gaussian adaptive thresholding to isolate ink strokes from uneven backgrounds.
    
    Args:
        img: Grayscale image array.
        block_size: Neighborhood block size (must be odd).
        c_offset: Constant subtracted from weighted mean.
    Returns:
        Binary image array (0 for ink, 255 for paper).
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_offset,
    )


def remove_ruled_lines(img_bin: np.ndarray) -> np.ndarray:
    """
    Detects and eliminates pre-printed notebook ruling lines using horizontal morphological kernels.
    
    Args:
        img_bin: Binary image (0=ink, 255=background).
    Returns:
        Binary image with horizontal ruling lines masked to white background (255).
    """
    inv = cv2.bitwise_not(img_bin)
    kernel_width = max(30, img_bin.shape[1] // 4)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = img_bin.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Avoid removing wide horizontal strokes of genuine letters by checking aspect ratio
        if w > img_bin.shape[1] * 0.2:
            cv2.rectangle(cleaned, (x, max(0, y - 1)), (x + w, min(img_bin.shape[0], y + h + 1)), 255, -1)
    return cleaned


def segment_lines(
    img_bin: np.ndarray,
    min_gap: int = 6,
    min_line_height: int = 12,
    noise_floor_ratio: float = 0.005,
) -> list[tuple[int, int]]:
    """
    Segments handwritten text lines using horizontal projection profiling with a dynamic noise floor.
    
    Noise-floor filtering prevents background speckles from bridging line gaps and creating
    catastrophic page-height bounding boxes.
    
    Args:
        img_bin: Binary image (0=ink, 255=background).
        min_gap: Minimum vertical blank space (in pixels) required to split adjacent lines.
        min_line_height: Minimum vertical height to consider a valid line strip.
        noise_floor_ratio: Threshold ratio of image width used to filter single-pixel speckles.
    Returns:
        List of (y_start, y_end) line strip coordinates.
    """
    proj = np.sum(img_bin == 0, axis=1)
    noise_floor = max(3, int(img_bin.shape[1] * noise_floor_ratio))
    
    lines = []
    in_line = False
    start = 0
    gap = 0
    
    for i, count in enumerate(proj):
        if count > noise_floor:
            if not in_line:
                start = i
                in_line = True
            gap = 0
        else:
            gap += 1
            if in_line and gap > min_gap:
                y0, y1 = start, i - gap
                if y1 - y0 >= min_line_height:
                    lines.append((y0, y1))
                in_line = False
                
    if in_line and (len(proj) - start) >= min_line_height:
        lines.append((start, len(proj)))
        
    return lines


def segment_words(
    line_img_bin: np.ndarray,
    gap_threshold: int = 15,
    min_word_width: int = 6,
    noise_floor: int = 2,
) -> list[tuple[int, int]]:
    """
    Segments individual words or glyph clusters along a line strip using vertical projection profiling.
    
    Args:
        line_img_bin: Binary line strip image (0=ink, 255=background).
        gap_threshold: Minimum horizontal gap (in pixels) defining a word boundary.
        min_word_width: Minimum pixel width to filter stray dots or punctuation fragments.
        noise_floor: Minimum ink pixel count required per column.
    Returns:
        List of (x_start, x_end) word boundary coordinates.
    """
    proj = np.sum(line_img_bin == 0, axis=0)
    words = []
    in_word = False
    start = 0
    gap = 0
    
    for i, count in enumerate(proj):
        if count > noise_floor:
            if not in_word:
                start = i
                in_word = True
            gap = 0
        else:
            gap += 1
            if in_word and gap > gap_threshold:
                x0, x1 = start, i - gap
                if x1 - x0 >= min_word_width:
                    words.append((x0, x1))
                in_word = False
                
    if in_word and (len(proj) - start) >= min_word_width:
        words.append((start, len(proj)))
        
    return words


def normalize_crop_with_padding(
    crop_gray: np.ndarray,
    target_size: tuple[int, int] = (256, 64),
    pad_value: int = 255,
) -> np.ndarray:
    """
    Scales a word/glyph crop into standard canvas dimensions while preserving aspect ratio.
    Prevents character squashing and geometric distortion by centering the scaled image with white margins.
    
    Args:
        crop_gray: Grayscale crop array (H, W).
        target_size: (target_width, target_height).
        pad_value: Background pixel value (255 for white).
    Returns:
        Padded grayscale image of shape (target_height, target_width).
    """
    target_w, target_h = target_size
    h, w = crop_gray.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w), pad_value, dtype=np.uint8)
        
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(crop_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def process_page(
    path: str,
    out_dir: str,
    page_id: str,
    is_ruled: bool = False,
    min_line_height: int = 12,
    min_word_width: int = 6,
) -> list[str]:
    """
    Full pipeline to load, deskew, binarize, and segment a single page scan into cropped word images.
    
    Args:
        path: Filepath to raw page image.
        out_dir: Target output directory for cropped PNG files.
        page_id: Identifier used in generated crop filenames.
        is_ruled: Whether page has printed ruling lines to suppress.
        min_line_height: Minimum height threshold for extracted lines.
        min_word_width: Minimum width threshold for extracted words.
    Returns:
        List of generated crop filenames.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"WARNING: Could not read image at {path}")
        return []
        
    img_deskewed = deskew(img)
    img_bin = binarize(img_deskewed)
    
    if is_ruled:
        img_bin = remove_ruled_lines(img_bin)
        
    os.makedirs(out_dir, exist_ok=True)
    saved_files = []
    
    lines = segment_lines(img_bin, min_line_height=min_line_height)
    for li, (y0, y1) in enumerate(lines):
        line_bin = img_bin[y0:y1, :]
        line_orig = img_deskewed[y0:y1, :]
        
        words = segment_words(line_bin, min_word_width=min_word_width)
        for wi, (x0, x1) in enumerate(words):
            crop = line_orig[:, x0:x1]
            fname = f"{page_id}_line{li:02d}_word{wi:02d}.png"
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, crop)
            saved_files.append(fname)
            
    return saved_files
