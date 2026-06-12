"""
Image Preprocessing, Heatmap Generation, Evidence Grid
"""
import cv2
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_image(image_path, output_dir=None):
    """
    Preprocess image for detection:
    - Resize if too large (keeps aspect ratio, max 1280px)
    - Normalize histogram for low-light images
    Returns path to preprocessed image (same as input if no changes needed).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    max_dim = 1280
    modified = False

    # Resize if too large
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        modified = True
        logger.info(f"Resized image from {w}x{h} to {img.shape[1]}x{img.shape[0]}")

    # Enhance dark images (CLAHE on L channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    mean_brightness = l_ch.mean()
    if mean_brightness < 80:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_ch)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        modified = True
        logger.info(f"Applied CLAHE (mean brightness was {mean_brightness:.1f})")

    if modified and output_dir:
        stem = Path(image_path).stem
        out_path = os.path.join(output_dir, f"pre_{stem}.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return out_path

    return image_path


def save_annotated_image(annotated_image, results_folder, original_filename):
    """Save the annotated image to results folder."""
    os.makedirs(results_folder, exist_ok=True)
    stem = Path(original_filename).stem
    out_path = os.path.join(results_folder, f"annotated_{stem}.jpg")
    cv2.imwrite(out_path, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info(f"Saved annotated image: {out_path}")
    return out_path


def generate_heatmap(image_path, detections, results_folder):
    """
    Generate a Gaussian heatmap overlaid on the image showing
    evidence concentration / hot zones.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    severity_weight = {'critical': 1.0, 'high': 0.75, 'medium': 0.5, 'low': 0.25}

    for d in detections:
        cx, cy = d['center']
        cx = max(0, min(cx, w-1))
        cy = max(0, min(cy, h-1))
        weight = severity_weight.get(d['severity'], 0.3)

        x1, y1, x2, y2 = d['bbox']
        radius = max(30, int(max(x2-x1, y2-y1) * 0.6))

        # Gaussian blob
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        gauss = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * (radius**2)))
        heat += gauss * weight

    if heat.max() > 0:
        heat = heat / heat.max()

    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.55, heat_color, 0.45, 0)

    os.makedirs(results_folder, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(results_folder, f"heatmap_{stem}.jpg")
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def generate_evidence_grid(image_path, detections, results_folder):
    """
    Generate a grid of cropped evidence items with labels.
    """
    img = cv2.imread(str(image_path))
    if img is None or not detections:
        return None

    h, w = img.shape[:2]
    thumb_size = 160
    padding = 6
    cols = min(4, len(detections))
    rows = (len(detections) + cols - 1) // cols

    severity_colors = {
        'critical': (0, 0, 200),
        'high':     (0, 60, 220),
        'medium':   (0, 140, 255),
        'low':      (0, 200, 200),
    }

    grid_w = cols * (thumb_size + padding) + padding
    grid_h = rows * (thumb_size + padding + 22) + padding
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 20  # dark background

    for idx, d in enumerate(detections):
        row = idx // cols
        col = idx % cols
        ox = padding + col * (thumb_size + padding)
        oy = padding + row * (thumb_size + padding + 22)

        x1, y1, x2, y2 = d['bbox']
        # Clamp
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)

        if x2c > x1c and y2c > y1c:
            crop = img[y1c:y2c, x1c:x2c]
            thumb = cv2.resize(crop, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        else:
            thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

        color = severity_colors.get(d['severity'], (128, 128, 128))
        cv2.rectangle(thumb, (0, 0), (thumb_size-1, thumb_size-1), color, 3)
        grid[oy:oy+thumb_size, ox:ox+thumb_size] = thumb

        # Label bar below thumb
        label_text = f"{d['label']} {d['confidence']}%"
        cv2.rectangle(grid, (ox, oy+thumb_size), (ox+thumb_size, oy+thumb_size+22), color, -1)
        cv2.putText(grid, label_text[:20],
                    (ox+3, oy+thumb_size+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(results_folder, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(results_folder, f"grid_{stem}.jpg")
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
