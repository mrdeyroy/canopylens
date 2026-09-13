"""
backend/detection.py
Reusable tree-crown detection and visualization module using DeepForest 2.1.0.
Integrates model caching, robust image loading, and clean OpenCV annotation.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Any, Union
import cv2
import numpy as np
import pandas as pd

# Global cached model instance for non-Streamlit environments
_CACHED_MODEL = None


def load_model():
    """
    Loads and caches the pretrained DeepForest model.
    Compatible with DeepForest 2.1.0 and earlier release versions.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    from deepforest import main as df_main
    model = df_main.deepforest()
    if hasattr(model, "use_release"):
        model.use_release()
    _CACHED_MODEL = model
    return _CACHED_MODEL


def detect_tree_crowns(
    image_input: Union[str, Path, np.ndarray],
    model: Optional[Any] = None,
    min_confidence: Optional[float] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Runs individual tree-crown detection on the input image.

    Args:
        image_input: Path to image file, or NumPy array (RGB or BGR).
        model: Pre-loaded DeepForest model instance. If None, loads cached model.

    Returns:
        Tuple of (predictions_df, image_bgr)
        predictions_df columns: ['tree_id', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'label']
    """
    if model is None:
        model = load_model()

    # Handle image input
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at '{image_path}'")
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"OpenCV could not read image from '{image_path}'. Ensure format is supported.")
        predictions = model.predict_image(path=str(image_path))
    elif isinstance(image_input, np.ndarray):
        # Assume input numpy array is RGB if passed from PIL / Streamlit
        if len(image_input.shape) == 2:
            image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        else:
            image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        # DeepForest predict_image accepts image array directly (expects RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictions = model.predict_image(image=image_rgb)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Standardize predictions DataFrame
    if predictions is None or len(predictions) == 0:
        clean_df = pd.DataFrame(columns=[
            "tree_id", "xmin", "ymin", "xmax", "ymax", "confidence", "label"
        ])
        return clean_df, image_bgr

    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    # Standardize column naming
    standardized = pd.DataFrame()
    standardized["tree_id"] = [f"T{i+1:03d}" for i in range(len(predictions))]
    standardized["xmin"] = predictions["xmin"].astype(float)
    standardized["ymin"] = predictions["ymin"].astype(float)
    standardized["xmax"] = predictions["xmax"].astype(float)
    standardized["ymax"] = predictions["ymax"].astype(float)

    if "score" in predictions.columns:
        standardized["confidence"] = predictions["score"].astype(float)
    elif "confidence" in predictions.columns:
        standardized["confidence"] = predictions["confidence"].astype(float)
    else:
        standardized["confidence"] = 1.0

    if "label" in predictions.columns:
        standardized["label"] = predictions["label"].astype(str)
    else:
        standardized["label"] = "Tree"

    # Optional confidence threshold filtering (preserves default behavior if None)
    if min_confidence is not None and min_confidence > 0:
        standardized = standardized[standardized["confidence"] >= min_confidence].reset_index(drop=True)
        standardized["tree_id"] = [f"T{i+1:03d}" for i in range(len(standardized))]

    return standardized, image_bgr


def draw_annotated_image(
    image_bgr: np.ndarray,
    predictions_df: pd.DataFrame,
    box_color: Tuple[int, int, int] = (0, 230, 115),  # Fresh emerald green (BGR)
    highlight_color: Tuple[int, int, int] = (0, 215, 255),  # High-visibility Gold/Yellow (BGR)
    show_labels: bool = True,
    show_tree_ids: bool = False,
    selected_tree_id: Optional[str] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Renders non-intrusive bounding boxes and confidence tags on the image.
    If selected_tree_id is specified, prominently highlights that individual tree.

    Returns:
        Annotated image in BGR format.
    """
    annotated = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # First draw all non-selected trees
    selected_row = None
    for _, row in predictions_df.iterrows():
        try:
            tid = str(row["tree_id"]) if "tree_id" in row else ""
            if selected_tree_id and tid == selected_tree_id:
                selected_row = row
                continue  # Draw selected tree on top later

            xmin = int(round(float(row["xmin"])))
            ymin = int(round(float(row["ymin"])))
            xmax = int(round(float(row["xmax"])))
            ymax = int(round(float(row["ymax"])))
            conf = float(row["confidence"]) if "confidence" in row else None

            # Draw crown box
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), box_color, thickness)

            if show_labels:
                if show_tree_ids and tid:
                    tag = f"{tid} ({conf:.2f})" if conf is not None else tid
                else:
                    tag = f"{conf:.2f}" if conf is not None else "tree"

                (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, font_thickness)
                label_y = max(ymin, th + baseline + 4)

                # Small label badge
                cv2.rectangle(
                    annotated,
                    (xmin, label_y - th - 4),
                    (xmin + tw + 4, label_y + baseline),
                    box_color,
                    -1
                )
                cv2.putText(
                    annotated,
                    tag,
                    (xmin + 2, label_y - 2),
                    font,
                    font_scale,
                    (10, 10, 10),
                    font_thickness,
                    cv2.LINE_AA
                )
        except Exception:
            continue

    # Draw highlighted selected tree on top
    if selected_row is not None:
        try:
            s_xmin = int(round(float(selected_row["xmin"])))
            s_ymin = int(round(float(selected_row["ymin"])))
            s_xmax = int(round(float(selected_row["xmax"])))
            s_ymax = int(round(float(selected_row["ymax"])))
            s_conf = float(selected_row["confidence"]) if "confidence" in selected_row else None
            s_tid = str(selected_row["tree_id"]) if "tree_id" in selected_row else ""

            # Bold highlighted bounding box
            hl_thickness = max(thickness + 2, 3)
            cv2.rectangle(annotated, (s_xmin, s_ymin), (s_xmax, s_ymax), highlight_color, hl_thickness)

            # Prominent badge
            s_tag = f"SELECTED: {s_tid} ({s_conf:.2f})" if s_conf is not None else f"SELECTED: {s_tid}"
            s_font_scale = 0.5
            (stw, sth), s_baseline = cv2.getTextSize(s_tag, font, s_font_scale, 2)
            s_label_y = max(s_ymin, sth + s_baseline + 6)

            cv2.rectangle(
                annotated,
                (s_xmin, s_label_y - sth - 6),
                (s_xmin + stw + 6, s_label_y + s_baseline),
                highlight_color,
                -1
            )
            cv2.putText(
                annotated,
                s_tag,
                (s_xmin + 3, s_label_y - 3),
                font,
                s_font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
        except Exception:
            pass

    return annotated


def extract_tree_crop(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: int = 12
) -> np.ndarray:
    """
    Extracts a high-resolution close-up cropped image of a single detected tree crown.

    Args:
        image: Image array (H, W, C).
        bbox: (xmin, ymin, xmax, ymax) in pixels.
        padding: Pixel padding around the bounding box.

    Returns:
        Cropped image slice.
    """
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    pad_xmin = max(0, int(round(xmin)) - padding)
    pad_ymin = max(0, int(round(ymin)) - padding)
    pad_xmax = min(w, int(round(xmax)) + padding)
    pad_ymax = min(h, int(round(ymax)) + padding)

    if pad_xmax > pad_xmin and pad_ymax > pad_ymin:
        return image[pad_ymin:pad_ymax, pad_xmin:pad_xmax]
    else:
        return image
