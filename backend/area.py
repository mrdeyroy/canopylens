"""
backend/area.py
Canopy area calculation and coverage estimation module for CanopyLens.
Implements transparent bounding-box proxy area calculations with strict
GSD handling and explicit scientific disclaimers.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


PROXY_DISCLAIMER = (
    "Canopy area is estimated from detected crown bounding boxes. This is a proxy and "
    "may overestimate actual crown area because bounding boxes include background pixels "
    "outside the natural elliptical or irregular crown boundary."
)


def calculate_tree_areas(
    predictions_df: pd.DataFrame,
    gsd_m: Optional[float] = None
) -> pd.DataFrame:
    """
    Computes individual bounding box pixel area and real-world area in m² (if GSD is available).

    Args:
        predictions_df: DataFrame containing at minimum 'xmin', 'ymin', 'xmax', 'ymax'.
        gsd_m: Ground Sampling Distance in meters/pixel (optional).

    Returns:
        Augmented DataFrame with 'pixel_area' and optional 'estimated_area_m2'.
    """
    df = predictions_df.copy()

    if df.empty:
        df["pixel_area"] = []
        if gsd_m is not None and gsd_m > 0:
            df["estimated_area_m2"] = []
        return df

    # Ensure numeric coordinates
    xmin = df["xmin"].astype(float)
    ymin = df["ymin"].astype(float)
    xmax = df["xmax"].astype(float)
    ymax = df["ymax"].astype(float)

    pixel_width = (xmax - xmin).clip(lower=0)
    pixel_height = (ymax - ymin).clip(lower=0)
    df["pixel_width"] = pixel_width
    df["pixel_height"] = pixel_height
    df["pixel_area"] = pixel_width * pixel_height

    if gsd_m is not None and gsd_m > 0:
        pixel_area_m2 = gsd_m ** 2
        df["estimated_area_m2"] = df["pixel_area"] * pixel_area_m2
    else:
        df["estimated_area_m2"] = None

    return df


def calculate_canopy_metrics(
    predictions_df: pd.DataFrame,
    gsd_m: Optional[float] = None,
    aoi_area_m2: Optional[float] = None,
    spatial_alignment_valid: bool = False
) -> Dict[str, Any]:
    """
    Aggregates tree count, canopy area proxy, coverage, and confidence metrics.

    Returns:
        dict containing metrics, availability flags, and honest explanatory messages.
    """
    df_with_areas = calculate_tree_areas(predictions_df, gsd_m=gsd_m)
    num_trees = len(df_with_areas)

    # Mean confidence
    if "score" in df_with_areas.columns and num_trees > 0:
        mean_confidence = float(df_with_areas["score"].mean() * 100.0)
    elif "confidence" in df_with_areas.columns and num_trees > 0:
        mean_confidence = float(df_with_areas["confidence"].mean() * 100.0)
    else:
        mean_confidence = 0.0

    total_pixel_area = float(df_with_areas["pixel_area"].sum()) if num_trees > 0 else 0.0
    avg_pixel_area = float(total_pixel_area / num_trees) if num_trees > 0 else 0.0

    # Real-world metric assessment
    has_real_world_area = (gsd_m is not None and gsd_m > 0 and num_trees > 0)
    if has_real_world_area:
        total_canopy_m2 = float(df_with_areas["estimated_area_m2"].sum())
        avg_crown_m2 = float(total_canopy_m2 / num_trees)
        area_status_message = (
            f"Calculated using Ground Sampling Distance (GSD): {gsd_m:.4f} m/pixel."
        )
    else:
        total_canopy_m2 = None
        avg_crown_m2 = None
        area_status_message = (
            "Real-world canopy area cannot be reliably calculated because image ground "
            "resolution / GSD is unavailable."
        )

    # Canopy coverage calculation
    canopy_coverage_pct = None
    coverage_status_message = ""

    if aoi_area_m2 is not None and aoi_area_m2 > 0:
        if total_canopy_m2 is not None:
            if spatial_alignment_valid:
                canopy_coverage_pct = float((total_canopy_m2 / aoi_area_m2) * 100.0)
                coverage_status_message = f"Canopy coverage over verified AOI: {canopy_coverage_pct:.2f}%"
            else:
                # If alignment cannot be established, do not fabricate an authoritative coverage figure
                coverage_status_message = (
                    "Canopy coverage cannot be reliably calculated because pixel-to-boundary "
                    "spatial alignment between the image and KML is unverified."
                )
        else:
            coverage_status_message = (
                "Canopy coverage cannot be calculated because real-world canopy area is unavailable."
            )
    else:
        coverage_status_message = "Area of Interest (AOI) boundary not provided."

    return {
        "tree_count": num_trees,
        "mean_confidence": mean_confidence,
        "total_pixel_area": total_pixel_area,
        "avg_pixel_area": avg_pixel_area,
        "total_canopy_m2": total_canopy_m2,
        "avg_crown_m2": avg_crown_m2,
        "gsd_m": gsd_m,
        "has_real_world_area": has_real_world_area,
        "area_status_message": area_status_message,
        "canopy_coverage_pct": canopy_coverage_pct,
        "coverage_status_message": coverage_status_message,
        "disclaimer": PROXY_DISCLAIMER,
        "augmented_df": df_with_areas
    }
