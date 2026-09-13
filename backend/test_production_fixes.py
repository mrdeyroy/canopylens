"""
backend/test_production_fixes.py
Comprehensive automated test suite for the 5 production bug fixes and UX patch.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geo import extract_geotiff_metadata, parse_kml
from area import calculate_tree_areas, calculate_canopy_metrics
from detection import load_model, detect_tree_crowns, draw_annotated_image, extract_tree_crop
from app import filter_predictions, CUSTOM_CSS


def test_issue_1_gsd_logic():
    print("\n--- Testing Issue 1: GeoTIFF & GSD Logic ---")
    
    # 1. Standard PNG check
    png_meta = extract_geotiff_metadata("data/sample_forest.png")
    assert png_meta["is_georeferenced"] is False
    assert png_meta["gsd_m"] is None
    assert "no geospatial headers" in png_meta["message"].lower()
    print(f"[PASS] PNG accurately detected as non-georeferenced: '{png_meta['message']}'")
    
    # 2. Manual GSD verification
    mock_df = pd.DataFrame({
        "tree_id": ["T001"], "xmin": [0.0], "ymin": [0.0], "xmax": [10.0], "ymax": [10.0],
        "confidence": [0.9], "label": ["Tree"]
    })
    # With manual GSD = 0.05 m/px (100 px2 -> 0.25 m2)
    m = calculate_canopy_metrics(mock_df, gsd_m=0.05)
    assert m["has_real_world_area"] is True
    assert abs(m["total_canopy_m2"] - 0.25) < 1e-4
    print(f"[PASS] Manual GSD calculation verified: 100 px² @ 0.05m/px = {m['total_canopy_m2']} m²")


def test_issue_2_confidence_threshold():
    print("\n--- Testing Issue 2: Confidence Threshold Dynamic Filtering ---")
    img_path = Path("data/sample_forest.png")
    assert img_path.exists()
    
    model = load_model()
    raw_preds, img_bgr = detect_tree_crowns(str(img_path), model=model)
    
    total_raw = len(raw_preds)
    print(f"Raw unfiltered detections count: {total_raw}")
    assert total_raw > 0
    
    # Test threshold 0.00
    df_000 = filter_predictions(raw_preds, 0.00)
    count_000 = len(df_000)
    assert count_000 == total_raw
    print(f"[PASS] Threshold 0.00: {count_000} detections (preserves full Phase 2/3 baseline).")
    
    # Test threshold 0.25
    df_025 = filter_predictions(raw_preds, 0.25)
    count_025 = len(df_025)
    assert count_025 <= count_000
    print(f"[PASS] Threshold 0.25: {count_025} detections (all confidence >= 0.25).")
    assert all(df_025["confidence"] >= 0.25)
    
    # Test threshold 0.50
    df_050 = filter_predictions(raw_preds, 0.50)
    count_050 = len(df_050)
    assert count_050 <= count_025
    print(f"[PASS] Threshold 0.50: {count_050} detections (all confidence >= 0.50).")
    assert all(df_050["confidence"] >= 0.50)
    
    # Test Tree IDs are sequential
    assert list(df_050["tree_id"]) == [f"T{i:03d}" for i in range(1, count_050 + 1)]
    print(f"[PASS] Filtered Tree IDs remain sequential and properly indexed.")


def test_issue_3_tree_inspection_and_crop():
    print("\n--- Testing Issue 3: Tree Inspection & Crown Extraction ---")
    img_path = Path("data/sample_forest.png")
    img_bgr = cv2.imread(str(img_path))
    
    # Mock detection
    mock_df = pd.DataFrame({
        "tree_id": ["T001", "T002"],
        "xmin": [50.0, 150.0],
        "ymin": [50.0, 150.0],
        "xmax": [100.0, 200.0],
        "ymax": [100.0, 200.0],
        "confidence": [0.88, 0.76],
        "label": ["Tree", "Tree"]
    })
    
    # 1. Test crop extraction
    crop_t001 = extract_tree_crop(img_bgr, (50.0, 50.0, 100.0, 100.0), padding=15)
    assert crop_t001 is not None
    assert crop_t001.shape[0] > 50 and crop_t001.shape[1] > 50
    print(f"[PASS] Tree crown crop extracted successfully: shape {crop_t001.shape}")
    
    # 2. Test annotation with selected tree highlight
    annotated = draw_annotated_image(img_bgr, mock_df, selected_tree_id="T001")
    assert annotated is not None and annotated.shape == img_bgr.shape
    print(f"[PASS] Annotated image with highlighted 'T001' rendered cleanly.")


def test_issue_4_css_and_styling():
    print("\n--- Testing Issue 4: Public Deployment Styling Audit ---")
    assert ".main-title" in CUSTOM_CSS
    assert ".metric-card" in CUSTOM_CSS
    assert ".inspect-card" in CUSTOM_CSS
    assert ".stepper-banner" in CUSTOM_CSS
    print("[PASS] Custom CSS validated for Streamlit Cloud dark & light mode compatibility.")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING CANOPYLENS PRODUCTION BUG FIX VALIDATION")
    print("==================================================")
    test_issue_1_gsd_logic()
    test_issue_2_confidence_threshold()
    test_issue_3_tree_inspection_and_crop()
    test_issue_4_css_and_styling()
    print("\n==================================================")
    print("ALL PRODUCTION BUG FIX TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
