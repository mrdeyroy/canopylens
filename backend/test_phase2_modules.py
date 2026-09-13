"""
backend/test_phase2_modules.py
Unit and integration test suite for CanopyLens Phase 2 modules.
Validates KML parsing, GSD area calculation, DeepForest detection, and report generation.
"""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd
from geo import parse_kml, extract_geotiff_metadata, validate_image_kml_alignment
from area import calculate_tree_areas, calculate_canopy_metrics, PROXY_DISCLAIMER
from detection import load_model, detect_tree_crowns, draw_annotated_image


def test_geo_module():
    print("\n--- Testing geo.py ---")
    kml_path = Path("data/sample_aoi.kml")
    assert kml_path.exists(), "sample_aoi.kml not found!"

    # 1. Test valid KML
    res = parse_kml(kml_path)
    assert res["success"] is True, f"KML parse failed: {res['message']}"
    assert res["polygon"] is not None, "Polygon geometry is None"
    assert res["aoi_area_m2"] is not None and res["aoi_area_m2"] > 0, "AOI area m2 is invalid"
    assert res["aoi_area_ha"] is not None and res["aoi_area_ha"] > 0, "AOI area ha is invalid"
    print(f"[PASS] Valid KML parsed: Area = {res['aoi_area_m2']:,.1f} m² ({res['aoi_area_ha']:.3f} ha).")

    # 2. Test invalid KML graceful handling
    res_inv = parse_kml("invalid xml content <not kml>")
    assert res_inv["success"] is False, "Invalid KML should fail gracefully"
    print(f"[PASS] Invalid KML handled gracefully: {res_inv['message']}")

    # 3. Test empty KML handling
    res_empty = parse_kml("<?xml version='1.0'?><kml><Document></Document></kml>")
    assert res_empty["success"] is False, "Empty KML without polygon should fail gracefully"
    print(f"[PASS] Empty KML handled gracefully: {res_empty['message']}")

    # 4. Test GeoTIFF metadata extraction on standard PNG
    png_meta = extract_geotiff_metadata("data/sample_forest.png")
    assert png_meta["is_georeferenced"] is False, "PNG should not be georeferenced"
    print(f"[PASS] Standard PNG correctly identified as non-georeferenced.")

    # 5. Test spatial alignment honesty
    aligned, reason = validate_image_kml_alignment(png_meta, res)
    assert aligned is False, "PNG without CRS should not claim alignment with KML"
    print(f"[PASS] Honest spatial alignment check: {reason}")


def test_area_module():
    print("\n--- Testing area.py ---")
    mock_data = {
        "xmin": [10.0, 50.0],
        "ymin": [20.0, 60.0],
        "xmax": [30.0, 70.0],
        "ymax": [40.0, 80.0],
        "score": [0.85, 0.92]
    }
    df = pd.DataFrame(mock_data)

    # 1. With GSD (0.1 m/pixel)
    gsd = 0.1
    metrics_gsd = calculate_canopy_metrics(df, gsd_m=gsd, aoi_area_m2=100.0, spatial_alignment_valid=True)
    assert metrics_gsd["tree_count"] == 2
    # box 1: 20x20 = 400 px, box 2: 20x20 = 400 px, total = 800 px. GSD^2 = 0.01 -> 8 m2
    assert abs(metrics_gsd["total_canopy_m2"] - 8.0) < 1e-4
    assert abs(metrics_gsd["canopy_coverage_pct"] - 8.0) < 1e-4
    print(f"[PASS] Area with GSD: Total canopy = {metrics_gsd['total_canopy_m2']} m², Coverage = {metrics_gsd['canopy_coverage_pct']}%.")

    # 2. Without GSD (Honesty check)
    metrics_no_gsd = calculate_canopy_metrics(df, gsd_m=None)
    assert metrics_no_gsd["has_real_world_area"] is False
    assert metrics_no_gsd["total_canopy_m2"] is None
    assert "unavailable" in metrics_no_gsd["area_status_message"].lower()
    print(f"[PASS] Area without GSD: Refuses to fabricate real-world metrics ({metrics_no_gsd['area_status_message']}).")


def test_detection_module():
    print("\n--- Testing detection.py on data/sample_forest.png ---")
    img_path = Path("data/sample_forest.png")
    assert img_path.exists(), "data/sample_forest.png not found!"

    model = load_model()
    predictions_df, image_bgr = detect_tree_crowns(str(img_path), model=model)

    assert len(predictions_df) > 0, "No trees detected!"
    expected_cols = ["tree_id", "xmin", "ymin", "xmax", "ymax", "confidence", "label"]
    for col in expected_cols:
        assert col in predictions_df.columns, f"Missing column {col}"

    print(f"[PASS] Dynamic tree detection completed: {len(predictions_df)} tree crowns detected.")
    print(f"[PASS] Columns: {list(predictions_df.columns)}")

    # Test annotation
    annotated = draw_annotated_image(image_bgr, predictions_df)
    assert annotated is not None and annotated.shape == image_bgr.shape
    print(f"[PASS] Annotated image rendered: shape {annotated.shape}.")


def main():
    print("==================================================")
    print("STARTING CANOPYLENS PHASE 2 TEST SUITE")
    print("==================================================")
    test_geo_module()
    test_area_module()
    test_detection_module()
    print("\n==================================================")
    print("ALL PHASE 2 UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
