"""
backend/test_phase3_edge_cases.py
Comprehensive Phase 3 test suite validating all deployment edge cases:
- Case A: Image + GSD
- Case B: Image without GSD
- Case C: Image + KML without georeferencing
- Case D: Invalid KML XML and empty geometries
- Case E: Zero detections handling (empty DataFrame)
- Case F: Export generation
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
import cv2

from geo import parse_kml, extract_geotiff_metadata, validate_image_kml_alignment
from area import calculate_tree_areas, calculate_canopy_metrics, PROXY_DISCLAIMER
from detection import detect_tree_crowns, draw_annotated_image
from app import generate_summary_report


def test_case_a_image_with_gsd():
    print("\n--- Test Case A: Image + GSD ---")
    mock_df = pd.DataFrame({
        "tree_id": ["T001", "T002"],
        "xmin": [10.0, 50.0],
        "ymin": [20.0, 60.0],
        "xmax": [30.0, 70.0],
        "ymax": [40.0, 80.0],
        "confidence": [0.85, 0.90],
        "label": ["Tree", "Tree"]
    })
    gsd = 0.10  # 10 cm/px
    metrics = calculate_canopy_metrics(mock_df, gsd_m=gsd, aoi_area_m2=1000.0, spatial_alignment_valid=True)
    
    assert metrics["has_real_world_area"] is True
    assert metrics["total_canopy_m2"] is not None
    # Box 1: 20x20 = 400px * 0.01 = 4.0 m2, Box 2: 20x20 = 400px * 0.01 = 4.0 m2 -> 8.0 m2
    assert abs(metrics["total_canopy_m2"] - 8.0) < 1e-4
    assert abs(metrics["canopy_coverage_pct"] - 0.8) < 1e-4
    print(f"[PASS] Case A Verified: Total canopy = {metrics['total_canopy_m2']:.2f} m², Coverage = {metrics['canopy_coverage_pct']:.2f}%.")


def test_case_b_image_without_gsd():
    print("\n--- Test Case B: Image without GSD (Honesty Mode) ---")
    mock_df = pd.DataFrame({
        "tree_id": ["T001"],
        "xmin": [10.0],
        "ymin": [20.0],
        "xmax": [30.0],
        "ymax": [40.0],
        "confidence": [0.85],
        "label": ["Tree"]
    })
    metrics = calculate_canopy_metrics(mock_df, gsd_m=None)
    assert metrics["has_real_world_area"] is False
    assert metrics["total_canopy_m2"] is None
    assert "unavailable" in metrics["area_status_message"].lower()
    assert metrics["total_pixel_area"] == 400.0
    print(f"[PASS] Case B Verified: Refuses to fabricate m² metrics. Retains pixel area = {metrics['total_pixel_area']} px².")


def test_case_c_unreferenced_image_with_kml():
    print("\n--- Test Case C: Standard Image + KML (Spatial Honesty) ---")
    kml_res = parse_kml("data/sample_aoi.kml")
    png_meta = extract_geotiff_metadata("data/sample_forest.png")
    
    aligned, reason = validate_image_kml_alignment(png_meta, kml_res)
    assert aligned is False
    assert "guarantee" in reason.lower() or "sufficient" in reason.lower()
    
    # Check that canopy metrics gracefully report unverified alignment
    mock_df = pd.DataFrame({
        "tree_id": ["T001"],
        "xmin": [0.0], "ymin": [0.0], "xmax": [10.0], "ymax": [10.0],
        "confidence": [0.9], "label": ["Tree"]
    })
    metrics = calculate_canopy_metrics(
        mock_df,
        gsd_m=0.1,
        aoi_area_m2=kml_res["aoi_area_m2"],
        spatial_alignment_valid=aligned
    )
    assert metrics["canopy_coverage_pct"] is None
    assert "unverified" in metrics["coverage_status_message"].lower()
    print(f"[PASS] Case C Verified: KML area calculated independently ({kml_res['aoi_area_ha']:.3f} ha), but coverage refused due to unverified alignment.")


def test_case_d_invalid_kml():
    print("\n--- Test Case D: Invalid & Malformed KML ---")
    res1 = parse_kml("completely invalid xml <<<")
    assert res1["success"] is False
    assert "failed" in res1["message"].lower()
    
    res2 = parse_kml("<?xml version='1.0'?><kml><Document><Placemark><name>No Geom</name></Placemark></Document></kml>")
    assert res2["success"] is False
    assert "no valid polygon" in res2["message"].lower()
    print(f"[PASS] Case D Verified: Malformed and empty KML structures handled cleanly.")


def test_case_e_zero_detections():
    print("\n--- Test Case E: Zero Detections (Empty DataFrame) ---")
    empty_df = pd.DataFrame(columns=["tree_id", "xmin", "ymin", "xmax", "ymax", "confidence", "label"])
    metrics = calculate_canopy_metrics(empty_df, gsd_m=0.1)
    
    assert metrics["tree_count"] == 0
    assert metrics["total_pixel_area"] == 0.0
    assert metrics["mean_confidence"] == 0.0
    assert metrics["total_canopy_m2"] == 0.0 or metrics["total_canopy_m2"] is None
    
    # Test annotation on empty df
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_annotated_image(blank_img, metrics["augmented_df"])
    assert annotated.shape == (100, 100, 3)
    print(f"[PASS] Case E Verified: Zero detections handled with zero division errors.")


def test_case_f_export_generation():
    print("\n--- Test Case F: Report & Export Generation ---")
    mock_df = pd.DataFrame({
        "tree_id": ["T001"],
        "xmin": [0.0], "ymin": [0.0], "xmax": [10.0], "ymax": [10.0],
        "confidence": [0.88], "label": ["Tree"],
        "pixel_area": [100.0], "estimated_area_m2": [1.0]
    })
    metrics = calculate_canopy_metrics(mock_df, gsd_m=0.1)
    report = generate_summary_report(
        image_name="test_image.png",
        kml_name="test_aoi.kml",
        metrics=metrics,
        kml_data={"aoi_area_m2": 5000.0, "aoi_area_ha": 0.5},
        gsd_m=0.1
    )
    assert "CANOPYLENS ANALYSIS REPORT" in report
    assert "Trees Detected:" in report
    assert "SCIENTIFIC LIMITATIONS" in report
    print("[PASS] Case F Verified: Scientific report text generated properly.")


def main():
    print("==================================================")
    print("STARTING CANOPYLENS PHASE 3 EDGE-CASE TEST SUITE")
    print("==================================================")
    test_case_a_image_with_gsd()
    test_case_b_image_without_gsd()
    test_case_c_unreferenced_image_with_kml()
    test_case_d_invalid_kml()
    test_case_e_zero_detections()
    test_case_f_export_generation()
    print("\n==================================================")
    print("ALL PHASE 3 EDGE-CASE TESTS PASSED PERFECTLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
