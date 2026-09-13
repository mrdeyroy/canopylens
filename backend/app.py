"""
backend/app.py
CanopyLens - Interactive Tree Crown Detection & Canopy Area Estimation Web App.
Built with Streamlit and DeepForest.
"""

import os
import sys
import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# Ensure backend directory is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from detection import load_model, detect_tree_crowns, draw_annotated_image
from geo import parse_kml, extract_geotiff_metadata, validate_image_kml_alignment
from area import calculate_canopy_metrics, PROXY_DISCLAIMER


# Set Streamlit page configuration
st.set_page_config(
    page_title="CanopyLens - Tree Crown Detection & Canopy Estimation",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean modern aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #2d6a4f;
        margin-bottom: 0.4rem;
    }
    .tagline {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .stepper-banner {
        background: #f0f7f4;
        border: 1px solid #cce3de;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 22px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 0.88rem;
    }
    .step-item {
        color: #1b4332;
    }
    .step-num {
        font-weight: 700;
        color: #2d6a4f;
        background: #e8f5e9;
        padding: 2px 7px;
        border-radius: 10px;
        margin-right: 4px;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.78rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1b4332;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #777;
        margin-top: 3px;
    }
    .disclaimer-box {
        background-color: #fffbf0;
        border-left: 4px solid #f59f00;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #664d03;
        margin: 16px 0;
        line-height: 1.5;
    }
    .demo-callout {
        background-color: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 15px;
        font-size: 0.85rem;
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_detector():
    """Initializes and caches DeepForest in Streamlit memory."""
    return load_model()


def generate_summary_report(
    image_name: str,
    kml_name: Optional[str],
    metrics: dict,
    kml_data: Optional[dict],
    gsd_m: Optional[float]
) -> str:
    """Generates a plain-text scientific analysis report."""
    aoi_str = "Not provided"
    if kml_data and kml_data.get("aoi_area_ha"):
        aoi_str = f"{kml_data['aoi_area_m2']:,.1f} m² ({kml_data['aoi_area_ha']:.3f} ha)"

    canopy_str = "Unavailable (GSD missing)"
    if metrics["total_canopy_m2"] is not None:
        canopy_str = f"{metrics['total_canopy_m2']:,.1f} m²"

    coverage_str = "Unavailable"
    if metrics["canopy_coverage_pct"] is not None:
        coverage_str = f"{metrics['canopy_coverage_pct']:.2f}%"

    gsd_str = f"{gsd_m:.4f} m/pixel" if gsd_m else "Not provided (pixel metrics only)"

    return f"""==================================================
CANOPYLENS ANALYSIS REPORT
==================================================

Project: CanopyLens
Purpose: Individual Tree Crown Detection & Canopy Area Estimation
Model: DeepForest 2.1.0 (RetinaNet / ResNet-50)

INPUTS
--------------------------------------------------
Input Image:       {image_name}
KML Boundary:      {kml_name or 'Not provided'}
Applied GSD:       {gsd_str}

RESULTS SUMMARY
--------------------------------------------------
Trees Detected:               {metrics['tree_count']}
Estimated Canopy Area:        {canopy_str}
Mean Detection Confidence:    {metrics['mean_confidence']:.1f}%
AOI Boundary Area:            {aoi_str}
Canopy Coverage:              {coverage_str}
Total Crown Pixel Area:       {metrics['total_pixel_area']:,.0f} px²
Average Crown Pixel Area:     {metrics['avg_pixel_area']:,.1f} px²

METHODOLOGY
--------------------------------------------------
- Detection: DeepForest object detector trained on airborne NEON forest datasets.
- Canopy Area: Sum of individual detected crown bounding boxes scaled by GSD².
- Coordinate Projection: Local UTM transformation for geodesic KML area accuracy.

SCIENTIFIC LIMITATIONS & HONESTY DECLARATION
--------------------------------------------------
1. Bounding-Box Proxy: Bounding boxes are rectangular approximations and include
   corner non-canopy pixels, typically overestimating isolated crown area by 20-25%.
2. Resolution Dependence: Sub-meter imagery is required. Low-resolution satellite
   imagery cannot resolve individual tree crowns.
3. Overlapping Canopies: Closed interlocking canopies may be subject to undercounting
   due to Non-Maximum Suppression (NMS).
4. Spatial Alignment: Standard PNG/JPG rasters lack coordinate metadata; alignment with
   KML boundaries cannot be spatially guaranteed without georeferenced GeoTIFF headers.
==================================================
"""


def main():
    # Header
    st.markdown('<div class="main-title">🌲 CanopyLens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Tree Crown Detection & Canopy Area Estimation</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="tagline">Upload high-resolution forest imagery to detect individual tree crowns and estimate canopy coverage.</div>',
        unsafe_allow_html=True
    )

    # 4-Step Visual Workflow Stepper
    st.markdown("""
    <div class="stepper-banner">
        <div class="step-item"><span class="step-num">STEP 1</span> Upload Forest Image</div>
        <div>➔</div>
        <div class="step-item"><span class="step-num">STEP 2</span> Optional: Upload KML Boundary</div>
        <div>➔</div>
        <div class="step-item"><span class="step-num">STEP 3</span> Provide GSD if required</div>
        <div>➔</div>
        <div class="step-item"><span class="step-num">STEP 4</span> Analyze Forest</div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # SIDEBAR CONTROLS
    # ----------------------------------------------------
    st.sidebar.header("📁 Step 1 & 2: Upload Data")

    # Demo Dataset Quick-Select
    st.sidebar.markdown("""
    <div class="demo-callout">
        <strong>🧪 Try the Demo Dataset</strong><br>
        Quickly test CanopyLens using the included benchmark data:
        <ul style="margin: 4px 0 0 16px; padding: 0;">
            <li><code>data/sample_forest.png</code> (OSBS)</li>
            <li><code>data/sample_aoi.kml</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    use_sample_image = st.sidebar.checkbox("Use Sample Forest Image", value=True)
    use_sample_kml = st.sidebar.checkbox("Use Sample KML Boundary", value=False)

    uploaded_image_file = st.sidebar.file_uploader(
        "Upload Forest Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="High-resolution aerial, drone, or ortho imagery (sub-meter resolution recommended)."
    )

    uploaded_kml_file = st.sidebar.file_uploader(
        "Upload Area of Interest (KML) - Optional",
        type=["kml"],
        help="Optional .kml boundary polygon defining the study site."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("📐 Step 3: Spatial Resolution (GSD)")

    gsd_mode = st.sidebar.radio(
        "Ground Sampling Distance Mode:",
        ["Auto-detect from GeoTIFF", "Manual GSD (m/pixel)", "No GSD (Pixel Analysis Only)"],
        index=1,
        help="GSD converts image pixel area to square meters. Without GSD, real-world metric area cannot be calculated."
    )

    manual_gsd = None
    if gsd_mode == "Manual GSD (m/pixel)":
        manual_gsd = st.sidebar.number_input(
            "GSD value (meters / pixel):",
            min_value=0.01,
            max_value=10.0,
            value=0.10,
            step=0.01,
            format="%.3f",
            help="Example: 0.10 m/pixel for high-res drone imagery, 0.50 m/pixel for aerial."
        )

    # Optional Advanced Settings
    with st.sidebar.expander("⚙️ Advanced Detection Settings", expanded=False):
        min_conf_input = st.slider(
            "Confidence Score Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            help="Filter detections by minimum model score. Default 0.00 preserves verified DeepForest outputs."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Step 4: Run Analysis")
    analyze_button = st.sidebar.button("Analyze Forest", type="primary", use_container_width=True)

    # Determine input image source
    sample_img_path = Path("data/sample_forest.png")
    sample_kml_path = Path("data/sample_aoi.kml")

    image_source_path = None
    image_bytes = None
    image_name = ""

    # Check file size safely
    MAX_FILE_SIZE_MB = 50.0
    if uploaded_image_file is not None:
        file_size_mb = len(uploaded_image_file.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"❌ Image is too large ({file_size_mb:.1f} MB). "
                f"Please upload a smaller image under {MAX_FILE_SIZE_MB:.0f} MB or crop the area of interest."
            )
            st.stop()
        image_bytes = uploaded_image_file.getvalue()
        image_name = uploaded_image_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_name).suffix) as tmp_file:
            tmp_file.write(image_bytes)
            image_source_path = tmp_file.name
    elif use_sample_image and sample_img_path.exists():
        image_source_path = str(sample_img_path)
        image_name = "sample_forest.png"
        with open(sample_img_path, "rb") as f:
            image_bytes = f.read()

    # Determine KML source
    kml_bytes = None
    kml_name = None
    if uploaded_kml_file is not None:
        kml_bytes = uploaded_kml_file.getvalue()
        kml_name = uploaded_kml_file.name
    elif use_sample_kml and sample_kml_path.exists():
        with open(sample_kml_path, "rb") as f:
            kml_bytes = f.read()
        kml_name = "sample_aoi.kml"

    # Main Analysis Flow
    if not image_source_path:
        st.info("👋 Welcome to CanopyLens! Upload an image on the sidebar or toggle 'Use Sample Forest Image' to begin.")
        st.stop()

    # Read image with PIL for inspection
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_np_rgb = np.array(pil_img.convert("RGB"))
        img_h, img_w = img_np_rgb.shape[:2]
    except Exception as e:
        st.error(f"❌ Failed to read image file: {e}")
        st.stop()

    # Check megapixel count for demo safety
    if img_w * img_h > 25_000_000:
        st.warning(
            f"⚠️ Large image detected ({img_w}x{img_h} = {img_w*img_h/1e6:.1f} MP). "
            "Processing may take extra time on CPU. Resolution has been preserved to maintain GSD fidelity."
        )

    # Inspect GeoTIFF metadata
    geotiff_meta = extract_geotiff_metadata(image_source_path)

    # Determine effective GSD
    effective_gsd = None
    if gsd_mode == "Auto-detect from GeoTIFF":
        if geotiff_meta.get("is_georeferenced") and geotiff_meta.get("gsd_m"):
            effective_gsd = geotiff_meta["gsd_m"]
            st.sidebar.success(f"Extracted GSD: {effective_gsd:.3f} m/px")
        else:
            st.sidebar.warning("No embedded GeoTIFF GSD found. Operating in pixel-only mode.")
    elif gsd_mode == "Manual GSD (m/pixel)":
        effective_gsd = manual_gsd
    else:
        effective_gsd = None

    # Parse KML if present
    kml_data = None
    spatial_alignment_valid = False
    alignment_reason = ""
    if kml_bytes:
        kml_data = parse_kml(kml_bytes)
        spatial_alignment_valid, alignment_reason = validate_image_kml_alignment(geotiff_meta, kml_data)

    # Run Detection
    with st.spinner("🌲 Loading DeepForest model & detecting tree crowns..."):
        try:
            model = get_cached_detector()
            min_conf_val = min_conf_input if min_conf_input > 0 else None
            try:
                predictions_df, image_bgr = detect_tree_crowns(
                    img_np_rgb,
                    model=model,
                    min_confidence=min_conf_val
                )
            except TypeError:
                predictions_df, image_bgr = detect_tree_crowns(img_np_rgb, model=model)
                if min_conf_val is not None and not predictions_df.empty:
                    conf_col = "confidence" if "confidence" in predictions_df.columns else "score"
                    predictions_df = predictions_df[predictions_df[conf_col] >= min_conf_val].reset_index(drop=True)
                    predictions_df["tree_id"] = [f"T{i+1:03d}" for i in range(len(predictions_df))]
        except Exception as e:
            st.error(f"❌ Tree detection failed: {e}")
            st.stop()

    # Calculate metrics
    aoi_m2 = kml_data.get("aoi_area_m2") if kml_data else None
    metrics = calculate_canopy_metrics(
        predictions_df,
        gsd_m=effective_gsd,
        aoi_area_m2=aoi_m2,
        spatial_alignment_valid=spatial_alignment_valid
    )

    # ----------------------------------------------------
    # DASHBOARD RESULTS SECTION
    # ----------------------------------------------------
    st.markdown("### 📊 Detection Results & Canopy Metrics")

    # Handle zero detections gracefully
    if metrics["tree_count"] == 0:
        st.info("ℹ️ **No individual tree crowns were detected above the current model threshold.** Try lowering the confidence threshold or ensuring the image contains distinct sub-meter tree crowns.")

    # KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trees Detected</div>
            <div class="metric-value">{metrics['tree_count']}</div>
            <div class="metric-sub">Individual Crowns</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if metrics["has_real_world_area"]:
            val_str = f"{metrics['total_canopy_m2']:,.1f} m²"
            sub_str = f"GSD: {effective_gsd:.3f} m/px"
        else:
            val_str = "Unavailable"
            sub_str = "Provide GSD or GeoTIFF"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Canopy Area (Proxy)</div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-sub">{sub_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if kml_data and kml_data.get("aoi_area_ha"):
            val_str = f"{kml_data['aoi_area_ha']:.2f} ha"
            sub_str = f"{kml_data['aoi_area_m2']:,.0f} m²"
        else:
            val_str = "N/A"
            sub_str = "No KML provided"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AOI Boundary</div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-sub">{sub_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if metrics["canopy_coverage_pct"] is not None:
            val_str = f"{metrics['canopy_coverage_pct']:.1f}%"
            sub_str = "Canopy / AOI"
        else:
            val_str = "Unavailable"
            sub_str = "Unverified alignment"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Canopy Coverage</div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-sub">{sub_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Confidence</div>
            <div class="metric-value">{metrics['mean_confidence']:.1f}%</div>
            <div class="metric-sub">DeepForest Score</div>
        </div>
        """, unsafe_allow_html=True)

    # Scientific Honesty & Limitations Notification Box
    st.markdown(f"""
    <div class="disclaimer-box">
        <strong>⚠️ Measurement Transparency:</strong> {metrics['area_status_message']}<br>
        <em>{metrics['disclaimer']}</em>
    </div>
    """, unsafe_allow_html=True)

    # Spatial alignment feedback if KML provided
    if kml_data:
        if not kml_data.get("success"):
            st.error(f"❌ **KML Error:** {kml_data.get('message', 'Invalid KML file.')}")
        elif not spatial_alignment_valid:
            st.warning(f"ℹ️ **Geospatial Notice:** {alignment_reason}")
        else:
            st.success(f"✅ **Geospatial Alignment:** {alignment_reason}")

    # ----------------------------------------------------
    # VISUALIZATION SECTION
    # ----------------------------------------------------
    st.markdown("### 🗺️ Visual Map & Detections")

    vcol1, vcol2 = st.columns([1, 4])
    with vcol1:
        show_labels = st.checkbox("Show Confidence Scores", value=True)
        show_ids = st.checkbox("Show Tree IDs", value=False)
        box_thickness = st.slider("Box Thickness", min_value=1, max_value=4, value=2)

    # Generate annotated image
    annotated_bgr = draw_annotated_image(
        image_bgr,
        metrics["augmented_df"],
        show_labels=show_labels,
        show_tree_ids=show_ids,
        thickness=box_thickness
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with vcol2:
        st.image(
            annotated_rgb,
            caption=f"CanopyLens Detection Output: {metrics['tree_count']} Individual Tree Crowns Identified",
            use_container_width=True
        )

    # ----------------------------------------------------
    # DATA TABLE & EXPORTS
    # ----------------------------------------------------
    st.markdown("### 📋 Detected Crown Attributes")

    display_df = metrics["augmented_df"].copy()

    # Reorder and format columns for clean display
    col_order = ["tree_id", "xmin", "ymin", "xmax", "ymax", "confidence", "pixel_area"]
    if "estimated_area_m2" in display_df.columns and effective_gsd:
        col_order.append("estimated_area_m2")

    existing_cols = [c for c in col_order if c in display_df.columns]
    table_df = display_df[existing_cols].rename(columns={
        "tree_id": "Tree ID",
        "xmin": "xmin (px)",
        "ymin": "ymin (px)",
        "xmax": "xmax (px)",
        "ymax": "ymax (px)",
        "confidence": "Confidence",
        "pixel_area": "Pixel Area (px²)",
        "estimated_area_m2": "Estimated Area (m²)"
    })

    st.dataframe(
        table_df,
        use_container_width=True,
        height=260
    )

    # Exports
    st.markdown("#### 📥 Export Suite")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # 1. Download Annotated Image
    is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
    if is_success:
        exp_col1.download_button(
            label="📷 Download Annotated Image",
            data=buffer.tobytes(),
            file_name="canopylens_detection.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    # 2. Download Predictions CSV
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    exp_col2.download_button(
        label="📄 Download Predictions CSV",
        data=csv_bytes,
        file_name="canopylens_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

    # 3. Download Summary Report
    report_text = generate_summary_report(
        image_name=image_name,
        kml_name=kml_name,
        metrics=metrics,
        kml_data=kml_data,
        gsd_m=effective_gsd
    )
    exp_col3.download_button(
        label="📑 Download Analysis Report",
        data=report_text.encode("utf-8"),
        file_name="canopylens_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    # ----------------------------------------------------
    # METHODOLOGY & LIMITATIONS ACCORDION
    # ----------------------------------------------------
    with st.expander("📖 Detailed Methodology, Mathematical Formulations & Limitations"):
        st.markdown("""
        #### Machine Learning Architecture
        - **Model:** Pretrained DeepForest 2.1.0 using a RetinaNet architecture with a ResNet-50 backbone.
        - **Training Domain:** Trained on airborne RGB and LiDAR annotations from the National Ecological Observatory Network (NEON).
        
        #### Bounding-Box Proxy Estimation
        - **Formula:** $\\text{Area}_{m^2} = (\\Delta x \\times \\Delta y) \\times \\text{GSD}^2$
        - **Geometric Overestimation:** Bounding boxes are rectangular and enclose background corners around circular/irregular tree crowns. For an isolated circular crown, a bounding box covers $\\frac{4}{\\pi} \\approx 1.27\\times$ the true circular area (a ~27% overestimate).
        
        #### Geospatial Alignment Honesty
        - Standard image formats (JPG, PNG) do not contain coordinate reference systems (CRS) or spatial bounding coordinates.
        - CanopyLens calculates KML boundary areas using geodesic UTM projections, but explicitly refuses to claim spatial alignment unless GeoTIFF georeferencing metadata is present.
        """)


if __name__ == "__main__":
    main()
