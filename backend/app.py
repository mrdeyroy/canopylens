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

from detection import load_model, detect_tree_crowns, draw_annotated_image, extract_tree_crop
from geo import parse_kml, extract_geotiff_metadata, validate_image_kml_alignment
from area import calculate_canopy_metrics, PROXY_DISCLAIMER


# Set Streamlit page configuration
st.set_page_config(
    page_title="CanopyLens - Tree Crown Detection & Canopy Estimation",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme-Safe CSS (Compatible with Light & Dark Modes in Streamlit Cloud)
CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d6a4f;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #40916c;
        margin-bottom: 0.3rem;
    }
    .tagline {
        font-size: 0.9rem;
        color: #777;
        margin-bottom: 1.0rem;
    }
    .stepper-banner {
        background: rgba(45, 106, 79, 0.08);
        border: 1px solid rgba(45, 106, 79, 0.25);
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 20px;
        font-size: 0.88rem;
    }
    .step-num {
        font-weight: 700;
        color: #2d6a4f;
        background: rgba(45, 106, 79, 0.15);
        padding: 2px 7px;
        border-radius: 6px;
        margin-right: 4px;
    }
    .metric-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        font-weight: 600;
        opacity: 0.8;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2d6a4f;
    }
    .metric-sub {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 2px;
    }
    .disclaimer-box {
        background: rgba(245, 159, 0, 0.08);
        border-left: 4px solid #f59f00;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.86rem;
        color: inherit;
        margin: 14px 0;
        line-height: 1.45;
    }
    .demo-callout {
        background: rgba(45, 106, 79, 0.07);
        border: 1px solid rgba(45, 106, 79, 0.2);
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 0.82rem;
    }
    .inspect-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_detector():
    """Initializes and caches DeepForest in Streamlit memory."""
    return load_model()


def generate_summary_report(
    image_name: str,
    kml_name: Optional[str],
    metrics: dict,
    kml_data: Optional[dict],
    gsd_m: Optional[float],
    min_confidence: float = 0.0
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
    conf_str = f"{min_confidence:.2f}" if min_confidence > 0 else "0.00 (All detections)"

    return f"""==================================================
CANOPYLENS ANALYSIS REPORT
==================================================

Project: CanopyLens
Purpose: Individual Tree Crown Detection & Canopy Area Estimation
Model: DeepForest 2.1.0 (RetinaNet / ResNet-50)

INPUTS & SETTINGS
--------------------------------------------------
Input Image:                  {image_name}
KML Boundary:                 {kml_name or 'Not provided'}
Applied GSD:                  {gsd_str}
Confidence Threshold:         {conf_str}

RESULTS SUMMARY
--------------------------------------------------
Trees Detected:               {metrics['tree_count']}
Estimated Canopy Area (Proxy):{canopy_str}
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


def filter_predictions(df: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    """Filters predictions DataFrame by confidence threshold and re-numbers Tree IDs."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["tree_id", "xmin", "ymin", "xmax", "ymax", "confidence", "label"])

    filtered = df.copy()
    if min_conf > 0.0:
        conf_col = "confidence" if "confidence" in filtered.columns else "score"
        filtered = filtered[filtered[conf_col] >= min_conf].reset_index(drop=True)

    filtered["tree_id"] = [f"T{i+1:03d}" for i in range(len(filtered))]
    return filtered


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
        <span><span class="step-num">STEP 1</span> Upload Forest Image</span>
        <span>&nbsp;➔&nbsp;</span>
        <span><span class="step-num">STEP 2</span> Optional: Upload KML Boundary</span>
        <span>&nbsp;➔&nbsp;</span>
        <span><span class="step-num">STEP 3</span> Configure GSD</span>
        <span>&nbsp;➔&nbsp;</span>
        <span><span class="step-num">STEP 4</span> Click Analyze Forest</span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "raw_predictions" not in st.session_state:
        st.session_state["raw_predictions"] = None
    if "image_rgb" not in st.session_state:
        st.session_state["image_rgb"] = None
    if "image_bgr" not in st.session_state:
        st.session_state["image_bgr"] = None
    if "image_name" not in st.session_state:
        st.session_state["image_name"] = ""
    if "input_signature" not in st.session_state:
        st.session_state["input_signature"] = None
    if "geotiff_meta" not in st.session_state:
        st.session_state["geotiff_meta"] = None
    if "kml_data" not in st.session_state:
        st.session_state["kml_data"] = None
    if "kml_name" not in st.session_state:
        st.session_state["kml_name"] = None

    # ----------------------------------------------------
    # SIDEBAR CONTROLS (Steps 1, 2, 3)
    # ----------------------------------------------------
    st.sidebar.header("📁 Step 1 & 2: Upload Data")

    # Demo Dataset Quick-Select
    st.sidebar.markdown("""
    <div class="demo-callout">
        <strong>🧪 Try the Demo Dataset</strong><br>
        Quickly test CanopyLens using the included benchmark data:
        <ul style="margin: 3px 0 0 14px; padding: 0;">
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

    # Determine input source
    sample_img_path = Path("data/sample_forest.png")
    sample_kml_path = Path("data/sample_aoi.kml")

    image_source_path = None
    image_bytes = None
    image_name = ""
    is_geotiff_file = False

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
        is_geotiff_file = Path(image_name).suffix.lower() in [".tif", ".tiff"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_name).suffix) as tmp_file:
            tmp_file.write(image_bytes)
            image_source_path = tmp_file.name
    elif use_sample_image and sample_img_path.exists():
        image_source_path = str(sample_img_path)
        image_name = "sample_forest.png"
        is_geotiff_file = False
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

    # Invalidate session if image changes
    current_sig = f"{image_name}_{len(image_bytes) if image_bytes else 0}_{kml_name}"
    if st.session_state["input_signature"] != current_sig:
        st.session_state["analysis_complete"] = False
        st.session_state["raw_predictions"] = None
        st.session_state["input_signature"] = current_sig

    st.sidebar.markdown("---")
    st.sidebar.header("📐 Step 3: Spatial Resolution (GSD)")

    # Smart GSD mode default based on uploaded file format
    gsd_default_index = 0 if is_geotiff_file else 1

    gsd_mode = st.sidebar.radio(
        "Ground Sampling Distance Mode:",
        ["Auto-detect from GeoTIFF", "Manual GSD (m/pixel)", "No GSD (Pixel Analysis Only)"],
        index=gsd_default_index,
        help=(
            "• Auto-detect: Use resolution embedded in a georeferenced GeoTIFF.\n"
            "• Manual GSD: Enter known ground resolution in meters/pixel.\n"
            "• Pixel Analysis Only: Calculate tree counts and pixel areas without converting to real-world area."
        )
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
    with st.sidebar.expander("⚙️ Detection Settings", expanded=False):
        min_conf_input = st.slider(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            help="Filter detections by minimum model score. Value 0.00 preserves all standard DeepForest outputs."
        )
        st.caption(f"Active threshold: **{min_conf_input:.2f}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Step 4: Run Analysis")
    analyze_button = st.sidebar.button("Analyze Forest", type="primary", use_container_width=True)

    # ----------------------------------------------------
    # WORKFLOW STATE MANAGEMENT (Issue 5: Only analyze on click)
    # ----------------------------------------------------
    if not image_source_path or not image_bytes:
        st.info("👋 **Welcome to CanopyLens!** Upload your forest imagery on the sidebar or use the demo dataset to begin.")
        st.stop()

    # Pre-read image to validate dimensions
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

    # Execute Analysis ONLY on explicit button click
    if analyze_button:
        with st.spinner("🌲 Loading DeepForest model & detecting tree crowns..."):
            try:
                model = get_cached_detector()
                # Run raw inference
                raw_df, image_bgr = detect_tree_crowns(img_np_rgb, model=model, min_confidence=None)
                
                # Inspect GeoTIFF metadata
                geotiff_meta = extract_geotiff_metadata(image_source_path)

                # Parse KML if present
                kml_data = None
                if kml_bytes:
                    kml_data = parse_kml(kml_bytes)

                # Store in session state
                st.session_state["raw_predictions"] = raw_df
                st.session_state["image_rgb"] = img_np_rgb
                st.session_state["image_bgr"] = image_bgr
                st.session_state["image_name"] = image_name
                st.session_state["geotiff_meta"] = geotiff_meta
                st.session_state["kml_data"] = kml_data
                st.session_state["kml_name"] = kml_name
                st.session_state["analysis_complete"] = True
            except Exception as e:
                st.error(f"❌ Tree detection failed: {e}")
                st.stop()

    # If analysis has not been run for the current input, show ready state
    if not st.session_state.get("analysis_complete") or st.session_state["raw_predictions"] is None:
        st.info("👉 **Ready to Analyze:** Your image is loaded. Select your GSD options in the sidebar and click **Analyze Forest** to run detection.")
        
        # Show image preview before analysis
        st.markdown("#### 📷 Image Preview")
        st.image(img_np_rgb, caption=f"{image_name} ({img_w}x{img_h} px)", use_container_width=True)
        st.stop()

    # ----------------------------------------------------
    # RENDER ACTIVE RESULTS (From Session State & Dynamic Filter)
    # ----------------------------------------------------
    raw_predictions = st.session_state["raw_predictions"]
    image_bgr = st.session_state["image_bgr"]
    image_rgb = st.session_state["image_rgb"]
    geotiff_meta = st.session_state["geotiff_meta"]
    kml_data = st.session_state["kml_data"]

    # Issue 2: Dynamically filter predictions based on current slider threshold
    filtered_predictions = filter_predictions(raw_predictions, min_conf_input)

    # Determine effective GSD
    effective_gsd = None
    if gsd_mode == "Auto-detect from GeoTIFF":
        if geotiff_meta and geotiff_meta.get("is_georeferenced") and geotiff_meta.get("gsd_m"):
            effective_gsd = geotiff_meta["gsd_m"]
        else:
            if not is_geotiff_file:
                st.sidebar.info("ℹ️ GeoTIFF metadata is not applicable for standard PNG/JPG rasters. Operating in pixel-only mode.")
            else:
                st.sidebar.warning("⚠️ No valid embedded metric GSD found in GeoTIFF. Operating in pixel-only mode.")
    elif gsd_mode == "Manual GSD (m/pixel)":
        effective_gsd = manual_gsd
    else:
        effective_gsd = None

    # Spatial alignment verification
    spatial_alignment_valid = False
    alignment_reason = ""
    if kml_data:
        spatial_alignment_valid, alignment_reason = validate_image_kml_alignment(geotiff_meta, kml_data)

    # Calculate metrics
    aoi_m2 = kml_data.get("aoi_area_m2") if kml_data else None
    metrics = calculate_canopy_metrics(
        filtered_predictions,
        gsd_m=effective_gsd,
        aoi_area_m2=aoi_m2,
        spatial_alignment_valid=spatial_alignment_valid
    )

    # ----------------------------------------------------
    # KPI METRIC CARDS
    # ----------------------------------------------------
    st.markdown("### 📊 Detection Results & Canopy Metrics")

    # Handle zero detections gracefully
    if metrics["tree_count"] == 0:
        st.info("ℹ️ **No individual tree crowns were detected above the current model threshold.** Try lowering the confidence threshold slider in the sidebar.")

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
            <div class="metric-sub">Threshold: ≥ {min_conf_input:.2f}</div>
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
    # ISSUE 3: INSPECT INDIVIDUAL TREE & SELECTION
    # ----------------------------------------------------
    st.markdown("### 🔍 Inspect Individual Tree")
    
    tree_id_list = list(metrics["augmented_df"]["tree_id"]) if not metrics["augmented_df"].empty else []
    select_options = ["None (Show All Detections)"] + tree_id_list

    selected_option = st.selectbox(
        "Select Tree ID to inspect & highlight on map:",
        options=select_options,
        index=0,
        help="Choose an individual tree crown to inspect its confidence, bounding box coordinates, and cropped close-up view."
    )

    selected_tree_id = None if selected_option == "None (Show All Detections)" else selected_option

    # If a tree is selected, show detail card and cropped view
    if selected_tree_id:
        tree_row = metrics["augmented_df"][metrics["augmented_df"]["tree_id"] == selected_tree_id].iloc[0]
        
        t_xmin = float(tree_row["xmin"])
        t_ymin = float(tree_row["ymin"])
        t_xmax = float(tree_row["xmax"])
        t_ymax = float(tree_row["ymax"])
        t_conf = float(tree_row["confidence"]) if "confidence" in tree_row else 1.0
        t_px_area = float(tree_row["pixel_area"])
        t_m2_area = float(tree_row["estimated_area_m2"]) if "estimated_area_m2" in tree_row and tree_row["estimated_area_m2"] is not None else None

        crop_col1, crop_col2 = st.columns([1, 2])
        
        with crop_col1:
            crop_img_rgb = extract_tree_crop(image_rgb, (t_xmin, t_ymin, t_xmax, t_ymax), padding=12)
            st.image(
                crop_img_rgb,
                caption=f"Crown Crop: {selected_tree_id}",
                use_container_width=True
            )

        with crop_col2:
            st.markdown(f"""
            <div class="inspect-card">
                <h4 style="margin-top:0; color:#2d6a4f;">🌲 Tree Details: {selected_tree_id}</h4>
                <p style="margin:4px 0;"><strong>Confidence Score:</strong> {t_conf:.3f} ({t_conf*100:.1f}%)</p>
                <p style="margin:4px 0;"><strong>Bounding Box (px):</strong> xmin={t_xmin:.0f}, ymin={t_ymin:.0f}, xmax={t_xmax:.0f}, ymax={t_ymax:.0f} (Size: {t_xmax-t_xmin:.0f}x{t_ymax-t_ymin:.0f} px)</p>
                <p style="margin:4px 0;"><strong>Pixel Area:</strong> {t_px_area:,.0f} px²</p>
                <p style="margin:4px 0;"><strong>Estimated Real Area:</strong> {f'{t_m2_area:.2f} m²' if t_m2_area is not None else 'Unavailable (No GSD)'}</p>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # VISUALIZATION SECTION
    # ----------------------------------------------------
    st.markdown("### 🗺️ Visual Map & Detections")

    vcol1, vcol2 = st.columns([1, 4])
    with vcol1:
        show_labels = st.checkbox("Show Confidence Scores", value=True)
        show_ids = st.checkbox("Show Tree IDs", value=False)
        box_thickness = st.slider("Box Thickness", min_value=1, max_value=4, value=2)

    # Generate annotated image with highlighted tree
    annotated_bgr = draw_annotated_image(
        image_bgr,
        metrics["augmented_df"],
        show_labels=show_labels,
        show_tree_ids=show_ids,
        selected_tree_id=selected_tree_id,
        thickness=box_thickness
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with vcol2:
        caption_text = f"CanopyLens Output: {metrics['tree_count']} Crowns"
        if selected_tree_id:
            caption_text += f" | Highlighted: {selected_tree_id}"
        st.image(
            annotated_rgb,
            caption=caption_text,
            use_container_width=True
        )

    # ----------------------------------------------------
    # DATA TABLE & EXPORTS
    # ----------------------------------------------------
    st.markdown("### 📋 Detected Crown Attributes")

    display_df = metrics["augmented_df"].copy()

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
        image_name=st.session_state["image_name"],
        kml_name=st.session_state["kml_name"],
        metrics=metrics,
        kml_data=kml_data,
        gsd_m=effective_gsd,
        min_confidence=min_conf_input
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
