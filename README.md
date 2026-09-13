# CanopyLens

**CanopyLens** is an automated tool designed to analyze high-resolution aerial, drone, and satellite forest imagery to detect individual tree crowns, count trees, and estimate canopy area coverage, with optional boundary delineation via KML files.

Built for ecological assessments, carbon project verification, and forestry analytics with an emphasis on **usability, transparency, and scientific honesty**.

---

## Features

- **Individual Tree Crown Detection**: Single-stage deep learning detection leveraging DeepForest (RetinaNet / ResNet-50) trained on diverse ecological biomes.
- **Dynamic Tree Counting**: Automated enumeration of detected tree crowns directly from model predictions.
- **KML Boundary Parsing**: Robust parsing of `.kml` polygon boundaries, projecting geographic coordinates to local UTM coordinate systems for accurate area calculations in square meters and hectares.
- **GSD-Aware Canopy Area Estimation**:
  - **Auto-detection**: Extracts pixel resolution from georeferenced GeoTIFF headers.
  - **Manual Entry**: Supports user-defined Ground Sampling Distance (GSD in meters/pixel).
  - **Scientific Honesty**: Refuses to invent square-meter metrics when GSD is unavailable, providing transparent pixel-based metrics instead.
- **Canopy Coverage Percentage**: Calculates canopy-to-AOI coverage percentage when spatial relationships are verified.
- **Confidence Scoring**: Displays individual and mean model detection scores.
- **Interactive Visual Map**: Clear OpenCV bounding box annotations with configurable badges and thickness.
- **Complete Export Suite**: Download annotated visualization imagery (`.jpg`), raw predictions data (`.csv`), and formatted scientific analysis summary reports (`.txt`).
- **Explicit Limitations & Disclaimers**: Discloses bounding-box overestimation biases and geospatial alignment constraints.

---

## Architecture

The processing pipeline flows linearly from image acquisition to metric synthesis:

```
Forest Image (PNG/JPG/GeoTIFF)
           │
           ▼
DeepForest 2.1.0 (RetinaNet + ResNet-50)
           │
           ▼
Tree Crown Bounding Boxes & Confidence Scores
           │
           ▼
Tree Count Enumeration
           │
           ▼
GSD / Spatial Metadata Resolution (Auto / Manual / None)
           │
           ▼
Canopy Area Proxy Calculation (m² & ha)
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
KML Boundary Parsing       Results Dashboard & Export Suite
(Local UTM Projection)     (Streamlit UI, CSV, JPG, Report)
```

---

## Running Locally

### 1. Prerequisites
- Python 3.10+ (tested and verified on **Python 3.12.9**).
- Windows, macOS, or Linux.

### 2. Environment Activation & Dependencies

Clone the repository and activate the existing virtual environment:

```bash
# On Windows (PowerShell / Command Prompt)
.\backend\.venv\Scripts\activate

# On Linux / macOS
source backend/.venv/bin/activate
```

If setting up fresh:
```bash
python -m venv backend/.venv
.\backend\.venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Launch the Web Application

From the project root:

```bash
.\backend\.venv\Scripts\streamlit run backend/app.py
```

Or from inside `backend/`:

```bash
cd backend
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### 4. Running the Phase 1 CLI Test Script

To run headless detection on a specific image via terminal:

```bash
python backend/test_detection.py data/sample_forest.png
```

---

## Methodology

- **Detection Model**: DeepForest 2.1.0 pre-trained model weights. Detects crown boundaries using a Feature Pyramid Network across multiple scales.
- **Bounding-Box Area Calculation**:
  $$\text{Pixel Area} = (x_{\max} - x_{\min}) \times (y_{\max} - y_{\min})$$
  $$\text{Metric Area } (m^2) = \text{Pixel Area} \times \text{GSD}^2$$
- **AOI Area Calculation**: Parses WGS84 polygon vertices from `.kml` files, identifies the local UTM zone based on polygon centroid longitude, and projects coordinates to compute accurate ground area.
- **Canopy Coverage**:
  $$\text{Coverage } (\%) = \frac{\text{Total Canopy Area } (m^2)}{\text{AOI Boundary Area } (m^2)} \times 100$$
  *Only computed when spatial alignment between image and KML is verified.*

Detailed mathematical derivations and references are documented in [`docs/methodology.md`](file:///c:/Users/Lenovo/Desktop/FloraCarbonAI/docs/methodology.md).

---

## Limitations

CanopyLens prioritizes honest scientific reporting over fabricated certainty:
1. **Bounding-Box Proxy Overestimation**: DeepForest provides bounding boxes, not pixel-wise crown segmentations. For roughly circular crowns, a bounding box encloses background corners, typically overestimating crown area by ~20–25%.
2. **Resolution Requirements**: Sub-meter spatial resolution (e.g., UAV/high-res aerial imagery with GSD 5–30 cm/px) is required. Low-resolution satellite imagery (Sentinel-2, Landsat) cannot resolve distinct crowns.
3. **Closed-Canopy Overlap**: Dense interlocking canopies may merge into single detections or suffer undercounting due to Non-Maximum Suppression (NMS).
4. **Spatial Alignment**: Ungeoreferenced rasters (PNG/JPG) lack spatial coordinate references; CanopyLens warns users that pixel-to-boundary overlay cannot be guaranteed rather than displaying a fake overlay.
