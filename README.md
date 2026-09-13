# 🌲 CanopyLens

### AI-powered individual tree crown detection, counting & canopy analysis

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![DeepForest](https://img.shields.io/badge/DeepForest-2.1.0-2E7D32?style=for-the-badge)](https://deepforest.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-Geospatial-139C5A?style=for-the-badge)](https://geopandas.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Turn high-resolution forest imagery into measurable tree-level insights.**

CanopyLens is an automated forest imagery analysis tool that detects individual tree crowns, counts detected trees, estimates canopy area, and optionally analyzes a user-defined area of interest using KML boundaries.

The project is designed around three principles:

**Usability · Transparency · Scientific Honesty**

A useful estimate is better than a precise-looking number with no scientific basis.

---

### 🚀 Live Demo


**[🌐 Try canopyLens](https://canopylens.streamlit.app/)**

**[▶ Launch Live Demo](https://drive.google.com/file/d/10HsVWouEIAvhXWrfmgDfpGBgNdMCnkJp/view?usp=sharing)**

### 📸 What CanopyLens Does

Upload forest imagery → Detect trees → Calculate metrics → Export results.

```text
┌─────────────────┐
│  Forest Image   │
│   PNG/JPG/TIFF  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  DeepForest AI  │
│ RetinaNet +     │
│ ResNet-50       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Tree Crown      │
│ Detection       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Tree Count +    │
│ Confidence      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ GSD / Spatial   │
│ Metadata        │
└────────┬────────┘
         ↓
┌────────────────────────┐
│ Canopy Area + Coverage │
└────────────┬───────────┘
             ↓
┌────────────────────────┐
│ Dashboard + CSV + JPG  │
│ + Scientific Report    │
└────────────────────────┘
```

---

# 🎯 The Problem

Forest monitoring often starts with a deceptively simple question:

> **How many trees are there, and how much canopy do they cover?**

Manually identifying individual trees from high-resolution imagery is time-consuming and difficult to scale.

CanopyLens provides an automated workflow for:

* 🌳 Detecting individual tree crowns
* 🔢 Counting detected trees
* 📐 Estimating canopy area
* 🗺️ Processing KML boundaries
* 📊 Calculating canopy coverage
* 🎯 Reporting model confidence
* 📥 Exporting analysis results

The project was built for a hiring challenge where the goal is not simply to produce a polished interface, but to build something that **actually works, can be used by another person, and communicates its limitations honestly**.

---

# ✨ Features

## 🌳 Individual Tree Crown Detection

Uses the pretrained **DeepForest 2.1.0** detection model based on RetinaNet with a ResNet-50 backbone.

The model produces bounding boxes and confidence scores for detected tree crowns.

---

## 🔢 Automatic Tree Counting

Every detected crown becomes an individual detection.

```text
Detected crowns
      ↓
Prediction filtering
      ↓
Tree detections
      ↓
Total tree count
```

The dashboard provides the final detected tree count alongside the visualized predictions.

---

## 📐 GSD-Aware Canopy Area

CanopyLens supports three spatial-resolution modes:

| Mode           | Source                | Result              |
| -------------- | --------------------- | ------------------- |
| 🛰️ Auto       | GeoTIFF metadata      | Metric area         |
| ✍️ Manual      | User-provided GSD     | Metric area         |
| ⚠️ Unavailable | No spatial resolution | Pixel-based metrics |

When GSD is unavailable, CanopyLens **does not fabricate square-meter measurements**.

Instead, it clearly reports pixel-based measurements.

---

## 🗺️ KML Boundary Support

Upload a `.kml` polygon representing the Area of Interest.

CanopyLens:

1. Parses WGS84 coordinates
2. Determines the appropriate local UTM zone
3. Projects geographic coordinates
4. Calculates polygon area
5. Reports area in:

   * Square meters
   * Hectares

This allows users to compare detected canopy area against a defined AOI.

---

## 📊 Canopy Coverage

When spatial alignment between the imagery and KML boundary is verified:

$$
Coverage(\%) =
\frac{Canopy\ Area}{AOI\ Area}
\times 100
$$

This provides an estimate of how much of the analyzed area is occupied by detected canopy.

---

## 🎯 Confidence Scoring

Each detection includes a model confidence score.

The dashboard provides:

* Individual detection confidence
* Mean confidence
* Number of detections
* Visual confidence badges

This makes model output easier to inspect instead of presenting the result as absolute truth.

---

## 🖼️ Interactive Visualization

Detected crowns are rendered directly on the source imagery using OpenCV.

The visualization includes:

* Bounding boxes
* Detection numbers
* Confidence scores
* Configurable annotation thickness
* Clear visual separation between detected trees

---

## 📥 Export Suite

Download the analysis results directly from the application.

### Available exports

| Export            | Format | Purpose              |
| ----------------- | ------ | -------------------- |
| Annotated Image   | `.jpg` | Visual inspection    |
| Predictions       | `.csv` | Further analysis     |
| Scientific Report | `.txt` | Reproducible summary |

---

# 🧠 Methodology

## Detection Model

CanopyLens uses:

**DeepForest 2.1.0**

with pretrained:

**RetinaNet + ResNet-50**

The model performs object detection across multiple image scales and returns bounding-box predictions for tree crowns.

---

## Canopy Area Calculation

For every detection:

$$
Pixel\ Area =
(x_{max}-x_{min})
\times
(y_{max}-y_{min})
$$

When GSD is known:

$$
Metric\ Area =
Pixel\ Area
\times
GSD^2
$$

where:

* `Pixel Area` = bounding-box area in pixels²
* `GSD` = ground sampling distance in meters/pixel
* `Metric Area` = estimated area in m²

---

## AOI Area Calculation

KML polygon coordinates are supplied in WGS84 geographic coordinates.

CanopyLens:

```text
WGS84 coordinates
       ↓
Polygon centroid
       ↓
UTM zone identification
       ↓
Local projected CRS
       ↓
Polygon area
       ↓
m² / hectares
```

Using a projected coordinate system allows the AOI area to be calculated in meaningful ground units.

---

# ⚠️ Scientific Honesty & Limitations

CanopyLens intentionally avoids presenting estimates as measurements when the required spatial information is unavailable.

## 1. Bounding-box area is a proxy

DeepForest provides **bounding boxes**, not pixel-level crown segmentation.

For approximately circular tree crowns, a rectangular bounding box contains background pixels.

Therefore:

> Bounding-box canopy area can overestimate the actual crown area.

For roughly circular crowns, the expected geometric overestimation can be approximately **20–25%**, depending on crown shape.

---

## 2. Image resolution matters

Individual crown detection requires sufficiently high-resolution imagery.

CanopyLens is intended for imagery with approximately:

**5–30 cm/pixel GSD**

such as:

* UAV imagery
* Drone imagery
* High-resolution aerial imagery
* Suitable high-resolution satellite imagery

Low-resolution imagery such as Sentinel-2 or Landsat generally cannot resolve individual tree crowns reliably.

---

## 3. Dense canopy can cause undercounting

In closed-canopy forests, neighboring crowns may overlap heavily.

This can result in:

* Merged detections
* Missed trees
* Undercounting
* Non-Maximum Suppression conflicts

Therefore, the detected count should not automatically be interpreted as the exact number of physical trees.

---

## 4. Spatial alignment matters

PNG and JPG images do not inherently contain geographic coordinates.

If a KML boundary is supplied alongside an ungeoreferenced image, CanopyLens cannot guarantee that the two datasets represent the same geographic footprint.

Rather than creating a visually convincing but potentially incorrect overlay, the system warns the user.

**No verified spatial relationship → no fabricated coverage percentage.**

---

# 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Forest Imagery     │
                    │ PNG / JPG / GeoTIFF  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     DeepForest       │
                    │ RetinaNet + ResNet50 │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Crown Bounding      │
                    │  Boxes + Confidence  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Tree Counting     │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Spatial Metadata Resolution│
                 │                            │
                 │ Auto / Manual / None       │
                 └──────────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
      ┌───────────────┐                  ┌─────────────────┐
      │ KML Boundary  │                  │ Results         │
      │ → UTM → AOI   │                  │ Dashboard       │
      │ Area          │                  │ CSV / JPG / TXT │
      └───────┬───────┘                  └─────────────────┘
              │
              ▼
      ┌─────────────────┐
      │ Canopy Coverage │
      │     (%)         │
      └─────────────────┘
```

---

# 🛠️ Tech Stack

### Machine Learning

* Python
* DeepForest 2.1.0
* RetinaNet
* ResNet-50

### Computer Vision

* OpenCV
* NumPy
* Pillow

### Geospatial

* KML parsing
* UTM projection
* GeoPandas / Shapely
* Raster spatial metadata

### Application

* Streamlit
* Python

### Data & Export

* Pandas
* CSV
* JPG
* TXT

---

# 📁 Project Structure

```text
CanopyLens/
│
├── backend/
│   ├── app.py
│   ├── test_detection.py
│   ├── requirements.txt
│   └── ...
│
├── data/
│   └── sample_forest.png
│
├── docs/
│   └── methodology.md
│
├── outputs/
│   └── ...
│
├── README.md
└── LICENSE
```

---

# ⚙️ Running Locally

## 1. Prerequisites

* Python 3.10+
* Windows / macOS / Linux
* Internet connection for initial model setup

Tested with:

```text
Python 3.12.9
```

---

## 2. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd CanopyLens
```

---

## 3. Create Virtual Environment

### Windows

```powershell
python -m venv backend/.venv

.\backend\.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv backend/.venv

source backend/.venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## 5. Launch CanopyLens

From the project root:

```powershell
.\backend\.venv\Scripts\streamlit run backend/app.py
```

Or:

```bash
cd backend
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 CLI Detection Test

CanopyLens also provides a headless detection test.

```bash
python backend/test_detection.py data/sample_forest.png
```

This is useful for verifying that the detection pipeline works independently from the Streamlit interface.

---

# 📊 Example Workflow

```text
1. Upload forest image
           ↓
2. Upload KML boundary (optional)
           ↓
3. Resolve GSD
   ├── GeoTIFF metadata
   ├── Manual input
   └── Pixel-only mode
           ↓
4. Run DeepForest
           ↓
5. Review detected crowns
           ↓
6. Inspect confidence scores
           ↓
7. Review canopy metrics
           ↓
8. Export results
```

---

# 🔬 Reproducibility

CanopyLens separates:

**Observed model output**

from

**Derived spatial estimates**

This distinction is important because model predictions, pixel measurements, GSD-derived measurements, and AOI-derived measurements have different levels of uncertainty.

Detailed mathematical derivations and methodology are available in:

`docs/methodology.md`

---

# 🧭 Future Improvements

Potential future development includes:

* [ ] Pixel-level crown segmentation
* [ ] Improved overlapping-crown separation
* [ ] Multi-model detection comparison
* [ ] GeoTIFF-native map visualization
* [ ] Automatic image/KML spatial alignment
* [ ] Tree density heatmaps
* [ ] Species-level classification
* [ ] Temporal forest-change analysis
* [ ] Cloud-based batch processing
* [ ] Carbon-stock estimation using species/biomass models

> These are future directions, not capabilities currently claimed by CanopyLens.

---

# 🎯 Why CanopyLens?

Most computer-vision demos focus on making predictions look impressive.

CanopyLens focuses on making the **result understandable and defensible**.

It asks:

* What did the model actually detect?
* How confident was it?
* Can the pixel measurement be converted into physical units?
* Is the KML actually spatially aligned?
* What assumptions were made?
* Where can the result be wrong?

That makes the tool more useful for real-world forest analysis than a system that simply outputs a large number and calls it accurate.

---

# 📜 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# 👨‍💻 Author

**Shibam Deyroy**

BCA Student · Full-Stack Developer · AI/ML Enthusiast

---

## ⭐ If you find CanopyLens useful

Give the repository a ⭐ and feel free to explore the implementation.

**Build it. Test it. Question the numbers.**
