# CanopyLens Methodology & Scientific Documentation

## 1. Overview
CanopyLens is an automated analytical tool designed to count individual tree crowns and estimate forest canopy cover from high-resolution aerial and UAV imagery, with optional boundary constraints provided via Keyhole Markup Language (KML).

This document details the underlying machine learning architecture, geometric area estimation methodology, geospatial coordinate transformations, and explicit scientific limitations.

---

## 2. Machine Learning Architecture: DeepForest

### 2.1 Model Selection & Backbone
CanopyLens uses the open-source **DeepForest 2.1.0** package developed by the Weecology Lab.
- **Model Family:** Single-stage object detector based on RetinaNet.
- **Feature Extractor:** ResNet-50 backbone pre-trained on ImageNet and fine-tuned with a Feature Pyramid Network (FPN) to handle multi-scale crown features.
- **Training Data:** Trained on thousands of hand-annotated and LiDAR-derived individual tree crowns across the National Ecological Observatory Network (NEON), encompassing diverse forest biomes ranging from open pine savannas to mixed hardwood stands.

### 2.2 Inference Pipeline
1. **Input Preprocessing:** The imagery is normalized to the expected RGB dynamic range.
2. **Feature Extraction:** Multi-scale feature maps are extracted across pyramid levels (P3 to P7).
3. **Anchor Classification & Regression:** Candidate anchors predict tree presence and regress bounding box coordinates:
   $$\text{Box} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}, \text{score}]$$
4. **Non-Maximum Suppression (NMS):** Overlapping proposal boxes with high Intersection-over-Union (IoU) are suppressed to minimize duplicate crown counting.

---

## 3. Canopy Area Estimation: Bounding-Box Proxy

### 3.1 Pixel Area Formulation
For each detected tree crown $i$, bounding dimensions are extracted:
$$\text{width}_i = x_{\max, i} - x_{\min, i}$$
$$\text{height}_i = y_{\max, i} - y_{\min, i}$$
$$\text{Area}_{\text{pixel}, i} = \text{width}_i \times \text{height}_i$$

### 3.2 Real-World Metric Conversion (GSD)
A Ground Sampling Distance (GSD) represents the ground distance covered by a single image pixel:
$$\text{GSD} = \text{meters / pixel}$$

When GSD is available:
$$\text{Area}_{m^2, i} = \text{Area}_{\text{pixel}, i} \times \text{GSD}^2$$
$$\text{Total Canopy Area}_{\text{proxy}} = \sum_{i=1}^{N} \text{Area}_{m^2, i}$$

### 3.3 Proxy Explanation & Bias Disclosure
> [!IMPORTANT]
> **Why this is a Proxy and NOT an Exact Segmentation:**
> Bounding boxes are rectangular representations enclosing an object. Natural tree crowns are approximately circular, elliptical, or irregular. Consequently:
> - A rectangular bounding box encloses background pixels (shadows, bare ground, or understory) in the corners.
> - For a circular crown of diameter $D$, the bounding box area is $D^2$, whereas the true circular area is $\frac{\pi}{4} D^2 \approx 0.785 D^2$.
> - Therefore, an uncorrected bounding-box proxy will typically overestimate single crown area by approximately 20–25% for isolated circular crowns, while overlapping crowns in dense clusters may experience undercounting due to NMS.
> - CanopyLens transparently labels all such metrics as **"Estimated Canopy Area (Bounding-Box Proxy)"**.

---

## 4. Geospatial Handling: KML and Georeferencing

### 4.1 Ground Sampling Distance Modes
CanopyLens operates under three transparent cases:

| Case | Input Condition | Methodology | Real-world Metric Output |
| :--- | :--- | :--- | :--- |
| **Case A** | GeoTIFF with spatial resolution tags | Auto-extract pixel resolution from affine transform | Full $m^2$ and hectare metrics |
| **Case B** | Standard raster (PNG/JPG) + User GSD | Apply user-specified GSD ($m/\text{pixel}$) | Full $m^2$ and hectare metrics |
| **Case C** | Standard raster without GSD | Pixel analysis only | Refuse to invent $m^2$; report pixel counts only |

### 4.2 KML Boundary Parsing & Area Calculation
1. **Coordinate Extraction:** Polygon / MultiPolygon boundaries are extracted from the KML file in geographic coordinates (WGS84, `EPSG:4326`).
2. **Projected Transformation:** Because geographic coordinates are angular (degrees), calculating geodesic area directly on an unprojected plane introduces substantial latitude distortion.
3. **UTM Projection:** CanopyLens dynamically calculates the local UTM zone based on polygon centroid longitude:
   $$\text{Zone} = \left\lfloor \frac{\text{longitude} + 180}{6} \right\rfloor + 1$$
4. **Area Calculation:** The polygon is transformed to the local UTM projected coordinate system (in meters), yielding accurate ground area in $m^2$ and hectares ($1\text{ ha} = 10,000\text{ m}^2$).

### 4.3 Image-to-KML Spatial Alignment
- If an image is a standard RGB PNG/JPG, there is no spatial reference system connecting pixel $(x, y)$ to ground $(\text{lon}, \text{lat})$.
- CanopyLens explicitly checks for geospatial bounds. If the image is not georeferenced, it warns:
  > *"KML boundary was successfully parsed, but the uploaded image does not contain sufficient geospatial metadata to guarantee pixel-to-boundary alignment."*
- We do **not** force a fake alignment between unreferenced images and real-world KML coordinates.

---

## 5. Canopy Coverage Calculation

Canopy coverage represents the percentage of ground area shaded or covered by tree crowns within a defined boundary:
$$\text{Canopy Coverage (\%)} = \left( \frac{\text{Total Canopy Area } (m^2)}{\text{AOI Ground Area } (m^2)} \right) \times 100$$

**Prerequisites for valid coverage calculation:**
1. A valid KML Area of Interest (AOI) must be supplied.
2. Real-world canopy area must be known (via GSD or GeoTIFF).
3. The image extent must correspond to the AOI.

If any prerequisite is missing, the system states that coverage cannot be reliably computed.

---

## 6. Known Model Limitations

1. **Resolution Thresholds:** DeepForest requires high-resolution imagery (sub-meter GSD, ideally $5\text{ cm} - 30\text{ cm/pixel}$ from drones or low-altitude aircraft). Standard 10m Sentinel-2 or 3m Planet imagery cannot resolve individual tree crowns.
2. **Overlapping / Closed Canopies:** In tropical rainforests or dense closed-canopy stands where crowns interlock continuously, individual crown boundary detection degrades, leading to clumped detections.
3. **Understory Exclusion:** Passive optical RGB sensors only detect the uppermost canopy layer; understory and intermediate trees occluded by dominant trees cannot be counted.
4. **Dead / Leafless Trees:** Standing dead timber (snags) or dormant deciduous trees during leaf-off seasons exhibit different spectral signatures and may be missed.
