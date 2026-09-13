# CanopyLens: Automated Tree Crown Detection & Canopy Area Estimation
**Hackathon Submission & Technical Brief (Max 2-Page Summary)**

---

## 1. Problem Statement
Accurate forest monitoring and canopy estimation are critical for biodiversity conservation, wildfire fuel mapping, and carbon sequestration verification. However, manual ground surveys are labor-intensive, hazardous, and spatially constrained. While high-resolution remote sensing (drones and airborne sensors) has proliferated, field practitioners lack transparent, accessible tools to convert raw imagery into individual tree counts and verifiable canopy metrics without expensive proprietary software or false assertions of accuracy.

## 2. Solution: CanopyLens
**CanopyLens** is a lightweight, open-source web application designed for ecological transparency. It allows any non-technical user to upload high-resolution forest imagery, optionally supply a KML boundary, detect individual tree crowns, compute transparent bounding-box canopy area proxies, and export audit-ready metrics—all while strictly communicating scientific limitations and avoiding fabricated geospatial certainty.

---

## 3. System Architecture

The pipeline processes data through sequential, decoupled stages:
```
Forest Imagery (PNG/JPG/GeoTIFF) ───► [ DeepForest 2.1.0 (RetinaNet + ResNet-50) ]
                                                        │
                                                        ▼
                                          [ Individual Crown Bounding Boxes ]
                                                        │
                                                        ▼
                                           [ Tree Enumeration (N = Count) ]
                                                        │
   [ Spatial Resolution / GSD ] ──────────────► [ Bounding-Box Area Proxy ]
                                                        │
   [ Optional KML Boundary ] ─────────────────► [ Local UTM Projection & AOI Area ]
                                                        │
                                                        ▼
                                         [ Results Dashboard & Export Suite ]
```

---

## 4. Tree Detection & Machine Learning
- **Model Framework:** Single-stage object detection via **DeepForest 2.1.0** (Weecology Lab), utilizing RetinaNet with a ResNet-50 backbone pre-trained on diverse airborne ecological datasets from the National Ecological Observatory Network (NEON).
- **Inference:** Detects multi-scale crown features across Feature Pyramid Network (FPN) levels. Bounding boxes are refined via Non-Maximum Suppression (NMS) to eliminate duplicate proposals.
- **Model Detections vs. Ground Truth:** Detections reflect model predictions given input spectral contrast and resolution; CanopyLens explicitly clarifies that model detections are estimates and not ground-truth census counts.

---

## 5. Canopy Area Estimation Methodology
- **Formulation:** Because bounding boxes are rectangular approximations enclosing natural crowns, each crown's area is computed as:
  $$\text{Pixel Area}_i = (x_{\max, i} - x_{\min, i}) \times (y_{\max, i} - y_{\min, i})$$
  $$\text{Area}_{m^2, i} = \text{Pixel Area}_i \times \text{GSD}^2$$
- **Proxy Disclosure:** Rectangular bounding boxes cover $\frac{4}{\pi} \approx 1.27\times$ the area of an ideal circular crown, typically overestimating isolated crown area by ~20–25%. CanopyLens transparently designates this metric as **"Estimated Canopy Area (Bounding-Box Proxy)"**.

---

## 6. Geospatial Handling & GSD Honesty
CanopyLens strictly rejects fabricated measurements:
1. **Case A (GeoTIFF):** Spatial resolution and CRS are automatically extracted from embedded affine transforms using `rasterio`.
2. **Case B (User-Supplied GSD):** Applied directly for calibrated drone surveys (e.g., $0.10\text{ m/px}$).
3. **Case C (No GSD):** The system outputs tree count and pixel area, but explicitly refuses to fabricate square-meter values: *"Real-world canopy area cannot be reliably calculated because image ground resolution / GSD is unavailable."*
4. **KML Coordinate Projection:** WGS84 coordinates from `.kml` files are dynamically projected into the local UTM zone based on centroid longitude ($\text{Zone} = \lfloor(\text{lon}+180)/6\rfloor + 1$), computing geodesic ground area in hectares. If an image is not georeferenced, the system computes the AOI area independently and warns that spatial alignment cannot be guaranteed.

---

## 7. User Workflow
1. **Step 1:** Upload high-resolution forest imagery (or click one-click demo data).
2. **Step 2:** (Optional) Upload KML study plot boundary.
3. **Step 3:** Confirm or enter Ground Sampling Distance (GSD).
4. **Step 4:** Click **"Analyze Forest"** to view KPI metrics, high-resolution visual detections, attributes table, and export results (CSV, annotated JPG, analysis report).

---

## 8. Verification & Example Result
On the benchmark Ordway-Swisher Biological Station test image (`data/sample_forest.png`, $400 \times 400$ pixels):
- **Model Detections:** **55** individual tree crowns.
- **Mean Confidence:** **78.4%**.
- **KML AOI Boundary (`data/sample_aoi.kml`):** **96,500.1 m² (9.650 ha)**.
- **Execution:** Zero runtime failures; predictions and annotated imagery exported successfully.

---

## 9. Limitations & Boundary Conditions
1. **Sub-meter Resolution:** Requires sub-meter imagery (5–30 cm/px); standard 10m satellite imagery cannot resolve crowns.
2. **Canopy Closure:** Continuous interlocking canopies in closed forests may lead to merged bounding boxes.
3. **Understory Visibility:** Passive optical sensors cannot capture understory trees occluded by dominant crowns.
4. **Spatial Overlays:** Standard rasters lack coordinate systems; unaligned overlays are refused to prevent misleading visualizations.
