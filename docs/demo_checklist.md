# CanopyLens 2-Minute Demo Checklist & Walkthrough

This checklist guides a presenter or judge through a live evaluation of CanopyLens.

---

## Pre-Flight Check (30 Seconds Before Demo)
1. Ensure the application is running:
   ```powershell
   .\backend\.venv\Scripts\python.exe -m streamlit run backend/app.py
   ```
2. Open your browser to `http://localhost:8501`.
3. Verify the browser shows **CanopyLens** with the 4-step workflow banner at the top.

---

## Primary 2-Minute Live Demo Flow

| Time | Action | What to Say / Highlight |
| :--- | :--- | :--- |
| **0:00 - 0:20** | **Introduction** | *"Welcome to CanopyLens. It is an automated tool to count individual tree crowns and estimate canopy area from high-resolution imagery, designed with strict scientific honesty."* |
| **0:20 - 0:40** | **Step 1 & 2: Load Data** | Point out the **"Try the Demo Dataset"** panel on the sidebar. Leave **"Use Sample Forest Image"** checked. Check **"Use Sample KML Boundary"**. Highlight: *"A user or judge can immediately test the pipeline without manual uploads, but can also upload custom TIFF/PNG/JPG and KML files."* |
| **0:40 - 0:55** | **Step 3: GSD Resolution** | Point to the **Ground Sampling Distance (GSD)** control set to `0.100 m/pixel`. Explain: *"CanopyLens refuses to invent square-meter measurements. If GSD is missing, it reports pixel counts rather than fabricated real-world figures."* |
| **0:55 - 1:15** | **Step 4: Execution** | Click **"Analyze Forest"**. Show the spinner loading the cached model. Point out the KPI cards: **55 Trees Detected**, **Mean Confidence 78.4%**, **Estimated Canopy Area ~1,850.7 m²**, **AOI Area 9.65 ha**. |
| **1:15 - 1:35** | **Visual Map & Transparency** | Scroll down to the **Visual Map**. Show green bounding boxes and confidence badges. Point to the **Disclaimer box**: *"We explicitly label this as an Estimated Canopy Area (Bounding-Box Proxy) because rectangular boxes overestimate true circular crown area by ~20-25%."* |
| **1:35 - 1:50** | **Exports & Data Table** | Scroll to the **Detected Crown Attributes** table showing individual dimensions, pixel areas, and metric areas. Point to the 3 download buttons: **Annotated Image**, **Predictions CSV**, and **Analysis Report**. Click one to demonstrate instant export. |
| **1:50 - 2:00** | **Conclusion** | *"CanopyLens is fully reproducible, runs locally or in the cloud, and prioritizes honest data over fabricated certainty. Thank you."* |

---

## Fallback Demo Flows

### Fallback A: Testing Without GSD (Honesty Mode)
1. In the sidebar, select **"No GSD (Pixel Analysis Only)"**.
2. Click **"Analyze Forest"**.
3. **Show**: The **Estimated Canopy Area** card displays **"Unavailable"** and clearly states: *"Real-world canopy area cannot be reliably calculated because image ground resolution / GSD is unavailable."*
4. **Talking point**: *"The tool admits what it cannot do—judges specifically requested that we not invent figures."*

### Fallback B: Spatial Alignment Warning
1. With a standard PNG image and KML boundary selected:
2. **Show**: The warning alert: *"ℹ️ Geospatial Notice: KML boundary was successfully parsed, but the uploaded image does not contain sufficient geospatial metadata (CRS & spatial extent) to guarantee pixel-to-boundary alignment."*
3. **Talking point**: *"Instead of drawing a fake alignment overlay across an unreferenced PNG, we compute the KML area independently and disclose the lack of georeferencing."*

### Fallback C: Offline / CLI Mode
If the browser or Streamlit cannot be opened, run the headless CLI test script directly in terminal:
```powershell
.\backend\.venv\Scripts\python.exe backend/test_detection.py data/sample_forest.png
```
This produces terminal logs, `data/predictions.csv`, and `data/detection_result.jpg`.
