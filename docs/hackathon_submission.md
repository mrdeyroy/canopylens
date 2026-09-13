# CanopyLens: Hackathon Submission Brief & Answers

---

### 3. 2-Page Explanation: Approach, Workflow, Architecture, and Technical Decisions

#### Overview & Approach
CanopyLens is a web application built to count individual trees and estimate their canopy area from aerial, drone, or satellite images. 

Traditional forest surveys require manual fieldwork, which takes days and can be dangerous. Standard satellite imagery tools often look only at greenness (like NDVI) rather than counting actual trees. CanopyLens solves this by running an open-source deep learning model (**DeepForest 2.1.0**) to find tree crowns, draw bounding boxes around them, and calculate the area they cover. 

We focused on three core principles:
1. **Making it easy for anyone to use** without technical training.
2. **Never faking numbers** (if ground resolution or map coordinates are missing, the tool clearly says so instead of guessing).
3. **Keeping the code fast and responsive** on standard computers.

---

#### System Architecture
The application runs in four clear steps:

1. **Input Stage:** The user uploads a forest image (PNG, JPG, or GeoTIFF) and can optionally choose an Area of Interest (AOI).
2. **Deep Learning Detection:** The image passes through DeepForest (RetinaNet with a ResNet-50 backbone pre-trained on airborne forest imagery from the NEON project). The model predicts bounding box coordinates and confidence scores for each tree.
3. **Measurement & Filtering Engine:**
   - **Confidence Filtering:** A dynamic slider lets users adjust the score threshold (from 0.00 to 1.00) instantly without re-running the neural network.
   - **Area of Interest Filtering:** If the user draws a custom box or uploads a KML polygon, detections outside that area are removed.
   - **Canopy Area Calculation:** Uses the Ground Sampling Distance (GSD in meters/pixel) to convert pixel area into square meters ($\text{Area} = \text{Pixel Area} \times \text{GSD}^2$).
4. **Display & Export:** An interactive map highlights detected trees, lets users select and zoom into individual trees, displays key metric cards, and allows downloading results as a CSV table, annotated image, or text report.

---

#### User Workflow
The user interface follows a 4-step sidebar workflow:

- **Step 1: Upload Image** — Upload any aerial or drone photo (or check a box to load the built-in sample image).
- **Step 2: Area of Interest (Optional)** — Choose between three simple options:
  - *Analyze Entire Image* (default, works immediately with no extra files).
  - *Upload KML Boundary* (for GIS boundary files).
  - *Draw Area on Image* (interactive sliders and presets like "Center 50%" to check a specific part of the image).
- **Step 3: Configure Ground Resolution (GSD)** — Pick auto-detect for GeoTIFFs, type a manual value (e.g., 0.10 meters/pixel for drone flights), or select "Pixel Only" mode.
- **Step 4: Run Analysis** — Click "Analyze Forest". The model runs once, and results appear instantly.

---

#### Key Technical Decisions

1. **Post-Inference Filtering for Real-Time Sliders:**
   DeepForest model predictions are saved in session state on the first run. Adjusting the confidence slider or AOI box filters the existing prediction table in memory in under 5 milliseconds. This avoids re-running the heavy model every time a slider moves.

2. **Honesty About Bounding-Box Area (Proxy):**
   Tree crowns are naturally circular or irregular, while bounding boxes are rectangles. A bounding box naturally includes empty corners and overestimates circular crown area by about 20% to 25%. We clearly label the number as **"Estimated Canopy Area (Bounding-Box Proxy)"** so users know it is a model proxy, not a laser-scanned measurement.

3. **Strict Geospatial Validation:**
   If a user uploads a standard JPG/PNG alongside a KML file, CanopyLens calculates the KML polygon area using local UTM projections, but clearly warns that the image cannot be mathematically aligned to the coordinates because standard JPGs have no coordinate header. We never fabricate coordinates.

4. **Interactive Tree Inspector:**
   Dense forests can have dozens of overlapping labels. We added a searchable Tree ID dropdown that highlights the selected tree in bright gold and shows a cropped close-up image along with its exact size and score.

---

### 4. What did you build? (3–5 sentences)
I built **CanopyLens**, an open-source web tool that automatically detects individual tree crowns and estimates forest canopy area from high-resolution aerial and drone images. Users can upload an image, optionally draw an area of interest or upload a KML boundary, and set the ground resolution to calculate real-world square meters. The tool provides interactive tree inspection, dynamic confidence filtering, and instant exports of prediction tables and summary reports. It is designed to work out-of-the-box for non-technical users while remaining scientifically honest about proxy limitations.

---

### 5. What doesn't work? (Honest Limitations & Constraints)

1. **Resolution Limit:** DeepForest requires sub-meter imagery (roughly 5 cm to 50 cm per pixel). Standard low-resolution satellite imagery (like 10-meter Sentinel-2) is too blurry to resolve single tree crowns.
2. **Dense Canopy Overlap:** In very thick, closed-canopy rainforests where tree branches completely intertwine, the detector may merge adjacent crowns into one box or miss suppressed trees underneath.
3. **No Understory Visibility:** Passive 2D RGB optical photos only see what is visible from above. Trees growing entirely under the top canopy layer cannot be seen or counted without 3D LiDAR.
4. **Bounding-Box Overestimation:** Because tree crowns are round or irregular and boxes are square, the calculated canopy area includes background corner pixels and typically overestimates area by 20% to 25%.
5. **Non-Georeferenced Alignment:** If a standard PNG or JPG is uploaded without embedded GeoTIFF tags, the app cannot guarantee spatial alignment with a KML polygon and will not calculate boundary coverage percentages to avoid misleading users.

---

### 6. What technologies did you use?
- **Python 3.12** — Core programming language.
- **DeepForest 2.1.0 (PyTorch & torchvision)** — Machine learning model (RetinaNet with ResNet-50 backbone) pre-trained for tree crown detection.
- **Streamlit** — Web application framework and user interface.
- **OpenCV & Pillow** — Image processing, bounding box rendering, and tree crop extraction.
- **Pandas & NumPy** — Data manipulation and metric aggregation.
- **Rasterio, Shapely, Pyproj & FastKML** — GeoTIFF metadata extraction, polygon parsing, and UTM coordinate area calculations.

---

### 7. Did you use AI coding tools? If yes, which ones?
Yes. I used **Google Antigravity (powered by Gemini)** as an AI pair-programming assistant to help write boilerplate code, build the Streamlit UI components, design modular test suites, and debug CSS styling issues for Streamlit Cloud deployment.

---

### 8. Anything else you'd like the team to know?
CanopyLens was designed with zero external database or cloud API dependencies, meaning it can run locally offline in remote field offices or deploy easily on free tiers of Streamlit Community Cloud and Hugging Face Spaces. The repository includes automated test suites covering edge cases such as empty images, missing GSDs, malformed KMLs, and custom drawn sub-regions.
