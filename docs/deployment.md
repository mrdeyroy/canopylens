# CanopyLens Public Deployment Guide

This guide details how to deploy CanopyLens to public cloud hosting platforms.

---

## 1. Selected Deployment Platforms

CanopyLens is built with standard Python and Streamlit, making it ideally suited for:
1. **Streamlit Community Cloud** (Recommended — free, native Streamlit support, zero-configuration GitHub integration).
2. **Hugging Face Spaces** (Alternative — free 16 GB RAM CPU tier, perfect for heavier PyTorch models).

---

## 2. Required Deployment Files

The repository includes all root-level deployment configuration files:
- [`app.py`](file:///c:/Users/Lenovo/Desktop/FloraCarbonAI/app.py): Root entrypoint delegating to `backend/app.py`.
- [`requirements.txt`](file:///c:/Users/Lenovo/Desktop/FloraCarbonAI/requirements.txt): Pinned production dependencies (using `opencv-python-headless` for headless cloud containers).
- [`packages.txt`](file:///c:/Users/Lenovo/Desktop/FloraCarbonAI/packages.txt): System-level Debian packages (`libgl1`, `libglib2.0-0`) required by OpenCV on Linux.
- [`.streamlit/config.toml`](file:///c:/Users/Lenovo/Desktop/FloraCarbonAI/.streamlit/config.toml): Production configuration disabling telemetry prompts and enabling headless mode.

---

## 3. Deployment Steps

### Option A: Streamlit Community Cloud (Recommended)

1. Push your CanopyLens repository to GitHub (ensure `backend/.venv` is not committed).
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **"New app"**.
4. Configure settings:
   - **Repository:** `<your-username>/CanopyLens`
   - **Branch:** `main`
   - **Main file path:** `app.py` (or `backend/app.py`)
   - **Python version:** `3.10`, `3.11`, or `3.12`
5. Click **"Deploy!"**.
6. Streamlit Cloud automatically reads `packages.txt` and `requirements.txt`, downloads the DeepForest weights on first run, and assigns a public URL (e.g. `https://canopylens.streamlit.app`).

### Option B: Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Select **Streamlit** SDK.
3. Push the repository files to the Space git remote.
4. HF Spaces automatically builds the container, allocates 16 GB RAM, and serves the application.

---

## 4. Resource & Memory Considerations
- **Model Size:** DeepForest's RetinaNet/ResNet-50 weights are ~160 MB.
- **Runtime RAM:** The active PyTorch process requires ~1.2 GB to 1.8 GB RAM during inference on 400x400 to 2000x2000 images.
- **Container Limit:** Streamlit Community Cloud provides 1 GB to 2 GB RAM on free tiers. Hugging Face Spaces provides 16 GB RAM, which is ideal for very large images.
- **Guardrails:** CanopyLens includes an automatic 50 MB upload size limit and a 25 megapixel warning to prevent out-of-memory (OOM) crashes in memory-constrained cloud containers.

---

## 5. Local Reproduction Command
To test the exact production launch locally:
```powershell
python -m streamlit run app.py --server.headless true
```
The app will bind to port 8501 without prompts.
