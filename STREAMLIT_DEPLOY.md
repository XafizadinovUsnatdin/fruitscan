## Streamlit Deployment Guide

This app supports YOLO (default) and CNN backends. For Streamlit Cloud, keep the small YOLO weights tracked and ready; CNN weights are large and not tracked.

### 1) What the app needs
- Code: `streamlit_app.py`
- YOLO weights: `solutions/yolo/weights/best.pt` (tracked; small)
- Optional CNN weights (large, not tracked by default): `solutions/cnn/fruit_cnn.h5` and `solutions/cnn/class_indices.json`
- Dependencies: `requirements.txt`

### 2) Prepare repository
1. Ensure `solutions/yolo/weights/best.pt` exists (your trained YOLO checkpoint). Commit/push this small file (allowed by `.gitignore`).
2. If you also want CNN in the cloud, add `solutions/cnn/fruit_cnn.h5` and `solutions/cnn/class_indices.json` (note: large; may exceed hosting limits).
3. Commit and push to GitHub.

### 3) Local test
```powershell
cd "D:\Study in Slovakia\3 4 5 Introduction to machine learning\FruitClassification"
.\fcenv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 4) Deploy on Streamlit Cloud
1. Create a new Streamlit app, point to your GitHub repo.
2. Main file: `streamlit_app.py`
3. Requirements file: `requirements.txt`
4. If weights are private/large, upload them via Streamlit Cloud “Files” (same paths as above) or host them remotely and update `streamlit_app.py` to download on startup.
5. Deploy.

### 5) Using the app
- Choose backend: “YOLO (best.pt)” or “CNN (fruit_cnn.h5)”
- Choose source: Upload image, camera snapshot, or live camera (live requires cv2).
- Results show predicted label + confidence; live view overlays text on frames.

### 6) Troubleshooting
- “Model not found”: verify files at `solutions/yolo/weights/best.pt` (and CNN files if using CNN).
- “Cannot open camera” on Cloud: live camera may be unavailable; use uploads instead.
- If TensorFlow is too heavy for Cloud, stick to YOLO-only by using the YOLO backend and omitting CNN weights.
