from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is required. Install with: pip install ultralytics opencv-python streamlit ultralytics"
    ) from exc


# ------------------------
# Model loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")
    return YOLO(str(path))


@st.cache_resource(show_spinner=False)
def load_cnn_model(weights_path: str, classes_path: str):
    w = Path(weights_path)
    c = Path(classes_path)
    if not w.exists():
        raise FileNotFoundError(f"CNN weights not found: {w}")
    if not c.exists():
        raise FileNotFoundError(f"Classes file not found: {c}")
    model = tf.keras.models.load_model(str(w))
    import json
    with open(c, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return model, idx_to_class


# ------------------------
# Predict helpers
# ------------------------
def predict_image_yolo(model, image_bgr: np.ndarray):
    results = model.predict(source=image_bgr, verbose=False)
    if not results:
        return None
    top = results[0].probs
    top_idx = int(top.top1)
    conf = float(top.top1conf)
    label = model.names.get(top_idx, f"class_{top_idx}")
    return label, conf


def predict_image_cnn(model, idx_to_class, image_bgr: np.ndarray, img_size=100):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    arr = resized.astype("float32") / 255.0
    preds = model.predict(arr[None, ...], verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = idx_to_class.get(top_idx, f"class_{top_idx}")
    conf = float(preds[top_idx])
    return label, conf


def main():
    st.set_page_config(page_title="Fruit Classifier", layout="wide")
    st.title("Fruit Classifier")
    st.write("Choose an image or use live camera.")

    backend = st.radio("Model", ["YOLOv8n-cls", "CNN (fruit_cnn.h5)"], horizontal=True)

    if backend == "YOLOv8n-cls":
        yolo_weights = "weights/best.pt"
        loader = lambda: load_yolo_model(yolo_weights)
    else:
        cnn_weights = "fruit_cnn.h5"
        classes_file = "class_indices.json"
        loader = lambda: load_cnn_model(cnn_weights, classes_file)

    # Load model once
    try:
        loaded = loader()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    # Normalize predict_fn for both backends
    if backend == "YOLOv8n-cls":
        predict_fn = lambda frame: predict_image_yolo(loaded, frame)
    else:
        cnn_model, idx_to_class = loaded
        predict_fn = lambda frame: predict_image_cnn(cnn_model, idx_to_class, frame, img_size=100)

    source = st.radio("Source", ["Image upload", "Camera snapshot", "Camera live"], horizontal=True)

    if source == "Image upload":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file:
            image_bytes = file.read()
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image_array is None:
                st.error("Could not read the image.")
            else:
                pred = predict_fn(image_array)
                if pred:
                    label, conf = pred
                    st.success(f"{label} ({conf*100:.2f}%)")
                st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), caption="Input image")

    elif source == "Camera snapshot":
        cam_image = st.camera_input("Take a photo")
        if cam_image:
            image_bytes = cam_image.getvalue()
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image_array is None:
                st.error("Could not read the camera image.")
            else:
                pred = predict_fn(image_array)
                if pred:
                    label, conf = pred
                    st.success(f"{label} ({conf*100:.2f}%)")
                st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), caption="Camera image")

    else:  # Camera live
        st.write("Start live camera.")
        if st.button("Start camera"):
            placeholder = st.empty()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open camera.")
            else:
                frame_count = 0
                max_live_frames = 200
                while frame_count < max_live_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    pred = predict_fn(frame)
                    label_text = ""
                    if pred:
                        label, conf = pred
                        label_text = f"{label} ({conf*100:.1f}%)"
                        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 220, 10), 2)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    placeholder.image(rgb, caption=label_text if label_text else "Camera")
                    frame_count += 1
                    time.sleep(0.05)
                cap.release()
                st.info(f"Finished reading {frame_count} frames.")

    # No extra footer text; interface kept minimal


if __name__ == "__main__":
    main()
