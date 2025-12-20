import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {int(v): k for k, v in class_to_idx.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Live fruit classification with webcam (uses trained CNN).")
    parser.add_argument("--model-path", default="fruit_cnn.h5", help="Path to trained model (.h5).")
    parser.add_argument("--classes-path", default="class_indices.json", help="Path to class index json.")
    parser.add_argument("--img-size", type=int, default=100, help="Input size used during training.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index for cv2.VideoCapture.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.model_path).exists():
        raise SystemExit(f"Model not found: {args.model_path}")
    if not Path(args.classes_path).exists():
        raise SystemExit(f"Classes file not found: {args.classes_path}")

    labels = load_labels(args.classes_path)
    model = tf.keras.models.load_model(args.model_path)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (args.img_size, args.img_size))
        tensor = resized.astype("float32") / 255.0
        preds = model.predict(np.expand_dims(tensor, axis=0), verbose=0)[0]

        idx = int(np.argmax(preds))
        label = labels.get(idx, "unknown")
        prob = float(preds[idx])

        text = f"{label}: {prob * 100:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 220, 10), 2)
        cv2.imshow("Fruit Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
