import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import tensorflow as tf


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {int(v): k for k, v in class_to_idx.items()}


def predict_image(model, idx_to_class, image_path, img_size):
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = idx_to_class.get(top_idx, "unknown")
    prob = float(preds[top_idx]) * 100
    return label, prob


def main():
    parser = argparse.ArgumentParser(description="Pick an image with a dialog and classify it.")
    parser.add_argument("--model-path", default="fruit_cnn.h5")
    parser.add_argument("--classes-path", default="class_indices.json")
    parser.add_argument("--img-size", type=int, default=100)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise SystemExit(f"Model not found: {args.model_path}")
    if not Path(args.classes_path).exists():
        raise SystemExit(f"Classes file not found: {args.classes_path}")

    model = tf.keras.models.load_model(args.model_path)
    idx_to_class = load_labels(args.classes_path)

    root = tk.Tk()
    root.withdraw()
    root.update()
    file_path = filedialog.askopenfilename(
        title="Select fruit image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
    )
    root.update()

    if not file_path:
        print("No image selected.")
        return

    label, prob = predict_image(model, idx_to_class, file_path, args.img_size)
    msg = f"{Path(file_path).name}\nPrediction: {label}\nConfidence: {prob:.2f}%"
    print(msg)
    messagebox.showinfo("Prediction", msg)


if __name__ == "__main__":
    main()
