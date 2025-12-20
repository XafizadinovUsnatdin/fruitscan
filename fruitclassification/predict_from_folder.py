import argparse
import json
from pathlib import Path

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
    return idx_to_class.get(top_idx, "unknown"), float(preds[top_idx])


def main():
    parser = argparse.ArgumentParser(description="Predict all images in a folder with trained fruit CNN.")
    parser.add_argument("--images", required=True, help="Folder containing images (flat).")
    parser.add_argument("--model-path", default="fruit_cnn.h5", help="Path to trained model.")
    parser.add_argument("--classes-path", default="class_indices.json", help="Path to class index json.")
    parser.add_argument("--img-size", type=int, default=100, help="Input size used during training.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process (0 = all).")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    idx_to_class = load_labels(args.classes_path)

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        raise SystemExit(f"Not a folder: {image_dir}")

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit("No images found in the provided folder.")

    for img_path in images:
        label, prob = predict_image(model, idx_to_class, img_path, args.img_size)
        print(f"{img_path.name}: {label} ({prob*100:.2f}%)")


if __name__ == "__main__":
    main()
