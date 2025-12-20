import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk


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


def build_ui(images, on_select):
    root = tk.Tk()
    root.title("Select a test image")
    root.geometry("1300x900")

    main_frame = ttk.Frame(root, padding=8)
    main_frame.pack(fill="both", expand=True)

    # Left: scrollable thumbnails
    thumb_container = ttk.Frame(main_frame)
    thumb_container.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(thumb_container)
    scrollbar = ttk.Scrollbar(thumb_container, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Right: preview
    preview_frame = ttk.Frame(main_frame, width=450, padding=8)
    preview_frame.pack(side="right", fill="y")
    preview_title = ttk.Label(preview_frame, text="No image selected", font=("Segoe UI", 12, "bold"))
    preview_title.pack(pady=(0, 8))
    preview_label = ttk.Label(preview_frame)
    preview_label.pack()
    preview_info = ttk.Label(preview_frame, text="", font=("Segoe UI", 11))
    preview_info.pack(pady=8)

    thumbs = []
    cols = 4
    for idx, img_path in enumerate(images):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pil_img.thumbnail((200, 200))
            tk_img = ImageTk.PhotoImage(pil_img)
            thumbs.append(tk_img)  # keep reference
        except Exception:
            continue

        frame = ttk.Frame(scroll_frame, borderwidth=1, relief="solid", padding=4)
        img_label = ttk.Label(frame, image=tk_img)
        img_label.image = tk_img  # keep reference on the widget too
        img_label.pack()
        name_label = ttk.Label(frame, text=img_path.name, wraplength=150)
        name_label.pack()

        def bind_click(widget, p=img_path):
            widget.bind("<Button-1>", lambda _e: on_select(p))
        bind_click(frame)
        bind_click(img_label)
        bind_click(name_label)

        r, c = divmod(idx, cols)
        frame.grid(row=r, column=c, padx=6, pady=6, sticky="n")

    return root, preview_label, preview_title, preview_info


def main():
    parser = argparse.ArgumentParser(description="Gallery picker for test images (click to classify).")
    parser.add_argument("--images-dir", default=str(Path("test") / "test"), help="Folder with test images.")
    parser.add_argument("--model-path", default="fruit_cnn.h5")
    parser.add_argument("--classes-path", default="class_indices.json")
    parser.add_argument("--img-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=100, help="Limit how many images to load/display (0 = all).")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    idx_to_class = load_labels(args.classes_path)

    image_dir = Path(args.images_dir)
    if not image_dir.is_dir():
        raise SystemExit(f"Not a folder: {image_dir}")

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit("No images found in the provided folder.")

    preview_photo = None

    def on_select(path: Path):
        nonlocal preview_photo
        label, prob = predict_image(model, idx_to_class, path, args.img_size)
        msg = f"{path.name}\nPrediction: {label}\nConfidence: {prob:.2f}%"
        print(msg)

        # Update preview image and labels
        try:
            pil_img = Image.open(path).convert("RGB")
            pil_img.thumbnail((500, 500))
            preview_photo = ImageTk.PhotoImage(pil_img)
            preview_label.configure(image=preview_photo)
            preview_label.image = preview_photo
        except Exception:
            preview_label.configure(image="")
            preview_label.image = None
        preview_title.config(text=path.name)
        preview_info.config(text=f"{label} ({prob:.2f}%)")

        messagebox.showinfo("Prediction", msg)

    root, preview_label, preview_title, preview_info = build_ui(images, on_select)
    root.mainloop()


if __name__ == "__main__":
    main()
