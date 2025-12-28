import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(img_size, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(img_size, img_size, 3)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def make_train_val_gens(data_root, img_size, batch_size, val_split):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=val_split,
    )
    train_gen = datagen.flow_from_directory(
        data_root,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        data_root,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    return train_gen, val_gen


def train(args):
    train_gen, val_gen = make_train_val_gens(args.data_root, args.img_size, args.batch_size, args.val_split)
    model = build_model(args.img_size, train_gen.num_classes)

    class StopAtAcc(tf.keras.callbacks.Callback):
        def __init__(self, target):
            super().__init__()
            self.target = target

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = logs.get("val_accuracy") or logs.get("accuracy")
            if acc is not None and acc >= self.target:
                self.model.stop_training = True
                print(f"\nReached target accuracy {acc:.4f} >= {self.target:.2f}, stopping training.")

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(args.model_path, save_best_only=True, monitor="val_loss"),
        StopAtAcc(args.target_acc),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch or None,
        validation_steps=args.validation_steps or None,
        callbacks=callbacks,
        verbose=1,
    )

    Path(args.classes_path).write_text(json.dumps(train_gen.class_indices, indent=2), encoding="utf-8")
    print(f"Saved model -> {args.model_path}")
    print(f"Saved classes -> {args.classes_path}")
    print(f"Final val acc: {history.history['val_accuracy'][-1]:.4f}")


def predict_webcam(args):
    if not Path(args.model_path).exists():
        raise SystemExit(f"Model not found: {args.model_path}")
    if not Path(args.classes_path).exists():
        raise SystemExit(f"Classes file not found: {args.classes_path}")

    with open(args.classes_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
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
        label = idx_to_class.get(idx, "unknown")
        prob = float(preds[idx])

        text = f"{label}: {prob * 100:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 220, 10), 2)
        cv2.imshow("Fruit Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_image(args):
    if not Path(args.image).exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not Path(args.model_path).exists():
        raise SystemExit(f"Model not found: {args.model_path}")
    if not Path(args.classes_path).exists():
        raise SystemExit(f"Classes file not found: {args.classes_path}")

    with open(args.classes_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    model = tf.keras.models.load_model(args.model_path)

    img = tf.keras.utils.load_img(args.image, target_size=(args.img_size, args.img_size))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(preds))
    print(f"Predicted: {idx_to_class.get(idx, 'unknown')} ({preds[idx] * 100:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on fruit dataset and run webcam demo.")
    parser.add_argument("--mode", choices=["train", "webcam", "predict"], required=True)
    parser.add_argument("--data-root", default=str(Path("train") / "train"),
                        help="Root folder with class subfolders (default: train/train).")
    parser.add_argument("--img-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Optional cap on train steps per epoch (0 = auto/len(dataset)).")
    parser.add_argument("--validation-steps", type=int, default=0,
                        help="Optional cap on validation steps per epoch (0 = auto/len(dataset)).")
    parser.add_argument("--target-acc", type=float, default=0.9,
                        help="Stop training early if val_accuracy reaches this value.")
    parser.add_argument("--model-path", default="fruit_cnn.h5")
    parser.add_argument("--classes-path", default="class_indices.json")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--image", help="Image path for predict mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "webcam":
        predict_webcam(args)
    elif args.mode == "predict":
        if not args.image:
            raise SystemExit("Predict mode requires --image.")
        predict_image(args)


if __name__ == "__main__":
    main()
