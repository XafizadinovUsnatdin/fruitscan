import argparse
import json
from pathlib import Path

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


def make_gens(train_dir, val_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


def train(args):
    train_gen, val_gen = make_gens(args.train_dir, args.val_dir, args.img_size, args.batch_size)
    model = build_model(args.img_size, train_gen.num_classes)

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(args.model_path, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    Path(args.classes_path).write_text(json.dumps(train_gen.class_indices, indent=2))
    print(f"Saved model -> {args.model_path}")
    print(f"Saved classes -> {args.classes_path}")
    final_acc = history.history["val_accuracy"][-1]
    print(f"Final val accuracy: {final_acc:.4f}")


def evaluate(args):
    model = tf.keras.models.load_model(args.model_path)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"Test loss: {loss:.4f} | Test acc: {acc:.4f}")


def predict_single(args):
    model = tf.keras.models.load_model(args.model_path)
    with open(args.classes_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    img = tf.keras.utils.load_img(args.image, target_size=(args.img_size, args.img_size))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    top_idx = int(np.argmax(preds))
    print(f"Predicted: {idx_to_class[top_idx]} ({preds[top_idx] * 100:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Fruit classification CNN (inspired by Kaggle 98% example).")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--train-dir", help="Training directory with class subfolders.")
    parser.add_argument("--val-dir", help="Validation directory with class subfolders.")
    parser.add_argument("--test-dir", help="Test directory with class subfolders for eval mode.")
    parser.add_argument("--image", help="Image path for predict mode.")
    parser.add_argument("--img-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model-path", default="fruit_cnn.h5")
    parser.add_argument("--classes-path", default="class_indices.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        if not args.train_dir or not args.val_dir:
            raise SystemExit("Training requires --train-dir and --val-dir.")
        train(args)
    elif args.mode == "eval":
        if not args.test_dir:
            raise SystemExit("Eval mode requires --test-dir.")
        evaluate(args)
    elif args.mode == "predict":
        if not args.image:
            raise SystemExit("Predict mode requires --image.")
        predict_single(args)


if __name__ == "__main__":
    main()
