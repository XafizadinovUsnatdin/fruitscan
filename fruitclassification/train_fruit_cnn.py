import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model(img_size, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to train folder (class subfolders).")
    parser.add_argument("--val_dir", required=True, help="Path to validation folder (class subfolders).")
    parser.add_argument("--img-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--model-path", default="fruit_cnn.h5")
    parser.add_argument("--classes-path", default="class_indices.json")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    model = build_model(args.img_size, num_classes=train_gen.num_classes)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(args.model_path, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    Path(args.classes_path).write_text(json.dumps(train_gen.class_indices, indent=2))
    print(f"Saved model to {args.model_path} and classes to {args.classes_path}")


if __name__ == "__main__":
    main()
