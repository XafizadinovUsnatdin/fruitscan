import json
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 100
MODEL_PATH = "fruit_cnn.h5"
CLASSES_PATH = "class_indices.json"


def load_labels(path):
    with open(path, "r") as f:
        class_to_idx = json.load(f)
    return {int(v): k for k, v in class_to_idx.items()}


def main():
    labels = load_labels(CLASSES_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        tensor = resized.astype("float32") / 255.0
        preds = model.predict(tensor[None, ...], verbose=0)[0]
        idx = int(np.argmax(preds))
        label = labels[idx]
        prob = float(preds[idx])

        text = f"{label}: {prob*100:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 220, 10), 2)
        cv2.imshow("Fruit Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
