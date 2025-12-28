"""
Simple YOLOv8/YOLO11 webcam (or file) detector for fruits.

Prereqs:
  pip install ultralytics opencv-python

Example (webcam):
  python fruit_yolo/webcam_yolo.py --model best.pt --source 0 --conf 0.5

Example (folder of images):
  python fruit_yolo/webcam_yolo.py --model best.pt --source test/test --conf 0.5
"""

import argparse
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description="YOLO detector for fruit webcam or image/video sources.")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model (trained detector).")
    parser.add_argument("--source", default="0", help="0 for default webcam, or path to image/video/folder/RTSP/URL.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="", help="Device to run on ('' autodetect, 'cpu', or '0', '1', ... for GPU).")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Install with: pip install ultralytics opencv-python")

    # Parse webcam index if the source is a simple digit
    src = args.source
    if src.isdigit():
        src = int(src)

    model = YOLO(args.model)

    # model.predict with stream=True yields results + frames
    for res in model.predict(
        source=src,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device or None,
        stream=True,
        verbose=False,
        show=False,
    ):
        frame = res.plot()
        cv2.imshow("YOLO Fruit Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
