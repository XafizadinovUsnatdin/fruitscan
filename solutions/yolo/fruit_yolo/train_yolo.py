import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Train YOLO detector for fruits.")
    parser.add_argument("--data", default="fruit_yolo/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="", help="Device ('' auto, 'cpu', or '0','1',...).")
    parser.add_argument("--project", default="runs", help="Project directory for outputs.")
    parser.add_argument("--name", default="fruit_yolo", help="Run name.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Install with: pip install ultralytics opencv-python")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
