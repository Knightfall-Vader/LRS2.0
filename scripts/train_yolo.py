import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 plate detector")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    if importlib.util.find_spec("ultralytics") is None:
        raise SystemExit("ultralytics is not installed. pip install ultralytics")

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found at {data_path}")

    model = YOLO(args.model)
    model.train(data=str(data_path), epochs=args.epochs, imgsz=args.img)


if __name__ == "__main__":
    main()
