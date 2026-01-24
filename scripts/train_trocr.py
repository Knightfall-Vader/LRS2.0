import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for license plates")
    parser.add_argument("--labels", default="dataset/ocr_labels.csv")
    parser.add_argument("--output", default="models/trocr")
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    args = parser.parse_args()

    if importlib.util.find_spec("transformers") is None:
        raise SystemExit("transformers is not installed. pip install transformers")
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("torch is not installed. pip install torch")

    from transformers import AutoProcessor, VisionEncoderDecoderModel

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"labels file not found at {labels_path}")

    df = pd.read_csv(labels_path)
    if "File" not in df.columns or "Plate text" not in df.columns:
        raise SystemExit("labels CSV must include 'File' and 'Plate text' columns")

    processor = AutoProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
