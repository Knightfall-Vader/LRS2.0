import io
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from app.schemas import PlateDetection, PlateInferenceResult, PlateRecognitionResult
from app.services.text_normalization import normalize_plate_text


class InferenceService:
    def __init__(self, yolo_weights: Path, trocr_dir: Path) -> None:
        self.yolo_weights = yolo_weights
        self.trocr_dir = trocr_dir
        self._yolo_model = None
        self._trocr_model = None
        self._trocr_processor = None

    def _load_yolo(self) -> None:
        if self._yolo_model is not None:
            return
        if not self.yolo_weights.exists():
            return
        if importlib.util.find_spec("ultralytics") is None:
            return
        from ultralytics import YOLO

        self._yolo_model = YOLO(str(self.yolo_weights))

    def _load_trocr(self) -> None:
        if self._trocr_model is not None:
            return
        if not self.trocr_dir.exists():
            return
        if importlib.util.find_spec("transformers") is None:
            return
        from transformers import AutoProcessor, VisionEncoderDecoderModel

        self._trocr_processor = AutoProcessor.from_pretrained(str(self.trocr_dir))
        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(str(self.trocr_dir))

    def _detect(self, image: Image.Image) -> List[PlateDetection]:
        self._load_yolo()
        if self._yolo_model is None:
            return []
        results = self._yolo_model.predict(source=np.array(image), verbose=False)
        detections: List[PlateDetection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    PlateDetection(
                        bbox_xyxy=[x1, y1, x2, y2],
                        confidence=float(box.conf[0].item()),
                    )
                )
        return detections

    def _crop_first(self, image: Image.Image, detections: List[PlateDetection]) -> Optional[Image.Image]:
        if not detections:
            return None
        x1, y1, x2, y2 = detections[0].bbox_xyxy
        return image.crop((x1, y1, x2, y2))

    def _recognize(self, plate_image: Image.Image) -> Optional[PlateRecognitionResult]:
        self._load_trocr()
        if self._trocr_model is None or self._trocr_processor is None:
            return None
        pixel_values = self._trocr_processor(images=plate_image, return_tensors="pt").pixel_values
        generated_ids = self._trocr_model.generate(pixel_values)
        text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        normalized = normalize_plate_text(text)
        return PlateRecognitionResult(text=normalized)

    def infer_bytes(self, content: bytes) -> PlateInferenceResult:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        detections = self._detect(image)
        plate_crop = self._crop_first(image, detections)
        recognition = self._recognize(plate_crop) if plate_crop else None
        return PlateInferenceResult(detections=detections, recognition=recognition)
