from typing import List, Optional

from pydantic import BaseModel, Field


class PlateDetection(BaseModel):
    bbox_xyxy: List[int] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float


class PlateRecognitionResult(BaseModel):
    text: str
    confidence: Optional[float] = None


class PlateInferenceResult(BaseModel):
    detections: List[PlateDetection]
    recognition: Optional[PlateRecognitionResult] = None
    authorized: Optional[bool] = None
    message: Optional[str] = None


class AuthorizedPlateRequest(BaseModel):
    plate_text: str


class AuthorizedPlateResponse(BaseModel):
    plate_text: str
    authorized: bool
