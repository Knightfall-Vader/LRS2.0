from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
import io
from PIL import Image, ImageDraw, ImageFont
import json

from app.config import settings
from app.schemas import (
    AuthorizedPlateRequest,
    AuthorizedPlateResponse,
    PlateInferenceResult,
)
from app.services.authorized_store import AuthorizedStore
from app.services.inference import InferenceService
from app.services.text_normalization import normalize_plate_text

app = FastAPI(title=settings.app_name)
store = AuthorizedStore(settings.authorized_plates_path)
service = InferenceService(settings.yolo_weights, settings.trocr_weights_dir)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.post("/infer/image", response_model=PlateInferenceResult)
def infer_image(file: UploadFile = File(...)) -> PlateInferenceResult:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    content = file.file.read()
    result = service.infer_bytes(content)
    if result.recognition is None:
        result.message = "No OCR model loaded. Install weights to enable recognition."
    if result.recognition is not None:
        result.authorized = store.is_authorized(result.recognition.text)
    return result

@app.post("/infer/visualize")
def infer_image_visualize(file: UploadFile = File(...)) -> Response:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    content = file.file.read()
    result = service.infer_bytes(content)
    
    # Load image for drawing
    image = Image.open(io.BytesIO(content)).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw bounding boxes and labels
    for i, detection in enumerate(result.detections):
        # Only draw detections above confidence threshold
        if detection.confidence < settings.confidence_threshold:
            continue
            
        x1, y1, x2, y2 = detection.bbox_xyxy
        
        # Draw rectangle (red color for general detections)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Add confidence score
        conf_text = f"Conf: {detection.confidence:.3f}"
        draw.text((x1, y1 - 20), conf_text, fill="red", font=font)
    
    # Add recognition result if available
    if result.recognition and result.detections:
        # Draw the first detection's bounding box in green if recognition succeeded
        x1, y1, x2, y2 = result.detections[0].bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # Add recognized text
        text = f"Plate: {result.recognition.text}"
        draw.text((x1, y1 - 40), text, fill="green", font=font)
    
    # Add legend
    legend_y = 10
    legend_lines = [
        "Legend:",
        f"Red Box: Detected Region (Conf > {settings.confidence_threshold})",
        "Green Box: Region used for OCR"
    ]
    for line in legend_lines:
        draw.text((10, legend_y), line, fill="blue", font=font)
        legend_y += 25
    
    # Convert image back to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")


@app.post("/authorized", response_model=AuthorizedPlateResponse)
def add_authorized_plate(payload: AuthorizedPlateRequest) -> AuthorizedPlateResponse:
    plate_text = store.add_plate(payload.plate_text)
    return AuthorizedPlateResponse(plate_text=plate_text, authorized=True)


@app.get("/authorized", response_model=list[AuthorizedPlateResponse])
def list_authorized_plates() -> list[AuthorizedPlateResponse]:
    return [AuthorizedPlateResponse(plate_text=plate, authorized=True) for plate in store.list_plates()]


@app.delete("/authorized/{plate_text}")
def remove_authorized_plate(plate_text: str) -> dict:
    store.remove_plate(plate_text)
    return {"removed": normalize_plate_text(plate_text)}


@app.post("/infer/stream")
def infer_stream_placeholder() -> dict:
    return {
        "status": "not_implemented",
        "message": "Camera stream ingestion will be added in a later milestone.",
    }
