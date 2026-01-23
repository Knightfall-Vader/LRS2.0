from fastapi import FastAPI, File, HTTPException, UploadFile

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
