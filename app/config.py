from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "LRS2.0"
    models_dir: Path = Path("models")
    yolo_weights: Path = Path("models/plate_detector.pt")
    trocr_weights_dir: Path = Path("models/trocr")
    authorized_plates_path: Path = Path("authorized_plates.json")
    input_size: int = 640

    class Config:
        env_prefix = "LRS_"


settings = Settings()
