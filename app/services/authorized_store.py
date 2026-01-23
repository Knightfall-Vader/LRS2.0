import json
from pathlib import Path
from typing import List

from app.services.text_normalization import normalize_plate_text


class AuthorizedStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            self._write([])

    def _write(self, plates: List[str]) -> None:
        self.path.write_text(json.dumps({"plates": plates}, indent=2))

    def _read(self) -> List[str]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        return data.get("plates", [])

    def list_plates(self) -> List[str]:
        return sorted(self._read())

    def add_plate(self, plate_text: str) -> str:
        normalized = normalize_plate_text(plate_text)
        plates = set(self._read())
        plates.add(normalized)
        self._write(sorted(plates))
        return normalized

    def remove_plate(self, plate_text: str) -> None:
        normalized = normalize_plate_text(plate_text)
        plates = set(self._read())
        plates.discard(normalized)
        self._write(sorted(plates))

    def is_authorized(self, plate_text: str) -> bool:
        normalized = normalize_plate_text(plate_text)
        plates = set(self._read())
        return normalized in plates
