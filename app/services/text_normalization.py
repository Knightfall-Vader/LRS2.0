import re

_ALLOWED_PATTERN = re.compile(r"[^A-Z0-9 ]+")


def normalize_plate_text(text: str) -> str:
    cleaned = text.upper()
    cleaned = _ALLOWED_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
