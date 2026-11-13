"""Shared paths and settings for the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "data" / "tabular"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
PLACEHOLDER = ROOT / "data" / "placeholder"
AUGMENTED = ROOT / "data" / "augmented"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

EXPRESSIONS = ["neutral", "smiling", "surprised"]
PHRASES = ["yes_approve", "confirm_transaction"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac"}

IMAGE_SIZE = 128
SAMPLE_RATE = 16_000
RANDOM_STATE = 42


def active_data_root() -> tuple[Path, str]:
    """Use real team data as soon as it exists; otherwise the labelled placeholders."""
    has_real = (RAW / "images").is_dir() and any(p.is_dir() for p in (RAW / "images").iterdir())
    if has_real:
        return RAW, "real"
    return PLACEHOLDER, "placeholder"
