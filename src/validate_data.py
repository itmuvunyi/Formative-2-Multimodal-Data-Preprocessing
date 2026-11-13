"""Check the rubric's collection rules before training on real samples."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import AUDIO_EXTS, EXPRESSIONS, IMAGE_EXTS, PHRASES, RAW


def problems_in(root: Path) -> list[str]:
    problems = []
    image_members = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    audio_dir = root / "audio"
    audio_members = sorted(p.name for p in audio_dir.iterdir() if p.is_dir()) if audio_dir.is_dir() else []
    if len(image_members) < 2:
        problems.append("Need at least 2 members in data/raw/images/")
    if image_members != audio_members:
        problems.append(f"Image members {image_members} differ from audio members {audio_members}")
    for m in image_members:
        stems = {p.stem for p in (root / "images" / m).iterdir() if p.suffix.lower() in IMAGE_EXTS}
        problems += [f"Missing image: images/{m}/{m}_{e}.jpg" for e in EXPRESSIONS if f"{m}_{e}" not in stems]
    for m in audio_members:
        stems = {p.stem for p in (root / "audio" / m).iterdir() if p.suffix.lower() in AUDIO_EXTS}
        problems += [f"Missing audio: audio/{m}/{m}_{ph}.wav" for ph in PHRASES if f"{m}_{ph}" not in stems]
    return problems


def warnings_in(root: Path) -> list[str]:
    notes = []
    demo = root / "demo"
    if not any(p.suffix.lower() in IMAGE_EXTS for p in (demo / "unauthorized").glob("*")) if demo.is_dir() else True:
        notes.append("No demo/unauthorized/unknown_face.* : the unauthorized demo has no face to test")
    if not any(p.suffix.lower() in AUDIO_EXTS for p in (demo / "unauthorized").glob("*")) if demo.is_dir() else True:
        notes.append("No demo/unauthorized/unknown_voice.* : the unauthorized demo has no voice to test")
    if not any((root / "unauthorized").glob("*")):
        notes.append("data/raw/unauthorized/ is empty: strangers are only rejected using genuine-score thresholds")
    return notes


def validate_or_exit(root: Path = RAW) -> None:
    problems = problems_in(root)
    for note in warnings_in(root):
        print(f"  note: {note}")
    if problems:
        print("Fix these before training:\n  - " + "\n  - ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    validate_or_exit()
    print("Data folder is complete.")
