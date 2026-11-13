"""Generate STAND-IN faces and voices so the code can run before real samples exist.

These drawings and synthetic tones are NOT the team's data and do not satisfy the
assignment's collection requirement. Any metric computed on them only proves the
code runs: the classes are artificially easy, so accuracy near 100% means nothing.
Once data/raw/images contains member folders, the pipeline ignores this folder.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import signal

from .audio_pipeline import save_wav
from .config import PLACEHOLDER, SAMPLE_RATE

PEOPLE = {
    "member_1": dict(skin=(224, 172, 135), hair=(40, 28, 20), face=(46, 58), eye=20, nose=8, f0=115, formant=1.00),
    "member_2": dict(skin=(141, 85, 57), hair=(15, 12, 12), face=(42, 62), eye=17, nose=11, f0=210, formant=1.15),
    "member_3": dict(skin=(190, 130, 95), hair=(70, 40, 25), face=(50, 54), eye=23, nose=6, f0=140, formant=0.92),
    "member_4": dict(skin=(105, 65, 45), hair=(25, 20, 30), face=(44, 56), eye=19, nose=9, f0=240, formant=1.22),
}
STRANGERS = {
    "unknown_1": dict(skin=(238, 200, 170), hair=(130, 90, 50), face=(52, 64), eye=25, nose=13, f0=95, formant=0.85),
    "unknown_2": dict(skin=(170, 110, 90), hair=(200, 200, 195), face=(38, 50), eye=15, nose=5, f0=175, formant=1.32),
    "unknown_3": dict(skin=(120, 90, 70), hair=(90, 30, 30), face=(55, 50), eye=26, nose=14, f0=280, formant=0.8),
}
VOWELS = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010),
          "o": (570, 840, 2410), "u": (300, 870, 2240), "r": (490, 1350, 1690)}
PHRASE_VOWELS = {"yes_approve": "iea" + "ru", "confirm_transaction": "ourae" + "ou",
                 "demo": "aeio"}


def draw_face(p: dict, expression: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = rng.integers(195, 235)
    img = np.full((256, 256, 3), bg, np.uint8)
    cx, cy = 128 + int(rng.integers(-6, 7)), 132 + int(rng.integers(-6, 7))
    ax, ay = p["face"]
    ax, ay = ax * 2, ay * 2
    cv2.ellipse(img, (cx, cy - ay // 3), (ax + 8, ay), 0, 180, 360, p["hair"], -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, p["skin"], -1, cv2.LINE_AA)
    eye_y, eye_x = cy - ay // 5, p["eye"] * 2
    eye_r = {"neutral": 8, "smiling": 6, "surprised": 12}[expression]
    brow = {"neutral": 20, "smiling": 22, "surprised": 32}[expression]
    for side in (-1, 1):
        cv2.circle(img, (cx + side * eye_x, eye_y), eye_r, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(img, (cx + side * eye_x, eye_y), eye_r // 2, (30, 25, 20), -1, cv2.LINE_AA)
        cv2.line(img, (cx + side * (eye_x - 12), eye_y - brow), (cx + side * (eye_x + 12), eye_y - brow),
                 p["hair"], 4, cv2.LINE_AA)
    nose = tuple(max(0, c - 30) for c in p["skin"])
    cv2.line(img, (cx, eye_y + 8), (cx - p["nose"], cy + 20), nose, 3, cv2.LINE_AA)
    cv2.line(img, (cx - p["nose"], cy + 20), (cx + 4, cy + 22), nose, 3, cv2.LINE_AA)
    mouth_y = cy + ay // 2
    if expression == "smiling":
        cv2.ellipse(img, (cx, mouth_y - 8), (26, 16), 0, 10, 170, (120, 40, 45), 4, cv2.LINE_AA)
    elif expression == "surprised":
        cv2.ellipse(img, (cx, mouth_y), (12, 17), 0, 0, 360, (90, 30, 35), -1, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - 22, mouth_y), (cx + 22, mouth_y), (120, 40, 45), 4, cv2.LINE_AA)
    light = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(1.0, 1.15), 256)[None, :, None]
    noisy = img * light + rng.normal(0, 7, img.shape)
    return np.clip(cv2.GaussianBlur(noisy.astype(np.float32), (3, 3), 0), 0, 255).astype(np.uint8)


def synth_voice(p: dict, phrase: str, seed: int) -> np.ndarray:
    """Source-filter synthesis: a glottal pulse train shaped by vowel formants."""
    rng = np.random.default_rng(seed)
    pieces = []
    for vowel in PHRASE_VOWELS[phrase]:
        n = int(SAMPLE_RATE * rng.uniform(0.17, 0.27))
        t = np.arange(n) / SAMPLE_RATE
        f0 = p["f0"] * rng.uniform(0.97, 1.03) * (1 + 0.012 * np.sin(2 * np.pi * 5.2 * t))
        phase = np.cumsum(f0) / SAMPLE_RATE
        source = signal.sawtooth(2 * np.pi * phase, width=0.1) + 0.015 * rng.normal(size=n)
        out = np.zeros(n)
        for formant, gain in zip(VOWELS[vowel], (1.0, 0.7, 0.35)):
            hz = min(formant * p["formant"], SAMPLE_RATE / 2 - 200)
            b, a = signal.iirpeak(hz, Q=hz / 110, fs=SAMPLE_RATE)
            out += gain * signal.lfilter(b, a, source)
        pieces += [out * np.hanning(n) ** 0.5, 0.002 * rng.normal(size=int(SAMPLE_RATE * 0.025))]
    y = np.concatenate(pieces)
    pad = 0.002 * rng.normal(size=SAMPLE_RATE // 5)
    return np.concatenate([pad, y / np.abs(y).max() * 0.85, pad])


def generate(root: Path = PLACEHOLDER) -> None:
    (root / "README.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "README.txt").write_text(
        "AUTO-GENERATED STAND-INS, NOT REAL PEOPLE. Do not submit these as team data.\n")

    def save_image(path: Path, img: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path, quality=95)

    for m, (name, person) in enumerate(PEOPLE.items()):
        for e, expression in enumerate(("neutral", "smiling", "surprised")):
            save_image(root / "images" / name / f"{name}_{expression}.jpg", draw_face(person, expression, 100 * m + e))
        for k, phrase in enumerate(("yes_approve", "confirm_transaction")):
            save_wav(root / "audio" / name / f"{name}_{phrase}.wav", synth_voice(person, phrase, 500 + 10 * m + k))
        save_image(root / "demo" / name / f"{name}_face.jpg", draw_face(person, "smiling", 9000 + m))
        save_wav(root / "demo" / name / f"{name}_voice.wav", synth_voice(person, "yes_approve", 9500 + m))
    for s, (name, person) in enumerate(STRANGERS.items()):
        if name == "unknown_3":
            save_image(root / "demo" / "unauthorized" / "unknown_face.jpg", draw_face(person, "neutral", 9900))
            save_wav(root / "demo" / "unauthorized" / "unknown_voice.wav", synth_voice(person, "yes_approve", 9901))
            continue
        for e, expression in enumerate(("neutral", "smiling")):
            save_image(root / "unauthorized" / f"{name}_{expression}.jpg", draw_face(person, expression, 7000 + 10 * s + e))
        save_wav(root / "unauthorized" / f"{name}_voice.wav", synth_voice(person, "yes_approve", 7500 + s))


if __name__ == "__main__":
    generate()
