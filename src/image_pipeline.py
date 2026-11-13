"""Task 2: load faces, augment them, extract features, write image_features.csv."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

from .config import AUGMENTED, EXPRESSIONS, IMAGE_EXTS, IMAGE_SIZE, PROCESSED

_LBP_POINTS, _LBP_RADIUS = 8, 1
_NAME = re.compile(r"^(?P<member>.+)_(?P<kind>" + "|".join(EXPRESSIONS) + r")(_\d+)?$")
_detector = None


def read_image(path: str | Path) -> np.ndarray:
    """RGB uint8. Applies EXIF rotation, which phone photos depend on."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def _face_detector():
    """OpenCV's bundled Haar cascade; None if this OpenCV build ships without it."""
    global _detector
    if _detector is None:
        data = getattr(cv2, "data", None)
        path = Path(getattr(data, "haarcascades", "")) / "haarcascade_frontalface_default.xml"
        _detector = cv2.CascadeClassifier(str(path)) if path.is_file() else False
    return _detector or None


def _center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return img[top: top + side, left: left + side]


def crop_face(img: np.ndarray, detect: bool = True) -> tuple[np.ndarray, bool | None]:
    """Return (face resized to IMAGE_SIZE, face_found). Falls back to a centre crop.

    face_found is None when no detection ran, because detect=False or no cascade exists.
    """
    box = None
    detector = _face_detector() if detect else None
    if detect and detector is None:
        warnings.warn("No Haar cascade in this OpenCV build (install opencv-python<5). "
                      "Using a centre crop and skipping the no-face check.", stacklevel=2)
    if detector is not None:
        scale = min(1.0, 800 / max(img.shape[:2]))
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        found = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(found):
            x, y, w, h = max(found, key=lambda b: b[2] * b[3])
            m = int(0.15 * h)  # keep some forehead and chin
            box = [max(0, int((y - m) / scale)), int((y + h + m) / scale),
                   max(0, int((x - m) / scale)), int((x + w + m) / scale)]
    crop = img[box[0]:box[1], box[2]:box[3]] if box else _center_square(img)
    size = (IMAGE_SIZE, IMAGE_SIZE)
    found = None if detector is None else box is not None
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA), found


def _rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment(img: np.ndarray) -> dict[str, np.ndarray]:
    """Five label-preserving variants, plus the original."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "original": img,
        "rotate_+12": _rotate(img, 12),
        "rotate_-12": _rotate(img, -12),
        "flip": cv2.flip(img, 1),
        "grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "brightness": cv2.convertScaleAbs(img, alpha=1.15, beta=18),
    }


def extract_features(img: np.ndarray) -> dict[str, float]:
    """HOG = face shape, LBP = skin/feature texture, histograms = tone and colour."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    feats: dict[str, float] = {}
    hog_vec = hog(equalized, orientations=9, pixels_per_cell=(24, 24), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)
    feats.update({f"hog_{i}": float(v) for i, v in enumerate(hog_vec)})
    lbp = local_binary_pattern(equalized, _LBP_POINTS, _LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=_LBP_POINTS + 2, range=(0, _LBP_POINTS + 2), density=True)
    feats.update({f"lbp_{i}": float(v) for i, v in enumerate(lbp_hist)})
    gray_hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    feats.update({f"gray_hist_{i}": float(v) for i, v in enumerate(gray_hist)})
    for c, name in enumerate("rgb"):
        hist, _ = np.histogram(img[..., c], bins=16, range=(0, 256), density=True)
        feats.update({f"{name}_hist_{i}": float(v) for i, v in enumerate(hist)})
    return feats


def list_enrolled(root: Path) -> list[dict]:
    items = []
    base = root / "images"
    for member_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(member_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            match = _NAME.match(path.stem)
            if not match or match["member"] != member_dir.name:
                raise ValueError(f"Bad image name {path}. Expected "
                                 f"{member_dir.name}_<{'|'.join(EXPRESSIONS)}>{path.suffix}")
            items.append({"path": path, "member": member_dir.name, "expression": match["kind"]})
    return items


def build_image_features(root: Path, source: str, save: bool = True) -> pd.DataFrame:
    rows = []
    enrolled = list_enrolled(root)
    unknown = [{"path": p, "member": "unauthorized", "expression": "unknown"}
               for p in sorted((root / "unauthorized").glob("*"))
               if p.suffix.lower() in IMAGE_EXTS]
    no_face = []
    for item in enrolled + unknown:
        face, found = crop_face(read_image(item["path"]), detect=source == "real")
        if found is False:
            no_face.append(item["path"].name)
        for aug_name, aug_img in augment(face).items():
            if save:
                out = AUGMENTED / "images" / item["member"] / f"{item['path'].stem}__{aug_name}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(aug_img).save(out, quality=90)
            rows.append({
                "image_id": f"{item['path'].stem}__{aug_name}", "member": item["member"],
                "expression": item["expression"], "augmentation": aug_name,
                "source_file": item["path"].name, "face_detected": -1 if found is None else int(found),
                "data_source": source, **extract_features(aug_img),
            })
    if no_face:
        warnings.warn(f"No face detected, used a centre crop: {no_face}", stacklevel=2)
    df = pd.DataFrame(rows)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "image_features.csv", index=False)
    return df


META_COLS = ["image_id", "member", "expression", "augmentation", "source_file", "face_detected", "data_source"]
