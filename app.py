#!/usr/bin/env python3
"""Command-line demo of the User Identity and Product Recommendation flow.

    Start -> Face model -> Product model -> Voice model -> Display product
                 | fail                         | fail
            Access denied                  Access denied

    python app.py                                   interactive menu
    python app.py --face me.jpg --voice me.wav      check your own files
    python app.py --scenario authorized             one scripted case
    python app.py --scenario all                    every scripted case
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import audio_pipeline, image_pipeline
from src.config import AUDIO_EXTS, IMAGE_EXTS, MODELS, PROCESSED, REPORTS, active_data_root
from src.tabular import latest_customer_features

SCENARIOS = ["authorized", "unknown_face", "unknown_voice", "voice_of_other_member"]
LINE = "-" * 62


class System:
    def __init__(self):
        missing = [n for n in ("face", "voice", "product") if not (MODELS / f"{n}_model.joblib").exists()]
        if missing:
            sys.exit(f"Missing models: {missing}. Run: python run_pipeline.py")
        self.face = joblib.load(MODELS / "face_model.joblib")
        self.voice = joblib.load(MODELS / "voice_model.joblib")
        self.product = joblib.load(MODELS / "product_model.joblib")
        self.merged = pd.read_csv(PROCESSED / "merged_dataset.csv", parse_dates=["purchase_date"])
        info = json.loads((REPORTS / "metrics.json").read_text())
        self.customers = info["member_customer_map"]
        self.source = info["face_voice_data_source"]
        report = info["product"]
        self.product_accuracy = report["cv_customer_grouped"][report["selected_model"]]["accuracy"]
        self.root, _ = active_data_root()

    def face_features(self, path) -> tuple[np.ndarray, bool | None]:
        crop, found = image_pipeline.crop_face(image_pipeline.read_image(path), detect=self.source == "real")
        feats = image_pipeline.extract_features(crop)
        return np.array([feats[c] for c in self.face.feature_cols_]), found

    def voice_features(self, path) -> np.ndarray:
        feats = audio_pipeline.extract_features(audio_pipeline.load_audio(path))
        return np.array([feats[c] for c in self.voice.feature_cols_])

    def recommend(self, member: str) -> list[tuple[str, float]]:
        row = latest_customer_features(self.merged, self.customers[member])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            proba = self.product.predict_proba(row)[0]
        order = np.argsort(proba)[::-1]
        return [(str(self.product.classes_[i]), float(proba[i])) for i in order]


def _gate(result: dict) -> str:
    return (f"confidence {result['confidence']:.2f} (needs {result['min_confidence']:.2f}), "
            f"similarity {result['similarity']:.2f} (needs {result['min_similarity']:.2f})")


def run_transaction(system: System, face_path, voice_path) -> dict:
    """Run the flow in the diagram's order. Returns the outcome for tests and the notebook."""
    print(f"\n[Step 1] Facial recognition: {Path(face_path).name}")
    features, found = system.face_features(face_path)
    if found is False:
        print("         No face found in the image.\n\n  >>> ACCESS DENIED at face check")
        return {"granted": False, "stopped_at": "face", "reason": "no_face"}
    face = system.face.identify(features)
    print(f"         {_gate(face)}")
    if face["member"] is None:
        print(f"         Not an enrolled member.\n\n  >>> ACCESS DENIED at face check")
        return {"granted": False, "stopped_at": "face", "reason": "unknown_face"}
    member = face["member"]
    print(f"         Recognised {member}. Product model unlocked.")

    print(f"\n[Step 2] Product recommendation model (customer {system.customers[member]})")
    ranking = system.recommend(member)
    print("         Prediction computed and held until voice approval.")

    print(f"\n[Step 3] Voice validation: {Path(voice_path).name}")
    voice = system.voice.identify(system.voice_features(voice_path))
    print(f"         {_gate(voice)}")
    if voice["member"] is None:
        print("         Not an approved voice.\n\n  >>> ACCESS DENIED at voice check. Prediction discarded.")
        return {"granted": False, "stopped_at": "voice", "reason": "unknown_voice"}
    if voice["member"] != member:
        print(f"         Voice belongs to {voice['member']}, but the face belongs to {member}.")
        print("\n  >>> ACCESS DENIED at voice check. Prediction discarded.")
        return {"granted": False, "stopped_at": "voice", "reason": "face_voice_mismatch"}
    print(f"         Voice matches {member}.")

    best, chance = ranking[0]
    print(f"\n[Step 4] ACCESS GRANTED for {member}\n\n  >>> PREDICTED PRODUCT: {best}  (probability {chance:.0%})")
    print("      Next: " + ", ".join(f"{name} {p:.0%}" for name, p in ranking[1:3]))
    print(f"      Tested accuracy of this model is about {system.product_accuracy:.0%}: treat it as a weak suggestion.")
    return {"granted": True, "member": member, "product": best, "probabilities": dict(ranking)}


def _first(folder: Path, exts: set[str]) -> Path | None:
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in exts)
    return files[0] if files else None


def scenario_files(system: System, name: str) -> tuple[Path, Path]:
    demo = system.root / "demo"
    members = system.face.classes_.tolist()
    first, second = members[0], members[1]

    def sample(member: str, exts: set[str], fallback: str) -> Path:
        path = _first(demo / member, exts)
        if path:
            return path
        path = _first(system.root / fallback / member, exts)
        print(f"  note: no unseen demo sample for {member}; using a training file, which flatters the result")
        return path

    def stranger(exts: set[str], word: str) -> Path:
        path = _first(demo / "unauthorized", exts)
        if not path:
            sys.exit(f"Add an unauthorized {word} sample to {demo / 'unauthorized'}")
        return path

    face, voice = sample(first, IMAGE_EXTS, "images"), sample(first, AUDIO_EXTS, "audio")
    if name == "unknown_face":
        face = stranger(IMAGE_EXTS, "face")
    elif name == "unknown_voice":
        voice = stranger(AUDIO_EXTS, "voice")
    elif name == "voice_of_other_member":
        voice = sample(second, AUDIO_EXTS, "audio")
    return face, voice


def run_scenario(system: System, name: str) -> dict:
    print(f"\n{LINE}\nSCENARIO: {name.replace('_', ' ')}\n{LINE}")
    return run_transaction(system, *scenario_files(system, name))


def banner(system: System) -> None:
    print(f"{LINE}\n  Identity and Product Recommendation System\n{LINE}")
    if system.source == "placeholder":
        print("  WARNING: running on generated stand-in faces and voices, not real people.")
    print("  Enrolled: " + ", ".join(system.face.classes_.tolist()))


def interactive(system: System) -> None:
    menu = {str(i + 1): s for i, s in enumerate(SCENARIOS)}
    while True:
        print()
        for key, name in menu.items():
            print(f"  {key}. Simulate: {name.replace('_', ' ')}")
        print("  5. Use my own face and voice files\n  q. Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "q":
            return
        if choice in menu:
            run_scenario(system, menu[choice])
        elif choice == "5":
            face = input("Face image path: ").strip().strip('"')
            voice = input("Voice file path: ").strip().strip('"')
            if Path(face).is_file() and Path(voice).is_file():
                run_transaction(system, face, voice)
            else:
                print("One of those files does not exist.")
        else:
            print("Please choose from the menu.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--face", help="path to a face image")
    parser.add_argument("--voice", help="path to a voice recording")
    parser.add_argument("--scenario", choices=SCENARIOS + ["all"], help="run a scripted demonstration")
    args = parser.parse_args()
    if bool(args.face) != bool(args.voice):
        parser.error("--face and --voice go together")
    system = System()
    banner(system)
    if args.face:
        return 0 if run_transaction(system, args.face, args.voice)["granted"] else 1
    if args.scenario:
        for name in SCENARIOS if args.scenario == "all" else [args.scenario]:
            run_scenario(system, name)
        return 0
    interactive(system)
    return 0


if __name__ == "__main__":
    sys.exit(main())
