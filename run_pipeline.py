#!/usr/bin/env python3
"""Run everything: merge -> features -> train -> save models and metrics.

    python run_pipeline.py

Real samples in data/raw/ are used automatically. Without them, labelled stand-ins are
generated so the code still runs, and every output says so.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib

from src import audio_pipeline, image_pipeline, models, tabular
from src.config import AUGMENTED, MODELS, PLACEHOLDER, REPORTS, active_data_root
from src.eda import plot_confusion
from src.placeholders import generate
from src.validate_data import validate_or_exit


def assign_customers(merged, members: list[str]) -> dict[str, int]:
    """Demo link between people and customers. The source data has no such link."""
    path = MODELS.parent / "data" / "members.json"
    mapping = json.loads(path.read_text()) if path.exists() else {}
    if set(mapping) != set(members):
        top = merged["customer_id"].value_counts(sort=True).sort_index(ascending=True) \
            .sort_values(ascending=False, kind="stable").index[:len(members)]
        mapping = {m: int(c) for m, c in zip(members, top)}
        path.write_text(json.dumps(mapping, indent=2) + "\n")
    return mapping


def main() -> None:
    root, source = active_data_root()
    if source == "placeholder":
        generate(PLACEHOLDER)
        print("!" * 74 + "\n  NO REAL SAMPLES IN data/raw/ : USING GENERATED STAND-INS.\n"
              "  Face and voice metrics below only prove the code runs. Do not report them.\n" + "!" * 74)
    else:
        validate_or_exit(root)
    print(f"[1/5] Merging tables (faces and voices from: {source})")
    merged, merge_log = tabular.build_dataset()
    print(f"      {merge_log['merge']['rows_after_merge']} rows, "
          f"{merge_log['merge']['customers_matched']} customers")

    shutil.rmtree(AUGMENTED, ignore_errors=True)
    print("[2/5] Image augmentation and features")
    images = image_pipeline.build_image_features(root, source)
    print(f"      image_features.csv: {images.shape[0]} rows, {images.shape[1]} columns")
    print("[3/5] Audio augmentation and features")
    audio = audio_pipeline.build_audio_features(root, source)
    print(f"      audio_features.csv: {audio.shape[0]} rows, {audio.shape[1]} columns")

    print("[4/5] Training face and voice models")
    face = models.train_identity(images, image_pipeline.META_COLS, "expression", "face")
    voice = models.train_identity(audio, audio_pipeline.META_COLS, "phrase", "voice")
    print("[5/5] Training product model (this runs a label-shuffle test, about a minute)")
    product, product_report = models.train_product(merged)

    members = face.classes_.tolist()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(face, MODELS / "face_model.joblib")
    joblib.dump(voice, MODELS / "voice_model.joblib")
    joblib.dump(product, MODELS / "product_model.joblib")
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "face_voice_data_source": source,
        "member_customer_map": assign_customers(merged, members),
        "merge": merge_log, "face": face.summary(), "voice": voice.summary(), "product": product_report,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_confusion(face.confusion_, members, "Face model, held-out expressions", "confusion_face.png")
    plot_confusion(voice.confusion_, voice.classes_.tolist(), "Voice model, held-out phrases", "confusion_voice.png")
    plot_confusion(product_report["confusion_matrix"], product_report["classes"],
                   "Product model, customer-grouped CV", "confusion_product.png")

    def line(name, summary):
        r = summary["cv"][summary["selected_model"]]
        return f"  {name:<8}{summary['selected_model']:<22}acc {r['accuracy']:.3f}  f1 {r['f1_macro']:.3f}  loss {r['log_loss']:.3f}"

    p = product_report["cv_customer_grouped"]
    best = p[product_report["selected_model"]]
    print("\nResults")
    print(line("face", metrics["face"]) + ("  [stand-in data]" if source == "placeholder" else ""))
    print(line("voice", metrics["voice"]) + ("  [stand-in data]" if source == "placeholder" else ""))
    print(f"  {'product':<8}{product_report['selected_model']:<22}acc {best['accuracy']:.3f}  "
          f"f1 {best['f1_macro']:.3f}  loss {best['log_loss']:.3f}")
    print(f"  baseline for product: acc {p['baseline_class_frequency']['accuracy']:.3f}, "
          f"loss {p['baseline_class_frequency']['log_loss']:.3f}")


if __name__ == "__main__":
    main()
