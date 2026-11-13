# Multimodal Identity and Product Recommendation

A command-line system that checks a person's **face**, runs a **product recommendation**, confirms their **voice**, and only then shows the prediction.

```
Start -> Face model -> Product model -> Voice model -> Display product
             | fail                         | fail
        Access denied                  Access denied
```

> **Data status:** check `face_voice_data_source` in `reports/metrics.json`. If it says `placeholder`, the face and voice results come from generated drawings and synthetic tones. They only prove the code runs and must not be reported as results.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py             # merge, build features, train, save everything
python app.py                      # interactive menu
python app.py --scenario all       # authorized and unauthorized simulations
python app.py --face me.jpg --voice me.wav
```

To add the team's data, follow [`data/raw/README.md`](data/raw/README.md), then run `python -m src.validate_data` and `python run_pipeline.py`.

## Repository layout

| Path | Contents |
|---|---|
| `data/tabular/` | The two source CSV files |
| `data/processed/merged_dataset.csv` | Merged dataset with engineered features |
| `data/processed/image_features.csv` | HOG, LBP and histogram features for each image and augmentation |
| `data/processed/audio_features.csv` | MFCC, roll-off, energy and pitch features for each clip and augmentation |
| `data/raw/` | Real team samples, once added |
| `data/placeholder/` | Generated stand-ins, ignored once real samples exist |
| `data/augmented/` | Augmented images and audio |
| `data/members.json` | Demo link from each member to a customer record; safe to edit |
| `src/` | Pipeline code |
| `run_pipeline.py` | Runs the three-model training pipeline |
| `app.py` | Command-line app |
| `notebooks/multimodal_pipeline.ipynb` | EDA, processing, evaluation and simulation |
| `models/` | Trained models |
| `reports/` | `metrics.json` and figures |

## Key design decisions

- **One row per transaction.** Social rows are aggregated per customer before an inner join, with a check that rejects any copied row.
- **No future information.** The model never sees the current purchase's amount or rating, and history features use earlier transactions only.
- **No augmentation leakage.** Testing holds out a whole expression or phrase, along with all of its augmented copies.
- **Strangers can be rejected.** A sample must pass both classifier confidence and similarity to the enrolled person.
- **Face and voice must match.** One member's face with another member's voice is denied.

## Known limits

- The product model is close to chance, as explained in the notebook. The cause is mainly the data.
- There is no liveness check, so a photo or a recording could pass.
- The voice check identifies the speaker, not the words spoken.
- Audio features use a custom NumPy and SciPy implementation of MFCCs, not librosa.
