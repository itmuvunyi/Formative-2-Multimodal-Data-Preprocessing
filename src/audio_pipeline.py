"""Task 3: load voices, augment them, extract features, write audio_features.csv.

The signal processing uses NumPy and SciPy directly, so the MFCC steps are visible
in this file and there is no compiled audio dependency to install. Formats other
than WAV are converted with ffmpeg, if it is installed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct
from scipy.io import wavfile

from .config import AUDIO_EXTS, AUGMENTED, PHRASES, PROCESSED, RANDOM_STATE, SAMPLE_RATE

N_FFT, HOP, N_MELS, N_MFCC = 512, 160, 40, 13
_NAME = re.compile(r"^(?P<member>.+)_(?P<kind>" + "|".join(PHRASES) + r")(_\d+)?$")


def _decode_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    data = data.astype(np.float64) if np.issubdtype(data.dtype, np.floating) else (
        (data.astype(np.float64) - 128) / 128 if data.dtype == np.uint8
        else data.astype(np.float64) / np.iinfo(data.dtype).max)
    return rate, data.mean(axis=1) if data.ndim == 2 else data


def load_audio(path: str | Path, trim: bool = True) -> np.ndarray:
    """Mono float audio at SAMPLE_RATE, peak-normalised, leading/trailing silence removed."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        rate, y = _decode_wav(path)
    else:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"{path.name}: install ffmpeg or export the recording as .wav")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clip.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), str(out)], check=True)
            rate, y = _decode_wav(out)
    if rate != SAMPLE_RATE:
        divisor = np.gcd(rate, SAMPLE_RATE)
        y = signal.resample_poly(y, SAMPLE_RATE // divisor, rate // divisor)
    peak = np.abs(y).max()
    if peak < 1e-4:
        raise ValueError(f"{path.name} is silent")
    y = y / peak
    return trim_silence(y) if trim else y


def _rms_frames(y: np.ndarray, frame: int = 400, hop: int = HOP) -> np.ndarray:
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    windows = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    return np.sqrt(np.mean(windows ** 2, axis=1))


def trim_silence(y: np.ndarray, db_below_peak: float = 35.0) -> np.ndarray:
    rms = _rms_frames(y)
    keep = np.flatnonzero(rms > rms.max() * 10 ** (-db_below_peak / 20))
    return y[max(0, keep[0] * HOP - 800): min(len(y), keep[-1] * HOP + 1200)]


def stft_power(y: np.ndarray) -> np.ndarray:
    _, _, z = signal.stft(y, SAMPLE_RATE, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    return np.abs(z) ** 2


def _mel_filterbank() -> np.ndarray:
    def to_mel(f):
        return 2595 * np.log10(1 + np.asarray(f) / 700)

    def to_hz(m):
        return 700 * (10 ** (np.asarray(m) / 2595) - 1)

    points = to_hz(np.linspace(to_mel(20), to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    bank = np.zeros((N_MELS, freqs.size))
    for i in range(N_MELS):
        low, mid, high = points[i:i + 3]
        bank[i] = np.clip(np.minimum((freqs - low) / (mid - low), (high - freqs) / (high - mid)), 0, None)
    return bank


_MEL = _mel_filterbank()


def mel_spectrogram_db(y: np.ndarray) -> np.ndarray:
    mel = _MEL @ stft_power(y)
    return 10 * np.log10(np.maximum(mel, 1e-10))


def mfcc(y: np.ndarray) -> np.ndarray:
    emphasised = np.append(y[0], y[1:] - 0.97 * y[:-1])
    return dct(mel_spectrogram_db(emphasised), type=2, axis=0, norm="ortho")[:N_MFCC]


def estimate_f0(y: np.ndarray, fmin: float = 70, fmax: float = 350) -> np.ndarray:
    """Autocorrelation pitch tracker. Returns Hz for voiced frames only."""
    frame, hop = 640, 320
    if len(y) < frame:
        return np.array([])
    lo, hi = int(SAMPLE_RATE / fmax), int(SAMPLE_RATE / fmin)
    windows = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop] * np.hanning(frame)
    threshold = np.sqrt(np.mean(windows ** 2, axis=1)).max() * 0.1
    out = []
    for w in windows:
        if np.sqrt(np.mean(w ** 2)) < threshold:
            continue
        ac = signal.fftconvolve(w, w[::-1])[frame - 1:]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[lag] / ac[0] > 0.3:
            out.append(SAMPLE_RATE / lag)
    return np.array(out)


def extract_features(y: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {}
    coeffs = mfcc(y)
    deltas = np.gradient(coeffs, axis=1)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(coeffs[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(coeffs[i].std())
        feats[f"delta_{i + 1}_std"] = float(deltas[i].std())
    power = stft_power(y)
    freqs = np.linspace(0, SAMPLE_RATE / 2, power.shape[0])
    total = power.sum(axis=0) + 1e-12
    rolloff = freqs[np.argmax(np.cumsum(power, axis=0) >= 0.85 * total, axis=0)]
    centroid = (freqs[:, None] * power).sum(axis=0) / total
    rms = _rms_frames(y)
    zero_crossings = np.abs(np.diff(np.sign(y))).mean() / 2
    f0 = estimate_f0(y)
    feats.update(
        spectral_rolloff_mean=float(rolloff.mean()), spectral_rolloff_std=float(rolloff.std()),
        spectral_centroid_mean=float(centroid.mean()), spectral_centroid_std=float(centroid.std()),
        energy_rms_mean=float(rms.mean()), energy_rms_std=float(rms.std()),
        zero_crossing_rate=float(zero_crossings),
        f0_mean=float(f0.mean()) if f0.size else 0.0, f0_std=float(f0.std()) if f0.size else 0.0,
    )
    return feats


def _phase_vocoder(y: np.ndarray, rate: float) -> np.ndarray:
    """Change duration by 1/rate without changing pitch."""
    n_fft, hop = 1024, 256
    _, _, z = signal.stft(y, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    steps = np.arange(0, z.shape[1] - 1, rate)
    expected = 2 * np.pi * hop * np.arange(z.shape[0]) / n_fft
    phase = np.angle(z[:, 0])
    out = np.zeros((z.shape[0], len(steps)), dtype=complex)
    for i, t in enumerate(steps):
        k, frac = int(t), t - int(t)
        magnitude = (1 - frac) * np.abs(z[:, k]) + frac * np.abs(z[:, k + 1])
        out[:, i] = magnitude * np.exp(1j * phase)
        advance = np.angle(z[:, k + 1]) - np.angle(z[:, k]) - expected
        phase = phase + expected + (advance + np.pi) % (2 * np.pi) - np.pi
    _, stretched = signal.istft(out, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    return stretched


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return _normalise(_phase_vocoder(y, rate))


def pitch_shift(y: np.ndarray, semitones: float) -> np.ndarray:
    factor = 2 ** (semitones / 12)
    stretched = _phase_vocoder(y, 1 / factor)
    return _normalise(signal.resample(stretched, max(int(len(stretched) / factor), 1)))


def add_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=y.size)
    noise *= np.sqrt(np.mean(y ** 2) / np.mean(noise ** 2)) * 10 ** (-snr_db / 20)
    return _normalise(y + noise)


def _normalise(y: np.ndarray) -> np.ndarray:
    return y / max(np.abs(y).max(), 1e-9)


def augment(y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "original": y,
        "pitch_+1": pitch_shift(y, 1.0),
        "pitch_-1": pitch_shift(y, -1.0),
        "stretch_0.9": time_stretch(y, 0.9),
        "stretch_1.1": time_stretch(y, 1.1),
        "noise_20db": add_noise(y, 20.0, RANDOM_STATE),
    }


def save_wav(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SAMPLE_RATE, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def list_enrolled(root: Path) -> list[dict]:
    items = []
    for member_dir in sorted(p for p in (root / "audio").iterdir() if p.is_dir()):
        for path in sorted(member_dir.iterdir()):
            if path.suffix.lower() not in AUDIO_EXTS:
                continue
            match = _NAME.match(path.stem)
            if not match or match["member"] != member_dir.name:
                raise ValueError(f"Bad audio name {path}. Expected "
                                 f"{member_dir.name}_<{'|'.join(PHRASES)}>{path.suffix}")
            items.append({"path": path, "member": member_dir.name, "phrase": match["kind"]})
    return items


def describe_clip(y: np.ndarray) -> dict[str, float]:
    """Numbers behind the plain-language reading of a waveform and spectrogram."""
    rms = _rms_frames(y)
    f0 = estimate_f0(y)
    quiet, loud = np.percentile(rms, 10), np.percentile(rms, 95)
    power = stft_power(y)
    freqs = np.linspace(0, SAMPLE_RATE / 2, power.shape[0])
    return {
        "duration_s": len(y) / SAMPLE_RATE,
        "median_pitch_hz": float(np.median(f0)) if f0.size else float("nan"),
        "loud_vs_quiet_db": float(20 * np.log10(loud / max(quiet, 1e-6))),
        "clipped_pct": float(100 * np.mean(np.abs(y) > 0.995)),
        "energy_below_4khz_pct": float(100 * power[freqs < 4000].sum() / power.sum()),
    }


def build_audio_features(root: Path, source: str, save: bool = True) -> pd.DataFrame:
    rows = []
    enrolled = list_enrolled(root)
    unknown = [{"path": p, "member": "unauthorized", "phrase": "unknown"}
               for p in sorted((root / "unauthorized").glob("*"))
               if p.suffix.lower() in AUDIO_EXTS]
    for item in enrolled + unknown:
        y = load_audio(item["path"])
        for aug_name, clip in augment(y).items():
            if save:
                save_wav(AUGMENTED / "audio" / item["member"] / f"{item['path'].stem}__{aug_name}.wav", clip)
            rows.append({
                "audio_id": f"{item['path'].stem}__{aug_name}", "member": item["member"],
                "phrase": item["phrase"], "augmentation": aug_name, "source_file": item["path"].name,
                "duration_s": round(len(clip) / SAMPLE_RATE, 3), "data_source": source,
                **extract_features(clip),
            })
    df = pd.DataFrame(rows)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "audio_features.csv", index=False)
    return df


META_COLS = ["audio_id", "member", "phrase", "augmentation", "source_file", "duration_s", "data_source"]
