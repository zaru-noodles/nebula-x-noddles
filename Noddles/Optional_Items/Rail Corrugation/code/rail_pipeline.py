"""Speed-normalised, side-aware Rail Corrugation feature pipeline."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SAMPLE_RATE = 10_000.0
WHEEL_DIAMETER_M = 0.85
TEETH = 90
LABELS = ("Normal", "Side I", "Side II")
SIGNAL_RE = re.compile(r"^(Vibration|Shock) of bearing in position (\d+) of car (\d+)$")
WAVELENGTH_BANDS = ((.015, .03), (.03, .06), (.06, .12), (.12, .25), (.25, .50))


def list_csv_files(path: str) -> list[str]:
    files = glob.glob(os.path.join(path, "*.csv")) if os.path.isdir(path) else [path]
    def number(p: str):
        match = re.search(r"(\d+)", os.path.basename(p))
        return int(match.group(1)) if match else 0
    return sorted(files, key=number)


def estimate_speed_mps(rotating_speed: np.ndarray) -> float:
    binary = np.nan_to_num(rotating_speed) > .5
    transitions = int(np.count_nonzero(binary[1:] != binary[:-1]))
    duration = len(binary) / SAMPLE_RATE
    revolutions = transitions / (2 * TEETH)
    return float(revolutions * np.pi * WHEEL_DIAMETER_M / max(duration, 1e-9))


BASE_STATS = (
    "rms_median", "rms_p90", "rms_max", "kurtosis_median",
    "kurtosis_p90", "crest_median", "rms_cv", "spectral_entropy",
    "dominant_wavelength", "band_0", "band_1", "band_2", "band_3", "band_4",
)
GROUPS = tuple(f"{kind.lower()}_{side}" for kind in ("Vibration", "Shock") for side in ("I", "II"))
FEATURE_NAMES = tuple(
    ["speed_mps"]
    + [f"{group}_{stat}" for group in GROUPS for stat in BASE_STATS]
    + [f"contrast_{kind.lower()}_{stat}" for kind in ("Vibration", "Shock") for stat in BASE_STATS]
)


def _group_features(values: np.ndarray, speed: float) -> dict[str, float]:
    eps = 1e-12
    values = np.nan_to_num(values.astype(float))
    values -= values.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(values ** 2, axis=0))
    std = np.std(values, axis=0)
    kurt = np.mean((values / np.maximum(std, eps)) ** 4, axis=0)
    crest = np.max(np.abs(values), axis=0) / np.maximum(rms, eps)
    spectrum = np.abs(np.fft.rfft(values, axis=0)) ** 2
    psd = np.median(spectrum, axis=1)
    psd[0] = 0
    total = max(float(psd.sum()), eps)
    probability = psd / total
    nonzero = probability > 0
    entropy = -float(np.sum(probability[nonzero] * np.log(probability[nonzero]))) / max(np.log(len(psd)), 1.0)
    freq = np.fft.rfftfreq(len(values), d=1/SAMPLE_RATE)
    dominant_idx = int(np.argmax(psd[1:]) + 1) if len(psd) > 1 else 0
    dominant_wavelength = speed / freq[dominant_idx] if dominant_idx and speed > 0 else 0.0
    out = {
        "rms_median": float(np.median(rms)), "rms_p90": float(np.quantile(rms, .9)),
        "rms_max": float(np.max(rms)), "kurtosis_median": float(np.median(kurt)),
        "kurtosis_p90": float(np.quantile(kurt, .9)), "crest_median": float(np.median(crest)),
        "rms_cv": float(np.std(rms) / max(np.mean(rms), eps)),
        "spectral_entropy": entropy, "dominant_wavelength": float(dominant_wavelength),
    }
    wavelength = np.divide(speed, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    for i, (low, high) in enumerate(WAVELENGTH_BANDS):
        out[f"band_{i}"] = float(psd[(wavelength >= low) & (wavelength < high)].sum() / total)
    return out


def extract_features(path: str) -> dict[str, float]:
    df = pd.read_csv(path, dtype=np.float32)
    if "Rotating speed" not in df:
        raise ValueError(f"Missing 'Rotating speed' in {path}")
    speed = estimate_speed_mps(df["Rotating speed"].to_numpy())
    grouped: dict[tuple[str, str], list[str]] = {(k, s): [] for k in ("Vibration", "Shock") for s in ("I", "II")}
    for col in df.columns:
        match = SIGNAL_RE.match(col)
        if not match:
            continue
        kind, position, _ = match.groups()
        side = "I" if int(position) % 2 else "II"
        grouped[(kind, side)].append(col)
    result: dict[str, float] = {"speed_mps": speed}
    by_group = {}
    for (kind, side), columns in grouped.items():
        if not columns:
            raise ValueError(f"No {kind} channels for Side {side} in {path}")
        stats = _group_features(df[columns].to_numpy(), speed)
        by_group[(kind, side)] = stats
        prefix = f"{kind.lower()}_{side}"
        result.update({f"{prefix}_{name}": value for name, value in stats.items()})
    for kind in ("Vibration", "Shock"):
        for stat in BASE_STATS:
            a, b = by_group[(kind, "I")][stat], by_group[(kind, "II")][stat]
            if stat.startswith(("rms", "band")):
                contrast = np.log((max(a, 0)+1e-9)/(max(b, 0)+1e-9))
            else:
                contrast = a-b
            result[f"contrast_{kind.lower()}_{stat}"] = float(contrast)
    return result


@dataclass
class BinaryModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    @classmethod
    def from_dict(cls, obj: dict) -> "BinaryModel":
        return cls(np.array(obj["mean"]), np.array(obj["scale"]), np.array(obj["weights"]), obj["intercept"])

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        z = np.nan_to_num((x-self.mean)/self.scale)
        return 1/(1+np.exp(-np.clip(z @ self.weights + self.intercept, -40, 40)))


@dataclass
class RailModel:
    fault: BinaryModel
    side: BinaryModel
    fault_threshold: float
    side_threshold: float

    @classmethod
    def load(cls, path: str) -> "RailModel":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if obj["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Rail model feature schema does not match pipeline")
        return cls(BinaryModel.from_dict(obj["fault_model"]), BinaryModel.from_dict(obj["side_model"]), obj["fault_threshold"], obj["side_threshold"])

    def predict_features(self, features: dict[str, float]) -> str:
        x = np.array([[features[n] for n in FEATURE_NAMES]])
        if self.fault.probabilities(x)[0] < self.fault_threshold:
            return "Normal"
        return "Side I" if self.side.probabilities(x)[0] >= self.side_threshold else "Side II"

    def predict_file(self, path: str) -> str:
        return self.predict_features(extract_features(path))
