"""Physics-informed fatigue-damage features and regression inference."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


def list_csv_files(path: str) -> list[str]:
    files = glob.glob(os.path.join(path, "*.csv")) if os.path.isdir(path) else [path]
    return sorted(files, key=lambda p: os.path.basename(p).lower())


def load_stress(path: str) -> np.ndarray:
    data = pd.read_csv(path, header=None).iloc[:, 0]
    stress = pd.to_numeric(data, errors="coerce").dropna().to_numpy(float)
    if stress.size < 4:
        raise ValueError(f"SHM file has fewer than four numeric samples: {path}")
    # Offsets do not contribute to alternating stress in an uncorrected S-N model.
    return stress - np.median(stress)


def turning_points(signal: np.ndarray) -> np.ndarray:
    """Return endpoints and local extrema after removing repeated values."""
    x = signal[np.r_[True, np.diff(signal) != 0]]
    if x.size < 3:
        return x
    delta = np.diff(x)
    extrema = np.r_[True, delta[:-1] * delta[1:] < 0, True]
    return x[extrema]


def rainflow_ranges(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Four-point rainflow count returning cycle ranges and counts (1 or .5)."""
    stack: list[float] = []
    ranges: list[float] = []
    counts: list[float] = []
    for point in turning_points(signal):
        stack.append(float(point))
        while len(stack) >= 3:
            older = abs(stack[-2] - stack[-3])
            newer = abs(stack[-1] - stack[-2])
            if newer < older:
                break
            if len(stack) == 3:
                ranges.append(older); counts.append(.5); stack.pop(0)
            else:
                ranges.append(older); counts.append(1.0)
                last = stack[-1]
                del stack[-3:]
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        ranges.append(abs(b-a)); counts.append(.5)
    return np.asarray(ranges), np.asarray(counts)


EXPONENTS = (2, 3, 4, 5, 6, 7, 8)
FEATURE_NAMES = tuple([f"log_damage_m{m}" for m in EXPONENTS] + [
    "log_rms", "log_std", "log_max_abs", "log_q90", "log_q95",
    "log_q99", "log_mean_abs", "crest", "kurtosis", "cycle_count",
    "spectral_low", "spectral_mid", "spectral_high",
])


def extract_features_from_signal(stress: np.ndarray) -> dict[str, float]:
    ranges, counts = rainflow_ranges(stress)
    amplitude = ranges / 2.0
    eps = 1e-12
    f: dict[str, float] = {}
    for m in EXPONENTS:
        proxy = np.sum(counts * np.power(amplitude, m))
        f[f"log_damage_m{m}"] = float(np.log(max(proxy, eps)))
    abs_x = np.abs(stress)
    rms = float(np.sqrt(np.mean(stress ** 2)))
    std = float(np.std(stress))
    for name, value in (
        ("log_rms", rms), ("log_std", std), ("log_max_abs", np.max(abs_x)),
        ("log_q90", np.quantile(abs_x, .90)), ("log_q95", np.quantile(abs_x, .95)),
        ("log_q99", np.quantile(abs_x, .99)), ("log_mean_abs", np.mean(abs_x)),
    ):
        f[name] = float(np.log(max(value, eps)))
    f["crest"] = float(np.max(abs_x) / max(rms, eps))
    f["kurtosis"] = float(np.mean((stress / max(std, eps)) ** 4))
    f["cycle_count"] = float(np.log1p(np.sum(counts)))
    # Normalised frequency bands are robust when the physical sample rate is absent.
    sample = stress if stress.size <= 131072 else stress[::int(np.ceil(stress.size / 131072))]
    power = np.abs(np.fft.rfft(sample - sample.mean())) ** 2
    total = max(float(power[1:].sum()), eps)
    n = len(power)
    f["spectral_low"] = float(power[1:max(2, n//20)].sum() / total)
    f["spectral_mid"] = float(power[max(2, n//20):max(3, n//4)].sum() / total)
    f["spectral_high"] = float(power[max(3, n//4):].sum() / total)
    return f


def extract_features(path: str) -> dict[str, float]:
    return extract_features_from_signal(load_stress(path))


@dataclass
class SHMModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    minimum: float

    @classmethod
    def load(cls, path: str) -> "SHMModel":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if obj["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("SHM model feature schema does not match pipeline")
        return cls(np.array(obj["mean"]), np.array(obj["scale"]), np.array(obj["coefficients"]), obj["intercept"], obj.get("minimum", 1e-9))

    def predict_features(self, features: dict[str, float]) -> float:
        x = np.array([features[n] for n in FEATURE_NAMES])
        z = np.nan_to_num((x-self.mean)/self.scale)
        return max(float(np.exp(np.clip(self.intercept + z @ self.coefficients, -30, 30))), self.minimum)

    def predict_file(self, path: str) -> float:
        return self.predict_features(extract_features(path))
