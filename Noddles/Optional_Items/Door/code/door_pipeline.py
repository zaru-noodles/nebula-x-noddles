"""Door cycle segmentation and abnormal-resistance classification.

The stream is segmented at timestamp discontinuities, then each complete
open/close cycle is represented by current, voltage, back-EMF, position and
phase features.  A small standardised logistic model is deliberately used so
inference only requires NumPy and pandas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIME = "Datetime"
CURRENT = "Motor current(mA)"
VOLTAGE = "Motor Voltage(10mV)"
EMF = "Motor electrodynamic force"
POSITION = "Door leaf position"
LABELS = ("Normal", "Abnormal resistance")


def parse_timestamp(value: str) -> pd.Timestamp:
    """Parse the native Y-M-D-h-m-s-ms timestamp without locale ambiguity."""
    parts = [int(x) for x in str(value).split("-")]
    if len(parts) != 7:
        return pd.Timestamp(value)
    y, month, day, hour, minute, second, millis = parts
    return pd.Timestamp(y, month, day, hour, minute, second, millis * 1000)


def parse_timestamps(values: pd.Series) -> pd.Series:
    return values.astype(str).map(parse_timestamp)


def load_stream(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {TIME, CURRENT, VOLTAGE, EMF, POSITION}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Door input is missing columns: {sorted(missing)}")
    df = df.copy()
    df["_time"] = parse_timestamps(df[TIME])
    return df


def infer_gap_threshold(times: pd.Series) -> pd.Timedelta:
    """Choose a data-derived threshold that stays well above the 20 ms cadence."""
    positive = times.diff().dropna()
    positive = positive[positive > pd.Timedelta(0)]
    cadence = positive.median() if not positive.empty else pd.Timedelta(milliseconds=20)
    return max(pd.Timedelta(seconds=1), cadence * 20)


def segment_stream(df: pd.DataFrame, gap_threshold: pd.Timedelta | None = None) -> list[pd.DataFrame]:
    threshold = gap_threshold or infer_gap_threshold(df["_time"])
    new_segment = df["_time"].diff().gt(threshold).fillna(True)
    ids = new_segment.cumsum()
    return [part.copy() for _, part in df.groupby(ids, sort=True) if len(part) >= 3]


def infer_operation(segment: pd.DataFrame) -> str:
    opening_votes = pd.to_numeric(segment.get("Door is opening", 0), errors="coerce").fillna(0).sum()
    closing_votes = pd.to_numeric(segment.get("Door is closing", 0), errors="coerce").fillna(0).sum()
    if opening_votes != closing_votes:
        return "Open" if opening_votes > closing_votes else "Close"
    pos = pd.to_numeric(segment[POSITION], errors="coerce")
    return "Open" if pos.iloc[-1] > pos.iloc[0] else "Close"


def _safe_quantile(x: np.ndarray, q: float) -> float:
    finite = x[np.isfinite(x)]
    return float(np.quantile(finite, q)) if finite.size else 0.0


FEATURE_NAMES = (
    "is_open", "duration", "n_rows", "current_mean", "current_min",
    "current_peak", "current_rms", "current_q10", "current_q50",
    "current_q90", "current_area", "voltage_mean", "voltage_rms",
    "emf_mean", "emf_rms", "position_range", "position_speed_mean",
    "position_speed_peak", "stall_fraction", "phase_current_0",
    "phase_current_1", "phase_current_2", "phase_current_3",
    "phase_current_4", "current_position_slope",
)


def extract_features(segment: pd.DataFrame) -> dict[str, float]:
    current = pd.to_numeric(segment[CURRENT], errors="coerce").to_numpy(float)
    voltage = pd.to_numeric(segment[VOLTAGE], errors="coerce").to_numpy(float)
    emf = pd.to_numeric(segment[EMF], errors="coerce").to_numpy(float)
    pos = pd.to_numeric(segment[POSITION], errors="coerce").to_numpy(float)
    elapsed = (segment["_time"] - segment["_time"].iloc[0]).dt.total_seconds().to_numpy(float)
    duration = max(float(elapsed[-1]), 1e-6)
    dt = np.diff(elapsed)
    dp = np.diff(pos)
    valid_dt = np.where(dt > 0, dt, np.nan)
    speed = np.abs(dp / valid_dt)
    abs_current = np.abs(current)
    result = {
        "is_open": float(infer_operation(segment) == "Open"),
        "duration": duration,
        "n_rows": float(len(segment)),
        "current_mean": float(np.nanmean(current)),
        "current_min": float(np.nanmin(current)),
        "current_peak": float(np.nanmax(abs_current)),
        "current_rms": float(np.sqrt(np.nanmean(current ** 2))),
        "current_q10": _safe_quantile(current, .10),
        "current_q50": _safe_quantile(current, .50),
        "current_q90": _safe_quantile(current, .90),
        "current_area": float(np.trapezoid(np.nan_to_num(abs_current), elapsed)),
        "voltage_mean": float(np.nanmean(voltage)),
        "voltage_rms": float(np.sqrt(np.nanmean(voltage ** 2))),
        "emf_mean": float(np.nanmean(emf)),
        "emf_rms": float(np.sqrt(np.nanmean(emf ** 2))),
        "position_range": float(np.nanmax(pos) - np.nanmin(pos)),
        "position_speed_mean": float(np.nanmean(speed)) if speed.size else 0.0,
        "position_speed_peak": float(np.nanmax(speed)) if speed.size else 0.0,
        "stall_fraction": float(np.mean(np.abs(dp) < 1e-9)) if dp.size else 0.0,
    }
    progress = np.linspace(0.0, 1.0, len(segment), endpoint=True)
    for i in range(5):
        mask = (progress >= i / 5) & (progress <= (i + 1) / 5 if i == 4 else progress < (i + 1) / 5)
        result[f"phase_current_{i}"] = float(np.nanmean(abs_current[mask]))
    valid = np.isfinite(pos) & np.isfinite(current)
    result["current_position_slope"] = (
        float(np.polyfit(pos[valid], current[valid], 1)[0])
        if valid.sum() >= 3 and np.nanstd(pos[valid]) > 0 else 0.0
    )
    return result


@dataclass
class DoorModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    threshold: float = 0.5

    @classmethod
    def load(cls, path: str) -> "DoorModel":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if obj["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Door model feature schema does not match pipeline")
        return cls(np.array(obj["mean"]), np.array(obj["scale"]), np.array(obj["weights"]), obj["intercept"], obj.get("threshold", .5))

    def probability(self, feature: dict[str, float]) -> float:
        x = np.array([feature[n] for n in FEATURE_NAMES], dtype=float)
        z = np.nan_to_num((x - self.mean) / self.scale)
        logit = float(np.clip(z @ self.weights + self.intercept, -40, 40))
        return 1.0 / (1.0 + np.exp(-logit))


def predict_stream(path: str, model: DoorModel) -> pd.DataFrame:
    stream = load_stream(path)
    rows = []
    for segment in segment_stream(stream):
        probability = model.probability(extract_features(segment))
        rows.append({
            "start_time": str(segment[TIME].iloc[0]),
            "end_time": str(segment[TIME].iloc[-1]),
            "prediction": LABELS[int(probability >= model.threshold)],
        })
    return pd.DataFrame(rows, columns=["start_time", "end_time", "prediction"])


def iou_weighted_f1(truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Official greedy same-label IoU-weighted F1."""
    candidates = []
    for ti, t in truth.iterrows():
        ts, te = parse_timestamp(t["start_time"]), parse_timestamp(t["end_time"])
        true_label = t.get("status", t.get("prediction"))
        for pi, p in predictions.iterrows():
            if p["prediction"] != true_label:
                continue
            ps, pe = parse_timestamp(p["start_time"]), parse_timestamp(p["end_time"])
            intersection = max(pd.Timedelta(0), min(te, pe) - max(ts, ps)).total_seconds()
            union = (te - ts).total_seconds() + (pe - ps).total_seconds() - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > 0:
                candidates.append((iou, ti, pi))
    used_t, used_p, total = set(), set(), 0.0
    for iou, ti, pi in sorted(candidates, reverse=True):
        if ti not in used_t and pi not in used_p:
            used_t.add(ti); used_p.add(pi); total += iou
    if not len(truth) or not len(predictions):
        return 0.0
    recall, precision = total / len(truth), total / len(predictions)
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0
