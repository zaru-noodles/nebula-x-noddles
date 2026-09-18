"""Chronological validation and final training for the Door model."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import door_pipeline as door

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "../../../../PS3/02_Datasets/Door"))
MODEL_PATH = os.path.abspath(os.path.join(HERE, "../model/model.json"))


def build_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    stream = door.load_stream(os.path.join(DATA, "Train.csv"))
    segments = door.segment_stream(stream)
    labels = pd.read_csv(os.path.join(DATA, "Train_Segments_Answer.csv"))
    if len(segments) != len(labels):
        raise ValueError(f"Detected {len(segments)} segments but found {len(labels)} labels")
    x = np.array([[door.extract_features(s)[n] for n in door.FEATURE_NAMES] for s in segments])
    y = (labels["status"] == "Abnormal resistance").astype(float).to_numpy()
    return x, y, labels


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 4000, lr: float = .03, reg: float = .08):
    mean, scale = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    z = np.nan_to_num((x - mean) / scale)
    w, b = np.zeros(z.shape[1]), 0.0
    class_w = np.where(y > 0, .5 / max(y.mean(), 1e-6), .5 / max(1-y.mean(), 1e-6))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(z @ w + b, -30, 30)))
        err = (p - y) * class_w
        w -= lr * ((z.T @ err) / len(y) + reg * w / len(y))
        b -= lr * err.mean()
    return mean, scale, w, b


def classification_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    pred = p >= .5
    tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    x, y, labels = build_dataset()
    print("=== Contiguous chronological 5-fold validation ===")
    fold_ids = np.array_split(np.arange(len(y)), 5)
    all_y, all_p = [], []
    for i, valid in enumerate(fold_ids, 1):
        train = np.setdiff1d(np.arange(len(y)), valid)
        mean, scale, w, b = fit_logistic(x[train], y[train])
        p = 1 / (1 + np.exp(-np.clip(np.nan_to_num((x[valid]-mean)/scale) @ w + b, -30, 30)))
        all_y.extend(y[valid]); all_p.extend(p)
        _, _, f1 = classification_metrics(y[valid], p)
        print(f"Fold {i}: classification F1={f1:.4f}")
    precision, recall, f1 = classification_metrics(np.array(all_y), np.array(all_p))
    print(f"OOF abnormal precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}")
    oof_predictions = labels[["start_time", "end_time"]].copy()
    oof_predictions["prediction"] = np.where(np.asarray(all_p) >= .5, "Abnormal resistance", "Normal")
    official = door.iou_weighted_f1(labels, oof_predictions)
    print(f"Official end-to-end IoU-weighted F1={official:.4f}")

    mean, scale, w, b = fit_logistic(x, y)
    artifact = {"feature_names": list(door.FEATURE_NAMES), "mean": mean.tolist(), "scale": scale.tolist(), "weights": w.tolist(), "intercept": b, "threshold": .5}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
