"""Leave-one-file-out validation and fitting for the SHM damage model."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import shm_pipeline as shm

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "../../../../PS3/02_Datasets/SHM"))
MODEL_PATH = os.path.abspath(os.path.join(HERE, "../model/model.json"))
CACHE_PATH = os.path.join(HERE, ".cache", "features.csv")


def load_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = pd.read_csv(os.path.join(ROOT, "Train_Labels.csv"))
    if os.path.exists(CACHE_PATH):
        table = pd.read_csv(CACHE_PATH)
    else:
        rows = []
        for name in labels["filename"]:
            print(f"Extracting {name} ...")
            rows.append({"filename": name, **shm.extract_features(os.path.join(ROOT, "Train", name))})
        table = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        table.to_csv(CACHE_PATH, index=False)
    merged = labels.merge(table, on="filename", validate="one_to_one")
    return merged[list(shm.FEATURE_NAMES)].to_numpy(float), merged["damage"].to_numpy(float), merged["filename"].tolist()


def fit(x: np.ndarray, y: np.ndarray, alpha: float):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x-mean)/scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ np.log(y))
    return mean, scale, beta[1:], float(beta[0])


def predict(x: np.ndarray, model) -> np.ndarray:
    mean, scale, coef, intercept = model
    return np.exp(np.clip(intercept + ((x-mean)/scale) @ coef, -30, 30))


def mape(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y-pred)/np.abs(y)))


def main() -> None:
    x, y, names = load_features()
    print("=== Leave-one-file-out validation ===")
    best_alpha, best_pred, best_mape = None, None, np.inf
    for alpha in (.01, .1, 1.0, 10.0, 50.0, 100.0):
        out = np.zeros_like(y)
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            out[i] = predict(x[i:i+1], fit(x[keep], y[keep], alpha))[0]
        score = mape(y, out)
        print(f"alpha={alpha:6g} MAPE={score:.4f} official_score={max(0, 1-score):.4f}")
        if score < best_mape:
            best_alpha, best_pred, best_mape = alpha, out, score
    assert best_pred is not None
    order = np.argsort(y)
    for label, indices in zip(("low", "medium", "high"), np.array_split(order, 3)):
        print(f"{label:6s} damage MAPE={mape(y[indices], best_pred[indices]):.4f}")
    model = fit(x, y, float(best_alpha))
    mean, scale, coef, intercept = model
    artifact = {"feature_names": list(shm.FEATURE_NAMES), "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coef.tolist(), "intercept": intercept, "minimum": 1e-9, "ridge_alpha": best_alpha, "loocv_mape": best_mape}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
