"""Repeated stratified validation and training for Rail Corrugation."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import rail_pipeline as rail

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "../../../../PS3/02_Datasets/Rail_Corrugation"))
CACHE = os.path.join(HERE, ".cache", "features.csv")
MODEL_PATH = os.path.abspath(os.path.join(HERE, "../model/model.json"))


def load_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = pd.read_csv(os.path.join(ROOT, "Train_Labels.csv"))
    if os.path.exists(CACHE):
        features = pd.read_csv(CACHE)
    else:
        rows = []
        for i, name in enumerate(labels["filename"], 1):
            print(f"[{i}/{len(labels)}] Extracting {name} ...", flush=True)
            rows.append({"filename": name, **rail.extract_features(os.path.join(ROOT, "Train", name))})
        features = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        features.to_csv(CACHE, index=False)
    merged = labels.merge(features, on="filename", validate="one_to_one")
    return merged[list(rail.FEATURE_NAMES)].to_numpy(float), merged["label"].to_numpy(), merged["filename"].tolist()


def fit_binary(x: np.ndarray, y: np.ndarray, steps: int = 2500, lr: float = .025, reg: float = .2) -> dict:
    mean, scale = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    z = np.nan_to_num((x-mean)/scale)
    w, b = np.zeros(z.shape[1]), 0.0
    prevalence = np.clip(y.mean(), 1e-4, 1-1e-4)
    sample_w = np.where(y > 0, .5/prevalence, .5/(1-prevalence))
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(z @ w+b, -30, 30)))
        error = (p-y)*sample_w
        w -= lr*((z.T @ error)/len(y)+reg*w/len(y))
        b -= lr*error.mean()
    return {"mean": mean, "scale": scale, "weights": w, "intercept": float(b)}


def probability(model: dict, x: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((x-model["mean"])/model["scale"])
    return 1/(1+np.exp(-np.clip(z @ model["weights"]+model["intercept"], -30, 30)))


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for label in rail.LABELS:
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(int(idx))
    return [np.array(sorted(fold), dtype=int) for fold in folds]


def macro_f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    scores = {}
    for label in rail.LABELS:
        tp = np.sum((y == label) & (pred == label))
        fp = np.sum((y != label) & (pred == label))
        fn = np.sum((y == label) & (pred != label))
        scores[label] = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0
    return float(np.mean(list(scores.values()))), scores


def predict_labels(p_fault, p_side, fault_threshold, side_threshold):
    return np.where(p_fault < fault_threshold, "Normal", np.where(p_side >= side_threshold, "Side I", "Side II"))


def main() -> None:
    x, y, _ = load_features()
    fault_y = (y != "Normal").astype(float)
    side_y = (y == "Side I").astype(float)
    repeats = []
    print("=== Repeated stratified 5-fold validation ===")
    for seed in (7, 29, 71):
        pf, ps = np.zeros(len(y)), np.zeros(len(y))
        for valid in stratified_folds(y, 5, seed):
            train = np.setdiff1d(np.arange(len(y)), valid)
            fault_model = fit_binary(x[train], fault_y[train])
            fault_train = train[fault_y[train] == 1]
            side_model = fit_binary(x[fault_train], side_y[fault_train])
            pf[valid] = probability(fault_model, x[valid])
            ps[valid] = probability(side_model, x[valid])
        repeats.append((pf, ps))
    best = (-1.0, .5, .5, None)
    for ft in np.linspace(.15, .85, 29):
        for st in np.linspace(.25, .75, 21):
            scores = [macro_f1(y, predict_labels(pf, ps, ft, st))[0] for pf, ps in repeats]
            mean_score = float(np.mean(scores))
            if mean_score > best[0]:
                best = (mean_score, float(ft), float(st), scores)
    print(f"Best mean macro F1={best[0]:.4f}; fault threshold={best[1]:.3f}; side threshold={best[2]:.3f}")
    for i, ((pf, ps), score) in enumerate(zip(repeats, best[3]), 1):
        pred = predict_labels(pf, ps, best[1], best[2])
        _, per_class = macro_f1(y, pred)
        print(f"Repeat {i}: macro F1={score:.4f}; per-class={per_class}")
        print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())

    fault_model = fit_binary(x, fault_y)
    fault_indices = np.flatnonzero(fault_y == 1)
    side_model = fit_binary(x[fault_indices], side_y[fault_indices])
    serialise = lambda m: {"mean": m["mean"].tolist(), "scale": m["scale"].tolist(), "weights": m["weights"].tolist(), "intercept": m["intercept"]}
    artifact = {"feature_names": list(rail.FEATURE_NAMES), "fault_model": serialise(fault_model), "side_model": serialise(side_model), "fault_threshold": best[1], "side_threshold": best[2], "cv_macro_f1": best[0]}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
