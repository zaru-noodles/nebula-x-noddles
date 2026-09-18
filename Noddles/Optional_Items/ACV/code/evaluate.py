"""Leave-one-case-out validation and weight tuning for the ACV ranking model.

Implements the plan's ACV "Validation" section:
  - Hold out one entire case at a time.
  - Never split timestamps from the same case between training and validation.
  - Report mean faulty-car rank, top-1 rate, top-3 rate, and the official
    ranking score.

Feature extraction (slow: it reads the .xlsx files) is cached to disk so the
weight search below — which only needs to recombine already-computed
features many times — doesn't re-read Excel on every trial.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

import acv_pipeline as acv

TRAIN_DIR = "../../../../PS3/02_Datasets/ACV/Train"
LABELS_CSV = "../../../../PS3/02_Datasets/ACV/Train_Labels.csv"
CACHE_PATH = ".cache/train_features.pkl"
WEIGHTS_PATH = "../model/weights.json"


def rank_decay_score(rank: int, n: int) -> float:
    """Official ACV metric: score = (n - (rank - 1)) / n, rank is 1-indexed."""
    return (n - (rank - 1)) / n


def load_training_features(force_recompute: bool = False) -> dict[str, pd.DataFrame]:
    """Feature table per training filename, computed once and cached."""
    if not force_recompute and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    features_by_file: dict[str, pd.DataFrame] = {}
    for file_path in acv.list_case_files(TRAIN_DIR):
        name = os.path.basename(file_path)
        print(f"Computing features for {name} ...")
        features_by_file[name] = acv.compute_case_features(file_path)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(features_by_file, f)
    return features_by_file


def zscore_all(features_by_file: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pre-compute each case's (weight-independent) z-scored feature table.

    A weight search recombines these thousands of times; z-scoring itself
    (per-column median/MAD) must not be repeated on every trial, or the
    search becomes dominated by redundant work instead of the cheap
    weighted-sum it should be.
    """
    return {name: acv.zscore_features(features) for name, features in features_by_file.items()}


def evaluate_weights(
    zscores_by_file: dict[str, pd.DataFrame],
    labels: dict[str, str],
    weights: dict[str, float],
    file_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Per-file rank / score of the true faulty car under one weight vector."""
    files = file_subset if file_subset is not None else list(zscores_by_file)
    rows = []
    for name in files:
        z = zscores_by_file[name]
        scores = acv.combine_scores(z, weights)
        ranking = acv.rank_cars(scores)
        true_car = labels[name]
        n = len(ranking)
        rank = ranking.index(true_car) + 1 if true_car in ranking else None
        score = rank_decay_score(rank, n) if rank is not None else 0.0
        rows.append({"file": name, "true_car": true_car, "rank": rank, "n_cars": n, "score": score})
    return pd.DataFrame(rows)


def _sample_weights(rng: np.random.Generator) -> dict[str, float]:
    """A random non-negative weight vector over the feature set, summing to 1."""
    raw = rng.dirichlet(np.ones(len(acv.FEATURE_COLUMNS)))
    return dict(zip(acv.FEATURE_COLUMNS, raw))


def tune_weights(
    zscores_by_file: dict[str, pd.DataFrame],
    labels: dict[str, str],
    file_subset: list[str],
    n_trials: int = 4000,
    seed: int = 0,
) -> tuple[dict[str, float], float]:
    """Random search over weight vectors, maximizing mean rank-decay score.

    With only a handful of labelled cases and ~9 features, a full grid or
    gradient search isn't warranted; random search over the simplex is cheap
    now that it only recombines precomputed z-scores (see
    :func:`zscore_all`) and works well at this scale.
    """
    rng = np.random.default_rng(seed)
    best_weights = acv.DEFAULT_WEIGHTS
    best_mean_score = evaluate_weights(zscores_by_file, labels, best_weights, file_subset)["score"].mean()

    for _ in range(n_trials):
        candidate = _sample_weights(rng)
        mean_score = evaluate_weights(zscores_by_file, labels, candidate, file_subset)["score"].mean()
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_weights = candidate

    return best_weights, best_mean_score


def leave_one_case_out(zscores_by_file: dict[str, pd.DataFrame], labels: dict[str, str]) -> pd.DataFrame:
    """For each case, tune weights on the other cases only, then score it held-out."""
    files = list(zscores_by_file)
    results = []
    for held_out in files:
        train_files = [f for f in files if f != held_out]
        weights, _ = tune_weights(zscores_by_file, labels, train_files)
        fold_result = evaluate_weights(zscores_by_file, labels, weights, [held_out])
        results.append(fold_result.iloc[0])
    return pd.DataFrame(results)


def main() -> None:
    labels = pd.read_csv(LABELS_CSV).set_index("filename")["faulty_car"].astype(str).str.zfill(2).to_dict()
    features_by_file = load_training_features()
    zscores_by_file = zscore_all(features_by_file)

    print("\n=== Leave-one-case-out validation ===")
    loocv = leave_one_case_out(zscores_by_file, labels)
    print(loocv.to_string(index=False))

    mean_rank = loocv["rank"].mean()
    top1_rate = (loocv["rank"] == 1).mean()
    top3_rate = (loocv["rank"] <= 3).mean()
    mean_score = loocv["score"].mean()
    print(f"\nMean faulty-car rank: {mean_rank:.2f}")
    print(f"Top-1 rate: {top1_rate:.2%}")
    print(f"Top-3 rate: {top3_rate:.2%}")
    print(f"Mean official rank-decay score (LOOCV estimate): {mean_score:.4f}")

    print("\n=== Fitting final weights on all training cases ===")
    final_weights, final_mean_score = tune_weights(zscores_by_file, labels, list(zscores_by_file))
    print(f"In-sample mean score with tuned weights: {final_mean_score:.4f}")
    print("Weights:")
    for k, v in sorted(final_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:.4f}")

    with open(WEIGHTS_PATH, "w") as f:
        json.dump(final_weights, f, indent=2)
    print(f"\nSaved final weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
