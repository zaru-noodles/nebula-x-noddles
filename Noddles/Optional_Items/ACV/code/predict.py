"""CLI: rank every car in one or more ACV case files by fault likelihood.

Usage:
    python predict.py --input <file_or_dir> --output acv_predictions.csv

``--input`` may be a single .xlsx case file or a directory containing one or
more of them (e.g. the "Test" folder). Produces one row per file:
``file_id`` (source filename with extension) and ``ranked_cars`` (every car
identifier in that file, most- to least-likely faulty, ``|``-separated,
using the identifier exactly as it appears in the file's own headers) — the
schema required by the ACV info kit and the top-level submission spec.
"""

from __future__ import annotations

import argparse
import json
import os

import acv_pipeline as acv

DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "weights.json")


def load_weights(weights_path: str | None) -> dict[str, float]:
    """Tuned feature weights from disk, falling back to equal weighting.

    The tuned weights (see evaluate.py) are fit once on the labelled
    training cases and frozen here so predictions are reproducible without
    re-running the search on every invocation.
    """
    path = weights_path or DEFAULT_WEIGHTS_PATH
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(acv.DEFAULT_WEIGHTS)


def run(input_path: str, output_path: str, weights_path: str | None = None) -> None:
    weights = load_weights(weights_path)
    files = acv.list_case_files(input_path)
    if not files:
        raise SystemExit(f"No .xlsx case files found at: {input_path}")

    rows = []
    for file_path in files:
        file_id = os.path.basename(file_path)
        print(f"Ranking cars in {file_id} ...")
        ranking = acv.predict_case(file_path, weights)
        rows.append({"file_id": file_id, "ranked_cars": "|".join(ranking)})

    import pandas as pd

    pd.DataFrame(rows, columns=["file_id", "ranked_cars"]).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} row(s) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="A case .xlsx file, or a directory of them.")
    parser.add_argument("--output", required=True, help="Destination path for acv_predictions.csv.")
    parser.add_argument("--weights", default=None, help="Optional path to a feature-weights JSON file.")
    args = parser.parse_args()
    run(args.input, args.output, args.weights)


if __name__ == "__main__":
    main()
