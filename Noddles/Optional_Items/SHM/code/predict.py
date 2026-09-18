"""CLI: predict cumulative fatigue damage for one CSV or a directory.

Usage: python predict.py --input <file_or_dir> --output shm_predictions.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import shm_pipeline as shm

DEFAULT_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model.json"))


def run(input_path: str, output_path: str, model_path: str = DEFAULT_MODEL) -> None:
    model = shm.SHMModel.load(model_path)
    files = shm.list_csv_files(input_path)
    if not files:
        raise SystemExit(f"No CSV files found at: {input_path}")
    rows = []
    for path in files:
        print(f"Predicting {os.path.basename(path)} ...")
        rows.append({"file_id": os.path.basename(path), "prediction": model.predict_file(path)})
    pd.DataFrame(rows, columns=["file_id", "prediction"]).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} prediction(s) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    run(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
