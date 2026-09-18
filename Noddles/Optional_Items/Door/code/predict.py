"""CLI: segment a Door stream and classify each cycle.

Usage: python predict.py --input Test.csv --output door_predictions.csv
"""

from __future__ import annotations

import argparse
import os

import door_pipeline as door

DEFAULT_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model.json"))


def run(input_path: str, output_path: str, model_path: str = DEFAULT_MODEL) -> None:
    result = door.predict_stream(input_path, door.DoorModel.load(model_path))
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} detected cycle(s) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    run(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
