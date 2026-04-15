"""Minimal valid aerial-cactus generated solution for monitor smoke tests."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("./data")
OUTPUT_PATH = Path("submission.csv")


def main():
    matches = sorted(DATA_DIR.glob("**/sample_submission*.csv"))
    if not matches:
        raise FileNotFoundError("Could not find sample_submission*.csv under ./data")

    submission = pd.read_csv(matches[0])
    if len(submission.columns) < 2:
        raise ValueError(f"Sample submission must have at least two columns: {matches[0]}")

    prediction_col = "has_cactus" if "has_cactus" in submission.columns else submission.columns[-1]
    submission[prediction_col] = 0.5
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote smoke-test submission to {OUTPUT_PATH} rows={len(submission)}")


main()
