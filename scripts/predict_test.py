"""
Quick sanity check for the prediction pipeline.

Runs a few sample inputs through src/predict.py and prints the outputs.
Assumes you run this from the repo root so that final_model.pkl is found.
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import run_prediction


def main():
    samples = [
        (3, 120, 450),   # short trip, moderate miles/receipts
        (5, 350, 900),   # mid trip
        (8, 750, 1800),  # long trip, higher spend
        (2, 40, 150),    # very short, low spend
    ]
    for days, miles, receipts in samples:
        pred = run_prediction(days, miles, receipts)
        print(f"Input (days={days}, miles={miles}, receipts=${receipts}) -> Predicted reimbursement: ${pred}")


if __name__ == "__main__":
    main()
