"""
Sanity check for predict_rules.py (with post-processing).
Run from repo root: python scripts/predict_rules_test.py
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_rules import run_prediction


def main():
    samples = [
        (3, 120, 450),
        (5, 350, 900),
        (8, 750, 1800),
        (2, 40, 150),
    ]
    for days, miles, receipts in samples:
        pred = run_prediction(days, miles, receipts)
        print(f"Input (days={days}, miles={miles}, receipts=${receipts}) -> Predicted reimbursement: ${pred}")


if __name__ == "__main__":
    main()
