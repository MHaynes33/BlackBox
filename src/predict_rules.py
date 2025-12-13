import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------------------------------
# Feature Engineering (matches training pipeline)
# ----------------------------------------------------------
def create_features(df):
    df = df.copy()
    df["cost_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"].replace(0, np.nan)
    df["cost_per_mile"] = df["total_receipts_amount"] / df["miles_traveled"].replace(0, np.nan)
    df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"].replace(0, np.nan)
    df["cost_ratio"] = df["cost_per_day"] / df["cost_per_mile"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# ----------------------------------------------------------
# Legacy-style post-processing rules (tiers, tweaks, rounding)
# ----------------------------------------------------------
def apply_postprocessing(raw_pred, days, miles, receipts):
    # Linear calibration (fit on training predictions vs actuals)
    calibrated = 11.132107958925483 + 0.9941362426252188 * raw_pred
    adjusted = calibrated

    # Mileage tier adjustments (softened coefficients)
    if miles < 200:
        adjusted += 0.06 * miles
    elif miles < 600:
        adjusted += 0.04 * miles
    else:
        adjusted += 0.03 * miles

    # Trip duration tweaks (per-diem-like)
    if days <= 2:
        adjusted -= 5.0
    elif days >= 7:
        adjusted += 10.0

    # Diminishing returns on very high receipts
    if receipts > 2000:
        adjusted -= 0.01 * (receipts - 2000)

    # Floor at zero and round to cents
    adjusted = max(adjusted, 0.0)
    return round(float(adjusted), 2)

# ----------------------------------------------------------
# Prediction entrypoint
# ----------------------------------------------------------
def run_prediction(days, miles, receipts):
    model_path = Path(__file__).resolve().parent / "final_model_rules.pkl"
    model = joblib.load(model_path)

    data = pd.DataFrame([{
        "trip_duration_days": days,
        "miles_traveled": miles,
        "total_receipts_amount": receipts
    }])
    data = create_features(data)

    raw_pred = float(model.predict(data)[0])
    return apply_postprocessing(raw_pred, days=days, miles=miles, receipts=receipts)
