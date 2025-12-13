import numpy as np
import pandas as pd
import joblib
import argparse
import sys

# ==========================================================
# ACME Legacy System - Production Prediction Script
# Author: Ayushi Bohra
# Description: Loads the final stacked model and predicts
#              reimbursement from CLI inputs.
# ==========================================================


# ----------------------------------------------------------
# Feature Engineering Function (must match notebook exactly!)
# ----------------------------------------------------------
def create_features(df):
    df = df.copy()

    df["cost_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"].replace(0, np.nan)
    df["cost_per_mile"] = df["total_receipts_amount"] / df["miles_traveled"].replace(0, np.nan)
    df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"].replace(0, np.nan)

    df["cost_ratio"] = df["cost_per_day"] / df["cost_per_mile"]

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# ----------------------------------------------------------
# Main Prediction Function
# ----------------------------------------------------------
def run_prediction(days, miles, receipts):

    # Load trained model
    model = joblib.load("final_model.pkl")

    # Prepare input as DataFrame
    data = pd.DataFrame([{
        "trip_duration_days": days,
        "miles_traveled": miles,
        "total_receipts_amount": receipts
    }])

    # Apply feature engineering
    data = create_features(data)

    # Predict
    prediction = model.predict(data)[0]

    # Return rounded reimbursement
    return round(float(prediction), 2)


# ----------------------------------------------------------
# Safe CLI Parsing (Fixes Jupyter Argument Bug)
# ----------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="ACME Reimbursement Predictor")

    parser.add_argument("trip_duration_days", type=float, help="Total trip days")
    parser.add_argument("miles_traveled", type=float, help="Total miles traveled")
    parser.add_argument("total_receipts_amount", type=float, help="Receipts total amount")

    # This prevents Jupyter kernel args from breaking the script
    args, unknown = parser.parse_known_args()
    return args


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()

    result = run_prediction(
        args.trip_duration_days,
        args.miles_traveled,
        args.total_receipts_amount
    )

    print("\n==============================")
    print("   ACME FINAL PREDICTION")
    print("==============================")
    print(f"Trip Duration (days): {args.trip_duration_days}")
    print(f"Miles Traveled:       {args.miles_traveled}")
    print(f"Receipts Total ($):   {args.total_receipts_amount}")
    print("------------------------------")
    print(f"Predicted Reimbursement: ${result}")
    print("==============================\n")
