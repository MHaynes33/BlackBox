"""
Compute Phase 3 stacking ensemble metrics on a 75/25 holdout split.
Outputs MAE/RMSE/R^2 and within-$ thresholds, and writes data/phase3_predictions.csv.
"""

import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    data_path = Path("data/phase2_features_baseline_models.csv")
    df = pd.read_csv(data_path)

    features = [
        "trip_duration_days",
        "miles_traveled",
        "total_receipts_amount",
        "cost_per_day",
        "cost_per_mile",
        "miles_per_day",
        "cost_ratio",
    ]

    X = df[features]
    y = df["reimbursement"]

    # 75/25 holdout split (index-based to mirror prior runs)
    split = int(0.75 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base_models = [
        ("decision_tree", DecisionTreeRegressor(random_state=42)),
        ("random_forest", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("gradient_boosting", GradientBoostingRegressor(random_state=42)),
    ]
    meta_model = LinearRegression()
    model = StackingRegressor(estimators=base_models, final_estimator=meta_model, passthrough=True)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    abs_diff = np.abs(y_test.values - pred)

    mae = mean_absolute_error(y_test, pred)
    rmse = sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    within_0_01 = (abs_diff <= 0.01).mean()
    within_1 = (abs_diff <= 1.0).mean()
    within_5 = (abs_diff <= 5.0).mean()

    print("Phase 3 holdout metrics (75/25 split):")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R^2:   {r2:.4f}")
    print(f"Exact (<= $0.01): {within_0_01:.4f}")
    print(f"Close (<= $1):    {within_1:.4f}")
    print(f"Within $5:        {within_5:.4f}")
    print(f"Test rows:        {len(y_test)}")

    out = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": pred,
            "AbsDiff": abs_diff,
        }
    )
    out_path = Path("data/phase3_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
