# jobs/daily_summary.py
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8000/predict/combined"
REPORTS_PATH = Path("data/new_reports.parquet")

def run_daily_summary():
    if not REPORTS_PATH.exists():
        print("No new reports found.")
        return

    df = pd.read_parquet(REPORTS_PATH)
    risky_count = 0
    results = []

    for _, row in df.iterrows():
        payload = row.to_dict()  # API expects raw features
        r = requests.post(API_URL, json=payload)
        if r.status_code == 200:
            resp = r.json()
            preds = resp["predictions"]
            is_risky = resp["is_risky"]
            results.append({
                "date": datetime.now(),
                "lat": payload.get("Location_of_households_Latitude"),
                "lon": payload.get("Location_of_households_Longitude"),
                "category": payload.get("category", "unknown"),
                "rf_prediction": preds["rf_prediction"],
                "xgb_prediction": preds["xgb_prediction"],
                "is_risky": is_risky,
                "risk_score": resp["risk_score"]
            })
            if is_risky:
                risky_count += 1
        else:
            print(f"Error for row {row}: {r.text}")

    print("Households at risk today:", risky_count)

    # Save summary to parquet for dashboard
    summary_path = Path("data/daily_summary.parquet")
    df_summary = pd.DataFrame(results)
    if summary_path.exists():
        df_old = pd.read_parquet(summary_path)
        df_all = pd.concat([df_old, df_summary], ignore_index=True)
    else:
        df_all = df_summary
    df_all.to_parquet(summary_path, index=False)
