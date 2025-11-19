def risk_flag(preds: dict) -> bool:
    return preds.get("rf_prediction") == 1 or preds.get("xgb_prediction") == 1

def weighted_risk_score(preds: dict, weights=(0.3, 0.7)) -> float:
    return weights[0]*preds.get("rf_prediction", 0) + weights[1]*preds.get("xgb_prediction", 0)
