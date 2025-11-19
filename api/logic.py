# api/logic.py

def risk_flag(preds: dict) -> bool:
    """
    Binary risk flag.
    Returns True if either RF or XGB predicts class 1 (risky).
    """
    return preds.get("rf_prediction") == 1 or preds.get("xgb_prediction") == 1


def weighted_risk_score(preds: dict, weights=(0.3, 0.7)) -> float:
    """
    Continuous risk score combining RF and XGB predictions.
    Default weights: 30% RF, 70% XGB.
    """
    rf = preds.get("rf_prediction", 0)
    xgb = preds.get("xgb_prediction", 0)
    return weights[0] * rf + weights[1] * xgb


def should_alert(preds: dict, threshold: float = 0.5) -> bool:
    """
    Decide if an alert should be triggered.
    - Always alert if risk_flag is True.
    - Or if weighted risk score exceeds threshold.
    """
    if risk_flag(preds):
        return True
    return weighted_risk_score(preds) >= threshold
