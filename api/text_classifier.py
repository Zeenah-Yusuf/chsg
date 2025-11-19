import importlib.util

# Attempt a dynamic import of transformers to avoid static import errors in environments
# where the package is not installed; if unavailable, provide a small fallback.
if importlib.util.find_spec("transformers") is not None:
    transformers = importlib.import_module("transformers")
    _pipeline = getattr(transformers, "pipeline")

    # Load pretrained sentiment classifier (fallback)
    nlp_classifier = _pipeline("text-classification",
                              model="distilbert-base-uncased-finetuned-sst-2-english")
else:
    # Simple fallback classifier that mimics the transformers pipeline output shape.
    def nlp_classifier(text):
        t = text.lower()
        positives = {"good", "great", "safe", "recovered", "stable"}
        negatives = {"bad", "danger", "risk", "death", "deaths", "died", "sick", "critical"}
        score = 0
        for w in positives:
            if w in t:
                score += 1
        for w in negatives:
            if w in t:
                score -= 1
        label = "POSITIVE" if score >= 0 else "NEGATIVE"
        return [{"label": label, "score": 0.0}]

def classify_text(text: str) -> str:
    """
    Classify hazard reports into categories.
    Uses keyword rules first, then falls back to pretrained classifier.
    """
    t = text.lower()
    if "cholera" in t or "diarrhea" in t:
        return "cholera"
    if "fire" in t or "burn" in t:
        return "fire"
    if "injury" in t or "accident" in t or "hurt" in t:
        return "injury"
    # fallback: POSITIVE/NEGATIVE sentiment
    result = nlp_classifier(text)[0]
    return result["label"]
