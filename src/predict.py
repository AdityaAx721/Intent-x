import json
import joblib
from pathlib import Path

from src.preprocess import clean_text

# --------------------
# Config
# --------------------
CONFIDENCE_THRESHOLD = 0.30

# Paths
MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

# Load artifacts
model = joblib.load(MODEL_DIR / "intent_classifier.joblib")
vectorizer = joblib.load(DATA_DIR / "tfidf_vectorizer.joblib")

# Load intent mapping
with open(MODEL_DIR / "intent_mapping.json") as f:
    INTENT_MAP = json.load(f)


def predict_intent(text: str):
    """
    Predict intent with confidence & fallback handling
    """
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(vectorized_text)[0]
    pred_id = probs.argmax()
    confidence = probs[pred_id]

    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain_intent", confidence

    intent_name = INTENT_MAP[str(pred_id)]
    return intent_name, confidence
