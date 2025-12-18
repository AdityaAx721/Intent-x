import json
import joblib
from pathlib import Path

from src.preprocess import clean_text

CONFIDENCE_THRESHOLD = 0.30

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

model = joblib.load(MODEL_DIR / "intent_classifier.joblib")
vectorizer = joblib.load(DATA_DIR / "tfidf_vectorizer.joblib")

with open(MODEL_DIR / "intent_mapping.json") as f:
    INTENT_MAP = json.load(f)

with open(MODEL_DIR / "intent_groups.json") as f:
    INTENT_GROUPS = json.load(f)


def predict_intent(text: str):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(vectorized_text)[0]
    pred_id = probs.argmax()
    confidence = probs[pred_id]

    if confidence < CONFIDENCE_THRESHOLD:
        # Return top 3 possible intents
        top_ids = probs.argsort()[-3:][::-1]
        possible_intents = [INTENT_MAP[str(i)] for i in top_ids]
        return "uncertain_intent", confidence, possible_intents

    intent_name = INTENT_MAP[str(pred_id)]
    return intent_name, confidence, None
