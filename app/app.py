import streamlit as st
import joblib
from pathlib import Path

# Import the same cleaning function
from src.preprocess import clean_text

# Paths
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "intent_classifier.joblib")
    vectorizer = joblib.load(PROCESSED_DIR / "tfidf_vectorizer.joblib")
    return model, vectorizer


model, vectorizer = load_artifacts()

# UI
st.set_page_config(page_title="Intent X", page_icon="🧠")
st.title("Intent X — Intent Classification Engine")
st.write("Enter a message and the system will predict the user intent.")

user_input = st.text_area(
    "User Input",
    placeholder="e.g. I want to block my debit card immediately",
    height=120
)

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()

        st.success(f"Predicted Intent: **{prediction}**")
        st.info(f"Confidence: **{confidence:.2%}**")
