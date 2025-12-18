import sys
from pathlib import Path

# -------------------------------------------------
# Add project root to PYTHONPATH (Streamlit Cloud fix)
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from src.predict import predict_intent

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Intent X",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Intent X — Intent Classification Engine")
st.write("Enter a message and the system will predict the user intent.")

user_input = st.text_area(
    "User Input",
    placeholder="e.g. I want to block my debit card immediately",
    height=120
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        intent, confidence = predict_intent(user_input)

        if intent == "uncertain_intent":
            st.warning("I'm not confident enough to understand this request.")
            st.info("Please rephrase or provide more details.")
        else:
            st.success(
                f"🎯 Predicted Intent: **{intent.replace('_', ' ').title()}**"
            )
            st.info(f"📊 Confidence: **{confidence:.2%}**")
