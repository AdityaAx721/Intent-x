import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from src.predict import predict_intent

# UI
st.set_page_config(page_title="Intent X", page_icon="🧠")
st.title("Intent X — Intent Classification Engine")
st.write("Enter a message and the system will predict the user intent.")

user_input = st.text_area(
    "User Input",
    placeholder="e.g. I want to block my debit card immediately",
    height=120
)

# 👇 NOTHING RUNS UNTIL BUTTON IS CLICKED
if st.button("Predict Intent"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        intent, confidence, suggestions = predict_intent(user_input)

        if intent == "uncertain_intent":
            st.warning("🤔 I need a bit more clarity.")

            if suggestions:
                st.write("Did you mean one of these?")
                for s in suggestions:
                    st.write(f"- {s.replace('_', ' ').title()}")

            st.info("Please clarify your issue in more detail.")

        else:
            st.success(
                f"🎯 Predicted Intent: **{intent.replace('_', ' ').title()}**"
            )
            st.info(f"📊 Confidence: **{confidence:.2%}**")
