import streamlit as st
import joblib
import os
import numpy as np

# Load model and vectorizer with error handling
try:
    model = joblib.load("news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.stop()

# UI Setup
st.set_page_config(page_title="News Authenticity Detector", layout="centered")
st.title("📰 Fake or Real News Detector")
st.markdown("Enter a news article or headline to check if it's fake or real. Use the sample texts to test the model.")

# Sample Texts
sample_texts = [
    "Government launches new policy to improve healthcare access for all.",
    "Aliens have landed in New York City and taken over Times Square!",
    "Scientists discover water on Mars.",
    "Celebrity spotted buying groceries at local store."
]

sample_choice = st.selectbox("Or select a sample text to test:", ["Choose a sample..."] + sample_texts)

text_input = st.text_area("Your News Text", sample_choice if sample_choice != "Choose a sample..." else "", height=200)

if st.button("Check Authenticity"):
    if text_input.strip() == "":
        st.warning("Please enter text.")
    else:
        vect_input = vectorizer.transform([text_input])
        result = model.predict(vect_input)[0]
        confidence = np.max(model.predict_proba(vect_input)) * 100

        if result == 1:
            st.success(f"✅ This news appears to be **REAL** with a confidence of {confidence:.2f}%. ")
        elif result == 0:
            st.error(f"🚫 This news appears to be **FAKE** or AI-generated with a confidence of {confidence:.2f}%. ")
        else:
            st.warning("⚠️ Unexpected output from the model.")
