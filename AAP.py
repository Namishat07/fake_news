import streamlit as st
import joblib

# Load model
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.set_page_config(page_title="News Authenticity Detector", layout="wide")
st.title("📰 Fake or Real News Detector")
st.markdown("Enter a news article or headline to check if it's fake or AI-generated.")

text_input = st.text_area("Your News Text", height=200)

if st.button("Check"):
    if text_input.strip() == "":
        st.warning("Please enter text.")
    else:
        vect_input = vectorizer.transform([text_input])
        result = model.predict(vect_input)[0]

        if result == 1:
            st.success("✅ This news appears to be **REAL**.")
        else:
            st.error("🚫 This news appears to be **FAKE** or AI-generated.")
