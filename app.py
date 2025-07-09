import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]

# Configure Gemini
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("chat-bison-001")  # free model

# Streamlit UI
st.set_page_config(page_title="Venantie Medical Assistant", page_icon="🧬", layout="wide")
st.title("🧬 AI Medical Assistant")
st.markdown("This AI agent helps doctors interpret patients' lab results or reports.")

uploaded_file = st.file_uploader("Upload patient results (CSV or TXT)", type=["csv", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        result_text = df.to_string()
    else:
        result_text = uploaded_file.read().decode()

    st.success("File uploaded successfully!")

    extra_context = st.text_area("Any additional context (e.g., symptoms)?")

    if st.button("Interpret Results"):
        with st.spinner("Analyzing with Gemini AI..."):
            prompt = f"""
You are a professional medical assistant. Analyze the following patient test results and provide possible medical concerns, diagnoses, or treatment recommendations.

Patient Data:
{result_text}

Extra context:
{extra_context}
"""
            response = model.generate_content(prompt)
            st.markdown("### 🧠 AI Interpretation")
            st.write(response.text)
