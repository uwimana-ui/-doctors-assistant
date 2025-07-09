import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Load API key from Streamlit Secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# Use free Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Venantie Medical Assistant", page_icon="🧬", layout="wide")
st.title("🧬 AI Medical Assistant")
st.markdown("Upload lab results and get AI-based medical analysis.")

uploaded_file = st.file_uploader("📁 Upload patient results (CSV or TXT)", type=["csv", "txt"])

if uploaded_file:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Patient Data")
        st.dataframe(df.head())
        result_text = df.to_string()
    else:
        result_text = uploaded_file.read().decode()

    st.success("✅ File uploaded successfully!")

    extra_context = st.text_area("📝 Any additional symptoms or notes?")

    if st.button("🔍 Analyze with AI"):
        with st.spinner("Generating medical insights..."):
            prompt = f"""
You are a professional medical assistant. Analyze the following patient test results and provide possible diagnoses or treatment recommendations.

Patient Data:
{result_text}

Extra Context:
{extra_context}
"""
            response = model.generate_content(prompt)
            st.markdown("### 🧠 AI Medical Insights")
            st.write(response.text)
