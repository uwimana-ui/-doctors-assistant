import streamlit as st
import pandas as pd
import google.generativeai as genai

# Load Gemini API key from Streamlit Secrets
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

        # Limit rows to max 10 to reduce prompt size & API usage
        max_rows = 10
        if len(df) > max_rows:
            st.warning(f"⚠️ Showing only first {max_rows} rows to reduce API usage.")
            df = df.head(max_rows)

        st.subheader("📊 Patient Data Preview")
        st.dataframe(df)
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
            try:
                response = model.generate_content(prompt)
                st.markdown("### 🧠 AI Medical Insights")
                st.write(response.text)
            except Exception as e:
                st.error(f"⚠️ API Error: {e}\nPlease try again later or reduce the data size.")
