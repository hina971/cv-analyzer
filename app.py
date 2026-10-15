
import json
import os
from io import BytesIO

import streamlit as st
from docx import Document
from google import genai
from pypdf import PdfReader

st.set_page_config(page_title="CV Analyzer", page_icon="📄", layout="wide")

st.title("📄 CV Analyzer")
st.write("Upload your resume to get an estimated ATS score and improvement suggestions.")

import os
import streamlit as st

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GEMINI_API_KEY", "")

    return key.strip()

api_key = get_api_key()

if not api_key:
    st.error(
        "Gemini API key nahi mili. "
        "Streamlit Settings > Secrets mein "
        "GEMINI_API_KEY configure karein."
    )
    st.stop()

def extract_text(file):
    raw = file.getvalue()
    ext = file.name.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        reader = PdfReader(BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        doc = Document(BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext == "txt":
        text = raw.decode("utf-8", errors="replace")
    else:
        raise ValueError("Upload a PDF, DOCX, or TXT file.")

    if not text.strip():
        raise ValueError("No readable text found in this file.")

    return text[:40000]

uploaded = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"]
)

job_description = st.text_area(
    "Target job description (optional)",
    placeholder="Paste the job description here..."
)

if st.button("Analyze Resume", type="primary"):
    if uploaded is None:
        st.error("Please upload your resume first.")
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error("Gemini API key not found. Configure GEMINI_API_KEY in Streamlit Secrets.")
        st.stop()

    try:
        resume_text = extract_text(uploaded)

        prompt = f"""
Analyze this resume for ATS readiness.

Give an estimated score from 0 to 100. This is not an actual employer ATS score.
Identify strengths, weaknesses, missing relevant keywords, formatting concerns,
and practical improvements. Do not invent achievements or skills.

If a job description is supplied, evaluate alignment with that role.

Return valid JSON with these keys:
ats_score, summary, strengths, improvements,
keywords_found, keywords_to_consider.

Job description:
{job_description or "Not provided"}

Resume:
{resume_text}
"""

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )

        result = json.loads(response.text)

        score = result.get("ats_score")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
            st.error("Gemini returned an invalid ATS score. Please try again.")
            st.stop()

        st.subheader("Estimated ATS Score")
        st.metric("ATS Readiness", f"{score}/100")
        st.progress(score / 100)

        st.subheader("Summary")
        st.write(result.get("summary", ""))

        st.subheader("Strengths")
        for item in result.get("strengths", []):
            st.markdown(f"- {item}")

        st.subheader("Recommended Improvements")
        for item in result.get("improvements", []):
            if isinstance(item, dict):
                st.markdown(f"**{item.get('issue', 'Improvement')}**")
                st.write(item.get("suggested_fix", item.get("description", "")))
            else:
                st.markdown(f"- {item}")

        st.subheader("Keywords Found")
        st.write(", ".join(result.get("keywords_found", [])) or "None returned")

        st.subheader("Keywords to Consider")
        st.write(", ".join(result.get("keywords_to_consider", [])) or "None returned")

        st.download_button(
            "Download Analysis (JSON)",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="resume_analysis.json",
            mime="application/json",
        )

        st.caption("AI-generated estimate only. ATS systems vary, and results are not guaranteed.")

    except Exception as e:
        message = str(e)
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            st.error("Gemini API quota reached. Check your API quota and try again later.")
        elif "API key" in message or "UNAUTHENTICATED" in message:
            st.error("Gemini authentication failed. Check your API key.")
        else:
            st.error(f"Analysis failed: {message[:300]}")
