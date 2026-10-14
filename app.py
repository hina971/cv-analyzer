
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import json
import os
import re
from io import BytesIO

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

APP_TITLE = "Resume ATS Analyzer"
MODEL_NAME = "gemini-3.8-flash"
MAX_FILE_SIZE_MB = 10

st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .main .block-container {max-width: 1050px; padding-top: 2rem;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#eaf2ff,#f5edff); margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .35rem 0; color: #172554;}
      .hero p {margin: 0; color: #334155; font-size: 1.05rem;}
      .small-note {font-size: .88rem; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>📄 Resume ATS Analyzer</h1>'
    '<p>Upload a resume to get an estimated ATS-readiness score, practical improvements, and keyword suggestions.</p></div>',
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY")


def extract_resume_text(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT uploads."""
    raw = uploaded_file.getvalue()
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        reader = PdfReader(BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif extension == "docx":
        document = Document(BytesIO(raw))
        chunks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(chunks)
    elif extension == "txt":
        text = raw.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(
            "No readable text was found. If this is a scanned/image-only PDF, run OCR on it or upload a text-based PDF/DOCX."
        )
    if len(text) > 45_000:
        text = text[:45_000] + "\n\n[Resume text shortened for analysis.]"
    return text


def analyze_resume(resume_text: str, job_description: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a careful resume reviewer. Evaluate the resume for ATS readability and relevance.
Do not claim to know the exact score a real employer's ATS would give. Give an estimated score from 0 to 100 based only on the supplied text. If a job description is supplied, score alignment to it; otherwise score general ATS readiness.
Do not infer or score protected traits such as age, gender, race, religion, disability, or nationality. Do not invent facts or achievements. Distinguish missing information from information that may simply not be present in the extracted text.

Return ONLY valid JSON matching this schema:
{{
  "ats_score": integer from 0 to 100,
  "score_label": string,
  "summary": string,
  "category_scores": [{{"category": string, "score": integer from 0 to 100, "reason": string}}],
  "strengths": [string],
  "improvements": [{{"priority": "High" | "Medium" | "Low", "issue": string, "why_it_matters": string, "suggested_fix": string}}],
  "keywords_found": [string],
  "keywords_to_consider": [string],
  "formatting_checks": [{{"check": string, "status": "Good" | "Needs review" | "Not assessable", "note": string}}],
  "sample_rewrites": [{{"before_or_issue": string, "suggested_version": string}}],
  "disclaimer": string
}}
Use concise, specific feedback. Keep category_scores to 4-6 items, improvements to at most 8, strengths to at most 5, keywords lists to at most 15 each, formatting_checks to at most 6, and sample_rewrites to at most 4. If a metric cannot be assessed from plain text, say so instead of guessing. Keep rewritten bullets faithful to facts already in the resume; use placeholders like [X%] only when needed and clearly mark them as placeholders.

JOB DESCRIPTION (may be empty):
{job_description.strip() or "Not provided; assess general ATS readiness."}

RESUME TEXT:
{resume_text}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=5000,
        ),
    )
    if not response.text:
        raise RuntimeError("The AI returned an empty response. Please try again.")
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The AI response could not be read as JSON. Please try again.") from exc

    score = result.get("ats_score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise RuntimeError("The AI returned an invalid score. Please try again.")
    return result


def show_list(title: str, values):
    st.subheader(title)
    if not values:
        st.write("No items returned.")
        return
    for item in values:
        st.markdown(f"- {item}")


with st.sidebar:
    st.header("How it works")
    st.write("1. Upload your resume.\n2. Optionally paste a job description.\n3. Get an estimated score and targeted suggestions.")
    st.markdown("---")
    st.caption("Supported formats: PDF, DOCX, TXT · Maximum upload size: 10 MB")
    st.caption("Your resume text is sent to the Gemini API for analysis. Avoid uploading information you do not want processed by an external AI service.")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Use a text-based PDF or DOCX for the best extraction results.",
)
job_description = st.text_area(
    "Target job description (optional)",
    placeholder="Paste the job description here to check keyword and role alignment...",
    height=180,
)

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File is too large. Please upload a file smaller than {MAX_FILE_SIZE_MB} MB.")
        st.stop()
    st.success(f"Ready to analyze: {uploaded_file.name}")

analyze_clicked = st.button("Analyze resume", type="primary", disabled=uploaded_file is None, use_container_width=True)

if analyze_clicked:
    api_key = get_api_key()
    if not api_key:
        st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets or set it as an environment variable.")
        st.stop()

    try:
        with st.spinner("Reading your resume and reviewing ATS readiness..."):
            resume_text = extract_resume_text(uploaded_file)
            result = analyze_resume(resume_text, job_description, api_key)
        st.session_state["analysis_result"] = result
        st.session_state["analysis_filename"] = uploaded_file.name
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        # Show a useful message without exposing API keys or a full traceback.
        message = str(exc)
        if "API key" in message or "PERMISSION_DENIED" in message or "UNAUTHENTICATED" in message:
            st.error("Gemini authentication failed. Check that your API key is valid and enabled for the Gemini API.")
        elif "RESOURCE_EXHAUSTED" in message or "429" in message:
            st.error("Gemini rate limit or quota reached. Wait a little and try again, or check your API quota.")
        else:
            st.error(f"Analysis failed: {message[:400]}")

if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    st.markdown("---")
    st.caption(f"Analysis for: {st.session_state.get('analysis_filename', 'uploaded resume')}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Estimated ATS score", f"{result['ats_score']}/100")
        st.progress(result["ats_score"] / 100)
        st.write(result.get("score_label", "ATS readiness estimate"))
    with col2:
        st.subheader("Overall assessment")
        st.write(result.get("summary", "No summary returned."))

    categories = result.get("category_scores", [])
    if categories:
        st.subheader("Category breakdown")
        for category in categories:
            name = str(category.get("category", "Category"))
            score = category.get("score", 0)
            score = score if isinstance(score, int) and 0 <= score <= 100 else 0
            st.markdown(f"*{name} — {score}/100*")
            st.progress(score / 100)
            st.caption(category.get("reason", ""))

    left, right = st.columns(2)
    with left:
        show_list("Strengths", result.get("strengths", []))
        show_list("Keywords found", result.get("keywords_found", []))
    with right:
        show_list("Keywords to consider", result.get("keywords_to_consider", []))
        st.subheader("Formatting checks")
        for check in result.get("formatting_checks", []):
            st.markdown(f"*{check.get('status', 'Not assessable')}: {check.get('check', 'Check')}*")
            st.write(check.get("note", ""))

    st.subheader("Recommended improvements")
    improvements = result.get("improvements", [])
    if improvements:
        for i, item in enumerate(improvements, start=1):
            with st.expander(f"{i}. [{item.get('priority', 'Medium')}] {item.get('issue', 'Improvement')}"):
                st.write(item.get("why_it_matters", ""))
                st.markdown("*Suggested fix*")
                st.write(item.get("suggested_fix", ""))
    else:
        st.write("No improvements were returned.")

    rewrites = result.get("sample_rewrites", [])
    if rewrites:
        st.subheader("Example bullet improvements")
        for item in rewrites:
            st.markdown(f"*Current issue:* {item.get('before_or_issue', '')}")
            st.markdown(f"*Suggested version:* {item.get('suggested_version', '')}")

    st.info(result.get("disclaimer", "This is an AI-generated estimate, not a guarantee of how any employer's ATS will score your resume."))
    st.download_button(
        "Download analysis as JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="resume_ats_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown("<p class='small-note'>Note: ATS systems vary. This score is an AI-generated estimate, not a guarantee of screening or hiring outcomes.</p>", unsafe_allow_html=True)
