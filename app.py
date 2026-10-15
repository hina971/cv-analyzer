import io
import json
import re

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------- Settings ----------
DEFAULT_MODEL = "gemini-3.5-flash"  # change in the sidebar if Google renames it
MAX_RESUME_CHARS = 15000
MAX_JD_CHARS = 6000

# How much each part counts towards the final ATS score (total = 100)
WEIGHTS = {
    "keywords_and_skills": 30,
    "experience_and_impact": 25,
    "formatting_and_structure": 20,
    "clarity_and_grammar": 15,
    "contact_and_sections": 10,
}

LABELS = {
    "keywords_and_skills": "Keywords & Skills",
    "experience_and_impact": "Experience & Impact",
    "formatting_and_structure": "Formatting & Structure",
    "clarity_and_grammar": "Clarity & Grammar",
    "contact_and_sections": "Contact & Sections",
}


# ---------- Step 1: read the resume file ----------
def extract_text(uploaded_file) -> str:
    """Return plain text from a PDF, DOCX or TXT upload."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        text = "\n".join(parts)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")

    return text.strip()


# ---------- Step 2: ask Gemini ----------
def build_prompt(resume_text: str, job_description: str) -> str:
    jd_part = (
        f"JOB DESCRIPTION:\n{job_description[:MAX_JD_CHARS]}\n"
        if job_description.strip()
        else "JOB DESCRIPTION: not provided. Judge the resume for general ATS-friendliness."
    )
    return f"""You are an expert ATS (Applicant Tracking System) resume reviewer.
Review the resume below and reply with ONLY valid JSON (no markdown, no extra text).

Score each category from 0 to 100 using this scale:
90-100 excellent, 70-89 good, 50-69 average, below 50 weak. Be honest and strict.

Required JSON format:
{{
  "scores": {{
    "keywords_and_skills": <int 0-100>,
    "experience_and_impact": <int 0-100>,
    "formatting_and_structure": <int 0-100>,
    "clarity_and_grammar": <int 0-100>,
    "contact_and_sections": <int 0-100>
  }},
  "summary": "<2-3 sentence overall verdict>",
  "strengths": ["<short point>", "..."],
  "missing_keywords": ["<keyword or skill that should be added>", "..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "issue": "<what is wrong>", "fix": "<exactly what to do>"}}
  ],
  "rewrite_examples": [
    {{"before": "<weak line copied from the resume>", "after": "<stronger version with action verb and numbers>"}}
  ]
}}

Rules:
- "missing_keywords": if a job description is given, use its keywords; otherwise use common keywords for the candidate's field.
- Give 5-8 improvements, 2-3 rewrite examples, 3-5 strengths.
- Only quote lines that really exist in the resume. Never invent experience.

{jd_part}
RESUME:
{resume_text[:MAX_RESUME_CHARS]}
"""


def parse_json(raw: str) -> dict:
    """Pull a JSON object out of the model reply, even if it has ``` fences."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI did not return JSON.")
    return json.loads(raw[start : end + 1])


def clamp(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def clean_result(data: dict) -> dict:
    """Make the AI output safe to display and calculate the final ATS score."""
    raw_scores = data.get("scores") or {}
    scores = {key: clamp(raw_scores.get(key)) for key in WEIGHTS}
    overall = int(sum(scores[k] * w for k, w in WEIGHTS.items()) / sum(WEIGHTS.values()) + 0.5)

    def as_list(key):
        value = data.get(key)
        return value if isinstance(value, list) else []

    return {
        "overall": overall,
        "scores": scores,
        "summary": str(data.get("summary", "")),
        "strengths": [str(s) for s in as_list("strengths")],
        "missing_keywords": [str(k) for k in as_list("missing_keywords")],
        "improvements": [i for i in as_list("improvements") if isinstance(i, dict)],
        "rewrite_examples": [r for r in as_list("rewrite_examples") if isinstance(r, dict)],
    }


def analyze_resume(client, model: str, resume_text: str, job_description: str) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(resume_text, job_description),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return clean_result(parse_json(response.text))


# ---------- Step 3: the web page ----------
def get_api_key_from_secrets() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def score_label(score: int) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Needs work"
    return "Weak"


def show_results(result: dict) -> None:
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("ATS Score", f"{result['overall']} / 100")
        st.caption(score_label(result["overall"]))
    with col2:
        st.progress(result["overall"] / 100)
        if result["summary"]:
            st.write(result["summary"])

    st.subheader("Score breakdown")
    for key, label in LABELS.items():
        st.write(f"**{label}** ({WEIGHTS[key]}% weight): {result['scores'][key]}/100")
        st.progress(result["scores"][key] / 100)

    if result["strengths"]:
        st.subheader("What is working well")
        for item in result["strengths"]:
            st.write(f"- {item}")

    if result["missing_keywords"]:
        st.subheader("Keywords to add")
        st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))

    if result["improvements"]:
        st.subheader("Improvements")
        order = {"high": 0, "medium": 1, "low": 2}
        items = sorted(
            result["improvements"],
            key=lambda i: order.get(str(i.get("priority", "")).lower(), 3),
        )
        for item in items:
            st.markdown(
                f"**[{item.get('priority', '-')}] {item.get('issue', '')}**  \n"
                f"{item.get('fix', '')}"
            )

    if result["rewrite_examples"]:
        st.subheader("Example rewrites")
        for ex in result["rewrite_examples"]:
            st.markdown(f"Before: {ex.get('before', '')}")
            st.markdown(f"After: **{ex.get('after', '')}**")
            st.write("")


def main() -> None:
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="centered")
    st.title("📄 ATS Resume Checker")
    st.write("Upload your resume to get an ATS score and clear tips to improve it.")

    with st.sidebar:
        st.header("Settings")
        api_key = get_api_key_from_secrets()
        if not api_key:
            api_key = st.text_input("Gemini API key", type="password")
            st.caption("Get a free key at aistudio.google.com/apikey")
        model = st.text_input("Gemini model", value=DEFAULT_MODEL)
        st.caption("Your resume text is sent to Google's Gemini API for analysis.")

    uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional, but gives a much better score)",
        height=150,
        placeholder="Paste the job description here...",
    )

    if st.button("Analyze my resume", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return
        if uploaded is None:
            st.error("Please upload a resume first.")
            return

        try:
            with st.spinner("Reading your resume..."):
                resume_text = extract_text(uploaded)
            if len(resume_text) < 100:
                st.error(
                    "I could not read enough text. If your PDF is a scanned image, "
                    "please upload a text-based PDF or a DOCX file."
                )
                return

            with st.spinner("Analyzing with Gemini..."):
                client = genai.Client(api_key=api_key)
                result = analyze_resume(client, model.strip(), resume_text, job_description)
            show_results(result)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
