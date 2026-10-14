
import json
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

APP_TITLE = "Resume ATS Analyzer"
MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_SIZE = 10 * 1024 * 1024

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume ATS Analyzer")
st.write(
    "Upload your resume to get an estimated ATS score, "
    "resume improvements, and job-related keyword suggestions."
)


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def extract_resume_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    try:
        if extension == "pdf":
            reader = PdfReader(BytesIO(file_bytes))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        elif extension == "docx":
            document = Document(BytesIO(file_bytes))

            parts = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]

            for table in document.tables:
                for row in table.rows:
                    parts.append(
                        " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                        )
                    )

            text = "\n".join(parts)

        elif extension == "txt":
            text = file_bytes.decode(
                "utf-8",
                errors="replace",
            )

        else:
            raise ValueError(
                "Please upload a PDF, DOCX, or TXT file."
            )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not read the file. Please upload a valid "
            "PDF, DOCX, or TXT file."
        ) from exc

    text = text.strip()

    if not text:
        raise ValueError(
            "No readable text found. Scanned PDFs may need OCR."
        )

    return text[:45000]


def analyze_resume(resume_text, job_description, api_key):
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert resume reviewer.

Analyze the supplied resume for ATS readability and relevance.
Give an estimated score from 0 to 100. Do not claim this is the
exact score from any employer's ATS.

If a job description is supplied, evaluate resume alignment.
Otherwise, evaluate general ATS readiness.

Do not invent qualifications, experience, or achievements.
Do not score protected personal traits.
Keep feedback concise and practical.
Suggested rewrites must remain faithful to the resume.

Return only valid JSON with this structure:

{{
  "ats_score": 75,
  "score_label": "Good",
  "summary": "Overall assessment",
  "category_scores": [
    {{
      "category": "Keywords",
      "score": 75,
      "reason": "Explanation"
    }}
  ],
  "strengths": ["Strength"],
  "improvements": [
    {{
      "priority": "High",
      "issue": "Issue",
      "why_it_matters": "Explanation",
      "suggested_fix": "Suggested fix"
    }}
  ],
  "keywords_found": ["Python"],
  "keywords_to_consider": ["SQL"],
  "formatting_checks": [
    {{
      "check": "Section headings",
      "status": "Good",
      "note": "Explanation"
    }}
  ],
  "sample_rewrites": [
    {{
      "before_or_issue": "Original issue",
      "suggested_version": "Improved version"
    }}
  ],
  "disclaimer": "This is an AI-generated estimate."
}}

Limits:
- Maximum 6 category scores.
- Maximum 5 strengths.
- Maximum 8 improvements.
- Maximum 15 items in each keyword list.
- Maximum 6 formatting checks.
- Maximum 4 sample rewrites.
- Use only information supported by the resume.
- Use placeholders such as [X%] when a real metric is missing.
- If something cannot be assessed, say so.

JOB DESCRIPTION:
{job_description.strip() or "Not provided. Assess general ATS readiness."}

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
        raise RuntimeError("The AI returned an empty response.")

    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            "The AI returned an invalid response. Please try again."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("The AI returned an invalid result.")

    score = result.get("ats_score")

    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not 0 <= score <= 100
    ):
        raise RuntimeError("The AI returned an invalid ATS score.")

    result["ats_score"] = int(round(score))

    for field in (
        "category_scores",
        "strengths",
        "improvements",
        "keywords_found",
        "keywords_to_consider",
        "formatting_checks",
        "sample_rewrites",
    ):
        if not isinstance(result.get(field), list):
            result[field] = []

    return result


uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
)

job_description = st.text_area(
    "Job description (optional)",
    placeholder="Paste the job description here...",
    height=160,
)

if uploaded_file and uploaded_file.size > MAX_FILE_SIZE:
    st.error("File too large. Maximum allowed size is 10 MB.")
    st.stop()

if st.button(
    "Analyze Resume",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
):
    api_key = get_api_key()

    if not api_key:
        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY "
            "to Streamlit Secrets or your local .env file."
        )
        st.stop()

    try:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_resume_text(uploaded_file)
            result = analyze_resume(
                resume_text,
                job_description,
                api_key,
            )

        st.session_state["analysis_result"] = result
        st.session_state["analysis_filename"] = uploaded_file.name

    except ValueError as exc:
        st.error(str(exc))

    except Exception as exc:
        message = str(exc)
        upper_message = message.upper()

        if any(
            term in upper_message
            for term in (
                "API_KEY_INVALID",
                "INVALID API KEY",
                "UNAUTHENTICATED",
                "PERMISSION_DENIED",
            )
        ):
            st.error(
                "Gemini authentication failed. Check your API key "
                "and make sure the Gemini API is enabled."
            )

        elif "429" in message or "RESOURCE_EXHAUSTED" in upper_message:
            st.error(
                "Gemini quota or rate limit reached. Try again later "
                "or check your API quota."
            )

        elif "NOT_FOUND" in upper_message:
            st.error(
                f"Model '{MODEL_NAME}' was not found. Check the "
                "Gemini model name available to your API key."
            )

        else:
            st.error(
                "Analysis failed. Check your settings and try again. "
                f"Details: {message[:200]}"
            )


if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]

    st.divider()
    st.caption(
        "Resume: "
        + st.session_state.get("analysis_filename", "Uploaded resume")
    )

    left, right = st.columns([1, 2])

    with left:
        score = result["ats_score"]
        st.metric("Estimated ATS Score", f"{score}/100")
        st.progress(score / 100)
        st.write(result.get("score_label", "ATS readiness estimate"))

    with right:
        st.subheader("Overall Assessment")
        st.write(result.get("summary", "No summary available."))

    categories = result["category_scores"]

    if categories:
        st.subheader("Category Breakdown")

        for item in categories:
            if not isinstance(item, dict):
                continue

            category_score = item.get("score", 0)

            if (
                not isinstance(category_score, (int, float))
                or isinstance(category_score, bool)
                or not 0 <= category_score <= 100
            ):
                category_score = 0

            category_score = int(round(category_score))

            st.markdown(
                f"**{item.get('category', 'Category')}: "
                f"{category_score}/100**"
            )
            st.progress(category_score / 100)
            st.caption(item.get("reason", ""))

    st.subheader("Strengths")
    if result["strengths"]:
        for item in result["strengths"]:
            st.markdown(f"- {item}")
    else:
        st.write("No strengths returned.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Keywords Found")
        for keyword in result["keywords_found"]:
            st.markdown(f"- {keyword}")

    with col2:
        st.subheader("Keywords to Consider")
        for keyword in result["keywords_to_consider"]:
            st.markdown(f"- {keyword}")

    st.subheader("Formatting Checks")
    for item in result["formatting_checks"]:
        if not isinstance(item, dict):
            continue

        st.markdown(
            f"**{item.get('status', 'Not assessed')}: "
            f"{item.get('check', 'Check')}**"
        )
        st.write(item.get("note", ""))

    st.subheader("Recommended Improvements")

    if result["improvements"]:
        for index, item in enumerate(
            result["improvements"],
            start=1,
        ):
            if not isinstance(item, dict):
                continue

            title = (
                f"{index}. [{item.get('priority', 'Medium')}] "
                f"{item.get('issue', 'Improvement')}"
            )

            with st.expander(title):
                st.write(item.get("why_it_matters", ""))
                st.markdown("**Suggested Fix**")
                st.write(item.get("suggested_fix", ""))
    else:
        st.write("No improvements returned.")

    if result["sample_rewrites"]:
        st.subheader("Example Resume Rewrites")

        for item in result["sample_rewrites"]:
            if not isinstance(item, dict):
                continue

            st.markdown(
                f"**Original:** {item.get('before_or_issue', '')}"
            )
            st.markdown(
                f"**Suggested:** {item.get('suggested_version', '')}"
            )

    st.info(
        result.get(
            "disclaimer",
            "This AI-generated score is an estimate, not a guarantee "
            "of an employer's ATS result.",
        )
    )

    st.download_button(
        "Download Analysis as JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="resume_ats_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

st.caption(
    "ATS scores are estimates. Different employers may use "
    "different screening systems and criteria."
)
