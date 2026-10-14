```python
import json
import os
import re
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

# Load .env for local development
load_dotenv()

APP_TITLE = "Resume ATS Analyzer"
MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_SIZE_MB = 10

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main .block-container {
          max-width: 1050px;
          padding-top: 2rem;
      }
      .hero {
          padding: 1.4rem 1.6rem;
          border-radius: 16px;
          background: linear-gradient(120deg,#eaf2ff,#f5edff);
          margin-bottom: 1rem;
      }
      .hero h1 {
          margin: 0 0 .35rem 0;
          color: #172554;
      }
      .hero p {
          margin: 0;
          color: #334155;
          font-size: 1.05rem;
      }
      .small-note {
          font-size: .88rem;
          color: #64748b;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>📄 Resume ATS Analyzer</h1>'
    '<p>Upload a resume to get an estimated ATS-readiness score, '
    'practical improvements, and keyword suggestions.</p></div>',
    unsafe_allow_html=True,
)


def get_api_key() -> str:
    """Read API key from Streamlit Secrets or environment variables."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def extract_resume_text(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT files."""
    raw = uploaded_file.getvalue()
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        reader = PdfReader(BytesIO(raw))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    elif extension == "docx":
        document = Document(BytesIO(raw))
        chunks = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        for table in document.tables:
            for row in table.rows:
                chunks.append(
                    " | ".join(cell.text.strip() for cell in row.cells)
                )

        text = "\n".join(chunks)

    elif extension == "txt":
        text = raw.decode("utf-8", errors="replace")

    else:
        raise ValueError(
            "Unsupported file type. Upload PDF, DOCX, or TXT."
        )

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError(
            "No readable text found. If your PDF is scanned, "
            "use OCR or upload a text-based PDF/DOCX."
        )

    if len(text) > 45000:
        text = text[:45000] + "\n\n[Resume text shortened for analysis.]"

    return text


def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
) -> dict:
    """Analyze resume using Gemini."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a careful resume reviewer. Evaluate the resume for ATS
readability and relevance.

Give an estimated score from 0 to 100 based only on the supplied text.
Do not claim to know the exact score used by an employer's ATS.

If a job description is supplied, evaluate alignment to it.
Otherwise, evaluate general ATS readiness.

Do not infer or score protected traits such as age, gender, race,
religion, disability, or nationality.
Do not invent facts or achievements.
Distinguish missing information from information not visible in
the extracted text.

Return ONLY valid JSON matching this schema:

{{
  "ats_score": 80,
  "score_label": "Good ATS readiness",
  "summary": "Brief overall assessment",
  "category_scores": [
    {{
      "category": "Keywords",
      "score": 80,
      "reason": "Explanation"
    }}
  ],
  "strengths": ["Strength 1"],
  "improvements": [
    {{
      "priority": "High",
      "issue": "Issue",
      "why_it_matters": "Explanation",
      "suggested_fix": "Suggested change"
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

Requirements:
- Category scores: maximum 6.
- Improvements: maximum 8.
- Strengths: maximum 5.
- Each keyword list: maximum 15.
- Formatting checks: maximum 6.
- Sample rewrites: maximum 4.
- Use concise, specific feedback.
- If something cannot be assessed from plain text, say so.
- Keep suggested rewrites faithful to the supplied resume.
- Use placeholders like [X%] only when needed and clearly label them.

JOB DESCRIPTION:
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
        raise RuntimeError(
            "The AI returned an empty response. Please try again."
        )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The AI response could not be read as JSON."
        ) from exc

    score = result.get("ats_score")

    if (
        not isinstance(score, int)
        or isinstance(score, bool)
        or not 0 <= score <= 100
    ):
        raise RuntimeError("The AI returned an invalid ATS score.")

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
    st.write(
        "1. Upload your resume.\n"
        "2. Optionally paste a job description.\n"
        "3. Get your estimated score and suggestions."
    )
    st.markdown("---")
    st.caption(
        "Supported formats: PDF, DOCX, TXT | Maximum size: 10 MB"
    )
    st.caption(
        "Resume text is sent to the Gemini API for analysis. "
        "Avoid uploading information you do not want processed "
        "by an external AI service."
    )


uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Use a text-based PDF or DOCX for best results.",
)

job_description = st.text_area(
    "Target job description (optional)",
    placeholder="Paste the job description here...",
    height=180,
)

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(
            f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB."
        )
        st.stop()

    st.success(f"Ready to analyze: {uploaded_file.name}")


analyze_clicked = st.button(
    "Analyze resume",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)


if analyze_clicked:
    api_key = get_api_key()

    if not api_key:
        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY "
            "to Streamlit Secrets or your local .env file."
        )
        st.stop()

    try:
        with st.spinner(
            "Reading your resume and reviewing ATS readiness..."
        ):
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
        # Do not display API credentials or a full traceback.
        message = str(exc)
        upper_message = message.upper()

        if any(
            term in upper_message
            for term in (
                "API KEY",
                "PERMISSION_DENIED",
                "UNAUTHENTICATED",
                "INVALID API KEY",
            )
        ):
            st.error(
                "Gemini authentication failed. Check that your API "
                "key is valid and enabled for the Gemini API."
            )

        elif (
            "RESOURCE_EXHAUSTED" in upper_message
            or "429" in message
        ):
            st.error(
                "Gemini quota or rate limit reached. Check your "
                "API quota and try again later."
            )

        elif (
            "NOT_FOUND" in upper_message
            or (
                "MODEL" in upper_message
                and "NOT FOUND" in upper_message
            )
        ):
            st.error(
                f"Gemini model '{MODEL_NAME}' was not found or is "
                "unavailable. Check the supported model name."
            )

        else:
            st.error(f"Analysis failed: {message[:400]}")


if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]

    st.markdown("---")
    st.caption(
        "Analysis for: "
        + st.session_state.get(
            "analysis_filename",
            "uploaded resume",
        )
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Estimated ATS score",
            f"{result['ats_score']}/100",
        )
        st.progress(result["ats_score"] / 100)
        st.write(
            result.get("score_label", "ATS readiness estimate")
        )

    with col2:
        st.subheader("Overall assessment")
        st.write(result.get("summary", "No summary returned."))

    categories = result.get("category_scores", [])

    if categories:
        st.subheader("Category breakdown")

        for category in categories:
            name = str(category.get("category", "Category"))
            score = category.get("score", 0)

            if (
                not isinstance(score, int)
                or isinstance(score, bool)
                or not 0 <= score <= 100
            ):
                score = 0

            st.markdown(f"**{name} — {score}/100**")
            st.progress(score / 100)
            st.caption(category.get("reason", ""))

    left, right = st.columns(2)

    with left:
        show_list(
            "Strengths",
            result.get("strengths", []),
        )
        show_list(
            "Keywords found",
            result.get("keywords_found", []),
        )

    with right:
        show_list(
            "Keywords to consider",
            result.get("keywords_to_consider", []),
        )

        st.subheader("Formatting checks")

        for check in result.get("formatting_checks", []):
            st.markdown(
                f"**{check.get('status', 'Not assessable')}: "
                f"{check.get('check', 'Check')}**"
            )
            st.write(check.get("note", ""))

    st.subheader("Recommended improvements")

    improvements = result.get("improvements", [])

    if improvements:
        for i, item in enumerate(improvements, start=1):
            with st.expander(
                f"{i}. [{item.get('priority', 'Medium')}] "
                f"{item.get('issue', 'Improvement')}"
            ):
                st.write(item.get("why_it_matters", ""))
                st.markdown("**Suggested fix**")
                st.write(item.get("suggested_fix", ""))

    else:
        st.write("No improvements were returned.")

    rewrites = result.get("sample_rewrites", [])

    if rewrites:
        st.subheader("Example bullet improvements")

        for item in rewrites:
            st.markdown(
                f"**Current issue:** "
                f"{item.get('before_or_issue', '')}"
            )
            st.markdown(
                f"**Suggested version:** "
                f"{item.get('suggested_version', '')}"
            )

    st.info(
        result.get(
            "disclaimer",
            "This is an AI-generated estimate, not a guarantee "
            "of how any employer's ATS will score your resume.",
        )
    )

    st.download_button(
        "Download analysis as JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="resume_ats_analysis.json",
        mime="application/json",
        use_container_width=True,
    )


st.markdown(
    "<p class='small-note'>Note: ATS systems vary. This score is "
    "an AI-generated estimate, not a guarantee of screening or "
    "hiring outcomes.</p>",
    unsafe_allow_html=True,
)
```
