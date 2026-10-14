
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
    """Read the Gemini key from Streamlit Cloud Secrets or local environment."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return str(key).strip()
    except (KeyError, FileNotFoundError, Exception):
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()


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
