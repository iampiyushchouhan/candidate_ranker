"""
Job Description Parser: Extract structured requirements from DOCX or text.
"""

import os
import re
from config import DEFAULT_JOB_DESCRIPTION


def parse_job_description_from_docx(docx_path: str) -> dict:
    """Parse a job description DOCX file and extract structured requirements."""
    try:
        import docx
        doc = docx.Document(docx_path)
        full_text = "\n".join([p.text for p in doc.paragraphs])
        return parse_job_description_text(full_text)
    except Exception as e:
        print(f"Warning: Could not parse DOCX ({e}). Using default JD.")
        return DEFAULT_JOB_DESCRIPTION


def parse_job_description_text(text: str) -> dict:
    """Parse raw text of a job description into structured requirements."""
    jd = DEFAULT_JOB_DESCRIPTION.copy()
    jd["raw_text"] = text

    # Try to extract title from text
    title_patterns = [
        r'(?:job\s*title|position|role)\s*[:\-]\s*(.+)',
        r'(?:hiring|looking\s+for)\s+(?:a\s+)?(.+?)(?:\.|,|\n)',
    ]
    text_lower = text.lower()
    for pattern in title_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted_title = match.group(1).strip().title()
            if len(extracted_title) < 60:
                jd["title"] = extracted_title
                break

    # Extract skills mentioned in text
    from config import AI_ML_CORE_SKILLS
    found_skills = []
    for skill in AI_ML_CORE_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    if found_skills:
        jd["required_skills"] = list(set(jd.get("required_skills", []) + found_skills))

    # Extract experience requirements
    exp_match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)\s*years?\s*(?:of\s*)?experience', text_lower)
    if exp_match:
        jd["experience_range"] = {
            "min": int(exp_match.group(1)),
            "max": int(exp_match.group(2))
        }

    return jd


def get_job_description(docx_path: str = None, text: str = None) -> dict:
    """
    Get structured job description from either a DOCX file path or raw text.
    Falls back to defaults if neither is provided.
    """
    if docx_path and os.path.exists(docx_path):
        return parse_job_description_from_docx(docx_path)
    elif text:
        return parse_job_description_text(text)
    else:
        return DEFAULT_JOB_DESCRIPTION
