
"""Rule-based profile extraction for Group 13 career matching."""

import re
import numpy as np

from career_match_engine import normalize_skill_list


SKILL_CATALOG = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle",
    "Excel", "Power BI", "Tableau", "Looker",
    "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "TensorFlow", "PyTorch", "Scikit-learn",
    "Pandas", "NumPy", "Data Analysis", "Data Visualization",
    "Statistics", "Data Modeling", "ETL", "Apache Spark",
    "Hadoop", "AWS", "Azure", "Google Cloud", "GCP",
    "Docker", "Kubernetes", "Linux", "Git", "GitHub",
    "REST API", "FastAPI", "Flask", "Django",
    "React", "Angular", "Node.js", "HTML", "CSS",
    "Cybersecurity", "Networking", "Cloud Computing",
    "Communication", "Project Management",
    "Agile", "Scrum", "Business Analysis", "Reporting",
    "Salesforce", "ServiceNow", "PowerShell"
]

SKILL_ALIASES = {
    "power bi": ["powerbi", "microsoft power bi"],
    "python": ["python3"],
    "aws": ["amazon web services"],
    "scikit-learn": ["sklearn"],
    "excel": ["microsoft excel", "ms excel"],
    "javascript": ["java script"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "google cloud": ["google cloud platform"],
    "rest api": ["restful api", "rest apis"]
}


def normalize_document_text(text):
    """Normalize whitespace and escaped newline sequences."""
    text = str(text or "")
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    return re.sub(r"[ \t]+", " ", text)


def extract_skills(text):
    """Find known skills using boundary-aware matching."""
    text = normalize_document_text(text).lower()
    found = []

    for skill in SKILL_CATALOG:
        canonical = skill.lower()
        variants = [canonical] + SKILL_ALIASES.get(canonical, [])

        for variant in variants:
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(variant)
                + r"(?![a-z0-9])"
            )
            if re.search(pattern, text):
                found.append(canonical)
                break

    return normalize_skill_list(found)



DEGREE_PATTERNS = [
    (r"\b(ph\.?d|doctorate|doctoral)\b", "PhD"),
    (r"\b(master(?:'s|s)?|m\.?sc|m\.?tech|mba|mca)\b",
     "Master of Science"),
    (r"\b(bachelor(?:'s|s)?|b\.?sc|b\.?tech|b\.?e)\b",
     "Bachelor of Science"),
    (r"\bdiploma\b", "Diploma"),
    (r"\bcertificate\b", "Certificate")
]

MAJOR_CATALOG = [
    "Computer Science", "Information Technology",
    "Information Systems", "Data Science",
    "Computer Engineering", "Software Engineering",
    "Electrical Engineering", "Business Analytics",
    "Mathematics", "Statistics", "Cybersecurity"
]


def extract_degree(text):
    """Return highest recognized degree mentioned in text."""
    text = normalize_document_text(text).lower()

    for pattern, degree in DEGREE_PATTERNS:
        if re.search(pattern, text):
            return degree

    return ""


def extract_major(text):
    """Identify a known academic field."""
    text = normalize_document_text(text).lower()

    for major in MAJOR_CATALOG:
        if re.search(
            r"(?<![a-z0-9])" + re.escape(major.lower()) + r"(?![a-z0-9])",
            text
        ):
            return major

    return ""





def extract_years_experience(text):
    """Estimate employment experience from resume dates or JD requirements."""
    import re
    import numpy as np
    from datetime import datetime

    text = normalize_document_text(text)

    # Recognize explicit job-description requirements.
    requirement_patterns = [
        r"\bexperience\s+required\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
        r"\brequired\s+experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
        r"\bat\s+least\s+(\d+(?:\.\d+)?)\s*years?\b",
        r"\bminimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*years?\b",
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:professional\s+)?experience\b"
    ]

    for pattern in requirement_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    month_pattern = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)"
    )

    date_pattern = (
        rf"({month_pattern})\.?\s+(\d{{4}})"
        rf"\s*[-–—]\s*"
        rf"({month_pattern})\.?\s+(\d{{4}})|"
        rf"({month_pattern})\.?\s+(\d{{4}})"
        rf"\s*[-–—]\s*(present|current|ongoing)"
    )

    employment_terms = [
        "engineer", "developer", "technician", "analyst",
        "intern", "consultant", "specialist", "administrator",
        "research assistant", "associate"
    ]

    education_terms = [
        "bachelor", "master", "university", "college",
        "intermediate education", "high school"
    ]

    lines = text.splitlines()
    intervals = []

    today = datetime.today()

    for i, line in enumerate(lines):
        match = re.search(date_pattern, line, re.IGNORECASE)

        if not match:
            continue

        # Check the date line AND nearby preceding lines.

        # Check the date line first, then preceding lines for job titles.
        current_line = line.lower()
        context = " ".join(lines[max(0, i - 2):i + 1]).lower()

        if any(term in current_line for term in employment_terms):
            # The date line itself identifies an employment role.
            pass
        elif any(term in context for term in education_terms):
            # Exclude education dates.
            continue
        elif not any(term in context for term in employment_terms):
            continue

        groups = match.groups()

        if groups[0] is not None:
            start_month, start_year = groups[0], groups[1]
            end_month, end_year = groups[2], groups[3]
        else:
            start_month, start_year = groups[4], groups[5]
            end_month = today.strftime("%b")
            end_year = str(today.year)

        def month_number(value):
            return datetime.strptime(value[:3].title(), "%b").month

        start = int(start_year) * 12 + month_number(start_month)
        end = int(end_year) * 12 + month_number(end_month)

        if start <= end:
            intervals.append((start, end))

    if not intervals:
        return np.nan

    # Merge overlapping employment periods.
    intervals.sort()
    merged = []

    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], end)
            )
        else:
            merged.append((start, end))

    total_months = sum(
        end - start + 1 for start, end in merged
    )

    return round(total_months / 12, 2)






def extract_candidate_profile(resume_text):
    """Extract structured candidate fields from resume text."""
    return {
        "skills": extract_skills(resume_text),
        "degree": extract_degree(resume_text),
        "major": extract_major(resume_text),
        "experience_years": extract_years_experience(resume_text)
    }


def extract_job_profile(job_text, job_title="Uploaded Job"):
    """Extract structured job fields from a job description."""
    text = normalize_document_text(job_text)

    # Use skills mentioned in explicit requirement lines where available.
    required_sections = []
    related_sections = []

    for line in text.splitlines():
        lower_line = line.lower().strip()

        if any(term in lower_line for term in [
            "required skills", "requirements:", "must have",
            "minimum qualifications"
        ]):
            required_sections.append(line)

        elif any(term in lower_line for term in [
            "preferred skills", "nice to have",
            "desired skills", "preferred qualifications"
        ]):
            related_sections.append(line)

    required_skills = extract_skills(
        "\n".join(required_sections)
    )

    related_skills = extract_skills(
        "\n".join(related_sections)
    )

    # Fallback when job descriptions lack labeled skill sections.
    if not required_skills:
        required_skills = extract_skills(text)

    # Do not list a required skill again as preferred.
    related_skills = sorted(
        set(related_skills) - set(required_skills)
    )

    return {
        "job_title": job_title,
        "required_skills": required_skills,
        "related_skills": related_skills,
        "education_requirement": extract_degree(text) + (
            " in " + extract_major(text) if extract_major(text) else ""
        ),
        "experience_years": extract_years_experience(text)
    }
