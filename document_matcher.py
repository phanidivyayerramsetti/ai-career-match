
"""End-to-end document matching for the Group 13 project."""

from document_extractor import extract_document_text
from profile_extractor import (
    extract_candidate_profile,
    extract_job_profile
)
from career_match_engine import predict_ai_match


def analyze_resume_and_job_text(
    resume_text,
    job_description,
    package,
    job_title="Uploaded Job"
):
    """Extract profiles and calculate an AI compatibility score."""

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    candidate = extract_candidate_profile(resume_text)

    job = extract_job_profile(
        job_description,
        job_title=job_title
    )

    if not candidate["skills"]:
        raise ValueError(
            "No recognized candidate skills were found. "
            "Please review the resume or enter skills manually."
        )

    if not job["required_skills"]:
        raise ValueError(
            "No recognized job skills were found. "
            "Please review the job description."
        )

    prediction = predict_ai_match(
        candidate,
        job,
        package
    )

    return {
        "candidate_profile": candidate,
        "job_profile": job,
        "prediction": prediction
    }


def analyze_resume_file_and_job_text(
    resume_bytes,
    resume_filename,
    job_description,
    package,
    job_title="Uploaded Job"
):
    """Analyze a PDF/DOCX/TXT resume against job-description text."""

    resume_text = extract_document_text(
        resume_bytes,
        resume_filename
    )

    return analyze_resume_and_job_text(
        resume_text,
        job_description,
        package,
        job_title
    )


def analyze_resume_and_job_files(
    resume_bytes,
    resume_filename,
    job_bytes,
    job_filename,
    package,
    job_title="Uploaded Job"
):
    """Analyze uploaded resume and job-description documents."""

    resume_text = extract_document_text(
        resume_bytes,
        resume_filename
    )

    job_text = extract_document_text(
        job_bytes,
        job_filename
    )

    return analyze_resume_and_job_text(
        resume_text,
        job_text,
        package,
        job_title
    )
