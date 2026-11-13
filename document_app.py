
"""Document-based AI Career Match application."""

from pathlib import Path
import sys

import joblib
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from document_extractor import extract_document_text
from profile_extractor import (
    extract_candidate_profile,
    extract_job_profile
)
from career_match_engine import predict_ai_match


st.set_page_config(
    page_title="AI Career Match — Document Upload",
    page_icon="📄",
    layout="wide"
)

st.title("AI-Powered Career Match & Skill Gap Analysis")
st.write(
    "Upload your resume and enter a job description "
    "to automatically extract qualifications and calculate "
    "an AI compatibility score."
)


@st.cache_resource
def load_model():
    return joblib.load(
        PROJECT_DIR / "models" / "career_match_model.joblib"
    )


package = load_model()

left, right = st.columns(2)

with left:
    st.subheader("1. Upload Your Resume")

    resume_file = st.file_uploader(
        "Resume PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key="resume"
    )

with right:
    st.subheader("2. Provide Job Description")

    job_title = st.text_input(
        "Job Title",
        value="Data Analyst"
    )

    job_input_mode = st.radio(
        "Job description input",
        ["Paste text", "Upload document"],
        horizontal=True
    )

    if job_input_mode == "Paste text":
        job_description = st.text_area(
            "Paste the complete job description",
            height=220
        )
    else:
        job_file = st.file_uploader(
            "Job Description PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            key="job_description_upload"
        )
        job_description = None


if st.button("Extract Resume and Job Information", type="primary"):

    if resume_file is None:
        st.warning("Please upload your resume first.")

    else:
        try:
            resume_text = extract_document_text(
                resume_file.getvalue(),
                resume_file.name
            )

            if job_input_mode == "Upload document":
                if job_file is None:
                    st.warning("Please upload a job description.")
                    st.stop()

                job_description = extract_document_text(
                    job_file.getvalue(),
                    job_file.name
                )

            if not job_description or not job_description.strip():
                st.warning("Please provide a job description.")
                st.stop()

            st.session_state["candidate"] = (
                extract_candidate_profile(resume_text)
            )

            st.session_state["job"] = (
                extract_job_profile(
                    job_description,
                    job_title=job_title.strip() or "Uploaded Job"
                )
            )

            st.success("Information extracted successfully!")

        except Exception as exc:
            st.error(f"Document processing failed: {exc}")


if "candidate" in st.session_state and "job" in st.session_state:

    st.divider()
    st.subheader("3. Review Extracted Information")

    candidate = st.session_state["candidate"]
    job = st.session_state["job"]

    with st.form("review_form"):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Candidate Profile")

            candidate_skills = st.text_area(
                "Candidate Skills (comma-separated)",
                value=", ".join(candidate["skills"])
            )

            candidate_degree = st.text_input(
                "Candidate Degree",
                value=candidate["degree"]
            )

            candidate_major = st.text_input(
                "Candidate Major",
                value=candidate["major"]
            )

            candidate_experience = st.number_input(
                "Candidate Experience (years)",
                min_value=0.0,
                value=float(candidate["experience_years"])
                if candidate["experience_years"] ==
                   candidate["experience_years"] else 0.0,
                step=0.5
            )

        with col2:
            st.markdown("#### Job Requirements")

            required_skills = st.text_area(
                "Required Skills (comma-separated)",
                value=", ".join(job["required_skills"])
            )

            related_skills = st.text_area(
                "Preferred/Related Skills",
                value=", ".join(job["related_skills"])
            )

            education_requirement = st.text_input(
                "Education Requirement",
                value=job["education_requirement"]
            )

            job_experience = st.number_input(
                "Required Experience (years)",
                min_value=0.0,
                value=float(job["experience_years"])
                if job["experience_years"] ==
                   job["experience_years"] else 0.0,
                step=0.5
            )

        analyze = st.form_submit_button(
            "Calculate AI Match Score",
            type="primary"
        )

    if analyze:

        candidate = {
            "skills": [
                skill.strip()
                for skill in candidate_skills.split(",")
                if skill.strip()
            ],
            "degree": candidate_degree,
            "major": candidate_major,
            "experience_years": candidate_experience
        }

        job = {
            "job_title": job["job_title"],
            "required_skills": [
                skill.strip()
                for skill in required_skills.split(",")
                if skill.strip()
            ],
            "related_skills": [
                skill.strip()
                for skill in related_skills.split(",")
                if skill.strip()
            ],
            "education_requirement": education_requirement,
            "experience_years": job_experience
        }

        if not candidate["skills"] or not job["required_skills"]:
            st.error(
                "Candidate and required job skills cannot be empty."
            )
            st.stop()

        try:
            prediction = predict_ai_match(
                candidate,
                job,
                package
            )

            st.divider()
            st.subheader("4. AI Match Results")

            st.metric(
                "Predicted AI Match Score",
                f"{prediction['ai_match_percentage']:.2f}%"
            )

            st.progress(
                min(max(
                    prediction["ai_match_score"], 0.0
                ), 1.0)
            )

            gap = prediction["skill_gap_analysis"]

            st.write("**Skill Gap Summary**")
            st.info(gap["summary"])

            st.write("**Exact Skill Matches**")
            st.write(gap["matched_skills"])

            st.write("**Partial Skill Matches**")
            st.write(gap["partial_matches"])

            st.write("**Missing Skills**")
            st.write(gap["missing_skills"])

            st.write("**Learning Recommendations**")
            st.write(gap["learning_recommendations"])

            st.caption(
                "The score is a model-estimated compatibility "
                "measure, not a hiring probability."
            )

        except Exception as exc:
            st.error(f"AI matching failed: {exc}")
