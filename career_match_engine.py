
import ast
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MISSING_VALUES = {
    "", "nan", "none", "null", "n/a", "na", "not available"
}

SKILL_PRIORITY = {
    "python": "High",
    "sql": "High",
    "machine learning": "High",
    "aws": "High",
    "java": "High",
    "kubernetes": "High",
    "power bi": "Medium",
    "tableau": "Medium",
    "excel": "Medium",
    "docker": "Medium",
    "git": "Medium",
    "communication": "Medium"
}

SKILL_RELATIONSHIPS = {
    "python": {"pandas", "numpy", "scikit-learn"},
    "machine learning": {"scikit-learn", "deep learning"},
    "sql": {"mysql", "postgresql", "sql server"},
    "aws": {"cloud computing", "amazon ec2"},
    "power bi": {"tableau", "data visualization"},
    "excel": {"google sheets", "spreadsheet analysis"},
    "java": {"spring boot", "spring framework"},
    "javascript": {"typescript", "react", "node.js"},
    "docker": {"kubernetes", "containerization"},
    "kubernetes": {"docker", "containerization"},
    "tensorflow": {"keras", "pytorch"}
}

LEARNING_RESOURCES = {
    "python": (
        "Python programming fundamentals",
        "https://docs.python.org/3/tutorial/"
    ),
    "sql": (
        "SQL queries, joins, and database fundamentals",
        "https://www.postgresql.org/docs/"
    ),
    "power bi": (
        "Power BI dashboards and data modeling",
        "https://learn.microsoft.com/power-bi/"
    ),
    "kubernetes": (
        "Kubernetes concepts, pods, and deployments",
        "https://kubernetes.io/docs/tutorials/"
    ),
    "docker": (
        "Docker containers and images",
        "https://docs.docker.com/get-started/"
    ),
    "machine learning": (
        "Supervised learning and model evaluation",
        "https://scikit-learn.org/stable/user_guide.html"
    ),
    "aws": (
        "AWS cloud fundamentals",
        "https://skillbuilder.aws/"
    ),
    "excel": (
        "Excel formulas, pivot tables, and analysis",
        "https://support.microsoft.com/excel"
    ),
    "tableau": (
        "Tableau visual analytics",
        "https://www.tableau.com/learn"
    ),
    "communication": (
        "Professional communication and collaboration",
        None
    )
}


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        value = " ".join(
            str(item) for item in value if item is not None
        )

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower()

    if text in MISSING_VALUES:
        return ""

    return re.sub(r"\s+", " ", text)


def parse_list_field(value):
    if isinstance(value, (list, tuple, set)):
        items = list(value)

    elif value is None:
        return []

    else:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if not text or text.lower() in MISSING_VALUES:
            return []

        if text.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple, set)):
                    items = list(parsed)
                else:
                    items = [parsed]

            except (ValueError, SyntaxError):
                items = re.split(r"[\n,;]+", text)

        else:
            items = re.split(r"[\n,;]+", text)

    cleaned_items = []

    for item in items:
        normalized = clean_text(item)

        if normalized and normalized not in cleaned_items:
            cleaned_items.append(normalized)

    return cleaned_items


def normalize_skill_list(value):
    aliases = {
        "py": "python",
        "python3": "python",
        "powerbi": "power bi",
        "ms excel": "excel",
        "microsoft excel": "excel",
        "amazon web services": "aws",
        "sklearn": "scikit-learn"
    }

    normalized_skills = []

    for skill in parse_list_field(value):
        skill = aliases.get(skill, skill)

        if skill not in normalized_skills:
            normalized_skills.append(skill)

    return sorted(normalized_skills)


def normalize_education(value):
    return " ".join(parse_list_field(value))


def calculate_skill_match(candidate_skills, job_skills):
    candidate_set = set(
        normalize_skill_list(candidate_skills)
    )

    job_set = set(
        normalize_skill_list(job_skills)
    )

    if not job_set:
        return np.nan

    return len(
        candidate_set.intersection(job_set)
    ) / len(job_set)


def calculate_tfidf_similarity(
    candidate_texts,
    job_texts,
    vectorizer
):
    candidate_texts = pd.Series(
        candidate_texts
    ).fillna("")

    job_texts = pd.Series(
        job_texts
    ).fillna("")

    candidate_vectors = vectorizer.transform(
        candidate_texts
    )

    job_vectors = vectorizer.transform(
        job_texts
    )

    similarities = np.asarray(
        candidate_vectors.multiply(
            job_vectors
        ).sum(axis=1)
    ).ravel()

    missing_jobs = job_texts.str.strip().eq(
        ""
    ).to_numpy()

    similarities[missing_jobs] = np.nan

    return similarities


def extract_degree_level(text):
    text = clean_text(text)

    if not text:
        return np.nan

    if re.search(
        r"\b(ph\.?d|doctorate|doctoral)\b",
        text
    ):
        return 5

    if re.search(
        r"\b(master|masters|m\.?sc|m\.?tech|m\.?a|mba|mca|m\.?e)\b",
        text
    ):
        return 4

    if re.search(
        r"\b(bachelor|bachelors|b\.?sc|b\.?tech|b\.?a|b\.?e|bba|bca|b\.?com|honours|honors)\b",
        text
    ):
        return 3

    if "diploma" in text:
        return 2

    if "certificate" in text or "certification" in text:
        return 1

    return np.nan


def calculate_degree_compatibility(
    candidate_degree,
    required_degree
):
    candidate_level = extract_degree_level(
        candidate_degree
    )

    required_level = extract_degree_level(
        required_degree
    )

    if pd.isna(candidate_level) or pd.isna(required_level):
        return np.nan

    if candidate_level >= required_level:
        return 1.0

    if candidate_level == required_level - 1:
        return 0.5

    return 0.0


def calculate_experience_match(
    candidate_years,
    required_years
):
    if pd.isna(candidate_years) or pd.isna(required_years):
        return np.nan

    candidate_years = float(candidate_years)
    required_years = float(required_years)

    if candidate_years < 0 or required_years < 0:
        return np.nan

    if required_years == 0:
        return 1.0

    return min(
        1.0,
        candidate_years / required_years
    )


def load_model_package(model_path):
    return joblib.load(model_path)


def extract_matching_features(
    candidate,
    job,
    package
):
    candidate_skills = normalize_skill_list(
        candidate.get("skills", [])
    )

    required_skills = normalize_skill_list(
        job.get("required_skills", [])
    )

    related_skills = normalize_skill_list(
        job.get("related_skills", [])
    )

    candidate_text = " ".join(candidate_skills)
    required_text = " ".join(required_skills)
    related_text = " ".join(related_skills)

    candidate_major = normalize_education(
        candidate.get("major", "")
    )

    job_education = clean_text(
        job.get("education_requirement", "")
    )

    skill_vectorizer = package[
        "skill_vectorizer"
    ]

    major_vectorizer = package[
        "major_vectorizer"
    ]

    if not candidate_major or not job_education:
        major_similarity = np.nan

    else:
        major_similarity = calculate_tfidf_similarity(
            [candidate_major],
            [job_education],
            major_vectorizer
        )[0]

    features = {
        "required_skill_match": calculate_skill_match(
            candidate_skills,
            required_skills
        ),

        "related_skill_match": calculate_skill_match(
            candidate_skills,
            related_skills
        ),

        "required_skill_tfidf": calculate_tfidf_similarity(
            [candidate_text],
            [required_text],
            skill_vectorizer
        )[0],

        "related_skill_tfidf": calculate_tfidf_similarity(
            [candidate_text],
            [related_text],
            skill_vectorizer
        )[0],

        "degree_compatibility": calculate_degree_compatibility(
            candidate.get("degree", ""),
            job_education
        ),

        "major_tfidf_similarity": major_similarity,

        "experience_match": calculate_experience_match(
            candidate.get("experience_years", np.nan),
            job.get("experience_years", np.nan)
        )
    }

    return pd.DataFrame(
        [features],
        columns=package["feature_columns"]
    )


def analyze_candidate_skills(
    candidate_skills,
    required_skills
):
    candidate_set = set(
        normalize_skill_list(candidate_skills)
    )

    required_set = set(
        normalize_skill_list(required_skills)
    )

    matched = sorted(
        candidate_set & required_set
    )

    missing = sorted(
        required_set - candidate_set
    )

    percentage = (
        np.nan
        if not required_set
        else len(matched) / len(required_set) * 100
    )

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": percentage
    }


def identify_partial_matches(
    candidate_skills,
    required_skills
):
    candidate_set = set(
        normalize_skill_list(candidate_skills)
    )

    required_set = set(
        normalize_skill_list(required_skills)
    )

    matches = []

    for required_skill in sorted(required_set):
        if required_skill in candidate_set:
            continue

        related = SKILL_RELATIONSHIPS.get(
            required_skill,
            set()
        )

        found = sorted(
            candidate_set.intersection(related)
        )

        if found:
            matches.append({
                "required_skill": required_skill,
                "candidate_related_skills": found
            })

    return matches


def generate_skill_gap_explanation(
    candidate_skills,
    required_skills
):
    analysis = analyze_candidate_skills(
        candidate_skills,
        required_skills
    )

    partial = identify_partial_matches(
        candidate_skills,
        required_skills
    )

    partial_lookup = {
        item["required_skill"]:
        item["candidate_related_skills"]
        for item in partial
    }

    gaps = []

    for skill in analysis["missing_skills"]:
        priority = SKILL_PRIORITY.get(
            skill,
            "Low"
        )

        related = partial_lookup.get(
            skill,
            []
        )

        gaps.append({
            "skill": skill,
            "priority": priority,
            "match_type": (
                "Partial" if related else "Missing"
            ),
            "related_candidate_skills": related
        })

    order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }

    gaps.sort(
        key=lambda item: (
            order[item["priority"]],
            item["skill"]
        )
    )

    recommendations = []

    for gap in gaps:
        topic, resource = LEARNING_RESOURCES.get(
            gap["skill"],
            (
                f"Learn and practice {gap['skill']}",
                None
            )
        )

        recommendations.append({
            "skill": gap["skill"],
            "priority": gap["priority"],
            "match_type": gap["match_type"],
            "learning_topic": topic,
            "resource": resource
        })

    exact_count = len(
        analysis["matched_skills"]
    )

    partial_count = sum(
        g["match_type"] == "Partial"
        for g in gaps
    )

    missing_count = sum(
        g["match_type"] == "Missing"
        for g in gaps
    )

    percentage = analysis[
        "match_percentage"
    ]

    if pd.isna(percentage):
        summary = (
            "Required job skills are unavailable, "
            "so exact skill compatibility cannot "
            "be calculated."
        )

    else:
        summary = (
            f"The candidate exactly matches "
            f"{exact_count} required skills "
            f"({percentage:.1f}%). "
            f"There are {partial_count} "
            f"related-skill matches and "
            f"{missing_count} skills without "
            f"a known match."
        )

    return {
        "summary": summary,

        "matched_skills": analysis[
            "matched_skills"
        ],

        "partial_matches": [
            gap for gap in gaps
            if gap["match_type"] == "Partial"
        ],

        "missing_skills": [
            gap for gap in gaps
            if gap["match_type"] == "Missing"
        ],

        "learning_recommendations": recommendations
    }


def predict_ai_match(candidate, job, package):
    raw_features = extract_matching_features(
        candidate,
        job,
        package
    )

    columns = package[
        "feature_columns"
    ]

    transformed = package[
        "imputer"
    ].transform(
        raw_features[columns]
    )

    processed = pd.DataFrame(
        transformed,
        columns=columns
    )

    score = float(
        np.clip(
            package["model"].predict(
                processed
            )[0],
            0.0,
            1.0
        )
    )

    return {
        "job_title": job.get(
            "job_title",
            "Unknown Job"
        ),

        "ai_match_score": round(
            score,
            4
        ),

        "ai_match_percentage": round(
            score * 100,
            2
        ),

        "features": raw_features.iloc[
            0
        ].to_dict(),

        "skill_gap_analysis": generate_skill_gap_explanation(
            candidate.get("skills", []),
            job.get("required_skills", [])
        )
    }


def rank_jobs_by_readiness(
    candidate,
    jobs,
    package
):
    rows = []

    for job in jobs:
        result = predict_ai_match(
            candidate,
            job,
            package
        )

        analysis = analyze_candidate_skills(
            candidate.get("skills", []),
            job.get("required_skills", [])
        )

        coverage = analysis[
            "match_percentage"
        ]

        exact_count = len(
            analysis["matched_skills"]
        )

        required_count = len(
            normalize_skill_list(
                job.get("required_skills", [])
            )
        )

        partial_count = len(
            identify_partial_matches(
                candidate.get("skills", []),
                job.get("required_skills", [])
            )
        )

        if required_count == 0:
            readiness = (
                "Insufficient job skill information"
            )

        elif exact_count == 0:
            readiness = "Major skill gaps"

        elif exact_count / required_count >= 0.75:
            readiness = "Strong skill coverage"

        elif exact_count / required_count >= 0.50:
            readiness = "Moderate skill coverage"

        else:
            readiness = "Limited skill coverage"

        rows.append({
            "Job Title": result["job_title"],

            "AI Match (%)": result[
                "ai_match_percentage"
            ],

            "Exact Skill Coverage (%)": coverage,

            "Exact Matches": exact_count,

            "Partial Matches": partial_count,

            "Readiness": readiness
        })

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    # Corrected: rank by AI Match Score, highest first
    ranking = ranking.sort_values(
        by="AI Match (%)",
        ascending=False,
        kind="stable",
        na_position="last"
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "Recommendation Rank",
        range(1, len(ranking) + 1)
    )

    return ranking
