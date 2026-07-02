"""
Feature Extractor — Extracts 40+ structured features from each candidate.
Used in Stage 3 re-ranking after semantic retrieval.
"""

import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    JOB_DESCRIPTION, AI_ML_CORE_SKILLS, CV_SPEECH_ROBOTICS_SKILLS,
    NON_TECH_SKILLS, CONSULTING_COMPANIES, CONSULTING_INDUSTRIES,
    RELEVANT_TITLES, IRRELEVANT_TITLES, PROFICIENCY_SCORES,
    EDUCATION_TIER_SCORES, DEGREE_SCORES,
)


def _clean(text: str) -> str:
    return (text or "").lower().strip()


def _skill_in_set(skill_name: str, skill_set: set) -> bool:
    """Check if a skill name matches any item in a set (case-insensitive, substring)."""
    s = _clean(skill_name)
    for target in skill_set:
        t = _clean(target)
        if s == t or s in t or t in s:
            return True
    return False


def extract_features(candidate: dict) -> dict:
    """
    Extract a comprehensive feature dictionary from a candidate profile.
    Returns a flat dict of numeric features suitable for scoring.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    features = {}

    # =====================================================================
    # TITLE FEATURES
    # =====================================================================
    current_title = _clean(profile.get("current_title", ""))
    features["title_is_relevant"] = 1.0 if any(
        _clean(t) in current_title or current_title in _clean(t)
        for t in RELEVANT_TITLES
    ) else 0.0
    features["title_is_irrelevant"] = 1.0 if any(
        _clean(t) in current_title or current_title in _clean(t)
        for t in IRRELEVANT_TITLES
    ) else 0.0

    # Check for AI/ML specific title keywords
    ai_title_keywords = ["ai", "ml", "machine learning", "data scientist",
                         "deep learning", "nlp", "research", "data engineer"]
    features["title_has_ai_keyword"] = 1.0 if any(k in current_title for k in ai_title_keywords) else 0.0

    # Career title progression — how many roles had AI/ML titles
    ai_title_count = 0
    for job in career:
        jt = _clean(job.get("title", ""))
        if any(k in jt for k in ai_title_keywords):
            ai_title_count += 1
    features["career_ai_title_count"] = ai_title_count
    features["career_ai_title_ratio"] = ai_title_count / max(len(career), 1)

    # =====================================================================
    # SKILL FEATURES
    # =====================================================================
    ai_skills = [s for s in skills if _skill_in_set(s.get("name", ""), AI_ML_CORE_SKILLS)]
    cv_speech_skills = [s for s in skills if _skill_in_set(s.get("name", ""), CV_SPEECH_ROBOTICS_SKILLS)]
    non_tech = [s for s in skills if _skill_in_set(s.get("name", ""), NON_TECH_SKILLS)]

    features["total_skills"] = len(skills)
    features["ai_ml_skill_count"] = len(ai_skills)
    features["cv_speech_skill_count"] = len(cv_speech_skills)
    features["non_tech_skill_count"] = len(non_tech)
    features["ai_skill_ratio"] = len(ai_skills) / max(len(skills), 1)

    # Required skill coverage
    required = set(JOB_DESCRIPTION.get("required_skills", []))
    preferred = set(JOB_DESCRIPTION.get("preferred_skills", []))
    req_matched = sum(1 for s in skills if _skill_in_set(s.get("name", ""), required))
    pref_matched = sum(1 for s in skills if _skill_in_set(s.get("name", ""), preferred))
    features["required_skills_matched"] = req_matched
    features["preferred_skills_matched"] = pref_matched
    features["required_skill_coverage"] = req_matched / max(len(required), 1)
    features["preferred_skill_coverage"] = pref_matched / max(len(preferred), 1)

    # Skill depth for AI skills
    if ai_skills:
        prof_scores = [PROFICIENCY_SCORES.get(s.get("proficiency", "beginner"), 0.2) for s in ai_skills]
        durations = [s.get("duration_months", 0) for s in ai_skills]
        endorsements = [s.get("endorsements", 0) for s in ai_skills]
        features["avg_ai_proficiency"] = sum(prof_scores) / len(prof_scores)
        features["max_ai_proficiency"] = max(prof_scores)
        features["avg_ai_duration_months"] = sum(durations) / len(durations)
        features["avg_ai_endorsements"] = sum(endorsements) / len(endorsements)
        features["total_ai_endorsements"] = sum(endorsements)
    else:
        features["avg_ai_proficiency"] = 0.0
        features["max_ai_proficiency"] = 0.0
        features["avg_ai_duration_months"] = 0.0
        features["avg_ai_endorsements"] = 0.0
        features["total_ai_endorsements"] = 0

    # Skill assessment scores
    if assessments:
        assessment_vals = list(assessments.values())
        features["has_assessments"] = 1.0
        features["avg_assessment_score"] = sum(assessment_vals) / len(assessment_vals)
        features["max_assessment_score"] = max(assessment_vals)
        features["num_assessments"] = len(assessment_vals)
    else:
        features["has_assessments"] = 0.0
        features["avg_assessment_score"] = 0.0
        features["max_assessment_score"] = 0.0
        features["num_assessments"] = 0

    # NLP/IR specific skills (what the JD actually wants)
    nlp_ir_keywords = ["nlp", "natural language", "information retrieval", "search",
                       "ranking", "recommendation", "embeddings", "vector",
                       "faiss", "milvus", "elasticsearch", "bm25", "retrieval",
                       "sentence transformer", "rag"]
    nlp_ir_count = sum(1 for s in skills if any(k in _clean(s.get("name", "")) for k in nlp_ir_keywords))
    features["nlp_ir_skill_count"] = nlp_ir_count

    # =====================================================================
    # EXPERIENCE FEATURES
    # =====================================================================
    years_exp = profile.get("years_of_experience", 0)
    features["years_experience"] = years_exp

    exp_range = JOB_DESCRIPTION.get("experience_range", {"min": 5, "max": 9})
    ideal = JOB_DESCRIPTION.get("ideal_experience", {"min": 6, "max": 8})
    features["exp_in_ideal_range"] = 1.0 if ideal["min"] <= years_exp <= ideal["max"] else 0.0
    features["exp_in_acceptable_range"] = 1.0 if exp_range["min"] <= years_exp <= exp_range["max"] else 0.0

    # Distance from ideal range
    if years_exp < ideal["min"]:
        features["exp_distance"] = ideal["min"] - years_exp
    elif years_exp > ideal["max"]:
        features["exp_distance"] = years_exp - ideal["max"]
    else:
        features["exp_distance"] = 0.0

    # Career stability
    features["num_roles"] = len(career)
    if career:
        durations = [j.get("duration_months", 0) for j in career]
        features["avg_tenure_months"] = sum(durations) / len(durations)
        features["min_tenure_months"] = min(durations)
        features["max_tenure_months"] = max(durations)
        features["short_stint_count"] = sum(1 for d in durations if d < 18)
        features["is_currently_employed"] = 1.0 if any(j.get("is_current", False) for j in career) else 0.0
    else:
        features["avg_tenure_months"] = 0
        features["min_tenure_months"] = 0
        features["max_tenure_months"] = 0
        features["short_stint_count"] = 0
        features["is_currently_employed"] = 0.0

    # Title-chaser detection (JD says: switching every 1.5 years = bad)
    if len(career) >= 3 and years_exp > 0:
        avg_years_per_role = years_exp / len(career)
        features["is_title_chaser"] = 1.0 if avg_years_per_role < 1.5 else 0.0
    else:
        features["is_title_chaser"] = 0.0

    # =====================================================================
    # CONSULTING-ONLY CHECK (JD hard disqualifier)
    # =====================================================================
    consulting_roles = 0
    product_roles = 0
    for job in career:
        company_lower = _clean(job.get("company", ""))
        industry_lower = _clean(job.get("industry", ""))
        if any(c in company_lower for c in CONSULTING_COMPANIES) or \
           any(c in industry_lower for c in CONSULTING_INDUSTRIES):
            consulting_roles += 1
        else:
            product_roles += 1

    current_company_lower = _clean(profile.get("current_company", ""))
    features["is_consulting_only"] = 1.0 if (
        consulting_roles > 0 and product_roles == 0
    ) else 0.0
    features["consulting_role_ratio"] = consulting_roles / max(len(career), 1)
    features["has_product_experience"] = 1.0 if product_roles > 0 else 0.0
    features["current_is_consulting"] = 1.0 if any(
        c in current_company_lower for c in CONSULTING_COMPANIES
    ) else 0.0

    # =====================================================================
    # CAREER DESCRIPTION ANALYSIS
    # =====================================================================
    all_descriptions = " ".join(_clean(j.get("description", "")) for j in career)

    # Check if descriptions mention AI/ML work
    ml_desc_keywords = [
        "machine learning", "deep learning", "neural network", "model training",
        "model deployment", "nlp", "natural language", "embedding",
        "retrieval", "ranking", "search", "recommendation",
        "classification", "regression", "prediction", "inference",
        "transformer", "bert", "gpt", "llm", "fine-tun",
        "evaluation", "a/b test", "feature engineer",
        "pipeline", "data pipeline", "ml pipeline",
        "vector", "faiss", "elasticsearch",
        "tensorflow", "pytorch", "scikit",
    ]
    non_ml_desc_keywords = [
        "accounting", "financial reporting", "tax filing", "audit",
        "brand design", "packaging design", "creative direction",
        "customer support", "support agent", "ticket",
        "seo strategy", "content writing", "editorial",
        "mechanical engineering", "cad", "solidworks", "ansys",
        "civil engineering", "construction", "structural",
        "supply chain", "warehouse", "fulfillment",
        "sales cycle", "quota", "prospecting", "negotiation",
    ]

    ml_keyword_count = sum(1 for kw in ml_desc_keywords if kw in all_descriptions)
    non_ml_keyword_count = sum(1 for kw in non_ml_desc_keywords if kw in all_descriptions)
    features["desc_ml_keyword_count"] = ml_keyword_count
    features["desc_non_ml_keyword_count"] = non_ml_keyword_count
    features["desc_ml_ratio"] = ml_keyword_count / max(ml_keyword_count + non_ml_keyword_count, 1)

    # Title-description consistency (honeypot check)
    features["title_desc_mismatch"] = 1.0 if (
        features["title_has_ai_keyword"] == 1.0 and ml_keyword_count < 2 and non_ml_keyword_count >= 2
    ) else 0.0

    # =====================================================================
    # LOCATION FEATURES
    # =====================================================================
    location = _clean(profile.get("location", ""))
    country = _clean(profile.get("country", ""))
    features["is_india"] = 1.0 if country == "india" else 0.0

    preferred_locs = JOB_DESCRIPTION.get("preferred_locations", [])
    features["is_preferred_location"] = 1.0 if any(
        _clean(loc) in location for loc in preferred_locs
    ) else 0.0

    willing_relocate = signals.get("willing_to_relocate", False)
    features["willing_to_relocate"] = 1.0 if willing_relocate else 0.0

    # Location score combining country + city + relocate
    if features["is_preferred_location"] == 1.0:
        features["location_score"] = 1.0
    elif features["is_india"] == 1.0 and features["willing_to_relocate"] == 1.0:
        features["location_score"] = 0.7
    elif features["is_india"] == 1.0:
        features["location_score"] = 0.5
    elif features["willing_to_relocate"] == 1.0:
        features["location_score"] = 0.3
    else:
        features["location_score"] = 0.15

    # =====================================================================
    # BEHAVIORAL SIGNAL FEATURES
    # =====================================================================
    features["profile_completeness"] = signals.get("profile_completeness_score", 0) / 100.0
    features["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    features["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0)
    features["avg_response_time_hours"] = signals.get("avg_response_time_hours", 200)
    features["interview_completion_rate"] = signals.get("interview_completion_rate", 0)
    features["offer_acceptance_rate"] = max(0, signals.get("offer_acceptance_rate", 0))
    features["github_activity"] = max(0, signals.get("github_activity_score", 0))
    features["has_github"] = 1.0 if signals.get("github_activity_score", -1) >= 0 else 0.0
    features["connection_count"] = signals.get("connection_count", 0)
    features["endorsements_received"] = signals.get("endorsements_received", 0)
    features["profile_views_30d"] = signals.get("profile_views_received_30d", 0)
    features["saved_by_recruiters_30d"] = signals.get("saved_by_recruiters_30d", 0)
    features["search_appearances_30d"] = signals.get("search_appearance_30d", 0)

    # Notice period (JD says < 30 days preferred)
    notice = signals.get("notice_period_days", 90)
    features["notice_period_days"] = notice
    features["notice_under_30"] = 1.0 if notice <= 30 else 0.0
    features["notice_under_60"] = 1.0 if notice <= 60 else 0.0

    # Recency (days since last active)
    last_active = signals.get("last_active_date", "")
    try:
        if last_active:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            ref_dt = datetime(2026, 6, 1)
            days_inactive = (ref_dt - last_dt).days
            features["days_since_active"] = max(0, days_inactive)
        else:
            features["days_since_active"] = 999
    except (ValueError, TypeError):
        features["days_since_active"] = 999

    features["recently_active"] = 1.0 if features["days_since_active"] <= 60 else 0.0

    # Verification signals
    verified_count = 0
    if signals.get("verified_email", False): verified_count += 1
    if signals.get("verified_phone", False): verified_count += 1
    if signals.get("linkedin_connected", False): verified_count += 1
    features["verification_count"] = verified_count
    features["verification_score"] = verified_count / 3.0

    # Work mode
    pref_mode = signals.get("preferred_work_mode", "")
    features["prefers_hybrid_or_flexible"] = 1.0 if pref_mode in ("hybrid", "flexible") else 0.0

    # =====================================================================
    # EDUCATION FEATURES
    # =====================================================================
    if education:
        best_tier = max(EDUCATION_TIER_SCORES.get(e.get("tier", "unknown"), 0.3) for e in education)
        best_degree = max(DEGREE_SCORES.get(e.get("degree", ""), 0.4) for e in education)

        # Field relevance
        relevant_fields = {"computer science", "artificial intelligence", "machine learning",
                          "data science", "mathematics", "statistics", "computer engineering",
                          "information technology", "electronics", "electrical engineering"}
        has_relevant_field = any(
            _clean(e.get("field_of_study", "")) in relevant_fields or
            any(rf in _clean(e.get("field_of_study", "")) for rf in relevant_fields)
            for e in education
        )
        features["edu_tier_score"] = best_tier
        features["edu_degree_score"] = best_degree
        features["edu_field_relevant"] = 1.0 if has_relevant_field else 0.0
    else:
        features["edu_tier_score"] = 0.3
        features["edu_degree_score"] = 0.4
        features["edu_field_relevant"] = 0.0

    # Certifications
    features["cert_count"] = len(certs)
    ai_cert_keywords = ["aws", "google", "azure", "tensorflow", "pytorch",
                        "machine learning", "data science", "kubernetes", "docker"]
    relevant_certs = sum(1 for c in certs if any(
        k in _clean(c.get("name", "")) or k in _clean(c.get("issuer", ""))
        for k in ai_cert_keywords
    ))
    features["relevant_cert_count"] = relevant_certs

    return features
