"""
Honeypot Detector — Identifies candidates with subtly impossible profiles.
JD says: ~80 honeypots in dataset. >10% in top 100 = disqualified.

Honeypot signals:
- "8 years of experience at a company founded 3 years ago"
- "expert proficiency in 10 skills with 0 years used"
- Title-description mismatch
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_honeypot(candidate: dict, features: dict = None) -> dict:
    """
    Analyze a candidate for honeypot signals.
    Returns dict with: is_honeypot, confidence, flags, penalty
    """
    flags = []
    penalty = 1.0

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    # =========================================================================
    # CHECK 1: Expert proficiency with zero usage duration
    # "expert proficiency in 10 skills with 0 years used"
    # =========================================================================
    expert_zero_duration = 0
    for skill in skills:
        prof = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)
        if prof in ("expert", "advanced") and duration <= 3 and endorsements <= 2:
            expert_zero_duration += 1

    if expert_zero_duration >= 5:
        flags.append(f"{expert_zero_duration} skills claimed as expert/advanced with near-zero usage")
        penalty *= 0.1
    elif expert_zero_duration >= 3:
        flags.append(f"{expert_zero_duration} skills with inflated proficiency claims")
        penalty *= 0.4

    # =========================================================================
    # CHECK 2: Experience duration impossibility
    # "8 years at a company founded 3 years ago"
    # =========================================================================
    for job in career:
        duration_months = job.get("duration_months", 0)
        # If someone claims 8+ years (96 months) at any single role, suspicious
        if duration_months > 120:
            flags.append(f"Impossibly long tenure: {duration_months} months at {job.get('company', '?')}")
            penalty *= 0.2

    # =========================================================================
    # CHECK 3: Assessment score contradictions
    # Claims expert but scores < 30 on assessment
    # =========================================================================
    assessment_contradictions = 0
    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        if name in assessments:
            score = assessments[name]
            if prof in ("expert", "advanced") and score < 25:
                assessment_contradictions += 1
                flags.append(f"'{name}': claims {prof} but scored {score}/100")

    if assessment_contradictions >= 3:
        penalty *= 0.1
    elif assessment_contradictions >= 2:
        penalty *= 0.3
    elif assessment_contradictions >= 1:
        penalty *= 0.6

    # =========================================================================
    # CHECK 4: Title-description mismatch
    # AI title but descriptions about accounting/marketing/etc.
    # =========================================================================
    if features and features.get("title_desc_mismatch", 0) == 1.0:
        flags.append("AI/ML title but career descriptions are non-technical")
        penalty *= 0.1

    # =========================================================================
    # CHECK 5: Non-tech title with many AI skills (keyword stuffer)
    # =========================================================================
    if features:
        is_irrelevant = features.get("title_is_irrelevant", 0)
        ai_count = features.get("ai_ml_skill_count", 0)
        desc_ml = features.get("desc_ml_keyword_count", 0)
        if is_irrelevant == 1.0 and ai_count >= 5 and desc_ml < 2:
            flags.append(f"Non-tech title with {ai_count} AI skills but no evidence in career")
            penalty *= 0.05

    # =========================================================================
    # CHECK 6: All skills at same proficiency (suspicious uniformity)
    # =========================================================================
    if len(skills) >= 8:
        proficiencies = [s.get("proficiency", "") for s in skills]
        unique_prof = set(proficiencies)
        if len(unique_prof) == 1 and len(skills) >= 10:
            flags.append(f"All {len(skills)} skills at same proficiency level '{proficiencies[0]}'")
            penalty *= 0.5

    # =========================================================================
    # Final
    # =========================================================================
    is_honeypot = penalty < 0.3
    confidence = max(0, 1.0 - penalty) if is_honeypot else 0.0

    return {
        "is_honeypot": is_honeypot,
        "confidence": round(confidence, 3),
        "flags": flags,
        "penalty": round(max(0.0, min(1.0, penalty)), 3),
    }
