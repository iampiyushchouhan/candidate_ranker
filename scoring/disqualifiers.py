"""
Hard Disqualifiers — Based on the actual JD's explicit "do not want" section.
These are not soft penalties; they are near-elimination filters.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONSULTING_COMPANIES, CV_SPEECH_ROBOTICS_SKILLS


def apply_disqualifiers(candidate: dict, features: dict) -> dict:
    """
    Check all hard disqualifiers from the JD.

    Returns dict with:
        - multiplier: float (0.0 = fully disqualified, 1.0 = no issue)
        - reasons: list of disqualification reasons
        - is_disqualified: bool
    """
    reasons = []
    multiplier = 1.0

    # =====================================================================
    # 1. CONSULTING-ONLY CAREER
    # JD: "People who have only worked at consulting firms (TCS, Infosys,
    # Wipro, Accenture, etc.) in their entire career"
    # Exception: "If you're currently at one of these companies but have
    # prior product-company experience, that's fine."
    # =====================================================================
    if features.get("is_consulting_only", 0) == 1.0:
        reasons.append("Consulting-only career (no product company experience)")
        multiplier *= 0.03

    # Even non-consulting-only, heavy consulting ratio is a yellow flag
    elif features.get("consulting_role_ratio", 0) > 0.7:
        reasons.append("Predominantly consulting career")
        multiplier *= 0.3

    # =====================================================================
    # 2. CV/SPEECH/ROBOTICS WITHOUT NLP/IR
    # JD: "People whose primary expertise is computer vision, speech, or
    # robotics without significant NLP/IR exposure."
    # =====================================================================
    cv_count = features.get("cv_speech_skill_count", 0)
    nlp_ir_count = features.get("nlp_ir_skill_count", 0)
    if cv_count >= 3 and nlp_ir_count == 0:
        reasons.append("Primary expertise in CV/Speech/Robotics without NLP/IR exposure")
        multiplier *= 0.1

    # =====================================================================
    # 3. IRRELEVANT TITLE — NOT an engineer/data person at all
    # JD targets AI Engineer; Marketing Manager / Accountant / etc. = bad
    # =====================================================================
    if features.get("title_is_irrelevant", 0) == 1.0 and features.get("title_is_relevant", 0) == 0.0:
        # Check if they at least have some AI career history
        if features.get("career_ai_title_count", 0) == 0 and features.get("desc_ml_keyword_count", 0) < 3:
            reasons.append(f"Non-technical title with no AI/ML career evidence")
            multiplier *= 0.02

    # =====================================================================
    # 4. TITLE-CHASER
    # JD: "If your career trajectory shows you optimizing for titles by
    # switching companies every 1.5 years"
    # =====================================================================
    if features.get("is_title_chaser", 0) == 1.0 and features.get("num_roles", 0) >= 4:
        reasons.append("Title-chaser pattern (frequent company switches)")
        multiplier *= 0.4

    # =====================================================================
    # 5. TITLE-DESCRIPTION MISMATCH (honeypot)
    # AI title but non-AI work descriptions
    # =====================================================================
    if features.get("title_desc_mismatch", 0) == 1.0:
        reasons.append("Title-description mismatch (possible honeypot)")
        multiplier *= 0.05

    # =====================================================================
    # 6. SKILL STUFFING — many AI skills but no evidence
    # High AI skill claims but zero assessments, zero description evidence
    # =====================================================================
    ai_skills = features.get("ai_ml_skill_count", 0)
    desc_ml = features.get("desc_ml_keyword_count", 0)
    desc_non_ml = features.get("desc_non_ml_keyword_count", 0)
    if ai_skills >= 5 and desc_ml < 2 and desc_non_ml >= 3:
        reasons.append(f"Skill stuffing: {ai_skills} AI skills claimed but career descriptions are non-technical")
        multiplier *= 0.05

    # =====================================================================
    # 7. EXPERIENCE OUT OF RANGE
    # JD says 5-9 years. Way outside = penalty (not full disqualification)
    # =====================================================================
    years = features.get("years_experience", 0)
    if years < 2:
        reasons.append(f"Too junior ({years:.1f} years, need 5-9)")
        multiplier *= 0.1
    elif years < 4:
        reasons.append(f"Below experience range ({years:.1f} years, need 5-9)")
        multiplier *= 0.4
    elif years > 15:
        reasons.append(f"Significantly over-experienced ({years:.1f} years)")
        multiplier *= 0.5

    # =====================================================================
    # 8. NOT IN INDIA (soft penalty, not full disqualification)
    # JD: "Outside India: case-by-case, but we don't sponsor work visas."
    # =====================================================================
    if features.get("is_india", 0) == 0.0 and features.get("willing_to_relocate", 0) == 0.0:
        reasons.append("Not in India and not willing to relocate")
        multiplier *= 0.3
    elif features.get("is_india", 0) == 0.0 and features.get("willing_to_relocate", 0) == 1.0:
        reasons.append("Not in India but willing to relocate")
        multiplier *= 0.6

    # =====================================================================
    # 9. INACTIVE / UNREACHABLE
    # JD: "Active on Redrob platform"
    # =====================================================================
    if features.get("days_since_active", 999) > 180:
        reasons.append("Inactive for 6+ months")
        multiplier *= 0.3
    if features.get("recruiter_response_rate", 0) < 0.1:
        reasons.append("Very low recruiter response rate (<10%)")
        multiplier *= 0.5

    # =====================================================================
    # 10. LONG NOTICE PERIOD
    # JD: "sub-30-day notice preferred; 30+ = higher bar"
    # =====================================================================
    notice = features.get("notice_period_days", 90)
    if notice > 120:
        reasons.append(f"Very long notice period ({notice} days)")
        multiplier *= 0.6

    return {
        "multiplier": max(0.0, min(1.0, multiplier)),
        "reasons": reasons,
        "is_disqualified": multiplier < 0.1,
    }
