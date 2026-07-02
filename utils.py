"""
Utility functions for text processing and similarity computation.
"""

import re
import math
from collections import Counter


def clean_text(text: str) -> str:
    """Lowercase, remove special chars, normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer after cleaning."""
    return clean_text(text).split()


def compute_tfidf_vectors(documents: list) -> tuple:
    """
    Compute TF-IDF vectors for a list of documents.
    Returns (tfidf_vectors, vocabulary).
    Each vector is a dict {term: tfidf_score}.
    """
    # Compute document frequencies
    df = Counter()
    doc_tokens = []
    for doc in documents:
        tokens = tokenize(doc)
        doc_tokens.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1

    n_docs = len(documents)
    tfidf_vectors = []

    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = {}
        for term, count in tf.items():
            tf_score = count / total
            idf_score = math.log((n_docs + 1) / (df[term] + 1)) + 1
            vector[term] = tf_score * idf_score
        tfidf_vectors.append(vector)

    vocabulary = set()
    for v in tfidf_vectors:
        vocabulary.update(v.keys())

    return tfidf_vectors, vocabulary


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Compute cosine similarity between two sparse vectors (dicts)."""
    if not vec1 or not vec2:
        return 0.0

    # Find common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())
    if not common_terms:
        return 0.0

    dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def fuzzy_match_score(s1: str, s2: str) -> float:
    """Simple fuzzy match score between two strings using character overlap."""
    s1 = clean_text(s1)
    s2 = clean_text(s2)
    if not s1 or not s2:
        return 0.0

    # Token-level Jaccard similarity
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def skill_name_match(skill_name: str, target_skills: set) -> float:
    """
    Check if a skill name matches any target skill.
    Returns 1.0 for exact match, 0.5-0.8 for partial/fuzzy match, 0.0 for no match.
    """
    skill_lower = skill_name.lower().strip()
    targets_lower = {t.lower().strip() for t in target_skills}

    # Exact match
    if skill_lower in targets_lower:
        return 1.0

    # Check if skill is a substring of any target or vice versa
    for target in targets_lower:
        if skill_lower in target or target in skill_lower:
            return 0.8

    # Token overlap
    skill_tokens = set(skill_lower.split())
    for target in targets_lower:
        target_tokens = set(target.split())
        overlap = skill_tokens & target_tokens
        if overlap and len(overlap) >= len(skill_tokens) * 0.5:
            return 0.6

    return 0.0


def extract_text_from_candidate(candidate: dict) -> str:
    """Extract all textual content from a candidate profile for TF-IDF."""
    parts = []

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    parts.append(profile.get("current_industry", ""))

    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))

    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    for edu in candidate.get("education", []):
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))

    return " ".join(parts)


def days_since(date_str: str, reference_date: str = "2026-06-01") -> int:
    """Calculate days between a date string and reference date."""
    try:
        from datetime import datetime
        if not date_str:
            return 999
        d = datetime.strptime(date_str, "%Y-%m-%d")
        ref = datetime.strptime(reference_date, "%Y-%m-%d")
        return (ref - d).days
    except (ValueError, TypeError):
        return 999


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
