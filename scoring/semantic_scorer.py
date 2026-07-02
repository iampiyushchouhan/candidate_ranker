"""
Semantic Scorer — Sentence-Transformer based embedding and similarity.
Uses all-MiniLM-L6-v2 (22M params, 384-dim, CPU-friendly).
"""

import os
import sys
import numpy as np
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM


_model = None


def get_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Try to load from local cache first
        model_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_cache, exist_ok=True)

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=model_cache)
    return _model


def encode_texts(texts: List[str], batch_size: int = 256, show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of texts into embedding vectors.
    Returns numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine sim via dot product
    )
    return embeddings


def encode_single(text: str) -> np.ndarray:
    """Encode a single text string into an embedding vector."""
    return encode_texts([text])[0]


def cosine_similarity_batch(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query embedding and all corpus embeddings.
    Since embeddings are L2-normalized, this is just a dot product.
    Returns array of shape (len(corpus_embeddings),).
    """
    # Both are L2-normalized, so dot product = cosine similarity
    return np.dot(corpus_embeddings, query_embedding)


def retrieve_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 500,
) -> tuple:
    """
    Retrieve top-k most similar items from corpus.
    Returns (indices, scores) — both sorted by descending similarity.
    """
    similarities = cosine_similarity_batch(query_embedding, corpus_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    return top_indices, top_scores


def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of a candidate for embedding.
    Includes: headline, summary, all career descriptions, skills, education fields.
    """
    parts = []

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(f"Current role: {profile.get('current_title', '')} at {profile.get('current_company', '')}")
    parts.append(f"Industry: {profile.get('current_industry', '')}")

    # Career history — descriptions are the richest signal
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        industry = job.get("industry", "")
        parts.append(f"{title} at {company} ({industry}): {desc}")

    # Skills with proficiency
    skill_strs = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        skill_strs.append(f"{name} ({prof})")
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs))

    # Education
    for edu in candidate.get("education", []):
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        parts.append(f"Education: {degree} in {field} from {institution}")

    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(f"Certification: {cert.get('name', '')} by {cert.get('issuer', '')}")

    return " ".join(p for p in parts if p)


def save_embeddings(embeddings: np.ndarray, path: str):
    """Save embeddings to a .npy file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: str) -> Optional[np.ndarray]:
    """Load embeddings from a .npy file, or return None if not found."""
    if os.path.exists(path):
        return np.load(path)
    return None
