# Project Context: Redrob AI Candidate Ranker (Neural Edition)

## 1. Problem Statement

Build an AI system that ranks 100K candidates for the role of **Senior AI Engineer — Founding Team at Redrob AI**. The system must:
- Use **semantic understanding** (not just keyword matching) to evaluate candidates
- Detect and avoid ~80 **honeypot/trap candidates** (>10% honeypots in top 100 = disqualified)
- Run within **5 minutes on a 16 GB CPU-only machine** (no GPUs, no network, no hosted LLM APIs)
- Output a **CSV with top 100 candidates**, ranked best-fit first, with reasoning

## 2. Architecture: 3-Stage Neural Pipeline

```
Stage 1: Pre-compute embeddings (one-time, via precompute.py)
    └─ sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim, CPU)
    └─ Saves: embeddings/candidate_embeddings.npy (~150MB for 100K)

Stage 2: Dense semantic retrieval (< 1 min)
    └─ Cosine similarity: JD embedding vs all 100K candidate embeddings
    └─ Output: top 500 shortlist

Stage 3: Neural re-ranking (< 2 min)
    └─ For each of 500 candidates:
        ├─ Extract 60+ structured features (feature_extractor.py)
        ├─ Compute weighted re-rank score (semantic + skills + career + behavioral)
        ├─ Apply hard disqualifiers (disqualifiers.py)
        ├─ Apply honeypot penalty (honeypot_detector.py)
        └─ Generate natural-language reasoning
    └─ Sort → top 100
```

## 3. Directory Structure

```
candidate_ranker/
├── app.py                      # Streamlit app (upload-only UI)
├── rank.py                     # CLI: python rank.py --candidates X --out Y
├── precompute.py               # Pre-compute embeddings for full dataset
├── config.py                   # Actual JD, weights, disqualifier criteria
├── job_parser.py               # JD parser (largely unused — JD is hardcoded)
├── candidate_loader.py         # JSONL/JSON loader
├── ranking_engine.py           # 3-stage neural pipeline
├── requirements.txt            # Dependencies
├── utils.py                    # Text utilities
├── CONTEXT.md                  # This file
├── models/                     # Local model cache (auto-downloaded)
├── embeddings/                 # Pre-computed embeddings
└── scoring/
    ├── __init__.py
    ├── semantic_scorer.py      # Sentence-transformer encoding + cosine retrieval
    ├── feature_extractor.py    # 60+ structured feature extraction
    ├── disqualifiers.py        # 10 hard disqualifiers from JD
    └── honeypot_detector.py    # Trap candidate detection
```

## 4. Re-Ranking Weights

| Component | Weight | What It Scores |
|-----------|--------|----------------|
| Semantic Similarity | 30% | Cosine sim between candidate text embedding and JD embedding |
| Skills Fit | 20% | Required/preferred coverage, NLP/IR skills, proficiency, assessments |
| Career Trajectory | 20% | AI title history, description relevance, product experience, stability |
| Behavioral Signals | 15% | Response rate, recency, GitHub, notice period, engagement |
| Experience Fit | 10% | Years in ideal/acceptable range |
| Education | 5% | Tier, degree level, field relevance, certifications |

## 5. Hard Disqualifiers (from actual JD)

1. Consulting-only career (TCS/Infosys/Wipro/Accenture with no product experience)
2. CV/Speech/Robotics primary without NLP/IR exposure
3. Non-technical title with no AI/ML career evidence
4. Title-chaser (avg < 1.5 years per role with 4+ roles)
5. Title-description mismatch (honeypot)
6. Skill stuffing (many AI skills, no career evidence)
7. Too junior (< 2 years) or too senior (> 15 years)
8. Not in India and not willing to relocate
9. Inactive 6+ months or < 10% response rate
10. Very long notice period (> 120 days)

## 6. How to Run

```bash
# Install
pip install -r requirements.txt

# Pre-compute embeddings (one-time, ~20 min for 100K)
python precompute.py --candidates candidates.jsonl

# Rank (< 5 min with pre-computed embeddings)
python rank.py --candidates candidates.jsonl --out submission.csv --xlsx report.xlsx

# Streamlit app
streamlit run app.py

# Validate
python validate_submission.py submission.csv
```
