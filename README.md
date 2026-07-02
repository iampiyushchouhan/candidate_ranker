# 🧠 Redrob AI Candidate Ranker — Neural Edition

An intelligent, high-performance candidate discovery and ranking system built for the Redrob AI Founding Team challenge. This system replaces traditional, easily fooled keyword-matching filters with a **3-stage Neural Ranking Pipeline** that evaluates candidates based on semantic compatibility, career trajectory, behavioral platform signals, and strict requirements.

---

## 🌐 Web View & Deployment

### Streamlit Web App
The system is deployed on Streamlit, providing an interactive dashboard to upload candidate files, view real-time rankings, perform candidate deep-dives, and download color-coded XLSX report files.

> [!IMPORTANT]
> **Warm-up Latency:** When starting the Streamlit application for the first time, it takes **30–60 seconds** to download and initialize the local sentence-transformer embedding model (`all-MiniLM-L6-v2`) in memory. Subsequent runs and searches are near-instantaneous.

---

## 🛠️ How the Model Works (The AI/Neural Network Architecture)

Instead of counting keyword frequency (which favors spam/trap profiles), the ranker treats candidate matching as a **Dense Retrieval and Multi-Signal Re-Ranking** problem.

```
                                 [Job Description]
                                         │
                                         ▼
[100,000+ Candidates] ──► [Stage 1: Semantic Embedding (all-MiniLM-L6-v2)]
                                         │
                                         ▼
                          [Stage 2: Dense Cosine Retrieval] (Matches top 500)
                                         │
                                         ▼
                          [Stage 3: Neural Re-Ranking] (Enforces JD Rules)
                                         │
                                         ├───► Apply 10 Hard Disqualifiers
                                         ├───► Penalize Honeypots & Traps
                                         │
                                         ▼
                               [Top 100 Shortlist]
```

### Stage 1: Dense Semantic Representation (Neural Network)
*   **The Neural Model:** We utilize **`all-MiniLM-L6-v2`** from Hugging Face's `sentence-transformers` library (a distilled version of the BERT architecture with 22 Million parameters).
*   **Text Encoding:** The model maps candidates' headlines, summaries, educational history, skill listings, and entire career descriptions into a dense, high-dimensional vector space (384 dimensions). This captures the *context* and *semantic meaning* of their career rather than just their raw wording.

### Stage 2: Dense Cosine Retrieval
*   We project the Job Description into the same 384-dimensional vector space.
*   We run a batch cosine-similarity dot product between the JD vector and all candidate vectors.
*   This instantly narrows down the 100,000 candidate pool to the **top 500 semantically closest candidates** in under 1 second.

### Stage 3: Feature Extraction & Neural Re-Ranking
For each of the 500 shortlisted candidates, we extract **60+ structured parameters** across:
1.  **Skills Fit (20%):** Required and preferred skill coverage, skill depth/proficiency, and platform assessment score validation.
2.  **Career Trajectory (20%):** AI/ML title history, product vs. consulting roles, career tenure, and stability.
3.  **Behavioral Signals (15%):** Redrob platform activity, recruiter response rates, notice periods, and GitHub contribution activity.
4.  **Experience Fit (10%):** Closeness of absolute years of experience to the 5–9 years target.
5.  **Education & Certifications (5%):** Tier of university, field of study relevance, and cloud/ML certifications.

#### Hard Disqualifiers & Honeypots (Multipliers):
We apply strict multipliers ($0.0 - 1.0$) to filter out unsuitable/trap profiles:
*   **Consulting-Only Trap:** Candidates who have only worked at service/consulting companies (TCS, Wipro, Infosys, etc.) with no product-company exposure are disqualified.
*   **CV/Speech-Only:** Candidates focusing exclusively on CV/speech/robotics with zero NLP/IR skills.
*   **Honeypots:** Candidates claiming expert skills with zero years used, or claiming impossibly long tenures (e.g., 8 years at a 2-year-old company).

---

## 📦 Project Directory Structure

```
candidate_ranker/
│
├── config.py                 # Core scoring weights, classification rules, and JD text
├── utils.py                  # Text cleaning and Jaccard token-matching utilities
├── job_parser.py             # Parses job description files
├── candidate_loader.py       # Optimized JSONL/JSON streaming candidate loader
├── ranking_engine.py         # Main controller orchestrating the 3-stage neural pipeline
├── app.py                    # Streamlit interactive application code
├── rank.py                   # Command Line Interface (CLI) script
├── precompute.py             # Script to pre-encode the candidate pool to .npy files
├── requirements.txt          # Python packages list
├── CONTEXT.md                # Developer memory and system context document
└── scoring/                  # Pipeline utility packages
    ├── __init__.py           
    ├── semantic_scorer.py      # Embedding model downloader, encoder, and cosine retrieval
    ├── feature_extractor.py    # Computes 60+ numeric features per profile
    ├── disqualifiers.py        # Validates candidate profiles against the 10 JD rules
    └── honeypot_detector.py    # Catches mock/trap profiles using logic conflicts
```

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-compute Embeddings (One-Time)
Run this script to download the MiniLM model weights and pre-compute embeddings for the full 100K dataset. This generates `/embeddings/candidate_embeddings.npy`:
```bash
python precompute.py --candidates path/to/candidates.jsonl
```

### 3. Run Ranking CLI (< 2 Minutes)
```bash
python rank.py --candidates path/to/candidates.jsonl --out submission.csv --xlsx report.xlsx
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

---

## ☁️ How to Deploy on Streamlit Cloud

1.  **Push your repository to GitHub** (Ensure you include `.gitignore` so local `.npy` embedding cache and large `candidates.jsonl` files are not pushed to Git).
2.  Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3.  Click **New app**, select your repository, the `main` branch, and set the file path to `candidate_ranker/app.py`.
4.  Click **Deploy**. On the first boot, Streamlit will download the `sentence-transformers` library and the model weights (taking about 30-60 seconds).
5.  Upload your candidates file (JSON, JSONL, or JSONL.gz) and run the ranker directly in the browser!
