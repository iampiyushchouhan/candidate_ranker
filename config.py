"""
Configuration — Redrob AI Candidate Ranker (Neural Edition)
Based on the ACTUAL job description: Senior AI Engineer — Founding Team at Redrob AI.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RETRIEVAL_TOP_K = 500        # Shortlist size from dense retrieval
FINAL_TOP_N = 100            # Final output size

# ============================================================================
# SCORING WEIGHTS for Re-Ranking (Stage 3)
# ============================================================================
RERANK_WEIGHTS = {
    "semantic_similarity": 0.30,
    "skills_fit": 0.20,
    "career_trajectory": 0.20,
    "behavioral_signals": 0.15,
    "experience_fit": 0.10,
    "education": 0.05,
}

# ============================================================================
# ACTUAL JOB DESCRIPTION — Senior AI Engineer, Founding Team, Redrob AI
# Extracted from job_description.docx
# ============================================================================
JOB_DESCRIPTION = {
    "title": "Senior AI Engineer",
    "company": "Redrob AI",
    "company_stage": "Series A AI-native talent intelligence platform",
    "location": "Pune/Noida, India (Hybrid)",
    "employment_type": "Full-time",
    "experience_range": {"min": 5, "max": 9},
    "ideal_experience": {"min": 6, "max": 8},
    "ideal_applied_ml_years": {"min": 4, "max": 5},

    # What the JD says the role actually does
    "role_summary": (
        "Own the intelligence layer of Redrob's product — ranking, retrieval, "
        "and matching systems that decide what recruiters see when they search "
        "for candidates and what candidates see when they search for roles. "
        "Ship a v2 ranking system using embeddings, hybrid retrieval, and "
        "LLM-based re-ranking. Set up evaluation infrastructure (offline "
        "benchmarks, online A/B testing, recruiter-feedback loops)."
    ),

    # ---- MUST-HAVE SKILLS (hard requirements from JD) ----
    "required_skills": [
        "Python",
        "Embeddings", "Sentence Transformers", "BGE", "E5",
        "Vector Databases", "Pinecone", "Weaviate", "Qdrant", "Milvus",
        "FAISS", "OpenSearch", "Elasticsearch",
        "Hybrid Search", "BM25", "Information Retrieval",
        "Ranking Systems", "Search Systems", "Recommendation Systems",
        "Evaluation Frameworks", "NDCG", "MRR", "MAP", "A/B Testing",
        "NLP", "Natural Language Processing",
        "Machine Learning", "Deep Learning",
        "RAG", "Retrieval Augmented Generation",
    ],

    # ---- NICE-TO-HAVE SKILLS ----
    "preferred_skills": [
        "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT",
        "Learning to Rank", "XGBoost", "LightGBM",
        "HR Tech", "Recruiting Tech", "Marketplace Products",
        "Distributed Systems", "Inference Optimization",
        "Open Source Contributions",
        "PyTorch", "TensorFlow", "Transformers", "Hugging Face",
        "MLOps", "MLflow", "Weights & Biases", "Kubeflow",
        "Docker", "Kubernetes", "AWS", "GCP", "Azure",
        "Spark", "Airflow", "Data Pipelines",
        "Statistical Modeling", "Feature Engineering",
        "Prompt Engineering", "LLMs",
    ],

    # ---- SKILLS THE JD EXPLICITLY DOES NOT WANT ----
    "unwanted_primary_domains": [
        "Computer Vision", "Image Classification", "Object Detection",
        "Image Segmentation", "OpenCV", "OCR",
        "Speech Recognition", "TTS", "Speech Synthesis",
        "Robotics", "Control Systems",
        "GANs", "Generative Adversarial Networks",
    ],

    # Location preferences
    "preferred_locations": [
        "Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Delhi NCR",
        "Gurgaon", "Gurugram", "Bangalore", "Bengaluru", "Chennai",
    ],
    "preferred_country": "India",

    # JD raw text for semantic encoding
    "raw_text": (
        "Senior AI Engineer Founding Team at Redrob AI. "
        "Own the intelligence layer: ranking, retrieval, and matching systems. "
        "Production experience with embeddings-based retrieval systems "
        "(sentence-transformers, BGE, E5). "
        "Production experience with vector databases or hybrid search "
        "infrastructure (Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, FAISS). "
        "Strong Python and code quality. "
        "Hands-on experience designing evaluation frameworks for ranking systems "
        "(NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation). "
        "Ship a v2 ranking system using embeddings, hybrid retrieval, "
        "and LLM-based re-ranking. "
        "NLP, information retrieval, search, recommendation systems. "
        "LLM fine-tuning (LoRA, QLoRA, PEFT). "
        "Learning-to-rank models (XGBoost or neural). "
        "Distributed systems, large-scale inference optimization. "
        "6-8 years total experience, 4-5 in applied ML/AI at product companies. "
        "Located in or willing to relocate to Noida or Pune India."
    ),
}

# ============================================================================
# HARD DISQUALIFIERS — From the JD's explicit "do not want" section
# These result in near-zero scores (0.0 - 0.05 multiplier)
# ============================================================================

# Companies that are pure consulting / services firms
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "ltimindtree",
    "l&t infotech", "hexaware", "persistent", "cyient",
    "zensar", "birlasoft", "sonata software", "coforge", "niit",
}

# Industries considered pure services/consulting
CONSULTING_INDUSTRIES = {
    "it services", "consulting", "outsourcing", "bpo",
    "staffing", "recruitment services",
}

# Titles that indicate NOT an AI/ML practitioner
IRRELEVANT_TITLES = {
    "marketing manager", "hr manager", "sales executive",
    "operations manager", "accountant", "content writer",
    "graphic designer", "mechanical engineer", "civil engineer",
    "customer support", "project manager", "business analyst",
}

# Titles that ARE relevant (AI/ML/Data/Engineering)
RELEVANT_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior machine learning engineer", "senior ai engineer",
    "junior ml engineer", "data scientist", "senior data scientist",
    "lead data scientist", "deep learning engineer", "nlp engineer",
    "applied scientist", "research scientist", "research engineer",
    "mlops engineer", "ai researcher", "data science lead",
    "ml architect", "ai architect", "principal ml engineer",
    "data engineer", "analytics engineer", "backend engineer",
    "software engineer", "full stack developer", "platform engineer",
    "senior software engineer", "staff engineer",
}

# AI/ML core skill names (for counting/matching)
AI_ML_CORE_SKILLS = {
    "Machine Learning", "Deep Learning", "Neural Networks", "CNN", "RNN",
    "Transformers", "NLP", "Natural Language Processing",
    "LLMs", "Fine-tuning LLMs", "Prompt Engineering", "RAG",
    "Information Retrieval", "BM25", "Vector Databases",
    "Embeddings", "Sentence Transformers",
    "Recommendation Systems", "Ranking Systems", "Search Systems",
    "Feature Engineering", "Statistical Modeling",
    "TensorFlow", "PyTorch", "scikit-learn", "Hugging Face",
    "MLOps", "MLflow", "Weights & Biases", "Kubeflow",
    "FAISS", "Milvus", "Elasticsearch", "Pinecone", "Qdrant",
    "LoRA", "PEFT", "QLoRA",
    "Python", "Data Science", "Statistics",
    "XGBoost", "LightGBM", "Learning to Rank",
    "A/B Testing", "NDCG", "Evaluation Frameworks",
    "Spark", "Airflow", "Data Pipelines",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure",
}

# CV/Speech/Robotics skills (JD says these are NOT wanted as primary)
CV_SPEECH_ROBOTICS_SKILLS = {
    "Computer Vision", "Image Classification", "Object Detection",
    "Image Segmentation", "OpenCV", "OCR",
    "Speech Recognition", "TTS", "Speech Synthesis",
    "Robotics", "Control Systems",
    "GANs", "Generative Adversarial Networks",
}

# Non-technical skills
NON_TECH_SKILLS = {
    "Accounting", "Sales", "Marketing", "Content Writing", "SEO",
    "Photoshop", "Illustrator", "Figma", "PowerPoint", "Excel",
    "Six Sigma", "SAP", "Project Management",
    "HR", "Recruitment", "Supply Chain", "Logistics",
}

# ============================================================================
# SCORING THRESHOLDS
# ============================================================================
PROFICIENCY_SCORES = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
    "unknown": 0.3,
}

DEGREE_SCORES = {
    "Ph.D": 1.0,
    "M.Tech": 0.90,
    "M.S.": 0.85,
    "M.E.": 0.80,
    "M.Sc": 0.75,
    "MBA": 0.55,
    "B.Tech": 0.65,
    "B.E.": 0.60,
    "B.Sc": 0.50,
    "B.Com": 0.25,
    "B.A.": 0.25,
    "Diploma": 0.20,
}
