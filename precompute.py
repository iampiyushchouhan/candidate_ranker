#!/usr/bin/env python3
"""
Pre-computation Script — Embed all candidates using sentence-transformers.
Run this ONCE before ranking. The ranking step then uses cached embeddings.

Usage:
    python precompute.py --candidates candidates.jsonl
    python precompute.py --candidates candidates.jsonl --batch-size 512
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMBEDDING_DIM, JOB_DESCRIPTION
from scoring.semantic_scorer import encode_texts, build_candidate_text, encode_single, save_embeddings


def main():
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    parser.add_argument("--candidates", "-c", required=True, help="Path to candidates JSONL or JSON file")
    parser.add_argument("--output-dir", "-o", default=None, help="Directory to save embeddings (default: ./embeddings/)")
    parser.add_argument("--batch-size", "-b", type=int, default=256, help="Encoding batch size (default: 256)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load candidates
    # ------------------------------------------------------------------
    print(f"Loading candidates from: {args.candidates}")
    start = time.time()

    candidates = []
    candidate_ids = []

    if args.candidates.endswith(".jsonl") or args.candidates.endswith(".jsonl.gz"):
        import gzip
        opener = gzip.open if args.candidates.endswith(".gz") else open
        with opener(args.candidates, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                candidates.append(c)
                candidate_ids.append(c.get("candidate_id", ""))
    else:
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            candidate_ids = [c.get("candidate_id", "") for c in candidates]

    load_time = time.time() - start
    print(f"  Loaded {len(candidates):,} candidates in {load_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Build candidate text profiles
    # ------------------------------------------------------------------
    print("Building candidate text profiles...")
    start = time.time()
    texts = [build_candidate_text(c) for c in candidates]
    text_time = time.time() - start
    print(f"  Built {len(texts):,} text profiles in {text_time:.1f}s")
    print(f"  Avg text length: {sum(len(t) for t in texts) / len(texts):.0f} chars")

    # ------------------------------------------------------------------
    # Step 3: Encode with sentence-transformer
    # ------------------------------------------------------------------
    print(f"\nEncoding with sentence-transformers (batch_size={args.batch_size})...")
    print("  (This may take 15-25 minutes for 100K candidates on CPU)")
    start = time.time()
    embeddings = encode_texts(texts, batch_size=args.batch_size, show_progress=True)
    encode_time = time.time() - start
    print(f"\n  Encoded {len(embeddings):,} candidates in {encode_time:.1f}s")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Rate: {len(embeddings) / encode_time:.0f} candidates/sec")

    # ------------------------------------------------------------------
    # Step 4: Encode JD text
    # ------------------------------------------------------------------
    print("Encoding job description...")
    jd_text = JOB_DESCRIPTION.get("raw_text", "")
    jd_embedding = encode_single(jd_text)
    print(f"  JD embedding shape: {jd_embedding.shape}")

    # ------------------------------------------------------------------
    # Step 5: Save everything
    # ------------------------------------------------------------------
    emb_path = os.path.join(output_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(output_dir, "candidate_ids.json")
    jd_path = os.path.join(output_dir, "jd_embedding.npy")

    save_embeddings(embeddings, emb_path)
    print(f"  Saved candidate embeddings: {emb_path} ({os.path.getsize(emb_path) / 1024 / 1024:.1f} MB)")

    with open(ids_path, "w") as f:
        json.dump(candidate_ids, f)
    print(f"  Saved candidate IDs: {ids_path}")

    save_embeddings(jd_embedding, jd_path)
    print(f"  Saved JD embedding: {jd_path}")

    total_time = load_time + text_time + encode_time
    print(f"\n✅ Pre-computation complete! Total time: {total_time:.1f}s")
    print(f"   Embeddings saved to: {output_dir}/")


if __name__ == "__main__":
    main()
