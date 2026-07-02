#!/usr/bin/env python3
"""
CLI Ranking Script — Produces submission CSV from candidates file.
Usage: python rank.py --candidates candidates.jsonl --out submission.csv

The ranking step must complete within 5 minutes on a 16 GB CPU-only machine.
Pre-computed embeddings are used if available in ./embeddings/.
"""

import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FINAL_TOP_N
from ranking_engine import rank_candidates, results_to_submission_csv, results_to_xlsx


def main():
    parser = argparse.ArgumentParser(description="Redrob AI Candidate Ranker — Neural Pipeline")
    parser.add_argument("--candidates", "-c", required=True, help="Path to candidates JSONL or JSON file")
    parser.add_argument("--out", "-o", default="submission.csv", help="Output CSV path (default: submission.csv)")
    parser.add_argument("--xlsx", default=None, help="Also export XLSX report")
    parser.add_argument("--top-n", "-n", type=int, default=FINAL_TOP_N, help=f"Top N candidates (default: {FINAL_TOP_N})")
    parser.add_argument("--embeddings-dir", "-e", default=None, help="Directory with pre-computed embeddings")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit candidates (for testing)")
    args = parser.parse_args()

    # Resolve embeddings directory
    emb_dir = args.embeddings_dir
    if emb_dir is None:
        default_emb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
        if os.path.exists(os.path.join(default_emb, "candidate_embeddings.npy")):
            emb_dir = default_emb
            print(f"Using pre-computed embeddings from: {emb_dir}")
        else:
            print("No pre-computed embeddings found. Will compute on-the-fly.")

    # Load candidates
    print(f"\nLoading candidates from: {args.candidates}")
    start = time.time()

    candidates = []
    filepath = args.candidates

    if filepath.endswith(".jsonl.gz"):
        import gzip
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                    if args.max_candidates and len(candidates) >= args.max_candidates:
                        break
    elif filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                    if args.max_candidates and len(candidates) >= args.max_candidates:
                        break
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            if args.max_candidates:
                candidates = candidates[:args.max_candidates]

    load_time = time.time() - start
    print(f"  Loaded {len(candidates):,} candidates in {load_time:.1f}s")

    # Rank
    print("\n" + "=" * 70)
    print("  NEURAL RANKING PIPELINE")
    print("=" * 70)

    def progress(current, total, stage, elapsed):
        bar_len = 40
        pct = current / max(total, 1)
        filled = int(bar_len * pct)
        bar = '█' * filled + '░' * (bar_len - filled)
        sys.stdout.write(f'\r  [{bar}] {current:,}/{total:,} · {stage} · {elapsed:.0f}s')
        sys.stdout.flush()

    start = time.time()
    results = rank_candidates(
        candidates,
        embeddings_dir=emb_dir,
        progress_callback=progress,
        top_n=args.top_n,
    )
    rank_time = time.time() - start

    # Export
    results_to_submission_csv(results, args.out, top_n=args.top_n)

    if args.xlsx:
        results_to_xlsx(results, args.xlsx, top_n=args.top_n)

    # Print top 10
    print(f"\n{'=' * 70}")
    print(f"  TOP 10 CANDIDATES")
    print(f"{'=' * 70}")
    for r in results[:10]:
        hp = "⚠️HP" if r.get("is_honeypot") else "  "
        dq = "❌DQ" if r.get("is_disqualified") else "  "
        print(f"  #{r['rank']:3d}  {r['candidate_id']}  {r['final_score']:.4f}  "
              f"sem={r['semantic_similarity']:.3f}  {hp} {dq}  "
              f"{r['current_title'][:28]:<28s}  {r['name']}")

    print(f"\n  Total ranking time: {rank_time:.1f}s")
    print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
