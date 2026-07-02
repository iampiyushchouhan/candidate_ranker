"""
Candidate Loader: Stream and parse candidates from JSONL files.
"""

import json


def load_candidates_from_jsonl(filepath: str, max_candidates: int = None):
    """
    Generator that streams candidates from a JSONL file.
    Yields one candidate dict at a time to handle large files.
    """
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield candidate
                count += 1
                if max_candidates and count >= max_candidates:
                    break
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line: {e}")
                continue


def load_candidates_from_json(filepath: str, max_candidates: int = None):
    """
    Load candidates from a JSON array file (like sample_candidates.json).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    if max_candidates:
        return candidates[:max_candidates]
    return candidates


def count_lines(filepath: str) -> int:
    """Count number of lines in a file (for progress tracking)."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def load_candidates(filepath: str, max_candidates: int = None) -> list:
    """
    Load candidates from either JSONL or JSON file.
    Returns a list of candidate dicts.
    """
    if filepath.endswith('.jsonl'):
        return list(load_candidates_from_jsonl(filepath, max_candidates))
    elif filepath.endswith('.json'):
        return load_candidates_from_json(filepath, max_candidates)
    else:
        # Try JSONL first, then JSON
        try:
            return list(load_candidates_from_jsonl(filepath, max_candidates))
        except Exception:
            return load_candidates_from_json(filepath, max_candidates)
