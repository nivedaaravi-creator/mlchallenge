#!/usr/bin/env python3
"""
Stage 2: Blocking Layer
ML-Only Business Entity Resolution Pipeline

Purpose:
    Generate a high-recall, low-noise candidate set of (S1, S2/S3) pairs for the
    matching model.  The blocking stage determines the recall *ceiling* -- any true
    match that is not in the candidate set can never be recovered downstream.

Strategy -- Multi-Key Inverted Index:
    For every record we emit several cheap, robust "blocking keys".  Two records
    become a candidate pair iff they share at least one key.  Keys are designed so
    that even heavily-noisy record pairs almost always collide on one key.

    Keys emitted per record:
    1.  GN   -- country + geo_code + first name token
    2.  GN2  -- country + geo_code + second name token
    3.  SN   -- country + state + first name token
    4.  CN   -- country + city_word1 + first name token
    5.  BG   -- country + each consecutive name-token bigram (canonical order)
    6.  N4   -- country + 4-char prefix of every sorted name token
    7.  GO   -- country + geo_code  (broad fallback)

    Noise robustness:
    - geo_code falls back to state-level default when ZIP is missing.
    - name4 (N4) handles abbreviations: "CORP" == "CORPORATION"[:4].
    - Bigrams (BG) catch word-order transpositions and dropped tokens.

Parameters:
    MAX_BLOCK_SIZE   -- discard keys whose bucket > this (default 500).
    MAX_CANDS_PER_S1 -- hard cap on candidates per S1 entity (default 200).

CLI:
    python blocking.py block --s1 ... --s2 ... --s3 ... --output output/candidate_pairs.tsv
    python blocking.py eval  --candidates output/candidate_pairs.tsv
                             --ground-truth dataset/train/train_ground_truth.tsv
"""

import sys
import os
import time
import logging
from collections import defaultdict
from typing import Dict, List, Set, Optional

import pandas as pd

# Make the src package importable when run directly
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from preprocess import preprocess_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

MAX_BLOCK_SIZE   = 500      # discard keys whose bucket > this value
MAX_CANDS_PER_S1 = 200      # hard cap per S1 entity
CHUNKSIZE        = 200_000


# =====================================================================
# 1. BLOCKING KEY GENERATION
# =====================================================================

def _name4(name_tokens_str: str) -> str:
    """4-char prefix of every name token, sorted and joined.

    'PRABHAV BUSINESS CENTER' -> 'BUSI CENT PRAB'
    Robust to abbreviations and word-order variation.
    """
    tokens = name_tokens_str.split()
    return " ".join(sorted(t[:4] for t in tokens if t))


def generate_blocking_keys(row: pd.Series) -> List[str]:
    """Return all blocking keys for one preprocessed record row."""
    country   = str(row.get("country",     "") or "").strip()
    geo_code  = str(row.get("geo_code",    "") or "").strip()
    state     = str(row.get("state",       "") or "").strip()
    city      = str(row.get("city",        "") or "").strip()
    nm_tokens = str(row.get("name_tokens", "") or "").strip()

    city_w1 = city.split()[0] if city else ""
    ntoks   = nm_tokens.split()
    nm1     = ntoks[0] if len(ntoks) >= 1 else ""
    nm2     = ntoks[1] if len(ntoks) >= 2 else ""

    keys: List[str] = []

    # GN -- geo_code + name token 1 / 2
    if geo_code and nm1:
        keys.append(f"GN|{country}|{geo_code}|{nm1}")
    if geo_code and nm2:
        keys.append(f"GN|{country}|{geo_code}|{nm2}")

    # SN -- country + state + first name token
    if state and nm1:
        keys.append(f"SN|{country}|{state}|{nm1}")

    # CN -- country + city_word1 + first name token
    if city_w1 and nm1:
        keys.append(f"CN|{country}|{city_w1}|{nm1}")

    # BG -- name bigrams (canonical pair order catches transpositions)
    for i in range(len(ntoks) - 1):
        a, b = ntoks[i], ntoks[i + 1]
        pair = f"{min(a,b)}_{max(a,b)}"
        keys.append(f"BG|{country}|{pair}")

    # N4 -- abbreviation-robust name key
    n4 = _name4(nm_tokens)
    if n4:
        keys.append(f"N4|{country}|{n4}")

    # GO -- geo-only broad fallback
    if geo_code:
        keys.append(f"GO|{country}|{geo_code}")

    return keys


# =====================================================================
# 2. INDEX BUILD
# =====================================================================

def _index_dataframe(df: pd.DataFrame, index: Dict[str, List[str]]) -> None:
    """Add preprocessed records from df into the inverted index in-place."""
    for _, row in df.iterrows():
        eid = str(row["entity_id"])
        for key in generate_blocking_keys(row):
            index[key].append(eid)


# =====================================================================
# 3. CANDIDATE RETRIEVAL
# =====================================================================

def retrieve_candidates(
    s1_row: pd.Series,
    index: Dict[str, List[str]],
    max_block_size: int = MAX_BLOCK_SIZE,
    max_cands: int = MAX_CANDS_PER_S1,
) -> Set[str]:
    """Retrieve S2/S3 candidates for a single preprocessed S1 row."""
    candidates: Set[str] = set()
    s1_id = str(s1_row["entity_id"])

    for key in generate_blocking_keys(s1_row):
        bucket = index.get(key, [])
        if len(bucket) > max_block_size:
            continue            # too common -- not discriminating
        for eid in bucket:
            if eid != s1_id:
                candidates.add(eid)
        if len(candidates) >= max_cands:
            break

    return candidates


# =====================================================================
# 4. FULL PIPELINE
# =====================================================================

def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    max_block_size: int = MAX_BLOCK_SIZE,
    max_cands_per_s1: int = MAX_CANDS_PER_S1,
    chunksize: int = CHUNKSIZE,
) -> None:
    """End-to-end blocking: preprocess -> index S2/S3 -> retrieve for S1.

    Writes candidate_pairs.tsv incrementally to output_path.
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Step 1: Preprocess and index S2
    log.info("Building index from S2: %s", s2_path)
    index: Dict[str, List[str]] = defaultdict(list)
    for chunk_no, chunk in enumerate(
        pd.read_csv(s2_path, sep="\t", chunksize=chunksize, dtype=str), 1
    ):
        proc = preprocess_dataframe(chunk)
        _index_dataframe(proc, index)
        log.info("  S2 chunk %d done  (index keys: %d)", chunk_no, len(index))

    # Step 2: Preprocess and index S3
    log.info("Building index from S3: %s", s3_path)
    for chunk_no, chunk in enumerate(
        pd.read_csv(s3_path, sep="\t", chunksize=chunksize, dtype=str), 1
    ):
        proc = preprocess_dataframe(chunk)
        _index_dataframe(proc, index)
        log.info("  S3 chunk %d done  (index keys: %d)", chunk_no, len(index))

    log.info("Index built in %.1fs  |  unique keys: %d", time.time() - t0, len(index))

    # Step 3: Retrieve candidates for each S1 record, write incrementally
    log.info("Retrieving candidates for S1: %s", s1_path)
    total_s1    = 0
    total_cands = 0
    first_write = True

    for chunk_no, chunk in enumerate(
        pd.read_csv(s1_path, sep="\t", chunksize=chunksize, dtype=str), 1
    ):
        proc = preprocess_dataframe(chunk)
        out_rows = []
        for _, row in proc.iterrows():
            cands     = retrieve_candidates(row, index, max_block_size, max_cands_per_s1)
            cands_str = ",".join(sorted(cands)) if cands else ""
            out_rows.append({
                "source1_entity_id":    str(row["entity_id"]),
                "candidate_entity_ids": cands_str,
            })
            total_cands += len(cands)
        total_s1 += len(proc)

        out_df = pd.DataFrame(out_rows, columns=["source1_entity_id", "candidate_entity_ids"])
        out_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
            encoding="utf-8",
        )
        first_write = False

        log.info(
            "  S1 chunk %d done  |  entities: %d  |  avg cands: %.1f",
            chunk_no, total_s1, total_cands / max(total_s1, 1),
        )

    elapsed = time.time() - t0
    log.info(
        "Blocking complete in %.1fs  |  S1: %d  |  total cands: %d  |  avg: %.1f  |  -> %s",
        elapsed, total_s1, total_cands,
        total_cands / max(total_s1, 1), output_path,
    )


# =====================================================================
# 5. RECALL EVALUATOR
# =====================================================================

def evaluate_blocking_recall(
    candidate_path: str,
    ground_truth_path: str,
    sample_n: Optional[int] = None,
) -> Dict[str, float]:
    """Compute pair-level blocking recall against ground truth.

    Returns:
        recall             -- fraction of true pairs captured
        avg_cands_per_s1   -- mean candidate set size
        true_pairs_found   -- raw count of true pairs captured
        total_true_pairs   -- total ground-truth pairs
    """
    gt    = pd.read_csv(ground_truth_path, sep="\t", dtype=str)
    cands = pd.read_csv(candidate_path,    sep="\t", dtype=str)
    cands["candidate_entity_ids"] = cands["candidate_entity_ids"].fillna("")

    if sample_n:
        gt = gt.head(sample_n)

    cands_map: Dict[str, Set[str]] = {}
    for _, row in cands.iterrows():
        s1id = str(row["source1_entity_id"])
        ids  = set(str(row["candidate_entity_ids"]).split(",")) - {""}
        cands_map[s1id] = ids

    hit         = 0
    total_true  = 0
    total_cands = 0

    for _, row in gt.iterrows():
        s1id      = str(row["source1_entity_id"])
        true_str  = str(row.get("matched_entity_ids", "") or "")
        true_ids  = set(true_str.split(",")) - {""}
        predicted = cands_map.get(s1id, set())

        hit         += len(true_ids & predicted)
        total_true  += len(true_ids)
        total_cands += len(predicted)

    recall    = hit / max(total_true, 1)
    avg_cands = total_cands / max(len(gt), 1)

    log.info(
        "Recall: %.4f  |  Avg cands/S1: %.1f  |  True pairs: %d / %d",
        recall, avg_cands, hit, total_true,
    )

    return {
        "recall":           recall,
        "avg_cands_per_s1": avg_cands,
        "true_pairs_found": hit,
        "total_true_pairs": total_true,
    }


# =====================================================================
# 6. CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2: Blocking Layer -- generate candidate pairs"
    )
    sub = parser.add_subparsers(dest="command")

    pb = sub.add_parser("block", help="Run blocking and write candidate_pairs.tsv")
    pb.add_argument("--s1",             required=True)
    pb.add_argument("--s2",             required=True)
    pb.add_argument("--s3",             required=True)
    pb.add_argument("--output",         required=True)
    pb.add_argument("--max-block-size", type=int, default=MAX_BLOCK_SIZE)
    pb.add_argument("--max-cands",      type=int, default=MAX_CANDS_PER_S1)
    pb.add_argument("--chunksize",      type=int, default=CHUNKSIZE)

    pe = sub.add_parser("eval", help="Evaluate blocking recall vs ground truth")
    pe.add_argument("--candidates",   required=True)
    pe.add_argument("--ground-truth", required=True)
    pe.add_argument("--sample",       type=int, default=0)

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.command == "block":
        run_blocking(
            s1_path=args.s1,
            s2_path=args.s2,
            s3_path=args.s3,
            output_path=args.output,
            max_block_size=args.max_block_size,
            max_cands_per_s1=args.max_cands,
            chunksize=args.chunksize,
        )

    elif args.command == "eval":
        metrics = evaluate_blocking_recall(
            candidate_path=args.candidates,
            ground_truth_path=args.ground_truth,
            sample_n=args.sample or None,
        )
        print()
        print("=== Blocking Recall Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()
