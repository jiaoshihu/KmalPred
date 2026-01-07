#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import hashlib
import json
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def ensure_list(x: Any) -> List[int]:
    """Robustly convert a cell into list[int]."""
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, (int, float)) and pd.notna(x):
        return [int(x)]
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [int(t) for t in v]
            if isinstance(v, (int, float)):
                return [int(v)]
        except Exception:
            return []
    return []


def stable_seq_key(accession: str, seq: str) -> str:
    """Unique key to avoid overwriting when same accession has different sequences."""
    h = hashlib.sha1(seq.encode("utf-8")).hexdigest()[:10]
    return f"{accession}|{h}"


def clamp_window(center_1based: int, L: int, half: int) -> Tuple[int, int]:
    """Return clamped [start, end] in 1-based inclusive coordinates."""
    start = max(1, center_1based - half)
    end = min(L, center_1based + half)
    return start, end


def read_sites_from_excel(
    excel_path: str,
    sheet_name: str,
    col_accession: str = "Uniprot Accession",
    col_site: str = "Position",
    col_sequence: str = "Sequence",
) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    df = df.rename(
        columns={
            col_accession: "Accession",
            col_site: "Site",
            col_sequence: "Sequence",
        }
    )

    df = df.dropna(subset=["Accession", "Sequence", "Site"]).copy()
    df["Accession"] = df["Accession"].astype(str)
    df["Sequence"] = df["Sequence"].astype(str)
    df["Site"] = df["Site"].astype(int)

    # group to sites list per (Accession, Sequence)
    agg = (
        df.groupby(["Accession", "Sequence"])["Site"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .reset_index()
    )
    return agg


def collect_negatives(
    agg_df: pd.DataFrame,
    win: int = 35,
    enforce_K: bool = True,
    require_full_window: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build negative samples: all lysine positions NOT in positive sites.

    Output sample format (consistent with positive builder):
      {
        "protein_key": "<accession>|<sha1prefix>",
        "accession": "<accession>",
        "center": <1-based lys position>,
        "start": <1-based inclusive>,
        "end": <1-based inclusive>,
        "seq_len": <L>,
        "is_full_window": <bool>,
        "label": 0
      }
    """
    half = win // 2
    samples: List[Dict[str, Any]] = []

    for _, r in agg_df.iterrows():
        acc = str(r["Accession"])
        seq = str(r["Sequence"])
        L = len(seq)
        protein_key = stable_seq_key(acc, seq)

        pos_sites = r["Site"]
        pos_sites = ensure_list(pos_sites) if not isinstance(pos_sites, list) else [int(x) for x in pos_sites]
        pos_set = set(int(x) for x in pos_sites)

        # find all lysines in sequence (1-based)
        lys_positions = [i + 1 for i, aa in enumerate(seq) if aa.upper() == "K"]

        for p in lys_positions:
            if p in pos_set:
                continue

            if enforce_K and seq[p - 1].upper() != "K":
                # normally redundant (we already filtered by 'K'), but keep safe
                continue

            s, e = clamp_window(p, L, half)
            length = e - s + 1
            is_full = (length == win)

            if require_full_window and not is_full:
                continue

            samples.append(
                {
                    "protein_key": protein_key,
                    "accession": acc,
                    "center": p,
                    "start": s,
                    "end": e,
                    "seq_len": L,
                    "is_full_window": is_full,
                    "label": 0,
                }
            )

    return samples


def save_samples(samples: List[Dict[str, Any]], out_path: str):
    suffix = Path(out_path).suffix.lower()
    if suffix == ".pkl":
        with open(out_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif suffix == ".jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for obj in samples:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif suffix == ".csv":
        pd.DataFrame(samples).to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .pkl / .jsonl / .csv")


def load_positive_count(pos_path: str) -> int:
    """Optional: load positives (.pkl/.jsonl/.csv) and return count for matching."""
    p = Path(pos_path)
    suf = p.suffix.lower()
    if suf == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
        return len(data)
    if suf == ".jsonl":
        n = 0
        with open(p, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n
    if suf == ".csv":
        return len(pd.read_csv(p))
    raise ValueError("pos_path must be .pkl/.jsonl/.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Build negative (non-modified K) window indices from an Excel file.")
    p.add_argument("--excel", type=str, required=True, help="Input Excel path (e.g., Train.xlsx).")
    p.add_argument("--sheet", type=str, default="Sheet1", help="Excel sheet name.")
    p.add_argument("--out", type=str, required=True, help="Output path: .pkl / .jsonl / .csv")
    p.add_argument("--win", type=int, default=35, help="Window length (default: 35).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--shuffle", action="store_true", help="Shuffle before saving.")
    p.add_argument("--no_enforce_k", action="store_true", help="Do not enforce center residue is 'K'.")
    p.add_argument(
        "--keep_short_windows",
        action="store_true",
        help="Keep terminal lysines with shorter windows (no trimming).",
    )
    # Optional: make balanced negative set by matching positive count
    p.add_argument(
        "--match_positives",
        type=str,
        default=None,
        help="Path to positive samples (.pkl/.jsonl/.csv). If set, randomly sample same #negatives.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg = read_sites_from_excel(args.excel, args.sheet)

    neg_samples = collect_negatives(
        agg_df=agg,
        win=args.win,
        enforce_K=(not args.no_enforce_k),
        require_full_window=(not args.keep_short_windows),
    )

    if args.shuffle or args.match_positives:
        random.seed(args.seed)
        random.shuffle(neg_samples)

    if args.match_positives is not None:
        target_n = load_positive_count(args.match_positives)
        if target_n < len(neg_samples):
            neg_samples = neg_samples[:target_n]
        # if target_n > len(neg_samples) we just keep all and warn
        print(f"[INFO] match_positives={args.match_positives} -> target_neg={target_n}, kept={len(neg_samples)}")

    save_samples(neg_samples, str(out_path))
    print(f"[OK] negatives={len(neg_samples)} saved to {out_path}")


if __name__ == "__main__":
    main()
