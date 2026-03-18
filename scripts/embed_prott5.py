#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


def read_fasta_file(fasta_file: str) -> Dict[str, str]:
    """Read FASTA into {seq_id: sequence}."""
    fasta_dict: Dict[str, str] = {}
    seq_id = None
    seq_list: List[str] = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    fasta_dict[seq_id] = "".join(seq_list)
                seq_id = line[1:].split()[0]
                seq_list = []
            else:
                seq_list.append(line)

    if seq_id is not None:
        fasta_dict[seq_id] = "".join(seq_list)

    return fasta_dict


def _normalize_aa_sequence(seq: str) -> str:
    """
    ProtT5 推荐：
    - 全部大写
    - 将罕见/歧义氨基酸 U, Z, O, B 替换为 X
    - tokenizer 输入时每个残基之间加空格
    """
    seq = seq.strip().upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def _protT5_preprocess_for_tokenizer(seq: str) -> str:
    """Insert whitespace between residues: 'ACD' -> 'A C D'."""
    return " ".join(list(seq))


def _sliding_windows(L: int, win: int = 1000, stride: int = 900) -> List[Tuple[int, int]]:
    """Return [start, end) windows; last window aligns to the end to avoid missing tail."""
    if L <= win:
        return [(0, L)]
    starts = list(range(0, max(1, L - win + 1), stride))
    if starts[-1] + win < L:
        starts.append(L - win)
    return [(s, min(s + win, L)) for s in starts]


@torch.inference_mode()
def _encode_chunk(
    subseq: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    fp16: bool = False,
) -> torch.Tensor:
    """
    Encode one subsequence with ProtT5 encoder, return per-residue embeddings: [len(subseq), D]
    Notes:
      - We follow common ProtT5 usage: add_special_tokens=True, then slice to residue length
        (i.e., ignore EOS and any padding).
    """
    txt = _protT5_preprocess_for_tokenizer(subseq)
    ids = tokenizer(
        [txt],
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    # autocast only on CUDA
    if fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    hidden = outputs.last_hidden_state[0]  # [T, D]
    L = len(subseq)
    return hidden[:L, :].detach()  # drop EOS/pad by slicing to residue length


def _encode_sequence(
    seq: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    l_max: int = 1024,
    win: int = 1000,
    stride: int = 900,
    fp16: bool = False,
) -> np.ndarray:
    """
    Encode full sequence into per-residue embeddings [L, D].
    - If L <= l_max: one pass
    - Else: sliding windows + overlap averaging
    """
    seq = _normalize_aa_sequence(seq)
    L = len(seq)
    D = model.config.d_model

    if L <= l_max:
        emb = _encode_chunk(seq, tokenizer, model, device, fp16=fp16)
        return emb.float().cpu().numpy()  # [L, D]

    # long sequence: overlap-average
    acc = np.zeros((L, D), dtype=np.float32)
    cnt = np.zeros((L,), dtype=np.int32)

    for s, e in _sliding_windows(L, win=win, stride=stride):
        subseq = seq[s:e]
        chunk = _encode_chunk(subseq, tokenizer, model, device, fp16=fp16)  # [e-s, D]
        chunk_np = chunk.float().cpu().numpy()
        acc[s:e, :] += chunk_np
        cnt[s:e] += 1

    out = acc / cnt[:, None]
    return out.astype(np.float32)  # [L, D]


def get_plm_representation(
    sequences_dict: Dict[str, str],
    model_name_or_path: str,
    device: torch.device,
    fp16: bool = False,
    l_max: int = 1024,
    win: int = 1000,
    stride: int = 900,
) -> Dict[str, np.ndarray]:
    """
    Input: {protein_id: raw_sequence}
    Output: {protein_id: np.ndarray [L, D]} per-residue embeddings
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name_or_path)

    model.eval().to(device)
    if fp16 and device.type == "cuda":
        model.half()

    reps: Dict[str, np.ndarray] = {}
    for name, raw_seq in sequences_dict.items():
        reps[name] = _encode_sequence(
            raw_seq,
            tokenizer=tokenizer,
            model=model,
            device=device,
            l_max=l_max,
            win=win,
            stride=stride,
            fp16=fp16,
        )
    return reps


def parse_args():
    p = argparse.ArgumentParser(description="Extract ProtT5 per-residue embeddings and save as pickle.")
    p.add_argument("--fasta", type=str, required=True, help="Input FASTA file.")
    p.add_argument("--out", type=str, required=True, help="Output pickle path, e.g. reps.pkl")
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="Rostlab/prot_t5_xl_uniref50",
        help="HuggingFace model name or local path.",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--fp16", action="store_true", help="Use fp16 inference on GPU to save VRAM.")
    p.add_argument("--l_max", type=int, default=1024, help="Max residue length for single-pass encoding.")
    p.add_argument("--win", type=int, default=1000, help="Sliding window length for long sequences.")
    p.add_argument("--stride", type=int, default=900, help="Sliding window stride for long sequences.")
    return p.parse_args()


def main():
    args = parse_args()
    fasta_path = Path(args.fasta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    sequences_data = read_fasta_file(str(fasta_path))
    reps = get_plm_representation(
        sequences_data,
        model_name_or_path=args.model_name_or_path,
        device=device,
        fp16=args.fp16,
        l_max=args.l_max,
        win=args.win,
        stride=args.stride,
    )

    with open(out_path, "wb") as f:
        pickle.dump(reps, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved {len(reps)} proteins to: {out_path}")


if __name__ == "__main__":
    main()
