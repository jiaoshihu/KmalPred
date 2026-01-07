from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _regions_to_entries(regions_obj: Any, default_label: int) -> List[Tuple[str, int, int, int]]:
    """
    Convert region objects to entries: (pid, start, end, label)
    Supports:
      1) list of {pid: (start,end)}
      2) list of dict with keys start/end and optional protein_key/accession/pid/label
    """
    entries: List[Tuple[str, int, int, int]] = []

    if isinstance(regions_obj, list) and len(regions_obj) > 0 and isinstance(regions_obj[0], dict):
        # case 1: list[{pid:(s,e)}]
        # or case 2: list[{start:end:...}]
        for item in regions_obj:
            if len(item) == 1 and list(item.values())[0] and isinstance(list(item.values())[0], tuple):
                # old style: {pid:(s,e)}
                for pid, (s, e) in item.items():
                    entries.append((str(pid), int(s), int(e), int(default_label)))
            else:
                # new style
                pid = item.get("protein_key") or item.get("pid") or item.get("accession")
                if pid is None:
                    raise ValueError("Region dict missing protein id field (protein_key/pid/accession).")
                s = int(item["start"])
                e = int(item["end"])
                lab = int(item.get("label", default_label))
                entries.append((str(pid), s, e, lab))
        return entries

    raise ValueError("Unsupported region file format.")


class ProteinRegionDataset(Dataset):
    def __init__(
        self,
        emb_file: str,
        pos_region_file: str,
        neg_region_file: str,
        max_len: int = 35,
        neg_limit: int | None = None,
        shuffle_before_limit: bool = False,
        seed: int = 50,
    ):
        self.embeddings: Dict[str, np.ndarray] = load_pkl(emb_file)

        pos_regions = load_pkl(pos_region_file)
        neg_regions = load_pkl(neg_region_file)

        pos_entries = _regions_to_entries(pos_regions, default_label=1)
        neg_entries = _regions_to_entries(neg_regions, default_label=0)

        # limit negatives
        if neg_limit is not None and len(neg_entries) > neg_limit:
            if shuffle_before_limit:
                g = torch.Generator().manual_seed(seed)
                idx = torch.randperm(len(neg_entries), generator=g).tolist()
                neg_entries = [neg_entries[i] for i in idx[:neg_limit]]
            else:
                neg_entries = neg_entries[:neg_limit]

        self.entries = pos_entries + neg_entries
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        pid, start, end, label = self.entries[idx]
        if pid not in self.embeddings:
            raise KeyError(f"Protein id '{pid}' not found in embeddings file.")

        emb = self.embeddings[pid]  # [L, D] numpy
        start_idx = start - 1
        end_idx = end  # python slice is exclusive end -> keep end residue

        region = emb[start_idx:end_idx]  # [len_region, D]
        region = torch.tensor(region, dtype=torch.float32)

        L, D = region.shape
        if L >= self.max_len:
            region = region[: self.max_len]
            mask = torch.ones(self.max_len, dtype=torch.bool)
        else:
            pad = torch.zeros((self.max_len - L, D), dtype=torch.float32)
            region = torch.cat([region, pad], dim=0)
            mask = torch.cat([torch.ones(L, dtype=torch.bool), torch.zeros(self.max_len - L, dtype=torch.bool)])

        y = torch.tensor(label, dtype=torch.float32)
        return region, y, mask, pid, (start, end)
