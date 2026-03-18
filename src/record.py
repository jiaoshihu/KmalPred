from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_metrics_table(rows: list[dict], out_path: str):
    """
    rows: list of dict metrics per epoch (already includes losses)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if out.suffix.lower() == ".xlsx":
        df.to_excel(out, index=False)
    else:
        df.to_csv(out, index=False)


