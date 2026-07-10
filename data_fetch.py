"""
data_fetch.py — resolve + load the UAP dataset artifacts from anywhere.

The two bulk datasets (parsed_files_distance_embeds, final_ufoseti_dataset)
are too large for git (the .h5 originals are gitignored; the bigger one is
1.8 GB). They are published as zstd parquet (~175 MB / ~19 MB) in a Hugging
Face dataset repo and fetched on demand with local caching, so any deploy
(HF Space, Docker, Railway, remote cron) gets them without bundling.

Resolution order for a dataset name:
    1. An explicit path that exists (absolute or cwd-relative), as passed.
    2. `$UAP_DATASET_DIR/<name>.h5` or `.parquet`
    3. `<repo root>/<name>.h5` or `.parquet` (local dev keeps working)
    4. `hf_hub_download` from `$UAP_HF_DATASET_REPO`
       (default: UFOSINT/uap-datasets, repo_type=dataset) — cached in
       ~/.cache/huggingface, so the download happens once per machine.

Private-repo access needs a token: `hf auth login` locally, or set
`HF_TOKEN` in the deploy's env/secrets.

Usage:
    from data_fetch import load_uap_dataset, get_dataset_path

    df = load_uap_dataset("parsed_files_distance_embeds")   # DataFrame
    p  = get_dataset_path("final_ufoseti_dataset")          # local Path
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
_HF_REPO_DEFAULT = "UFOSINT/uap-datasets"

KNOWN_DATASETS = {"parsed_files_distance_embeds", "final_ufoseti_dataset"}


def _strip_ext(name: str) -> str:
    for ext in (".h5", ".parquet"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def get_dataset_path(name_or_path: str) -> Path:
    """Return a local path for a dataset, downloading from HF Hub if needed."""
    # 1. Caller passed a real path (uploaded file, env-provided absolute path…)
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p

    stem = _strip_ext(Path(name_or_path).name)

    # 2./3. Local candidates — prefer h5 (original) so local dev sees no change.
    search_dirs = []
    if os.environ.get("UAP_DATASET_DIR"):
        search_dirs.append(Path(os.environ["UAP_DATASET_DIR"]))
    search_dirs.append(_REPO_ROOT)
    for d in search_dirs:
        for ext in (".h5", ".parquet"):
            cand = d / f"{stem}{ext}"
            if cand.is_file():
                return cand

    # 4. HF Hub (parquet only — that's what the dataset repo publishes).
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"Dataset '{stem}' not found locally and huggingface_hub is not "
            "installed to fetch it. `pip install huggingface_hub` or place "
            f"{stem}.h5/.parquet in the repo root."
        ) from e

    repo_id = os.environ.get("UAP_HF_DATASET_REPO", _HF_REPO_DEFAULT)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{stem}.parquet",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN") or None,  # None → cached login
        )
    )


def load_uap_dataset(name_or_path: str, key: str = "df") -> pd.DataFrame:
    """Load a dataset by name or path, whichever format is available.

    Drop-in replacement for the `pd.read_hdf(path, key='df')` helpers that
    were duplicated across the Streamlit pages. Parquet stores the embedding
    vectors as list<float> — they are converted back to float32 ndarrays so
    downstream `np.stack(df['embeddings'])` code behaves identically.
    """
    path = get_dataset_path(name_or_path)
    if path.suffix == ".h5":
        return pd.read_hdf(path, key=key)
    df = pd.read_parquet(path)
    for col in df.columns:
        sample = df[col].dropna()
        if len(sample) and isinstance(sample.iloc[0], (list, np.ndarray)):
            df[col] = df[col].apply(
                lambda v: np.asarray(v, dtype=np.float32) if v is not None else None
            )
    return df
