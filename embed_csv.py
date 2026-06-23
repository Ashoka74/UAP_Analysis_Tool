"""
embed_csv.py — batch-embed a CSV column and save to HDF5

Usage:
    uv run python embed_csv.py --input data.csv --column description
    uv run python embed_csv.py --input data.csv --column description --output out.h5 --batch-size 128 --prompt web_search_query
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_ID = "microsoft/harrier-oss-v1-0.6b"


def load_model(device: str) -> SentenceTransformer:
    print(f"Loading {MODEL_ID} on {device}…")
    t0 = time.time()
    model_kwargs = {"dtype": "auto"}
    if device == "cuda":
        model_kwargs["device_map"] = "cuda"  # load directly into VRAM, skip CPU copy
    model = SentenceTransformer(MODEL_ID, model_kwargs=model_kwargs)
    if device != "cuda":
        model.to(device)
    print(f"Model ready in {time.time() - t0:.1f}s")
    return model


def embed(model: SentenceTransformer, texts: list[str], batch_size: int,
          prompt_name: str | None) -> np.ndarray:
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    kwargs = {"prompt_name": prompt_name} if prompt_name else {}
    all_embs = []
    for batch in tqdm(batches, desc="Encoding", unit="batch"):
        with torch.no_grad():
            all_embs.append(model.encode(batch, show_progress_bar=False, **kwargs))
    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser(description="Embed a CSV column with harrier-oss-v1-0.6b")
    parser.add_argument("--input",      required=True,  help="Input CSV file")
    parser.add_argument("--column",     required=True,  help="Column to embed")
    parser.add_argument("--output",     default=None,   help="Output .h5 file (default: <input>_embeddings.h5)")
    parser.add_argument("--key",        default="df",   help="HDF5 key (default: df)")
    parser.add_argument("--batch-size", default=256,    type=int, help="Batch size (default: 256)")
    parser.add_argument("--prompt",     default=None,   choices=["web_search_query"],
                        help="Prompt name for query encoding (omit for documents)")
    args = parser.parse_args()

    # Output path
    out_path = args.output or args.input.rsplit(".", 1)[0] + "_embeddings.h5"

    # Load CSV
    print(f"Reading {args.input}…")
    df = pd.read_csv(args.input)
    if args.column not in df.columns:
        print(f"ERROR: column '{args.column}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    print(f"{len(df):,} rows loaded. Embedding column: '{args.column}'")

    texts = df[args.column].fillna("").astype(str).tolist()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(device)

    # Embed
    t0 = time.time()
    embeddings = embed(model, texts, batch_size=args.batch_size, prompt_name=args.prompt)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — shape {embeddings.shape} "
          f"({len(texts)/elapsed:.0f} texts/s)")

    # Save
    df["embeddings"] = embeddings.tolist()
    df.to_hdf(out_path, key=args.key, mode="w")
    print(f"Saved → {out_path}  (key='{args.key}')")


if __name__ == "__main__":
    main()
