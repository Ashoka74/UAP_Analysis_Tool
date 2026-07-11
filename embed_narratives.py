"""Embed columns for every row with Harrier-OSS (GPU), normalized.
Saves narr_emb.npy aligned to the CSV row index so cos(i,j) = emb[i] @ emb[j].
Reused by the autoresearch confirmer and the full-dataset apply step."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import make_timeline as mt
import argparse

MODEL = "microsoft/harrier-oss-v1-270m"
OUT = "narr_emb.npy"


def get_embed_model(device):
    import torch
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if device == "cuda":
        model_kwargs["device_map"] = "cuda"
    m = SentenceTransformer(MODEL, model_kwargs=model_kwargs)
    if device != "cuda":
        m.to(device)
    m.max_seq_length = 512
    return m


def embed_all(src=mt.SRC, out=OUT, columns=None):
    if columns is None:
        columns = [mt._NARR]

    df = pd.read_csv(src)

    # Construct texts from columns
    texts = []
    for _, row in df.iterrows():
        if len(columns) == 1:
            col = columns[0]
            val = row[col] if col in row.index else ""
            val_str = str(val).strip() if pd.notna(val) else ""
            texts.append(val_str)
        else:
            parts = []
            for col in columns:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        val_str = str(val).strip()
                        if val_str:
                            # Normalize year/month/day floats if needed
                            if col in ["date_time.year", "date_time.month", "date_time.day"]:
                                try:
                                    float_val = float(val)
                                    if float_val.is_integer():
                                        val_str = str(int(float_val))
                                except ValueError:
                                    pass
                            parts.append(f"{col}: {val_str}")
            texts.append(" | ".join(parts))

    dev = "cuda"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"

    m = get_embed_model(device=dev)
    emb = m.encode_document(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    np.save(out, emb)
    print(f"saved {out} {emb.shape} (device={dev})")
    return emb


def main():
    parser = argparse.ArgumentParser(description="Embed columns for every row with Harrier-OSS (GPU), normalized.")
    parser.add_argument("--input", "-i", default=mt.SRC, help="Path to input CSV dataset")
    parser.add_argument("--output", "-o", default=OUT, help="Path to output .npy embeddings file")
    parser.add_argument(
        "--columns", "-c",
        nargs="+",
        default=[mt._NARR],
        help="Space- or comma-separated list of columns to concatenate for embedding"
    )
    args = parser.parse_args()

    # Parse columns: handle comma-separated strings
    cols = []
    for c in args.columns:
        if "," in c:
            cols.extend([x.strip() for x in c.split(",") if x.strip()])
        else:
            cols.append(c.strip())

    embed_all(src=args.input, out=args.output, columns=cols)


if __name__ == "__main__":
    main()

