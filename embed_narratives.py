"""Embed DenseNarrativeSection for every row with MiniLM (GPU), normalized.
Saves narr_emb.npy aligned to the CSV row index so cos(i,j) = emb[i] @ emb[j].
Reused by the autoresearch confirmer and the full-dataset apply step."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import make_timeline as mt

MODEL = "all-MiniLM-L6-v2"
OUT = "narr_emb.npy"


def embed_all(src=mt.SRC, out=OUT):
    df = pd.read_csv(src)
    texts = df[mt._NARR].fillna("").astype(str).tolist()
    dev = "cuda"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    m = SentenceTransformer(MODEL, device=dev)
    emb = m.encode(texts, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False).astype("float32")
    np.save(out, emb)
    print(f"saved {out} {emb.shape} (device={dev})")
    return emb


if __name__ == "__main__":
    embed_all()
