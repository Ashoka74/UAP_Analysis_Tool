# Standard library
import concurrent.futures
import json
import logging
import os
import pickle
import textwrap
import time

# Third-party core
import numpy as np
import pandas as pd

# ML / NLP
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from statsmodels.api import stats
from statsmodels.graphics.mosaicplot import mosaic
import xgboost as xgb
from xgboost import plot_importance
from Levenshtein import distance

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import squarify

# Deep learning / embeddings
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim, pairwise_cos_sim

# UMAP + HDBSCAN
import umap
import fast_hdbscan as hdbscan

# Web / API
from requests.exceptions import HTTPError
from openai import OpenAI
from stqdm import stqdm
stqdm.pandas()
import streamlit as st

torch.set_float32_matmul_precision("medium")

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _model_kwargs = {"torch_dtype": torch.bfloat16}
        if _device == "cuda":
            _model_kwargs["device_map"] = "cuda"
        _embed_model = SentenceTransformer(
            "microsoft/harrier-oss-v1-270m", model_kwargs=_model_kwargs
        )
        if _device != "cuda":
            _embed_model.to(_device)
        # Harrier's prompts load automatically from its config; verify with:
        #   print(_embed_model.prompts)
        # Set max_seq_length explicitly — many models silently truncate otherwise.
        _embed_model.max_seq_length = 512
    return _embed_model


# GPU acceleration — cuML (UMAP/HDBSCAN/KMeans/PCA) and cuVS (ANN search)
# Falls back to CPU libraries if RAPIDS is not installed.
def _check_gpu() -> bool:
    try:
        import cupy as cp
        cp.zeros(1)
        return True
    except Exception:
        return False

_GPU_AVAILABLE = _check_gpu()

_CUML_AVAILABLE = False
if _GPU_AVAILABLE:
    try:
        from cuml.manifold import UMAP as _cuUMAP
        from cuml.cluster.hdbscan import HDBSCAN as _cuHDBSCAN
        from cuml.cluster import KMeans as _cuKMeans
        from cuml.decomposition import PCA as _cuPCA
        import rmm
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2**31)  # 2 GB pool
        _CUML_AVAILABLE = True
        logging.getLogger(__name__).info("cuML detected — GPU acceleration enabled for UMAP/HDBSCAN/KMeans/PCA")
    except ImportError:
        logging.getLogger(__name__).warning("cuML not found — install with: uv add --extra-index-url=https://pypi.nvidia.com cuml-cu12")

_CUVS_AVAILABLE = False
if _GPU_AVAILABLE:
    try:
        from cuvs.neighbors import cagra as _cagra
        _CUVS_AVAILABLE = True
        logging.getLogger(__name__).info("cuVS detected — GPU ANN search enabled")
    except ImportError:
        logging.getLogger(__name__).warning("cuVS not found — install with: uv add --extra-index-url=https://pypi.nvidia.com cuvs-cu12")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UAPAnalyzer:
    """
    A class for analyzing and clustering textual data within a pandas DataFrame using 
    Natural Language Processing (NLP) techniques and machine learning models.
    
    Attributes:
        data (pd.DataFrame): The dataset containing textual data for analysis.
        column (str): The name of the column in the DataFrame to be analyzed.
        embeddings (np.ndarray): The vector representations of textual data.
        reduced_embeddings (np.ndarray): The dimensionality-reduced embeddings.
        cluster_labels (np.ndarray): The labels assigned to each data point after clustering.
        cluster_terms (list): The list of terms associated with each cluster.
        tfidf_matrix (sparse matrix): The Term Frequency-Inverse Document Frequency (TF-IDF) matrix.
        models (dict): A dictionary to store trained machine learning models.
        evaluations (dict): A dictionary to store evaluation results of models.
        data_nums (pd.DataFrame): The DataFrame with numerical encoding of categorical data.
    """

    def __init__(self, data, column, has_embeddings=False):
        """
        Initializes the UAPAnalyzer with a dataset and a specified column for analysis.
        
        Args:
            data (pd.DataFrame): The dataset for analysis.
            column (str): The column within the dataset to analyze.
        """
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
        assert column in data.columns, f"Column '{column}' not found in DataFrame"
        self.has_embeddings = has_embeddings        
        self.data = data
        self.column = column
        self.embeddings = None
        self.reduced_embeddings = None      # 2-D, for visualisation
        self.cluster_embeddings = None      # higher-dim, for HDBSCAN (optional)
        self.cluster_labels = None
        self.cluster_names = None
        self.cluster_terms = None 
        self.cluster_terms_embeddings = None
        self.tfidf_matrix = None
        self.models = {}  # To store trained models
        self.evaluations = {}  # To store evaluation results
        self.data_nums = None  # Encoded numerical data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.preds = None
        self.new_dataset = None
        #self.cluster_names_ = pd.DataFrame()

        logging.info("UAPAnalyzer initialized")

    def preprocess_data(self, trim=False, has_embeddings=False, top_n=32,):
        """
        Preprocesses the data by optionally trimming the dataset to include only the top N labels and extracting embeddings.
        
        Args:
            trim (bool): Whether to trim the dataset to include only the top N labels.
            top_n (int): The number of top labels to retain if trimming is enabled.
        """
        logging.info("Preprocessing data")

        # if trim is True
        if trim:
            # Identify the top labels based on value counts
            top_labels = self.data[self.column].value_counts().nlargest(top_n).index.tolist()
            # Revise the column data, setting values to 'Other' if they are not in the top labels
            self.data[f'{self.column}_revised'] = np.where(self.data[self.column].isin(top_labels), self.data[self.column], 'Other')
        # Convert the column data to string type before passing to _extract_embeddings
        # This is useful especially if the data type of the column is not originally string
        string_data = self.data[f'{self.column}'].astype(str)
        # Extract embeddings from the revised and string-converted column data
        if has_embeddings:
            self.embeddings = self.data['embeddings'].to_list()
        else:
            model = get_embed_model()
            pool = model.start_multi_process_pool(target_devices=[_device])
            try:
                self.embeddings = self._extract_embeddings(string_data, pool=pool)
            finally:
                model.stop_multi_process_pool(pool)
        logging.info("Data preprocessing complete")


    def _extract_embeddings(self, data_column, pool=None):
        """
        Extracts embeddings from the given data column.

        Args:
            data_column (pd.Series): The column from which to extract embeddings.
            pool: Optional multi-process pool from model.start_multi_process_pool().

        Returns:
            np.ndarray: float32 embeddings of shape (n_samples, dim).
        """
        logging.info("Extracting embeddings")
        embs = get_embed_model().encode_document(
            data_column.tolist(),
            show_progress_bar=True,
            batch_size=256,
            pool=pool,
            chunk_size=5000,
        )
        return np.array(embs, dtype=np.float32)

    def reduce_dimensionality(self, method='UMAP', n_components=2,
                               pca_preprocess=True, pca_components=50, **kwargs):
        """
        Reduce the embeddings to `n_components` dims for VISUALISATION and store
        the result in ``self.reduced_embeddings`` (typically 2-D for plotting).

        When a GPU is available and cuML is installed, UMAP and PCA run on the GPU
        (60-300x faster than CPU).  An optional PCA pre-reduction step compresses
        high-dimensional embeddings (e.g. 768-d) to `pca_components` before UMAP,
        which dramatically speeds up the UMAP graph construction.

        Args:
            method (str): 'UMAP' or 'PCA'.
            n_components (int): Target dimensionality.
            pca_preprocess (bool): Apply PCA before UMAP when embeddings are wide.
            pca_components (int): PCA target dims used only as a pre-reduction step.
            **kwargs: Forwarded to the UMAP/PCA constructor.
        """
        logging.info(f"Reducing dimensionality (viz) using {method}")
        self.reduced_embeddings = self._reduce_embeddings(
            method=method, n_components=n_components,
            pca_preprocess=pca_preprocess, pca_components=pca_components, **kwargs,
        )
        logging.info(f"Viz embedding: {self.reduced_embeddings.shape[1]}-D "
                     f"({'GPU' if _CUML_AVAILABLE else 'CPU'})")

    def reduce_for_clustering(self, method='UMAP', n_components=10,
                              pca_preprocess=True, pca_components=50,
                              min_dist=0.0, **kwargs):
        """
        Reduce the embeddings to a higher-dimensional CLUSTERING space (default
        10-D) stored in ``self.cluster_embeddings``. When set, :meth:`cluster_data`
        runs HDBSCAN on this instead of the 2-D viz embedding.

        Clustering on a ~10-D UMAP rather than the 2-D scatter is the practice the
        UMAP authors recommend: 2-D over-compresses the manifold and produces
        spurious density merges/splits, whereas a higher-dim embedding preserves
        topology for cleaner clusters. It also makes the cuML GPU HDBSCAN path
        worthwhile — the GPU gate in :meth:`cluster_data` wants dim > 5.
        ``min_dist`` defaults to 0.0, which packs points tightly (better for
        density-based clustering than the viz default of 0.1).

        Note: because clusters are found in N-D but plotted in the separate 2-D
        embedding, points that look adjacent in the scatter may belong to
        different clusters, and vice-versa.
        """
        logging.info(f"Reducing dimensionality (clustering) using {method}")
        self.cluster_embeddings = self._reduce_embeddings(
            method=method, n_components=n_components,
            pca_preprocess=pca_preprocess, pca_components=pca_components,
            min_dist=min_dist, **kwargs,
        )
        logging.info(f"Clustering embedding: {self.cluster_embeddings.shape[1]}-D "
                     f"({'GPU' if _CUML_AVAILABLE else 'CPU'})")

    def _reduce_embeddings(self, method='UMAP', n_components=2,
                           pca_preprocess=True, pca_components=50, **kwargs):
        """Shared UMAP/PCA reduction core. Returns a float32 ndarray (it does NOT
        store the result). GPU-accelerated via cuML when available. Used by both
        :meth:`reduce_dimensionality` (viz) and :meth:`reduce_for_clustering`."""
        embeddings = np.array(self.embeddings, dtype=np.float32)

        # PCA pre-reduction: collapse 768→50 before UMAP to cut cost ~50x
        if pca_preprocess and method == 'UMAP' and embeddings.shape[1] > pca_components:
            n_pca = min(pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
            if _CUML_AVAILABLE:
                pre_pca = _cuPCA(n_components=n_pca)
            else:
                pre_pca = PCA(n_components=n_pca)
            embeddings = pre_pca.fit_transform(embeddings)
            if hasattr(embeddings, 'get'):
                embeddings = embeddings.get()
            embeddings = np.array(embeddings, dtype=np.float32)
            logging.info(f"PCA pre-reduction to {n_pca} dims complete")

        if method == 'UMAP':
            if _CUML_AVAILABLE:
                reducer = _cuUMAP(n_components=n_components, **kwargs)
            else:
                reducer = umap.UMAP(n_components=n_components, **kwargs)
        elif method == 'PCA':
            if _CUML_AVAILABLE:
                reducer = _cuPCA(n_components=n_components)
            else:
                reducer = PCA(n_components=n_components)
        else:
            raise ValueError("Unsupported dimensionality reduction method")

        result = reducer.fit_transform(embeddings)
        if hasattr(result, 'get'):      # CuPy → NumPy
            result = result.get()
        return np.array(result, dtype=np.float32)

    def cluster_data(self, method='HDBSCAN', **kwargs):
        """
        Clusters the reduced dimensionality data using the specified clustering method.

        When cuML is available and the dataset is large enough (n*d > 50k), HDBSCAN
        and KMeans run on GPU.  For small/low-dim inputs the CPU implementation is used
        because GPU kernel launch overhead dominates.

        Args:
            method (str): 'HDBSCAN' or 'KMeans'.
            **kwargs: Forwarded to the clusterer constructor.
        """
        logging.info(f"Clustering data using {method}")
        # Cluster on the dedicated higher-dim clustering embedding when present
        # (better cluster quality + viable cuML GPU path); otherwise fall back to
        # the 2-D viz embedding (legacy behaviour).
        source = self.cluster_embeddings if self.cluster_embeddings is not None else self.reduced_embeddings
        embeddings = np.array(source, dtype=np.float32)

        # cuML HDBSCAN/KMeans only beats CPU when both N and dim are large.
        # Benchmarks: 50k×2D → GPU 40× slower; 300k×50D → GPU 13× faster.
        # Crossover is ~100k rows AND dim > 5.
        n, d = embeddings.shape
        _use_gpu_cluster = _CUML_AVAILABLE and (n > 100_000 and d > 5)

        if method == 'HDBSCAN':
            clusterer = _cuHDBSCAN(**kwargs) if _use_gpu_cluster else hdbscan.HDBSCAN(**kwargs)
        elif method == 'KMeans':
            clusterer = _cuKMeans(**kwargs) if _use_gpu_cluster else KMeans(**kwargs)
        else:
            raise ValueError("Unsupported clustering method")

        clusterer.fit(embeddings)
        labels = clusterer.labels_
        if hasattr(labels, 'get'):      # CuPy → NumPy
            labels = labels.get()
        self.cluster_labels = np.array(labels)
        logging.info(f"Data clustering complete using {method} ({'GPU' if _use_gpu_cluster else 'CPU'})")

    # ------------------------------------------------------------------
    # GPU-accelerated ANN search (cuVS CAGRA)
    # ------------------------------------------------------------------

    def build_search_index(self):
        """
        Build a GPU CAGRA ANN index over the raw embeddings for fast similarity search.

        Requires cuVS (uv add --extra-index-url=https://pypi.nvidia.com cuvs-cu12).
        Embeddings are L2-normalised before indexing so inner-product == cosine sim.
        """
        if not _CUVS_AVAILABLE:
            logging.warning("cuVS not available — search index not built")
            return
        import cupy as cp
        embs = cp.array(self.embeddings, dtype=cp.float32)
        norms = cp.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / (norms + 1e-8)
        index_params = _cagra.IndexParams(metric="inner_product")
        self._search_index = _cagra.build(index_params, embs_norm)
        self._search_embs_norm = embs_norm
        logging.info(f"CAGRA search index built on GPU ({embs.shape[0]} vectors, dim={embs.shape[1]})")

    def find_similar_reports(self, query_texts: list, k: int = 10):
        """
        Return the indices and cosine-similarity scores of the k most similar
        stored reports for each query string.

        Args:
            query_texts: List of query strings to encode and search.
            k: Number of nearest neighbours to return.

        Returns:
            distances: np.ndarray shape (len(query_texts), k) — cosine similarities.
            neighbors: np.ndarray shape (len(query_texts), k) — row indices into self.data.
        """
        if not _CUVS_AVAILABLE or not hasattr(self, '_search_index'):
            raise RuntimeError("Call build_search_index() first (requires cuVS).")
        import cupy as cp
        query_embs = get_embed_model().encode_query(query_texts, batch_size=256)
        q = cp.array(query_embs, dtype=cp.float32)
        norms = cp.linalg.norm(q, axis=1, keepdims=True)
        q_norm = q / (norms + 1e-8)
        distances, neighbors = _cagra.search(_cagra.SearchParams(), self._search_index, q_norm, k=k)
        return distances.get(), neighbors.get()

    def get_tf_idf_clusters(self, top_n=2):
        """
        Names clusters using the most frequent terms based on TF-IDF analysis.

        Args:
            top_n (int): The number of top terms to consider for naming each cluster.
        """
        logging.info("Naming clusters based on top TF-IDF terms.")

        # Ensure data has been clustered
        assert self.cluster_labels is not None, "Data has not been clustered yet."
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Fit the vectorizer to the text data and transform it into a TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(self.data[f'{self.column}'].astype(str))

        # Initialize an empty list to store the cluster terms
        self.cluster_terms = []

        for cluster_id in np.unique(self.cluster_labels):
            # Skip noise if present (-1 in HDBSCAN)
            if cluster_id == -1:
                continue

            # Find indices of documents in the current cluster
            indices = np.where(self.cluster_labels == cluster_id)[0]

            # Compute the mean TF-IDF score for each term in the cluster
            cluster_tfidf_mean = np.mean(tfidf_matrix[indices], axis=0)

            # Use the matrix directly for indexing if it does not support .toarray()
            # Ensure it's in a format that supports indexing, convert if necessary
            if hasattr(cluster_tfidf_mean, "toarray"):
                dense_mean = cluster_tfidf_mean.toarray().flatten()
            else:
                dense_mean = np.asarray(cluster_tfidf_mean).flatten()

            # Get the indices of the top_n terms
            top_n_indices = np.argsort(dense_mean)[-top_n:]

            # Get the corresponding terms for these top indices
            terms = vectorizer.get_feature_names_out()
            top_terms = [terms[i] for i in top_n_indices]

            # Join the top_n terms with a hyphen
            cluster_name = '-'.join(top_terms)

            # Append the cluster name to the list
            self.cluster_terms.append(cluster_name)

        # Convert the list of cluster terms to a categorical data type
        self.cluster_terms = pd.Categorical(self.cluster_terms)
        logging.info("Cluster naming completed.")

    def get_llm_clusters(
        self,
        sample_size: int = 30,
        max_words: int = 6,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        provider: str | None = None,
        max_chars: int = 500,
        max_clusters_per_call: int = 40,
        seed: int = 42,
    ):
        """
        Name clusters with an LLM instead of TF-IDF keyword extraction.

        For every non-noise cluster, up to ``sample_size`` report excerpts are
        sampled. ALL clusters' samples are then sent to the model in a SINGLE
        request (batched only when there are more than ``max_clusters_per_call``
        clusters) so it sees every cluster at once and can assign distinct,
        mutually non-duplicated labels of at most ``max_words`` words each.

        The result is stored in ``self.cluster_terms`` as a ``pd.Categorical``
        with one label per non-noise cluster id (sorted ascending) — the exact
        shape produced by :meth:`get_tf_idf_clusters`, so the downstream
        ``merge_similar_clusters`` / plotting / XGBoost code is unchanged.

        On any LLM failure the method falls back to TF-IDF naming, and finally
        to numeric ``"Cluster N"`` placeholders, so it never raises for an API
        problem.

        Args:
            sample_size (int): Max report excerpts sampled per cluster.
            max_words (int): Hard cap on the word count of every label.
            api_key (str | None): LLM key; falls back to ``config.API_KEY``.
            model (str): Chat model id — an OpenAI/DeepSeek key from ``_PRICING``
                or a Gemini model (e.g. ``"models/gemini-3.1-pro-preview"``).
            provider (str | None): Force ``"openai"``/``"deepseek"``/``"google"``
                (Google = Gemini); auto-detected from ``model`` when ``None``.
            max_chars (int): Per-excerpt truncation length (token-cost guard).
            max_clusters_per_call (int): Clusters per request before batching.
            seed (int): RNG seed for reproducible per-cluster sampling.
        """
        assert self.cluster_labels is not None, "Data has not been clustered yet."

        labels_arr = np.asarray(self.cluster_labels)
        # Match get_tf_idf_clusters: one name per non-noise cluster id, sorted.
        cluster_ids = [int(c) for c in np.unique(labels_arr) if c != -1]
        # Records which naming path actually produced the labels so callers can
        # surface a warning when the LLM path silently degraded:
        # "llm" | "tfidf_fallback" | "numeric_fallback" | "empty".
        self.naming_method = None
        self.naming_error = None
        if not cluster_ids:
            self.cluster_terms = pd.Categorical([])
            self.naming_method = "empty"
            logging.warning("get_llm_clusters: no non-noise clusters to name.")
            return

        texts = self.data[f'{self.column}'].astype(str).reset_index(drop=True)
        rng = np.random.default_rng(seed)

        # Sample up to `sample_size` excerpts per cluster (truncated for token cost).
        samples: dict[int, list[str]] = {}
        for cid in cluster_ids:
            idx = np.where(labels_arr == cid)[0]
            if len(idx) > sample_size:
                idx = rng.choice(idx, size=sample_size, replace=False)
            excerpts = []
            for i in idx:
                t = " ".join(texts.iloc[int(i)].split())  # collapse whitespace
                if len(t) > max_chars:
                    t = t[:max_chars] + "…"
                if t:
                    excerpts.append(t)
            samples[cid] = excerpts

        # Try the LLM; fall back to TF-IDF, then numeric, on any failure.
        names_by_id: dict[int, str] | None = None
        try:
            names_by_id = self._llm_label_clusters(
                samples, max_words=max_words, api_key=api_key, model=model,
                provider=provider, max_clusters_per_call=max_clusters_per_call,
            )
        except Exception as e:
            self.naming_error = str(e)
            logging.warning(f"get_llm_clusters: LLM labeling failed ({e}); "
                            "falling back to TF-IDF naming.")

        if names_by_id:
            ordered, used = [], set()
            for cid in cluster_ids:
                name = (names_by_id.get(cid) or "").strip()
                if not name:
                    name = f"Cluster {cid}"
                # Cap to max_words, then ensure global uniqueness ("avoid
                # duplicates") with a numeric suffix that still respects the cap.
                words = name.split()[:max_words]
                candidate, n = " ".join(words), 1
                while candidate.lower() in used:
                    n += 1
                    candidate = " ".join(words[:max(1, max_words - 1)] + [f"({n})"])
                used.add(candidate.lower())
                ordered.append(candidate)
            self.cluster_terms = pd.Categorical(ordered)
            self.naming_method = "llm"
            logging.info(f"get_llm_clusters: named {len(ordered)} clusters via {model}.")
            return

        # Fallback 1: TF-IDF keyword naming (no API needed).
        try:
            self.get_tf_idf_clusters(top_n=3)
            self.naming_method = "tfidf_fallback"
            return
        except Exception as e:
            logging.warning(f"get_llm_clusters: TF-IDF fallback failed ({e}); "
                            "using numeric names.")
        # Fallback 2: numeric placeholders.
        self.cluster_terms = pd.Categorical([f"Cluster {cid}" for cid in cluster_ids])
        self.naming_method = "numeric_fallback"

    def _llm_label_clusters(self, samples, max_words, api_key, model,
                            provider, max_clusters_per_call):
        """Map cluster id → short label via one (or a few batched) LLM call(s).

        Every cluster's samples go in a single request; batching only kicks in
        beyond ``max_clusters_per_call`` clusters, and each later batch is told
        which labels are already taken so names stay distinct across the whole
        set. Returns ``{cluster_id: label}`` (only ids the model labelled).
        """
        # Resolve provider (auto-detect from the model name when unset):
        # "openai" | "deepseek" (both via the OpenAI-compatible client) |
        # "google" (Gemini via google.generativeai), mirroring rag_search.py.
        if provider is None:
            if model.startswith("gemini") or model.startswith("models/gemini"):
                provider = "google"
            elif model in DEEPSEEK_MODELS:
                provider = "deepseek"
            else:
                provider = "openai"
        provider = provider.lower()

        # Resolve API key — fall back to the matching config.py key per provider.
        if api_key is None:
            try:
                from config import API_KEY as _OAI_KEY, GEMINI_KEY as _GEM_KEY
            except Exception:
                _OAI_KEY = _GEM_KEY = ""
            api_key = _GEM_KEY if provider == "google" else _OAI_KEY
        if not api_key:
            raise RuntimeError(
                f"No API key available for LLM cluster naming (provider={provider})."
            )

        sys_prompt = (
            "You label clusters of UAP (unidentified aerial phenomena) report "
            "excerpts. For each cluster, write ONE short, specific, human-readable "
            f"label of AT MOST {max_words} words capturing the theme that "
            "distinguishes that cluster (shapes, behaviours, locations, witnesses, "
            "effects, etc.). Do not number the labels and never reuse the same "
            "label for two clusters. Respond with ONLY a JSON object mapping each "
            "cluster id (as a string) to its label."
        )

        # Per-provider client setup.
        if provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gem_model = genai.GenerativeModel(
                model if model.startswith("models/") else f"models/{model}",
                system_instruction=sys_prompt,
            )
        else:
            # OpenAI and DeepSeek share the OpenAI-compatible client.
            if provider == "deepseek":
                client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE)
            else:
                client = OpenAI(api_key=api_key)
            api_model = _PRICING[model][0] if model in _PRICING else model

        cluster_ids = list(samples.keys())
        result: dict[int, str] = {}
        used_labels: list[str] = []

        for start in range(0, len(cluster_ids), max_clusters_per_call):
            batch = cluster_ids[start:start + max_clusters_per_call]
            lines = []
            if used_labels:
                lines.append(
                    "Labels already assigned to other clusters — do NOT repeat "
                    "any of these: " + "; ".join(used_labels) + "\n"
                )
            for cid in batch:
                ex = samples[cid]
                lines.append(f"### Cluster {cid} ({len(ex)} sample reports)")
                lines.extend(f"- {e}" for e in ex)
                lines.append("")
            ids_str = ", ".join(f'"{cid}"' for cid in batch)
            lines.append(
                f"Return a JSON object with exactly these keys: {ids_str}. "
                "Each value is that cluster's label."
            )
            user_prompt = "\n".join(lines)

            if provider == "google":
                resp = gem_model.generate_content(
                    user_prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    },
                )
                content = resp.text
            else:
                kwargs = dict(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                if provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content

            parsed = _extract_json(content) or {}
            for cid in batch:
                name = parsed.get(str(cid), parsed.get(cid))
                if isinstance(name, str) and name.strip():
                    clean = " ".join(name.strip().split()[:max_words])
                    result[cid] = clean
                    used_labels.append(clean)
        return result

    def cluster_levenshtein(self, cluster_terms, cluster_labels, char_diff_threshold=3):
        from Levenshtein import distance  # Make sure to import the correct distance function

        merge_map = {}
        # Iterate over term pairs and decide on merging based on the distance
        for idx, term1 in enumerate(cluster_terms):
            for jdx, term2 in enumerate(cluster_terms):
                if idx < jdx and distance(term1, term2) <= char_diff_threshold:  
                    labels_to_merge = [label for label, term_index in enumerate(cluster_labels) if term_index == jdx]
                    for label in labels_to_merge:
                        merge_map[label] = idx  # Map the label to use the term index of term1
                    logging.info(f"Merging '{term2}' into '{term1}'")
                    st.write(f"Merging '{term2}' into '{term1}'")
        # Update the cluster labels
        updated_cluster_labels = [merge_map.get(label, label) for label in cluster_labels]
        # Update string labels to reflect merged labels
        updated_string_labels = [cluster_terms[label] for label in updated_cluster_labels]
        return updated_string_labels

    def cluster_cosine(self, cluster_terms, cluster_labels, similarity_threshold):
        from sklearn.metrics.pairwise import cosine_similarity

        cluster_terms_embeddings = get_embed_model().encode_document(cluster_terms)
        # Compute cosine similarity matrix in a vectorized form
        cos_sim_matrix = cosine_similarity(cluster_terms_embeddings, cluster_terms_embeddings)

        merge_map = {}
        n_terms = len(cluster_terms)
        # Iterate only over upper triangular matrix excluding diagonal to avoid redundant computations and self-comparison
        for idx in range(n_terms):
            for jdx in range(idx + 1, n_terms):
                if cos_sim_matrix[idx, jdx] >= similarity_threshold:
                    labels_to_merge = [label for label, term_index in enumerate(cluster_labels) if term_index == jdx]
                    for label in labels_to_merge:
                        merge_map[label] = idx
                    st.write(f"Merging '{cluster_terms[jdx]}' into '{cluster_terms[idx]}'")
                    logging.info(f"Merging '{cluster_terms[jdx]}' into '{cluster_terms[idx]}'")
        # Update the cluster labels
        updated_cluster_labels = [merge_map.get(label, label) for label in cluster_labels]
        # Update string labels to reflect merged labels
        updated_string_labels = [cluster_terms[label] for label in updated_cluster_labels]
        # make a dataframe with index, cluster label and cluster term
        return updated_string_labels

    def merge_similar_clusters(self, cluster_terms, cluster_labels, distance_type='cosine', char_diff_threshold=3, similarity_threshold=0.92):
        if distance_type == 'levenshtein':
            return self.cluster_levenshtein(cluster_terms, cluster_labels, char_diff_threshold)
        elif distance_type == 'cosine':
            return self.cluster_cosine(cluster_terms, cluster_labels, similarity_threshold)


    def plot_embeddings2(self, title=None):
        assert self.reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert self.cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")

        fig = go.Figure()

        unique_cluster_terms = np.unique(self.cluster_terms)

        for cluster_term in unique_cluster_terms:
            if cluster_term != 'Noise':
                indices = np.where(np.array(self.cluster_terms) == cluster_term)[0]

                # Plot points in the current cluster
                fig.add_trace(
                    go.Scatter(
                        x=self.reduced_embeddings[indices, 0],
                        y=self.reduced_embeddings[indices, 1],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.8,
                        ),
                        name=cluster_term,
                        text=self.data[f'{self.column}'].iloc[indices], 
                        hoverinfo='text',
                    )
                )
            else:
                # Plot noise points differently if needed
                fig.add_trace(
                    go.Scatter(
                        x=self.reduced_embeddings[indices, 0],
                        y=self.reduced_embeddings[indices, 1],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.5,
                            color='grey'
                        ),
                        name='Noise',
                        text=[self.data[f'{self.column}'][i] for i in indices],  # Adjusted for potential pandas use
                        hoverinfo='text',
                    )
                )
            # else:
            #     indices = np.where(np.array(self.cluster_terms) == 'Noise')[0]

            #     # Plot noise points
            #     fig.add_trace(
            #         go.Scatter(
            #             x=self.reduced_embeddings[indices, 0],
            #             y=self.reduced_embeddings[indices, 1],
            #             mode='markers',
            #             marker=dict(
            #                 size=5,
            #                 opacity=0.8,
            #             ),
            #             name='Noise',
            #             text=self.data[f'{self.column}'].iloc[indices],
            #             hoverinfo='text',
            #         )
            #     )

        fig.update_layout(title=title, showlegend=True, legend_title_text='Top TF-IDF Terms')
        #return fig
        st.plotly_chart(fig, use_container_width=True)  
       #fig.show()
        #logging.info("Embeddings plotted with TF-IDF colors")

    def plot_embeddings3(self, title=None):
        assert self.reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert self.cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")

        fig = go.Figure()

        unique_cluster_terms = np.unique(self.cluster_terms)

        terms_order = {term: i for i, term in enumerate(np.unique(self.cluster_terms, return_index=True)[0])}
        #indices = np.argsort([terms_order[term] for term in self.cluster_terms])

        # Handling color assignment, especially for noise
        colors = {term: ('grey' if term == 'Noise' else None) for term in unique_cluster_terms}
        color_map = px.colors.qualitative.Plotly  # Default color map from Plotly Express for consistency

        # Apply a custom color map, handling 'Noise' specifically
        color_idx = 0
        for cluster_term in unique_cluster_terms:
            indices = np.where(np.array(self.cluster_terms) == cluster_term)[0]
            if cluster_term != 'Noise':
                marker_color = color_map[color_idx % len(color_map)]
                color_idx += 1
            else:
                marker_color = 'grey'

            fig.add_trace(
                go.Scatter(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=(0.5 if cluster_term == 'Noise' else 0.8),
                        color=marker_color
                    ),
                    name=cluster_term,
                    text=self.data[f'{self.column}'].iloc[indices],
                    hoverinfo='text'
                )
            )
        fig.data = sorted(fig.data, key=lambda trace: terms_order[trace.name])
        fig.update_layout(title=title if title else "Embeddings Visualized", showlegend=True, legend_title_text='Top TF-IDF Terms')
        st.plotly_chart(fig, use_container_width=True)


    def plot_embeddings(self, title=None):
        """
        Plots the reduced dimensionality embeddings with clusters indicated.
        
        Args:
            title (str): The title of the plot.
        """
        # Ensure dimensionality reduction and TF-IDF based cluster naming have been performed
        assert self.reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert self.cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")
        
        fig = go.Figure()
        
        #for i, term in enumerate(self.cluster_terms):
            # Indices of points in the current cluster
        #unique_cluster_ids = np.unique(self.cluster_labels[self.cluster_labels != -1])  # Exclude noise
        unique_cluster_terms = np.unique(self.cluster_terms)
        unique_cluster_labels = np.unique(self.cluster_labels)
            
        for i, (cluster_id, cluster_terms) in enumerate(zip(unique_cluster_labels, unique_cluster_terms)):
            indices = np.where(self.cluster_labels == cluster_id)[0]
            #indices = np.where(self.cluster_labels == i)[0]
            
            # Plot points in the current cluster
            fig.add_trace(
                go.Scatter(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    mode='markers',
                    marker=dict(
                        #color=i,
                        #colorscale='rainbow',
                        size=5,
                        opacity=0.8,
                    ),
                    name=cluster_terms,
                    text=self.data[f'{self.column}'].iloc[indices],
                    hoverinfo='text',
                )
            )
            
        
        fig.update_layout(title=title, showlegend=True, legend_title_text='Top TF-IDF Terms')
        st.plotly_chart(fig, use_container_width=True)
        logging.info("Embeddings plotted with TF-IDF colors")

    def plot_embeddings4(self, title=None, cluster_terms=None, cluster_labels=None, reduced_embeddings=None, column=None, data=None):
        """
        Plots the reduced dimensionality embeddings with clusters indicated.
        
        Args:
            title (str): The title of the plot.
        """
        # Ensure dimensionality reduction and TF-IDF based cluster naming have been performed
        assert reduced_embeddings is not None, "Dimensionality reduction has not been performed yet."
        assert cluster_terms is not None, "Cluster TF-IDF analysis has not been performed yet."

        logging.info("Plotting embeddings with TF-IDF colors")
        
        fig = go.Figure()
        
        # Determine unique cluster IDs and terms, and ensure consistent color mapping
        unique_cluster_ids = np.unique(cluster_labels)
        unique_cluster_terms = [cluster_terms[i] for i in unique_cluster_ids]#if i != -1]  # Exclude noise by ID

        color_map = px.colors.qualitative.Plotly  # Using Plotly Express's qualitative colors for consistency
        color_idx = 0
        
        # Map each cluster ID to a color
        cluster_colors = {}
        for cid in unique_cluster_ids:
            #if cid != -1:  # Exclude noise
                cluster_colors[cid] = color_map[color_idx % len(color_map)]
                color_idx += 1
            #else:
            #    cluster_colors[cid] = 'grey'  # Noise or outliers in grey

        for cluster_id, cluster_term in zip(unique_cluster_ids, unique_cluster_terms):
            indices = np.where(cluster_labels == cluster_id)[0]
            fig.add_trace(
                go.Scatter(
                    x=reduced_embeddings[indices, 0],
                    y=reduced_embeddings[indices, 1],
                    mode='markers',
                    marker=dict(
                        color=cluster_colors[cluster_id],
                        size=5,
                        opacity=0.8#if cluster_id != -1 else 0.5,
                    ),
                    name=cluster_term,
                    text=data[f'{column}'].iloc[indices],
                    hoverinfo='text',
                )
            )
            
        fig.update_layout(
            title=title if title else "Embeddings Visualized",
            showlegend=True,
            legend_title_text='Top TF-IDF Terms',
            legend=dict(
                traceorder='normal',  # 'normal' or 'reversed'; ensures that traces appear in the order they are added
                itemsizing='constant'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        logging.info("Embeddings plotted with TF-IDF colors")



def analyze_and_predict(data, analyzers, col_names):
    """
    Performs analysis on the data using provided analyzers and makes predictions on specified columns.

    Args:
        data (pd.DataFrame): The dataset for analysis.
        analyzers (list): A list of UAPAnalyzer instances.
        col_names (list): Column names to be analyzed and predicted.
    """
    new_data = pd.DataFrame()
    for i, (column, analyzer) in enumerate(zip(col_names, analyzers)):
        new_data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_terms']
        logging.info(f"Cluster terms extracted for {column}")

    new_data = new_data.fillna('null').astype('category')
    data_nums = new_data.apply(lambda x: x.cat.codes)

    for col in data_nums.columns:
        try:
            categories = new_data[col].cat.categories
            x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
            bst, accuracy, preds = train_xgboost(x_train, y_train, x_test, y_test, len(categories))
            plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col)
        except Exception as e:
            logging.error(f"Error processing {col}: {e}")
    return new_data

def train_xgboost(x_train, y_train, x_test, y_test, num_classes):
    """
    Trains an XGBoost model and evaluates its performance.

    Uses GPU training (tree_method='hist', device='cuda') when a CUDA GPU is
    available.  Early stopping avoids over-training and reduces wall time.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        num_classes (int): The number of unique classes in the target variable.

    Returns:
        bst (Booster): The trained XGBoost model.
        accuracy (float): The accuracy of the model on the test set.
        preds (np.ndarray): Predictions on the test set.
    """
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # Co-tuned with analysis_service._xgb_cv_accuracy (autoresearch sweep): a
    # shallower, slower, mildly-subsampled model generalises better on these
    # small, imbalanced categorical targets than the old depth-6/eta-0.3 default
    # (raised mean 5-fold CV accuracy 0.862 → 0.866) and shrinks the holdout↔CV
    # overfit gap. Keep these in sync with that function.
    params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'tree_method': 'hist',
        'device': 'cuda' if _GPU_AVAILABLE else 'cpu',
        'nthread': -1,
    }
    bst = xgb.train(
        dtrain=dtrain,
        params=params,
        num_boost_round=400,
        evals=[(dtest, 'eval')],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    preds = bst.predict(dtest)
    accuracy = accuracy_score(y_test, preds)

    logging.info(f"XGBoost trained with accuracy: {accuracy:.2f} ({'GPU' if _GPU_AVAILABLE else 'CPU'})")
    return bst, accuracy, preds

def plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col):
    """
    Plots the feature importance, confusion matrix, and contingency table.

    Args:
        bst (Booster): The trained XGBoost model.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        preds (np.array): Predictions made by the model.
        categories (Index): Category names for the target variable.
        accuracy (float): The accuracy of the model on the test set.
        col (str): The target column name being analyzed and predicted.
    """
    fig, axs = plt.subplots(1, 3, figsize=(25, 5), dpi=300)
    fig.suptitle(f'{col.split(sep=".")[-1]} prediction', fontsize=35)

    plot_importance(bst, ax=axs[0], importance_type='gain', show_values=False)
    conf_matrix = confusion_matrix(y_test, preds)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=axs[1])
    axs[1].set_title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
    # make axes rotated
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=30, ha='right')
    sorted_features = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: x[1], reverse=True)
    # The most important feature is the first element in the sorted list
    most_important_feature = sorted_features[0][0]
    # Create a contingency table
    contingency_table = pd.crosstab(new_data[col], new_data[most_important_feature])

    # resid pearson is used to calculate the residuals, which 
    table = stats.Table(contingency_table).resid_pearson
    #print(table)
    # Perform the chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # Print the results
    logging.info(f"Chi-squared test for {col} and {most_important_feature}: p-value = {p}")
    
    sns.heatmap(table, annot=True, cmap='Greens', ax=axs[2])
    # make axis rotated
    axs[2].set_yticklabels(axs[2].get_yticklabels(), rotation=30, ha='right')
    axs[2].set_title(f'Contingency Table between {col.split(sep=".")[-1]} and {most_important_feature.split(sep=".")[-1]}\np-value = {p}')    

    plt.tight_layout()
    #plt.savefig(f"{col}_{accuracy:.2f}_prediction_XGB.jpeg", dpi=300)
    return plt

def cramers_v(confusion_matrix):
    """Calculate Cramer's V statistic for categorical-categorical association."""
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]
    except ValueError:
        return 0.0
    n = confusion_matrix.sum().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    denom = min((k_corr-1), (r_corr-1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)

def plot_cramers_v_heatmap(data, significance_level=0.05):
    """Plot heatmap of Cramer's V statistic for each pair of categorical variables in a DataFrame."""
    # Initialize a DataFrame to store Cramer's V values
    cramers_v_df = pd.DataFrame(index=data.columns, columns=data.columns, data=np.nan)
    
    # Compute Cramer's V for each pair of columns
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:  # Avoid self-comparison
                confusion_matrix = pd.crosstab(data[col1], data[col2])
                chi2, p, dof, expected = chi2_contingency(confusion_matrix)
                # Check if the p-value is less than the significance level
                #if p < significance_level:
                #    cramers_v_df.at[col1, col2] = cramers_v(confusion_matrix)
                # alternatively, you can use the following line to include all pairs
                cramers_v_df.at[col1, col2] = cramers_v(confusion_matrix)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10), dpi=200)
    mask = np.triu(np.ones_like(cramers_v_df, dtype=bool))  # Mask for the upper triangle
    # make a max and min of the cmap
    sns.heatmap(cramers_v_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, mask=mask, square=True)
    plt.title(f"Heatmap of Cramér's V (p < {significance_level})")
    return plt


class UAPVisualizer:
    def __init__(self, data=None):
        pass  # Initialization can be added if needed

    def analyze_and_predict(self, data, analyzers, col_names):
        new_data = pd.DataFrame()
        for i, (column, analyzer) in enumerate(zip(col_names, analyzers)):
            new_data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_terms']
            logging.info(f"Cluster terms extracted for {column}")

        new_data = new_data.fillna('null').astype('category')
        data_nums = new_data.apply(lambda x: x.cat.codes)

        for col in data_nums.columns:
            try:
                categories = new_data[col].cat.categories
                x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
                bst, accuracy, preds = self.train_xgboost(x_train, y_train, x_test, y_test, len(categories))
                self.plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col)
            except Exception as e:
                logging.error(f"Error processing {col}: {e}")

    def train_xgboost(self, x_train, y_train, x_test, y_test, num_classes):
        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(x_test, label=y_test)

        params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'max_depth': 6,
            'eta': 0.3,
            'tree_method': 'hist',
            'device': 'cuda' if _GPU_AVAILABLE else 'cpu',
            'nthread': -1,
        }
        bst = xgb.train(
            dtrain=dtrain,
            params=params,
            num_boost_round=100,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        preds = bst.predict(dtest)
        accuracy = accuracy_score(y_test, preds)

        logging.info(f"XGBoost trained with accuracy: {accuracy:.2f} ({'GPU' if _GPU_AVAILABLE else 'CPU'})")
        return bst, accuracy, preds

    def plot_results(self, new_data, bst, x_test, y_test, preds, categories, accuracy, col):
        fig, axs = plt.subplots(1, 3, figsize=(25, 5))
        fig.suptitle(f'{col.split(sep=".")[-1]} prediction', fontsize=35)

        plot_importance(bst, ax=axs[0], importance_type='gain', show_values=False)
        conf_matrix = confusion_matrix(y_test, preds)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=axs[1])
        axs[1].set_title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')

        sorted_features = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: x[1], reverse=True)
        most_important_feature = sorted_features[0][0]
        contingency_table = pd.crosstab(new_data[col], new_data[most_important_feature])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        logging.info(f"Chi-squared test for {col} and {most_important_feature}: p-value = {p}")

        sns.heatmap(contingency_table, annot=True, cmap='Greens', ax=axs[2])
        axs[2].set_title(f'Contingency Table between {col.split(sep=".")[-1]} and {most_important_feature.split(sep=".")[-1]}\np-value = {p}')    

        plt.tight_layout()
        plt.savefig(f"{col}_{accuracy:.2f}_prediction_XGB.jpeg", dpi=300)
        plt.show()

    @staticmethod
    def cramers_v(confusion_matrix):
        try:
            chi2 = chi2_contingency(confusion_matrix)[0]
        except ValueError:
            return 0.0
        n = confusion_matrix.sum().sum()
        if n <= 1:
            return 0.0
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        denom = min((k_corr-1), (r_corr-1))
        if denom <= 0:
            return 0.0
        return np.sqrt(phi2corr / denom)

    def plot_cramers_v_heatmap(self, data, significance_level=0.05):
        cramers_v_df = pd.DataFrame(index=data.columns, columns=data.columns, data=np.nan)
        
        for col1 in data.columns:
            for col2 in data.columns:
                if col1 != col2:
                    confusion_matrix = pd.crosstab(data[col1], data[col2])
                    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
                    if p < significance_level:
                        cramers_v_df.at[col1, col2] = UAPVisualizer.cramers_v(confusion_matrix)
        
        plt.figure(figsize=(10, 8)),# facecolor="black")
        mask = np.triu(np.ones_like(cramers_v_df, dtype=bool))
        #sns.set_theme(style="dark", rc={"axes.facecolor": "black", "grid.color": "white", "xtick.color": "white", "ytick.color": "white", "axes.labelcolor": "white", "axes.titlecolor": "white"})
        # ax = sns.heatmap(cramers_v_df, annot=True, fmt=".1f", linewidths=.5, linecolor='white', cmap='coolwarm', annot_kws={"color":"white"}, cbar=True, mask=mask, square=True)
        # Customizing the color of the ticks and labels to white
        # plt.xticks(color='white')
        # plt.yticks(color='white')
        sns.heatmap(cramers_v_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, mask=mask, square=True)
        plt.title(f"Heatmap of Cramér's V (p < {significance_level})")
        plt.show()

    
    def plot_treemap(self, df, column, top_n=32):
        # Get the value counts and the top N labels
        value_counts = df[column].value_counts()
        top_labels = value_counts.iloc[:top_n].index
        

        # Use np.where to replace all values not in the top N with 'Other'
        revised_column = f'{column}_revised'
        df[revised_column] = np.where(df[column].isin(top_labels), df[column], 'Other')

        # Get the value counts including the 'Other' category
        sizes = df[revised_column].value_counts().values
        labels = df[revised_column].value_counts().index

        # Get a gradient of colors
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Get % of each category
        percents = sizes / sizes.sum()

        # Prepare labels with percentages
        labels = [f'{label}\n {percent:.1%}' for label, percent in zip(labels, percents)]

        # Plot the treemap
        squarify.plot(sizes=sizes, label=labels, alpha=0.7, pad=True, color=colors, text_kwargs={'fontsize': 10})

        ax = plt.gca()

        # Iterate over text elements and rectangles (patches) in the axes for color adjustment
        for text, rect in zip(ax.texts, ax.patches):
            background_color = rect.get_facecolor()
            r, g, b, _ = mcolors.to_rgba(background_color)
            brightness = np.average([r, g, b])
            text.set_color('white' if brightness < 0.5 else 'black')

            # Adjust font size based on rectangle's area and wrap long text
            coef = 0.8
            font_size = np.sqrt(rect.get_width() * rect.get_height()) * coef
            text.set_fontsize(font_size)
            wrapped_text = textwrap.fill(text.get_text(), width=20)
            text.set_text(wrapped_text)

        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.gcf().set_size_inches(20, 12)
        plt.show()



# ── API pricing (May 2026) ────────────────────────────────────────────────────
# key → (api_model_id, provider, input_$/1M, cached_$/1M, output_$/1M, batch_discount_fraction)
_PRICING: dict[str, tuple] = {
    # (name, provider, price_in/M, price_cached_in/M, price_out/M, batch_discount)
    # OpenAI
    "gpt-5.4-mini":       ("gpt-5.4-mini",       "openai",    0.400, 0.1000, 1.600, 0.50),  # pricing placeholder — verify at platform.openai.com/docs/pricing
    "gpt-4o-mini":        ("gpt-4o-mini",        "openai",    0.150, 0.0750, 0.600, 0.50),
    "gpt-4o":             ("gpt-4o",             "openai",    2.500, 1.2500, 10.00, 0.50),
    "gpt-4.1-mini":       ("gpt-4.1-mini",       "openai",    0.400, 0.1000, 1.600, 0.50),
    "gpt-4.1":            ("gpt-4.1",            "openai",    2.000, 0.5000, 8.000, 0.50),
    "gpt-3.5-turbo-0125": ("gpt-3.5-turbo-0125", "openai",    0.500, 0.2500, 1.500, 0.50),
    # DeepSeek — v4-flash and v4-pro are the current generation
    # deepseek-chat / deepseek-reasoner are deprecated 2026-07-24
    "deepseek-v4-flash":  ("deepseek-v4-flash",  "deepseek",  0.140, 0.0028, 0.280, 0.00),
    "deepseek-v4-pro":    ("deepseek-v4-pro",    "deepseek",  0.435, 0.0036, 0.870, 0.00),
    "deepseek-chat":      ("deepseek-chat",      "deepseek",  0.270, 0.0700, 1.100, 0.00),
    "deepseek-reasoner":  ("deepseek-reasoner",  "deepseek",  0.550, 0.1400, 2.190, 0.00),
}

OPENAI_MODELS    = [k for k, v in _PRICING.items() if v[1] == "openai"]
DEEPSEEK_MODELS  = [k for k, v in _PRICING.items() if v[1] == "deepseek"]
_DEEPSEEK_BASE   = "https://api.deepseek.com"
_BATCH_SENTINELS = {"__batch_ids__", "__batch_descs__"}


def _extract_json(text: str) -> dict | None:
    """
    Robustly extract a JSON object from a model response.

    Handles the common patterns that DeepSeek (and other models) produce even
    when ``response_format={"type": "json_object"}`` is requested:
      - Pure JSON (ideal)
      - Markdown fences: ```json ... ``` or ``` ... ```
      - JSON embedded in prose (find first { … last })
      - Single-quoted keys/values
    Returns None if all strategies fail.
    """
    if not isinstance(text, str):
        return None

    import re as _re

    def _try(s: str) -> dict | None:
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # single-quote fallback
        try:
            return json.loads(s.replace("'", '"'))
        except (json.JSONDecodeError, Exception):
            pass
        return None

    # 1. Direct parse
    result = _try(text)
    if result is not None:
        return result

    # 2. Strip markdown fences  ```json\n...\n```  or  ```\n...\n```
    fence = _re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=_re.IGNORECASE)
    fence = _re.sub(r"\s*```$", "", fence.strip())
    result = _try(fence)
    if result is not None:
        return result

    # 3. Extract first { ... last } block (handles prose wrapping)
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        result = _try(text[start : end + 1])
        if result is not None:
            return result

    return None


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Token count via tiktoken when available, char/4 otherwise."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost(
    descriptions: list[str],
    format_long,
    model: str = "gpt-4o-mini",
    use_cache: bool = True,
    use_batch: bool = False,
) -> dict:
    """
    Estimate API cost before running process_descriptions().

    The system prompt + format_long template is identical for every request
    (the cacheable prefix). With prompt caching, only the first request pays
    the full input price; all subsequent ones pay the cached rate for that
    prefix — a major saving at scale.

    Parameters
    ----------
    descriptions : list[str]
        The raw texts you plan to process.
    format_long :
        The JSON template (dict or string) passed to process_descriptions().
    model : str
        Key from _PRICING (e.g. "gpt-4o-mini", "deepseek-v4-flash").
    use_cache : bool
        Assume the prefix is cached after the first request.
    use_batch : bool
        Apply OpenAI Batch API 50% discount (OpenAI models only).

    Returns
    -------
    dict with full cost breakdown and token counts.
    """
    if model not in _PRICING:
        raise ValueError(f"Unknown model {model!r}. Choose from: {sorted(_PRICING)}")

    _, provider, p_in, p_cached, p_out, batch_disc = _PRICING[model]
    fmt_str = format_long if isinstance(format_long, str) else json.dumps(format_long)

    system = (
        "You are a structured-data extraction assistant. "
        "Extract UAP sighting information from the provided report "
        "and return it as JSON matching the given template exactly. "
        "Leave fields empty string or null when the information is absent. "
        "IMPORTANT — engagement_type constraint: exactly ONE field in "
        "engagement_type must be set to \"P\" (the primary engagement type). "
        "Any number of additional fields may be \"S\" (secondary). "
        "All remaining fields must be blank. "
        "Never assign \"P\" to more than one field. "
        "Never assign \"S\" to any field without also assigning \"P\" to exactly one field. "
        "If no engagement type applies, set no_engagement to \"P\" and leave every other field blank."
    )
    prefix_text = (
        f"{system}\n\n"
        f"Parse into this JSON structure (leave missing fields empty): {fmt_str}\n\nOutput:"
    )
    prefix_tokens = _count_tokens(prefix_text, model)
    desc_counts   = [_count_tokens(f"Input report: {d}\n\n", model) for d in descriptions]
    output_per_q  = _count_tokens(fmt_str, model)

    n = len(descriptions)
    total_input  = sum(prefix_tokens + t for t in desc_counts)
    total_output = n * output_per_q

    if use_cache and n > 1:
        cached_tokens   = prefix_tokens * (n - 1)
        uncached_tokens = total_input - cached_tokens
    else:
        cached_tokens   = 0
        uncached_tokens = total_input

    disc = batch_disc if use_batch and provider == "openai" else 0.0

    cost_in     = uncached_tokens / 1_000_000 * p_in     * (1 - disc)
    cost_cached = cached_tokens   / 1_000_000 * p_cached * (1 - disc)
    cost_out    = total_output    / 1_000_000 * p_out    * (1 - disc)
    total_usd   = cost_in + cost_cached + cost_out

    return {
        "total_usd":              round(total_usd, 4),
        "cost_input_usd":         round(cost_in, 4),
        "cost_cached_usd":        round(cost_cached, 4),
        "cost_output_usd":        round(cost_out, 4),
        "total_input_tokens":     total_input,
        "cached_tokens":          cached_tokens,
        "uncached_tokens":        uncached_tokens,
        "output_tokens":          total_output,
        "prefix_tokens":          prefix_tokens,
        "avg_desc_tokens":        int(sum(desc_counts) / max(1, n)),
        "output_tokens_per_query": output_per_q,
        "batch_discount_pct":     int(disc * 100),
        "model":                  model,
        "provider":               provider,
        "n_queries":              n,
    }


def _empty_of(template_value):
    """The 'blank' counterpart of a schema template value: nested dicts keep
    their full key structure (leaves -> ''), lists -> [], scalars -> ''."""
    if isinstance(template_value, dict):
        return {k: _empty_of(v) for k, v in template_value.items()}
    if isinstance(template_value, list):
        return []
    return ""


def schema_complete(record, template):
    """Pad a parsed record so every schema-template key exists (missing or None
    keys -> blank structure; nested dicts recursed; extra record keys kept).

    LLMs routinely drop keys they never assign despite "leave blank" prompting —
    and once a key is absent from EVERY record, pd.json_normalize produces no
    column at all, so downstream consumers (SCU gate, column mapping, XGBoost)
    silently lose the field. Padding makes absence explicit: the column exists
    and is blank."""
    if not isinstance(record, dict) or not isinstance(template, dict):
        return record
    out = dict(record)
    for k, tv in template.items():
        if k not in out or out[k] is None:
            out[k] = _empty_of(tv)
        elif isinstance(tv, dict) and isinstance(out[k], dict):
            out[k] = schema_complete(out[k], tv)
    return out


class FatalParseError(RuntimeError):
    """A non-retryable, run-fatal API failure — an unfunded/empty account
    (``insufficient_quota``), a bad key (401 / ``invalid_api_key``), or a
    billing/permission block. Retrying these just storms the API with a doomed
    call for every row (× MAX_RETRIES × max_workers), so the whole parse run
    aborts on the first one instead."""


# Substrings marking a permanent, batch-fatal failure (matched case-insensitively
# against the exception text). These are NOT transient — retrying can never help.
_FATAL_API_MARKERS = (
    "insufficient_quota", "exceeded your current quota", "invalid_api_key",
    "incorrect api key", "authenticationerror", "invalid authentication",
    "billing", "account_deactivated", "account deactivated",
    "access terminated", "permissiondeniederror",
)


def _is_fatal_api_error(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _FATAL_API_MARKERS)


class UAPParser:
    """
    Parse raw UAP report texts into structured JSON via OpenAI or DeepSeek.

    Providers
    ---------
    ``provider="openai"``   — concurrent requests via OpenAI Chat API.
                              Set ``use_batch=True`` for the Batch API (50% off,
                              async ≤24 h; batch_ids saved to checkpoint).
    ``provider="deepseek"`` — concurrent requests via DeepSeek (OpenAI-compatible).
                              No batch API; uses dynamic rate-limit back-off.

    Checkpointing
    -------------
    Pass ``checkpoint_path`` to persist every response immediately. On restart
    with the same path, completed requests are skipped. OpenAI Batch IDs are
    also stored so jobs can be resumed after a Streamlit rerun.

    Cost estimation
    ---------------
    Call the module-level ``estimate_cost(descriptions, format_long, model=...)``
    before running to get a full cost breakdown.

    Usage
    -----
        # Concurrent (OpenAI or DeepSeek)
        parser = UAPParser(api_key=KEY, model="gpt-4o-mini",
                           checkpoint_path="ckpt.json")
        parser.process_descriptions(texts, FORMAT_LONG)

        # OpenAI Batch API (async, 50% cheaper)
        parser = UAPParser(api_key=KEY, model="gpt-4o-mini",
                           use_batch=True, checkpoint_path="ckpt.json")
        parser.process_descriptions(texts, FORMAT_LONG)   # submits & polls

        # DeepSeek
        parser = UAPParser(api_key=DS_KEY, model="deepseek-v4-flash",
                           checkpoint_path="ckpt.json")
        parser.process_descriptions(texts, FORMAT_LONG)
    """

    _BATCH_CHUNK = 50_000   # OpenAI Batch API hard limit per job

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str | None = None,
        use_batch: bool = False,
        col=None,
        format_long=None,
        checkpoint_path: str | None = None,
    ):
        import threading as _threading
        # Auto-detect provider from model name
        if provider is None:
            provider = "deepseek" if model in DEEPSEEK_MODELS else "openai"

        self.model    = model
        self.provider = provider
        self.use_batch = use_batch and provider == "openai"
        self.col      = col
        # Schema template (dict or JSON string) — refreshed by every
        # process_descriptions call; used by parse_responses to schema-complete
        # records so LLM-dropped keys still yield (blank) columns.
        self.format_long = format_long
        self.checkpoint_path = checkpoint_path
        self.responses: dict[str, str] = {}
        self._responses_lock = _threading.Lock()
        # Circuit breaker: set on the first fatal API error (no quota / bad key)
        # so concurrent workers short-circuit instead of each hammering the API.
        self._fatal = _threading.Event()
        self.last_errors: list[str] = []   # populated by process_descriptions

        if provider == "deepseek":
            self.client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE)
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            self.client = OpenAI()

        if checkpoint_path:
            self._load_checkpoint()

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                self.responses.update(json.load(f))
            logging.info(f"Checkpoint: {len(self.responses)} entries from {self.checkpoint_path}")
        except FileNotFoundError:
            logging.info(f"No checkpoint at {self.checkpoint_path} — starting fresh")
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Could not read checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(self.responses, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logging.warning(f"Could not write checkpoint: {e}")

    def save_checkpoint(self) -> None:
        with self._responses_lock:
            self._save_checkpoint()

    # ── Shared prompt builder ─────────────────────────────────────────────

    def _build_messages(self, description: str, format_long) -> list:
        fmt = format_long if isinstance(format_long, str) else json.dumps(format_long)
        return [
            {"role": "system", "content": (
                "You are a structured-data extraction assistant. "
                "Extract UAP sighting information from the provided report "
                "and return it as JSON matching the given template exactly. "
                "Leave fields empty string or null when the information is absent. "
                "IMPORTANT — engagement_type constraint: exactly ONE field in "
                "engagement_type must be set to \"P\" (the primary engagement type). "
                "Any number of additional fields may be \"S\" (secondary). "
                "All remaining fields must be blank. "
                "Never assign \"P\" to more than one field. "
                "Never assign \"S\" to any field without also assigning \"P\" to exactly one field. "
                "no_engagement may only ever be \"P\" or blank — never \"S\". "
                "Set no_engagement to \"P\" only when no other engagement type applies, and leave every other field blank."
            )},
            {"role": "user", "content": (
                f"Input report: {description}\n\n"
                f"Parse into this JSON structure (leave missing fields empty): "
                f"{fmt}\n\nOutput:"
            )},
        ]

    # ── Single-request fetch (both providers) ─────────────────────────────

    def fetch_response(self, description: str, format_long) -> object | None:
        INITIAL_WAIT, MAX_WAIT, MAX_RETRIES = 5, 600, 10
        wait = INITIAL_WAIT
        last_exc = None
        for _ in range(MAX_RETRIES):
            # Bail immediately if a fatal error already aborted the run — an
            # unfunded account would otherwise fire one doomed call per row.
            if self._fatal.is_set():
                raise FatalParseError("aborted after an earlier fatal API error")
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=self._build_messages(description, format_long),
                )
            except HTTPError as e:
                if "TooManyRequests" in str(e):
                    logging.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    wait = min(wait * 2, MAX_WAIT)
                    last_exc = e
                else:
                    raise
            except Exception as e:
                err = str(e)
                # Permanent failures (no quota / bad key / billing / permission)
                # are NOT rate limits — never retry. Trip the breaker so the rest
                # of the concurrent run aborts instead of hammering the API.
                if _is_fatal_api_error(err):
                    self._fatal.set()
                    raise FatalParseError(err)
                low = err.lower()
                if "429" in err or "rate" in low or "concurren" in low:
                    logging.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    wait = min(wait * 2, MAX_WAIT)
                    last_exc = e
                elif any(code in err for code in ("500", "502", "503", "504")):
                    logging.warning(f"Server error ({err[:80]}) — retrying in {wait}s")
                    time.sleep(wait)
                    wait = min(wait * 2, MAX_WAIT)
                    last_exc = e
                else:
                    raise  # surfaces the real error into process_descriptions.last_errors
        if last_exc:
            raise last_exc
        return None

    # ── OpenAI Batch API ──────────────────────────────────────────────────

    def _submit_batch_openai(self, descriptions: list[str], format_long) -> list[str]:
        """Upload requests in 50k chunks and return list of batch IDs."""
        import io as _io
        batch_ids = []
        for start in range(0, len(descriptions), self._BATCH_CHUNK):
            chunk = descriptions[start : start + self._BATCH_CHUNK]
            lines = [
                json.dumps({
                    "custom_id": str(start + i),
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body": {
                        "model":           self.model,
                        "response_format": {"type": "json_object"},
                        "messages":        self._build_messages(desc, format_long),
                    },
                })
                for i, desc in enumerate(chunk)
            ]
            content  = "\n".join(lines).encode("utf-8")
            uploaded = self.client.files.create(
                file=(f"batch_{start}.jsonl", _io.BytesIO(content), "application/jsonl"),
                purpose="batch",
            )
            batch = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_ids.append(batch.id)
            logging.info(
                f"Submitted batch {batch.id} for chunk "
                f"[{start}:{start + len(chunk)}] ({len(chunk)} requests)"
            )
        return batch_ids

    def _poll_and_collect(self, batch_ids: list[str], descriptions: list[str]) -> None:
        """Poll all batch jobs until done, then store results in self.responses."""
        remaining = list(batch_ids)
        while remaining:
            still_pending = []
            for bid in remaining:
                batch  = self.client.batches.retrieve(bid)
                status = batch.status
                c      = batch.request_counts
                logging.info(
                    f"Batch {bid}: {status} "
                    f"({c.completed}/{c.total} done, {c.failed} failed)"
                )
                if status == "completed":
                    text = self.client.files.content(batch.output_file_id).text
                    n_saved = 0
                    for line in text.splitlines():
                        if not line.strip():
                            continue
                        result  = json.loads(line)
                        idx     = int(result["custom_id"])
                        desc    = descriptions[idx]
                        choices = (result.get("response") or {}).get("body", {}).get("choices", [])
                        if choices:
                            with self._responses_lock:
                                self.responses[desc] = choices[0]["message"]["content"]
                            n_saved += 1
                    self._save_checkpoint()
                    logging.info(f"Batch {bid} collected — {n_saved} results saved.")
                elif status in ("failed", "expired", "cancelled"):
                    logging.error(f"Batch {bid} ended with status: {status}")
                else:
                    still_pending.append(bid)
            remaining = still_pending
            if remaining:
                logging.info(f"{len(remaining)} batch(es) pending — waiting 60s")
                time.sleep(60)

    def _process_batch_openai(self, pending: list[str], format_long) -> None:
        """Full Batch API flow with checkpoint persistence for resume after restart."""
        raw_ids   = self.responses.pop("__batch_ids__",   None)
        raw_descs = self.responses.pop("__batch_descs__", None)

        if raw_ids and raw_descs:
            logging.info("Resuming existing batch job(s) from checkpoint.")
            batch_ids  = json.loads(raw_ids)
            all_descs  = json.loads(raw_descs)
        else:
            batch_ids = self._submit_batch_openai(pending, format_long)
            all_descs = pending
            with self._responses_lock:
                self.responses["__batch_ids__"]   = json.dumps(batch_ids)
                self.responses["__batch_descs__"] = json.dumps(all_descs)
                self._save_checkpoint()

        self._poll_and_collect(batch_ids, all_descs)

        with self._responses_lock:
            self.responses.pop("__batch_ids__",   None)
            self.responses.pop("__batch_descs__", None)
            self._save_checkpoint()

    def batch_status(self) -> list[dict] | None:
        """Non-blocking check of in-progress batch jobs. Returns None if none active."""
        raw = self.responses.get("__batch_ids__")
        if not raw:
            return None
        result = []
        for bid in json.loads(raw):
            try:
                b = self.client.batches.retrieve(bid)
                result.append({
                    "batch_id":  bid,
                    "status":    b.status,
                    "completed": b.request_counts.completed,
                    "total":     b.request_counts.total,
                    "failed":    b.request_counts.failed,
                })
            except Exception as e:
                result.append({"batch_id": bid, "status": "error", "error": str(e)})
        return result

    # ── Non-blocking Batch API (Streamlit-safe) ───────────────────────────

    def submit_batch_only(self, descriptions: list[str], format_long) -> str:
        """
        Submit to OpenAI Batch API and return immediately without polling.

        Batch IDs and the original description list are written to the checkpoint
        file so any future session can resume via ``fetch_batch_results()``.

        Safe to call from Streamlit — does not block or sleep.
        Returns the checkpoint path (create one if none was given).
        """
        if not self.checkpoint_path:
            import uuid as _uuid
            self.checkpoint_path = f"checkpoints/batch_{_uuid.uuid4().hex[:12]}.json"
            os.makedirs("checkpoints", exist_ok=True)

        pending = [d for d in descriptions
                   if d not in self.responses and d not in _BATCH_SENTINELS]
        if not pending:
            logging.info("All descriptions already processed — nothing to submit")
            return self.checkpoint_path

        if "__batch_ids__" in self.responses:
            logging.info("Batch already submitted — checkpoint has IDs, skipping resubmission")
            return self.checkpoint_path

        batch_ids = self._submit_batch_openai(pending, format_long)
        with self._responses_lock:
            self.responses["__batch_ids__"]   = json.dumps(batch_ids)
            self.responses["__batch_descs__"] = json.dumps(pending)
            self._save_checkpoint()

        logging.info(
            f"Batch submitted: {len(batch_ids)} job(s) covering {len(pending)} requests. "
            f"Checkpoint: {self.checkpoint_path}"
        )
        return self.checkpoint_path

    def fetch_batch_results(self) -> dict:
        """
        Non-blocking single poll of all pending batch jobs.

        Call this on reconnect (or via a Streamlit button) to check progress and
        collect finished results. Does NOT sleep — returns immediately with the
        current state of every batch job.

        Returns a dict:
            status   : "complete" | "in_progress" | "no_batch" | "error"
            batches  : list of per-job status dicts
            n_saved  : total new responses written to self.responses this call
            checkpoint: path to the checkpoint file (for display in the UI)

        When ``status == "complete"`` the sentinel keys are cleared from the
        checkpoint so a subsequent call correctly reports ``no_batch``.
        """
        raw_ids   = self.responses.get("__batch_ids__")
        raw_descs = self.responses.get("__batch_descs__")

        if not raw_ids:
            return {"status": "no_batch",
                    "message": "No active batch found in checkpoint.",
                    "checkpoint": self.checkpoint_path}

        batch_ids = json.loads(raw_ids)
        all_descs = json.loads(raw_descs) if raw_descs else []

        statuses: list[dict] = []
        n_saved_total = 0
        all_terminal  = True

        for bid in batch_ids:
            try:
                b  = self.client.batches.retrieve(bid)
                st = b.status
                c  = b.request_counts
                info: dict = {
                    "batch_id":  bid,
                    "status":    st,
                    "completed": c.completed,
                    "total":     c.total,
                    "failed":    c.failed,
                }
                if st == "completed":
                    text    = self.client.files.content(b.output_file_id).text
                    n_saved = 0
                    for line in text.splitlines():
                        if not line.strip():
                            continue
                        result  = json.loads(line)
                        idx     = int(result["custom_id"])
                        if idx >= len(all_descs):
                            continue
                        desc    = all_descs[idx]
                        choices = (result.get("response") or {}).get("body", {}).get("choices", [])
                        if choices:
                            with self._responses_lock:
                                self.responses[desc] = choices[0]["message"]["content"]
                            n_saved += 1
                    self._save_checkpoint()
                    info["results_saved"] = n_saved
                    n_saved_total += n_saved
                elif st in ("failed", "expired", "cancelled"):
                    info["error"] = f"Batch ended with non-success status: {st}"
                else:
                    all_terminal = False  # validating / in_progress / finalizing
                statuses.append(info)
            except Exception as e:
                statuses.append({"batch_id": bid, "status": "error", "error": str(e)})

        if all_terminal:
            with self._responses_lock:
                self.responses.pop("__batch_ids__",   None)
                self.responses.pop("__batch_descs__", None)
                self._save_checkpoint()

        return {
            "status":     "complete" if all_terminal else "in_progress",
            "batches":    statuses,
            "n_saved":    n_saved_total,
            "checkpoint": self.checkpoint_path,
        }

    # ── Main entry point ──────────────────────────────────────────────────

    def process_descriptions(
        self, descriptions, format_long, max_workers: int = 10,
        progress_callback=None,
    ) -> None:
        """
        Parse all descriptions not already in self.responses.

        Routes to:
        - OpenAI Batch API  if self.use_batch=True  (async, ≤24 h, blocking poll)
        - Concurrent single requests otherwise       (both providers)

        For non-blocking Batch API use (Streamlit), call ``submit_batch_only``
        then ``fetch_batch_results`` from a separate button/page rerun.
        """
        self.format_long = format_long   # remembered for parse_responses padding
        pending = [
            d for d in descriptions
            if d not in self.responses and d not in _BATCH_SENTINELS
        ]
        already_done = len(descriptions) - len(pending)
        if already_done:
            logging.info(f"Skipping {already_done} already-processed descriptions")
        if not pending:
            logging.info("All descriptions already processed — nothing to do")
            return

        if self.use_batch:
            self._process_batch_openai(pending, format_long)
            return

        self.last_errors.clear()
        self._fatal.clear()   # reset the circuit breaker for this run
        n_ok = n_fail = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_desc = {
                executor.submit(self.fetch_response, desc, format_long): desc
                for desc in pending
            }
            for future in stqdm(
                concurrent.futures.as_completed(future_to_desc),
                total=len(pending),
                desc=f"Parsing [{self.provider}/{self.model}]",
            ):
                desc = future_to_desc[future]
                try:
                    response = future.result()
                    text = response.choices[0].message.content if response else None
                    if text:
                        with self._responses_lock:
                            self.responses[desc] = text
                            self._save_checkpoint()
                        n_ok += 1
                    else:
                        n_fail += 1
                        msg = f"No response returned for: {str(desc)[:80]!r}"
                        self.last_errors.append(msg)
                        logging.warning(msg)
                except FatalParseError as exc:
                    # Unfunded account / bad key / billing — abort the whole run
                    # instead of firing a doomed call for every remaining row.
                    n_fail += 1
                    self.last_errors.append(f"FATAL: {exc}")
                    logging.error(f"Fatal API error — aborting parse run: {exc}")
                    for f in future_to_desc:
                        f.cancel()   # drop not-yet-started rows
                    if progress_callback is not None:
                        progress_callback(n_ok, n_fail, len(pending))
                    break
                except Exception as exc:
                    n_fail += 1
                    self.last_errors.append(str(exc))
                    logging.error(f"Error processing description: {exc}")
                if progress_callback is not None:
                    progress_callback(n_ok, n_fail, len(pending))

        logging.info(
            f"process_descriptions done — ok={n_ok} fail={n_fail} "
            f"model={self.model} provider={self.provider}"
        )

    # ── Post-processing ───────────────────────────────────────────────────

    def parse_responses(self, format_long=None) -> dict:
        """Extract JSON from every stored response. When the schema template is
        known (arg, or remembered from process_descriptions), each record is
        schema-completed so LLM-dropped keys still appear as blank fields —
        guaranteeing every schema column exists downstream."""
        template = format_long if format_long is not None else self.format_long
        if isinstance(template, str):
            try:
                template = json.loads(template)
            except (json.JSONDecodeError, TypeError):
                template = None
        if not isinstance(template, dict) or not template:
            template = None

        parsed, failed = {}, 0
        for k, v in self.responses.items():
            if k in _BATCH_SENTINELS:
                continue
            result = _extract_json(v)
            if result is not None:
                parsed[k] = schema_complete(result, template) if template else result
            else:
                failed += 1
                logging.warning(f"Could not parse response for key (first 120 chars): {str(v)[:120]!r}")
        logging.info(f"Parsed: {len(parsed)}  |  Failed: {failed}"
                     + ("  (schema-completed)" if template else ""))
        return parsed

    def responses_to_df(self, col: str, parsed_responses: dict) -> pd.DataFrame:
        """Expand one top-level JSON group into a flat DataFrame."""
        rows = {k: v[col] for k, v in parsed_responses.items() if col in v}
        df = pd.json_normalize(list(rows.values()))
        df.index = list(rows.keys())
        return df



