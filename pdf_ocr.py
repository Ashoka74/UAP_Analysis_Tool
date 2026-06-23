import json
import queue
import re
import threading

import numpy as np
import pandas as pd
import streamlit as st

st.title("PDF OCR — Table Extractor")
st.caption("Powered by LightOnOCR-1B + Microsoft Harrier embeddings.")


# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import fitz  # noqa: F401
except ImportError:
    st.error("**PyMuPDF is not installed.**\n\n```bash\npip install pymupdf\n```")
    st.stop()


# ── Cached models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading OCR model (one-time, ~2 GB)…")
def load_ocr_model():
    import torch
    from transformers import AutoProcessor, LightOnOcrForConditionalGeneration

    processor = AutoProcessor.from_pretrained("lightonai/LightOnOCR-1B-1025")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        "lightonai/LightOnOCR-1B-1025",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return processor, model


@st.cache_resource(show_spinner="Loading embedding model (one-time)…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "microsoft/harrier-oss-v1-0.6b",
        model_kwargs={"dtype": "auto"},
    )


# ── Core helpers ───────────────────────────────────────────────────────────────
def pdf_page_to_image(pdf_bytes: bytes, page_num: int, dpi: int):
    from PIL import Image
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc.load_page(page_num).get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    image = Image.fromarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    )
    n_pages = doc.page_count
    doc.close()
    return image, n_pages


def run_ocr(processor, model, image) -> str:
    import torch
    prompt = (
        "Extract all text from this page in natural reading order, "
        "using Markdown for tables and LaTeX for equations."
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
    return processor.decode(output[0], skip_special_tokens=True)


def parse_markdown_tables(text: str) -> list[pd.DataFrame]:
    tables = []
    for match in re.finditer(r"((?:\|.+\|\n?)+)", text, re.MULTILINE):
        lines = match.group().strip().split("\n")
        data_lines = [l for l in lines if not re.fullmatch(r"\|[\s:\-|]+\|", l.strip())]
        if len(data_lines) < 2:
            continue
        rows = [[c.strip() for c in ln.split("|")[1:-1]] for ln in data_lines]
        n_cols = len(rows[0])
        rows = [r for r in rows if len(r) == n_cols]
        if len(rows) < 2:
            continue
        tables.append(pd.DataFrame(rows[1:], columns=rows[0]))
    return tables


def ocr_pdf(pdf_bytes: bytes, page_start: int, page_end: int, dpi: int,
            processor, model, label: str) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Pipeline: a background thread renders PDF pages (CPU) while the main
    thread runs GPU inference, so rendering and inference overlap.

    render page N+1  ──────┐
                            ├─ overlapped
    OCR page N       ───────┘
    """
    _, n_pages = pdf_page_to_image(pdf_bytes, 0, dpi)
    page_end = min(page_end, n_pages - 1)
    pages = list(range(page_start, page_end + 1))
    n = len(pages)

    # Queue holds (page_num, PIL.Image) tuples; sentinel = None
    render_q: queue.Queue = queue.Queue(maxsize=2)  # max 2 pre-rendered ahead

    def _render_worker():
        for p in pages:
            image, _ = pdf_page_to_image(pdf_bytes, p, dpi)
            render_q.put((p, image))
        render_q.put(None)  # sentinel

    threading.Thread(target=_render_worker, daemon=True).start()

    tables, texts = [], []
    bar = st.progress(0, text=f"{label}: starting…")

    for i in range(n):
        item = render_q.get()
        if item is None:
            break
        p, image = item
        bar.progress(i / n, text=f"{label}: page {p + 1}/{n_pages}…")
        with st.expander(f"{label} — page {p + 1}", expanded=False):
            st.image(image, use_column_width=True)
        text = run_ocr(processor, model, image)
        texts.append(f"#### Page {p + 1}\n\n{text}")
        for t in parse_markdown_tables(text):
            t.insert(0, "_page", p + 1)
            tables.append(t)

    bar.progress(1.0, text=f"{label}: done.")
    return tables, texts


def show_tables(tables: list[pd.DataFrame], key_prefix: str):
    if not tables:
        st.info("No markdown tables found in OCR output.")
        return
    for idx, df in enumerate(tables):
        page_label = df["_page"].iloc[0] if "_page" in df.columns else "?"
        st.write(f"**Table {idx + 1}** — page {page_label}")
        display = df.drop(columns=[c for c in ("_page", "_file") if c in df.columns])
        st.dataframe(display, use_container_width=True)


# ── Embedding section (shared, shown after any OCR run) ───────────────────────
def embedding_section(combined: pd.DataFrame):
    st.markdown("---")
    st.subheader("Calculate Embeddings")

    text_cols = [c for c in combined.columns if c not in ("_page", "_file", "embeddings")]
    if not text_cols:
        st.warning("No text columns available.")
        return

    selected_cols = st.multiselect(
        "Columns to embed (values are concatenated per row)",
        options=text_cols,
        default=[text_cols[0]],
        key="embed_cols",
    )
    prompt_name = st.selectbox(
        "Encoding prompt",
        options=["none (document)", "web_search_query", "sts_query", "bitext_query"],
        help="Use 'none' for document/passage content. Use a query prompt only for query-side encoding.",
        key="embed_prompt",
    )

    if not st.button("Calculate Embeddings", key="btn_embed"):
        return

    if not selected_cols:
        st.warning("Select at least one column.")
        return

    texts = combined[selected_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    embed_model = load_embedding_model()
    with st.spinner(f"Encoding {len(texts)} rows…"):
        kwargs = {} if prompt_name == "none (document)" else {"prompt_name": prompt_name}
        embeddings = embed_model.encode(texts, show_progress_bar=False, **kwargs)

    combined = combined.copy()
    combined["embeddings"] = [json.dumps(e.tolist()) for e in embeddings]
    st.session_state["ocr_combined_with_embeddings"] = combined

    st.success(f"Embeddings added: {embeddings.shape[1]}-dim vectors for {len(texts)} rows.")
    st.dataframe(
        combined.assign(embeddings=combined["embeddings"].str[:60] + "…"),
        use_container_width=True,
    )
    st.download_button(
        label="Download table + embeddings as CSV",
        data=combined.to_csv(index=False).encode(),
        file_name="ocr_tables_with_embeddings.csv",
        mime="text/csv",
        key="dl_embed",
    )


# ── JSON helper ────────────────────────────────────────────────────────────────
def json_to_dataframes(raw: bytes) -> list[pd.DataFrame]:
    """
    Convert any JSON structure into one or more DataFrames.

    Handles:
    - list of dicts          → single flat DataFrame
    - dict of lists          → single DataFrame (pd.DataFrame(data))
    - nested dict            → pd.json_normalize, one DataFrame
    - dict of dicts          → one DataFrame per top-level key, or normalise whole thing
    - list of non-dict items → single-column DataFrame
    """
    data = json.loads(raw)
    tables = []

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            tables.append(pd.json_normalize(data))
        else:
            tables.append(pd.DataFrame({"value": data}))

    elif isinstance(data, dict):
        # Check if values are all lists of equal length → columnar format
        if all(isinstance(v, list) for v in data.values()):
            try:
                tables.append(pd.DataFrame(data))
            except ValueError:
                # Unequal lengths — normalise each key separately
                for key, val in data.items():
                    df = pd.json_normalize(val) if val and isinstance(val[0], dict) else pd.DataFrame({"value": val})
                    df.insert(0, "_key", key)
                    tables.append(df)
        # Dict of dicts → one table per key
        elif all(isinstance(v, dict) for v in data.values()):
            for key, val in data.items():
                df = pd.json_normalize([val])
                df.insert(0, "_key", key)
                tables.append(df)
        else:
            # Mixed / deeply nested — normalise whole thing
            tables.append(pd.json_normalize(data if isinstance(data, list) else [data]))

    return [df for df in tables if not df.empty]


# ── UI ─────────────────────────────────────────────────────────────────────────
mode = st.radio("Input mode", ["Single PDF upload", "Multiple PDF upload", "JSON file"], horizontal=True)
dpi = st.slider("Render DPI", 100, 300, 200, step=50,
                help="~1540 px longest side is optimal for LightOnOCR (PDF modes only)")

# Clear stale results when mode changes
if st.session_state.get("_ocr_mode") != mode:
    for key in ("ocr_tables", "ocr_texts", "ocr_combined_with_embeddings"):
        st.session_state.pop(key, None)
    st.session_state["_ocr_mode"] = mode

# ── Single PDF ─────────────────────────────────────────────────────────────────
if mode == "Single PDF upload":
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to get started.")
        st.stop()

    pdf_bytes = uploaded.read()
    _, n_pages = pdf_page_to_image(pdf_bytes, 0, dpi)
    st.write(f"**Pages detected:** {n_pages}")

    c1, c2 = st.columns(2)
    page_start = c1.number_input("First page (0-indexed)", 0, n_pages - 1, 0)
    page_end   = c2.number_input("Last page (inclusive)",  int(page_start), n_pages - 1, n_pages - 1)

    if st.button("Run OCR"):
        processor, ocr_model = load_ocr_model()
        tables, texts = ocr_pdf(
            pdf_bytes, int(page_start), int(page_end), dpi,
            processor, ocr_model, label=uploaded.name,
        )
        st.session_state["ocr_tables"] = tables
        st.session_state["ocr_texts"]  = texts
        st.session_state.pop("ocr_combined_with_embeddings", None)

    if "ocr_tables" in st.session_state:
        tables = st.session_state["ocr_tables"]
        texts  = st.session_state["ocr_texts"]

        st.subheader("Extracted Tables")
        show_tables(tables, key_prefix="single")

        with st.expander("Full OCR text", expanded=not bool(tables)):
            st.markdown("\n\n---\n\n".join(texts))

        if tables:
            combined = pd.concat(
                [df.drop(columns=[c for c in ("_page",) if c in df.columns]) for df in tables],
                ignore_index=True,
            )
            # Restore _page for reference
            combined_with_meta = pd.concat(tables, ignore_index=True)
            embedding_section(combined_with_meta)

            if "ocr_combined_with_embeddings" not in st.session_state:
                st.download_button(
                    "Download tables as CSV",
                    data=combined_with_meta.to_csv(index=False).encode(),
                    file_name=f"{uploaded.name}_tables.csv",
                    mime="text/csv",
                    key="dl_single",
                )

# ── Multiple PDF upload ────────────────────────────────────────────────────────
elif mode == "Multiple PDF upload":
    uploaded_files = st.file_uploader(
        "Select PDF files (Ctrl/Shift-click to pick multiple)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Select one or more PDF files to get started.")
        st.stop()

    st.write(f"**{len(uploaded_files)} PDF(s) selected:**")
    st.dataframe(
        pd.DataFrame({
            "File": [f.name for f in uploaded_files],
            "Size (KB)": [round(f.size / 1024, 1) for f in uploaded_files],
        }),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    page_start     = c1.number_input("First page per PDF (0-indexed)", min_value=0, value=0)
    page_end_input = c2.number_input("Last page per PDF (-1 = all)",   min_value=-1, value=-1)

    if st.button("Run OCR on all PDFs"):
        processor, ocr_model = load_ocr_model()
        all_tables: list[pd.DataFrame] = []
        all_texts:  list[str]          = []
        overall = st.progress(0, text="Starting batch OCR…")

        for f_idx, uploaded_file in enumerate(uploaded_files):
            overall.progress(f_idx / len(uploaded_files),
                             text=f"{uploaded_file.name} ({f_idx + 1}/{len(uploaded_files)})…")
            st.markdown(f"---\n### {uploaded_file.name}")
            pdf_bytes = uploaded_file.read()
            _, n_pages = pdf_page_to_image(pdf_bytes, 0, dpi)
            p_end = n_pages - 1 if page_end_input == -1 else min(int(page_end_input), n_pages - 1)

            tables, texts = ocr_pdf(
                pdf_bytes, int(page_start), p_end, dpi,
                processor, ocr_model, label=uploaded_file.name,
            )
            for t in tables:
                t.insert(0, "_file", uploaded_file.name)
            all_tables.extend(tables)
            all_texts.extend(texts)
            show_tables(tables, key_prefix=f"f{f_idx}")
            with st.expander(f"Full OCR text — {uploaded_file.name}", expanded=False):
                st.markdown("\n\n---\n\n".join(texts))

        overall.progress(1.0, text=f"All {len(uploaded_files)} PDFs processed.")
        st.session_state["ocr_tables"] = all_tables
        st.session_state["ocr_texts"]  = all_texts
        st.session_state.pop("ocr_combined_with_embeddings", None)

    if "ocr_tables" in st.session_state:
        all_tables = st.session_state["ocr_tables"]

        if all_tables:
            st.markdown("---")
            st.subheader("Combined export")
            combined = pd.concat(all_tables, ignore_index=True)
            st.dataframe(combined, use_container_width=True)

            embedding_section(combined)

            if "ocr_combined_with_embeddings" not in st.session_state:
                st.download_button(
                    "Download all tables as CSV",
                    data=combined.to_csv(index=False).encode(),
                    file_name="ocr_all_tables.csv",
                    mime="text/csv",
                    key="dl_folder_combined",
                )

# ── JSON file ──────────────────────────────────────────────────────────────────
elif mode == "JSON file":
    json_file = st.file_uploader(
        "Upload a JSON file",
        type=["json"],
        help="Accepts any JSON structure: list of dicts, dict of lists, nested objects, etc.",
    )
    if not json_file:
        st.info("Upload a JSON file to get started.")
        st.stop()

    try:
        tables = json_to_dataframes(json_file.read())
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")
        st.stop()

    if not tables:
        st.warning("No non-empty tables could be extracted from this JSON.")
        st.stop()

    # Store as ocr_tables so the shared embedding section works unchanged
    # Attach a dummy _page col so show_tables doesn't break
    tagged = []
    for i, df in enumerate(tables):
        df = df.copy()
        if "_page" not in df.columns:
            df.insert(0, "_page", i + 1)
        tagged.append(df)
    st.session_state["ocr_tables"] = tagged
    st.session_state["ocr_texts"]  = []

    st.subheader("Extracted Tables")
    for idx, df in enumerate(tagged):
        label = f"Table {idx + 1}"
        if "_key" in df.columns:
            label += f" — {df['_key'].iloc[0]}"
        st.write(f"**{label}** ({len(df)} rows × {len(df.columns)} cols)")
        display = df.drop(columns=[c for c in ("_page", "_key") if c in df.columns])
        st.dataframe(display, use_container_width=True)
        st.download_button(
            f"Download table {idx + 1} as CSV",
            data=display.to_csv(index=False).encode(),
            file_name=f"{json_file.name}_table_{idx + 1}.csv",
            mime="text/csv",
            key=f"dl_json_{idx}",
        )

    combined = pd.concat(tagged, ignore_index=True)
    embedding_section(combined)
