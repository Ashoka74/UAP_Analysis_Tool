# `pipeline/` — Document Preprocessing scripts (vendored)

These are the standalone scripts that turn raw government PDFs into the report
table consumed by the **UAP Feature Extraction** page. They used to live in an
external directory (`D:\divided` / `/mnt/d/divided`); they are now vendored here
so the **Document Preprocessing Pipeline** Streamlit page (`preprocessing.py`)
is self-contained — no external script folder, no separate interpreter.

## How they're run

`preprocessing.py` invokes each script with the **app's own interpreter**
(`sys.executable`) and an absolute path into this folder, while the **working
directory** (the configurable data dir, default `pipeline_data/`) is the cwd.
So the scripts read/write `raw/`, `pages_out/`, `concat/`, `extracted/`, … in
the working dir, and their code ships with the app. Each script is unmodified
from the original pipeline and remains runnable directly:

```bash
python pipeline/page_coverage.py --root /path/to/workdir
```

## Stages

| Stage | Scripts |
|-------|---------|
| 0 · Scrape | `download_uap_pdfs.py` |
| 1 · Layout | `split_pages.py`, `restructure_pages.py`, `reorganize.py` |
| 2 · OCR | `stamp_pages.py`, `find_ocr_targets.py`, `run_ocr.py`, `destamp_pages.py` |
| 3 · Assembly | `page_coverage.py`, `concat_pages.py` |
| 4 · Extraction | `extract_reports.py`, `pdf_to_reports.py`, `reconcile.py` |
| 5 · Table | `add_source_agency.py`, `analyze_reports.py`, `join_reports.py`, `map_yaml_to_reports.py` |
| Audit | `audit.py`, `find_missing_concat.py` |

## Dependencies

Pinned in the project's `pyproject.toml` / `requirements.txt`, so the app's
interpreter already has them: `pyyaml`, `google-genai`, `pypdf`, `mistralai`,
`openai`, `pdfplumber`, `reportlab`, `pdf2image`, `pillow`, `pymupdf`,
`requests`. `pdf2image` additionally needs the system **poppler** binary
(`apt-get install poppler-utils`).

API keys (entered in the page or via `st.secrets`): `MISTRAL_API_KEY` (OCR),
`GEMINI_API_KEY` (extraction), `NVIDIA_API_KEY` (optional NIM extraction).
