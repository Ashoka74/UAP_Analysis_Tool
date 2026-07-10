"""
Document Preprocessing Pipeline — Streamlit page.

Self-contained: the preprocessing scripts (scrape → layout → OCR → concat →
report extraction → table creation) are vendored in-repo under ``pipeline/`` and
run with the app's own interpreter, so this page no longer depends on an
external script directory (the old ``D:/divided``) or a separate Python env.
Only a *working directory* (where the PDFs and intermediate outputs live) is
configurable. Each step is run as a subprocess with its output streamed live.

The end product (extracted reports / parsed_reports.xlsx) is what the
"UAP Feature Extraction" page consumes — this page is the stage that produces
that table from raw government PDFs.
"""

import os
import sys
import shlex
import subprocess
from pathlib import Path

import streamlit as st

st.title("🧪 Document Preprocessing Pipeline")
st.caption(
    "Scrape → layout → OCR → concat → report extraction → table. Runs the "
    "pipeline scripts in your pipeline directory and streams their output. "
    "The table it produces feeds the **UAP Feature Extraction** page."
)

# Every script the pipeline expects — used for the presence check.
PIPELINE_SCRIPTS = [
    "download_uap_pdfs.py", "check_new_release.py", "embed_media.py",
    "restructure_pages.py", "split_pages.py",
    "reorganize.py", "stamp_pages.py", "find_ocr_targets.py", "run_ocr.py",
    "destamp_pages.py", "page_coverage.py", "concat_pages.py",
    "extract_reports.py", "pdf_to_reports.py", "reconcile.py", "audit.py",
    "find_missing_concat.py", "add_source_agency.py", "analyze_reports.py",
    "join_reports.py", "map_yaml_to_reports.py",
]


def _secret(*names):
    """Best-effort lookup of an API key from st.secrets; '' if unavailable."""
    try:
        for n in names:
            v = st.secrets.get(n)
            if v:
                return v
    except Exception:
        pass
    return ""


# Pipeline scripts are vendored beside this page, so there is no external
# script-directory dependency anymore.
SCRIPTS_DIR = Path(__file__).resolve().parent / "pipeline"
DEFAULT_WORKDIR = Path(__file__).resolve().parent / "pipeline_data"

# ── Configuration ──────────────────────────────────────────────────────────
with st.expander("⚙️ Configuration", expanded=True):
    work_dir = st.text_input(
        "Working directory (where the PDFs and intermediate outputs live)",
        value=st.session_state.get("prep_workdir", str(DEFAULT_WORKDIR)),
        key="prep_workdir",
        help="Every step runs with this as its working directory — the scripts "
             "read/write raw/, pages_out/, concat/, extracted/, … here. The "
             "pipeline *scripts* themselves now ship with the app in pipeline/.",
    )
    py_exe = st.text_input(
        "Python executable", value=st.session_state.get("prep_py", sys.executable),
        key="prep_py",
        help="Defaults to the app's own interpreter. Override only if the "
             "pipeline dependencies are installed in a different environment.",
    )
    k1, k2, k3 = st.columns(3)
    mistral_key = k1.text_input(
        "MISTRAL_API_KEY", _secret("MISTRAL_API_KEY"), type="password",
        help="Used by run_ocr.py (Mistral OCR).",
    )
    gemini_key = k2.text_input(
        "GEMINI_API_KEY", _secret("GEMINI_API_KEY", "GEMINI_KEY"), type="password",
        help="Used by extract_reports.py (Gemini report extraction).",
    )
    nvidia_key = k3.text_input(
        "NVIDIA_API_KEY", _secret("NVIDIA_API_KEY"), type="password",
        help="Optional — used by pdf_to_reports.py (NVIDIA NIM extraction).",
    )
    db_url = st.text_input(
        "DATABASE_URL", _secret("DATABASE_URL"), type="password",
        help="Optional — Neon Postgres DSN for embed_media.py (multimodal "
             "pgvector embeddings). The embed step self-skips without it.",
    )

    # Popover (not a nested expander — Streamlit forbids expander-in-expander).
    with st.popover("Pipeline dependencies"):
        st.code(
            f"{py_exe or 'python'} -m pip install pyyaml google-genai pypdf "
            "mistralai openai pdfplumber reportlab pdf2image pillow pymupdf requests",
            language="bash",
        )
        st.caption(
            "`pdf2image` also needs the system **poppler** binary "
            "(`apt-get install poppler-utils`). These deps are also pinned in "
            "the project's requirements, so the app's own interpreter has them."
        )

# Sanity check on the vendored scripts (should always pass).
_missing = [s for s in PIPELINE_SCRIPTS if not (SCRIPTS_DIR / s).is_file()]
if _missing:
    st.error(
        f"{len(PIPELINE_SCRIPTS) - len(_missing)}/{len(PIPELINE_SCRIPTS)} "
        f"bundled scripts found in `pipeline/`. Missing: {', '.join(_missing)}"
    )
else:
    st.success(f"All {len(PIPELINE_SCRIPTS)} pipeline scripts bundled in `pipeline/`.")

# Resolve (and create) the working directory — no external folder required.
_wdir = Path(work_dir).expanduser() if work_dir else None
if _wdir is None:
    st.error("Set a working directory above.")
    st.stop()
try:
    _wdir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    st.error(f"Could not create working directory `{work_dir}`: {e}")
    st.stop()
st.caption(f"Working directory: `{_wdir}`")

with st.expander("📋 Recommended sequence", expanded=False):
    st.markdown(
        "1. **Scrape** — `download_uap_pdfs.py`\n"
        "2. **Layout** — `restructure_pages.py --execute` → `reorganize.py --execute` "
        "(`split_pages.py` first for multi-page PDFs like the FBI reports)\n"
        "3. **OCR** — `stamp_pages.py` → `find_ocr_targets.py` → `run_ocr.py` "
        "→ `destamp_pages.py`\n"
        "4. **Assembly** — `page_coverage.py` → `concat_pages.py`\n"
        "5. **Extraction** — `extract_reports.py` (or `pdf_to_reports.py`) → `reconcile.py`\n"
        "6. **Table** — `add_source_agency.py` / `join_reports.py` / `map_yaml_to_reports.py`\n\n"
        "`stamp_pages.py` must run **after** `reorganize.py` so the frontmatter "
        "picks up agency/subtype/region from the path. For the stamp→concat→destamp "
        "flow: stamp new pages, concat with `--no-inline` (metadata bakes into "
        "`concat/`), then destamp the individual page files."
    )

st.divider()


# ── Subprocess runner ──────────────────────────────────────────────────────
def run_command(cmd_str: str, key: str) -> None:
    """Run a command in the pipeline directory, streaming output live."""
    cmd_str = (cmd_str or "").strip()
    if not cmd_str:
        st.warning("Empty command.")
        return
    try:
        parts = shlex.split(cmd_str)
    except ValueError as e:
        st.error(f"Could not parse command: {e}")
        return
    # Route a leading `python` / `python3` to the configured interpreter.
    if parts and parts[0] in ("python", "python3") and py_exe:
        parts[0] = py_exe
    # Resolve a bare pipeline-script filename (e.g. `run_ocr.py`) to its vendored
    # absolute path so the script runs from pipeline/ while its working dir stays
    # the configured data directory.
    if len(parts) >= 2 and parts[1].endswith(".py") and not any(
        sep in parts[1] for sep in ("/", "\\")
    ):
        _candidate = SCRIPTS_DIR / parts[1]
        if _candidate.is_file():
            parts[1] = str(_candidate)

    env = os.environ.copy()
    if mistral_key:
        env["MISTRAL_API_KEY"] = mistral_key
    if gemini_key:
        env["GEMINI_API_KEY"] = gemini_key
    if nvidia_key:
        env["NVIDIA_API_KEY"] = nvidia_key
    if db_url:
        env["DATABASE_URL"] = db_url

    buf: list[str] = []
    with st.status(f"Running: `{cmd_str}`", expanded=True) as status:
        out_box = st.empty()
        try:
            proc = subprocess.Popen(
                parts, cwd=str(_wdir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except FileNotFoundError as e:
            status.update(label=f"✗ launch failed — {cmd_str}", state="error")
            st.error(f"Could not launch (`{parts[0]}` not found?): {e}")
            return
        except Exception as e:
            status.update(label=f"✗ launch failed — {cmd_str}", state="error")
            st.error(f"Could not launch: {e}")
            return

        for line in proc.stdout:
            buf.append(line.rstrip("\n"))
            out_box.code("\n".join(buf[-400:]), language="text")
        proc.wait()
        if proc.returncode == 0:
            status.update(label=f"✓ done — `{cmd_str}`", state="complete")
        else:
            status.update(
                label=f"✗ exit code {proc.returncode} — `{cmd_str}`", state="error"
            )
    st.session_state[f"prep_log_{key}"] = "\n".join(buf)


def step(key: str, label: str, desc: str, default_cmd: str) -> None:
    """One pipeline step: description, editable command, Run button, last log."""
    st.markdown(f"**{label}**")
    st.caption(desc)
    cmd = st.text_input(
        "Command", value=default_cmd, key=f"prep_cmd_{key}",
        label_visibility="collapsed",
    )
    run = st.button("▶ Run", key=f"prep_btn_{key}")
    if run:
        run_command(cmd, key)
    elif st.session_state.get(f"prep_log_{key}"):
        # Popover, not an expander: step() runs inside a stage expander, and
        # Streamlit forbids nesting an expander inside another expander.
        with st.popover("Last run output"):
            st.code(st.session_state[f"prep_log_{key}"][-8000:], language="text")
    st.divider()


# ── Stage 0 — Scrape ───────────────────────────────────────────────────────
with st.expander("0 · Scrape — download primary-source PDFs", expanded=False):
    step(
        "check_release", "check_new_release.py",
        "Check war.gov for a new UAP release (manifest diff vs release_state.json). "
        "Detects + downloads new PDFs and runs ALL stages below automatically. "
        "Use --dry-run to only check, --force to reprocess, --stop-after layout "
        "to skip the API-key stages.",
        "python check_new_release.py --workdir . --dry-run",
    )
    step(
        "scrape", "download_uap_pdfs.py",
        "Legacy: downloads the war.gov release_1 PDFs (static list) into "
        "UAP_PDFs/ beside the script. Superseded by check_new_release.py.",
        "python download_uap_pdfs.py",
    )

# ── Stage 1 — Layout ───────────────────────────────────────────────────────
with st.expander("1 · Layout — organise PDFs into the document tree", expanded=False):
    step(
        "split", "split_pages.py",
        "Split multi-page PDFs into per-page archive structure "
        "(e.g. the FBI reports). Adjust --src to the folder of multi-page PDFs.",
        "python split_pages.py --src FBI/reports",
    )
    step(
        "restructure", "restructure_pages.py",
        "Move page PDFs into their page_XXXX/ subfolders. Run before reorganize.",
        "python restructure_pages.py --root . --execute",
    )
    step(
        "reorganize", "reorganize.py",
        "Move document folders into raw/<agency>/<subtype>/<region>/ so the "
        "path carries full context.",
        "python reorganize.py --root . --execute",
    )

# ── Stage 2 — OCR ──────────────────────────────────────────────────────────
with st.expander("2 · OCR — page PDFs to markdown text", expanded=False):
    step(
        "stamp", "stamp_pages.py",
        "Write stamped copies into pages_out/ with YAML frontmatter "
        "(agency/subtype/region). Run after reorganize so the path is populated.",
        "python stamp_pages.py --src raw --out pages_out",
    )
    step(
        "find_ocr", "find_ocr_targets.py",
        "Scan for page folders that have a PDF but no .md; writes ocr_targets.txt.",
        "python find_ocr_targets.py --root .",
    )
    step(
        "run_ocr", "run_ocr.py",
        "OCR every target PDF via Mistral OCR. Needs MISTRAL_API_KEY. "
        "Add --limit 5 to test on a few files first.",
        "python run_ocr.py --targets ocr_targets.txt",
    )
    step(
        "destamp", "destamp_pages.py",
        "Strip the YAML frontmatter back off the individual page .md files in "
        "pages_out/. Use --dry-run to preview, or --out clean_pages to keep copies.",
        "python destamp_pages.py --src pages_out",
    )

# ── Stage 3 — Assembly ─────────────────────────────────────────────────────
with st.expander("3 · Assembly — concatenate pages per document", expanded=False):
    step(
        "coverage", "page_coverage.py",
        "Report .md-vs-total page coverage per document (confirms OCR gaps closed).",
        "python page_coverage.py --root . --out-dir .",
    )
    step(
        "concat", "concat_pages.py",
        "Concatenate each document's pages into concat/. --no-inline keeps the "
        "stamped copies clean; --force rebuilds files from a previous run.",
        "python concat_pages.py --root . --src pages_out --no-inline --execute --force",
    )

# ── Stage 4 — Extraction ───────────────────────────────────────────────────
with st.expander("4 · Extraction — concatenated text to structured reports", expanded=False):
    step(
        "extract", "extract_reports.py",
        "Gemini extracts each distinct UAP report as schema-enforced JSON into "
        "extracted/. Needs GEMINI_API_KEY.",
        "python extract_reports.py --concat concat --out extracted",
    )
    step(
        "nim_extract", "pdf_to_reports.py",
        "Optional alternative — NVIDIA NIM two-pass extraction straight from a "
        "PDF. Needs NVIDIA_API_KEY. Replace the path with your input PDF.",
        "python pdf_to_reports.py path/to/document.pdf --out reports_out",
    )
    step(
        "reconcile", "reconcile.py",
        "Build the YAML ground-truth records from the CSV manifest, the raw "
        "page tree, and the extracted reports.",
        "python reconcile.py --csv uap-csv.csv --divided raw --reports reports_out --out-dir records",
    )

# ── Stage 5 — Table creation ───────────────────────────────────────────────
with st.expander("5 · Table creation — assemble the report table", expanded=False):
    step(
        "agency", "add_source_agency.py",
        "Add agency / collection columns derived from the source-file slugs.",
        "python add_source_agency.py --input parsed_reports.xlsx",
    )
    step(
        "analyze", "analyze_reports.py",
        "Sanity-check the extracted reports for truncation / merge anomalies.",
        "python analyze_reports.py --input extracted/_all_reports.json --out anomaly_report.md",
    )
    step(
        "join", "join_reports.py",
        "Left-join the parsed reports with raw_reports_table.csv (OCR + assessment).",
        "python join_reports.py --left parsed_reports_with_agency.xlsx "
        "--right raw_reports_table.csv --out joined_reports.xlsx",
    )
    step(
        "map_yaml", "map_yaml_to_reports.py",
        "Outer-join the YAML records onto the report table → enriched_reports.xlsx.",
        "python map_yaml_to_reports.py --records records --csv raw_reports_table.csv "
        "--out enriched_reports.xlsx",
    )
    st.info(
        "The resulting table (`parsed_reports.xlsx` / `enriched_reports.xlsx`) is "
        "what the **UAP Feature Extraction** page loads — upload it there to "
        "continue into schema parsing and the SCU filters."
    )

# ── Stage 6 — Multimodal embeddings ────────────────────────────────────────
with st.expander("6 · Embed — IMG/VID/AUD assets → Gemini → pgvector", expanded=False):
    step(
        "embed_media", "embed_media.py",
        "Embed the release's images/videos/audio (from uap-csv.csv) via "
        "gemini-embedding-2 into the Neon `embeddings` table. Needs "
        "GEMINI_API_KEY + DATABASE_URL above (+ ffmpeg for video). Videos "
        "come from the release's CloudFront bundle — multi-GB download. "
        "Use --dry-run to preview, --skip-video for images only.",
        "python embed_media.py --manifest uap-csv.csv --dry-run",
    )

# ── Audit ──────────────────────────────────────────────────────────────────
with st.expander("🔍 Audit tools", expanded=False):
    step(
        "audit", "audit.py",
        "Audit the corpus against the CSV manifest — coverage, redactions, gaps.",
        "python audit.py --csv uap-csv.csv --root . --out audit_report.md",
    )
    step(
        "missing_concat", "find_missing_concat.py",
        "List document folders that are missing from concat/.",
        "python find_missing_concat.py --root .",
    )

st.caption(
    "Each step runs a bundled `pipeline/` script with the working directory as "
    "its cwd — no external script folder required. Long steps (OCR, extraction) "
    "keep this tab busy until they finish; output streams live above."
)
