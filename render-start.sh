#!/usr/bin/env bash
# Render entrypoint for the FastAPI backend (see render.yaml).
#
# Downloads the .h5 datasets onto the persistent disk on first boot (only when
# a URL is provided and the file isn't already there), then starts uvicorn on
# the port Render injects via $PORT. Safe to run without any DATASET_URL_* set —
# it just skips the download and the app boots without the pre-parsed dataset.
set -euo pipefail

mkdir -p "${HF_HOME:-/data/huggingface}"

download_dataset() {
  local url="$1" dest="$2"
  if [ -z "${url:-}" ]; then
    echo "[start] no URL for ${dest} — skipping (feature degrades gracefully)"
    return 0
  fi
  if [ -f "${dest}" ]; then
    echo "[start] ${dest} already present ($(du -h "${dest}" | cut -f1)) — skipping download"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "[start] downloading ${dest} ..."
  curl -fL --retry 5 --retry-delay 5 -o "${dest}.partial" "${url}"
  mv "${dest}.partial" "${dest}"
  echo "[start] downloaded ${dest} ($(du -h "${dest}" | cut -f1))"
}

download_dataset "${DATASET_URL_WEST:-}" "${UAP_DATASET_WEST:-/data/parsed_files_distance_embeds.h5}"
download_dataset "${DATASET_URL_EAST:-}" "${UAP_DATASET_EAST:-/data/final_ufoseti_dataset.h5}"

echo "[start] launching uvicorn on port ${PORT:-8000}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
