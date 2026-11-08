#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/runtime_env.sh"
PYTHON_BIN="${DERMAGUARD_DERMFM_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(xinfu_env_bin dermfm-zero)/python"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m pad_multimodal.train "$@"
