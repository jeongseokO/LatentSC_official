#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of current working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log files (.out/.err)
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_PREFIX="${LOG_PREFIX:-run}"
TS="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"
OUT_FILE="${LOG_DIR}/${LOG_PREFIX}_${TS}.out"
ERR_FILE="${LOG_DIR}/${LOG_PREFIX}_${TS}.err"
exec > >(tee -a "$OUT_FILE") 2> >(tee -a "$ERR_FILE" >&2)

# Allow callers to override Python executable and args via env vars.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Set PYTHON_BIN or install python3." >&2
  exit 1
fi

# Optional Hugging Face token (for gated models)
HF_TOKEN="${HF_TOKEN:-}"
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
if [[ -n "$HF_TOKEN" && -z "$HUGGINGFACE_HUB_TOKEN" ]]; then
  HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

MODEL="${MODEL:-/workspace/LatentSC_official/outputs/llama3.1_8b_with_summary}"
DATASET="${DATASET:-MATH}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
NUM_SPECIAL_TOKENS="${NUM_SPECIAL_TOKENS:-6}"
NUM_PATH="${NUM_PATH:-10}"
AGGR="${AGGR:-mean}"
LSC_TEMP="${LSC_TEMP:-0.5}"
REMOVE_EOS="${REMOVE_EOS:-True}"
USE_EOS="${USE_EOS:-False}"
USE_EOS_SECOND="${USE_EOS_SECOND:-False}"
SEED="${SEED:-42}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  HF_TOKEN="$HF_TOKEN" \
  "$PYTHON_BIN" "${REPO_ROOT}/inference.py" \
  --model "$MODEL" \
  --num_special_tokens "$NUM_SPECIAL_TOKENS" \
  --num_path "$NUM_PATH" \
  --aggr "$AGGR" \
  --dataset "$DATASET" \
  --max_samples "$MAX_SAMPLES" \
  --LSC_TEMP "$LSC_TEMP" \
  --remove_eos "$REMOVE_EOS" \
  --use_eos "$USE_EOS" \
  --use_eos_second "$USE_EOS_SECOND" \
  --seed "$SEED"
