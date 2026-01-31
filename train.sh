#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of current working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log files (.out/.err)
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_PREFIX="${LOG_PREFIX:-train}"
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

# Training args (match train.py defaults)
MODEL="${MODEL:-llama3.1_8b}"
NUM_SPECIAL_TOKENS="${NUM_SPECIAL_TOKENS:-6}"
EPOCH="${EPOCH:-3}"
AGGR="${AGGR:-mean}" # mean | last
REMOVE_EOS="${REMOVE_EOS:-True}" # True | False
PUSH_TO_HUB="${PUSH_TO_HUB:-1}"   # 1 | 0 | True | False
HF_REPO_ID="${HF_REPO_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Use mixed_data.jsonl by default (override with USE_MIXED_DATA=0 or MIXED_DATA_PATH).
USE_MIXED_DATA="${USE_MIXED_DATA:-1}"
MIXED_DATA_PATH="${MIXED_DATA_PATH:-${REPO_ROOT}/dataset_generation/mixed_data.jsonl}"

WORK_DIR="$REPO_ROOT"
TEMP_DATA_DIR=""

cleanup_temp_dir() {
  if [[ -n "$TEMP_DATA_DIR" ]]; then
    rm -rf "$TEMP_DATA_DIR"
  fi
}

if [[ "$USE_MIXED_DATA" == "1" || "$USE_MIXED_DATA" == "true" ]]; then
  if [[ ! -f "$MIXED_DATA_PATH" ]]; then
    echo "Mixed data file not found: $MIXED_DATA_PATH" >&2
    exit 1
  fi

  family=""
  if [[ "$MODEL" == *"llama"* ]]; then
    family="llama"
  elif [[ "$MODEL" == *"qwen"* ]]; then
    family="qwen"
  fi

  TEMP_DATA_DIR="$(mktemp -d)"
  DATA_DIR="${TEMP_DATA_DIR}/dataset_generation"
  mkdir -p "$DATA_DIR"

  for ds in gsm8k math triviaqa mmlu mmlu_cot; do
    : > "${DATA_DIR}/undersampled_${ds}_${family}.jsonl"
  done

  ln -sf "$MIXED_DATA_PATH" "${DATA_DIR}/undersampled_gsm8k_${family}.jsonl"

  WORK_DIR="$TEMP_DATA_DIR"
  trap cleanup_temp_dir EXIT
fi

(
  cd "$WORK_DIR"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    HF_TOKEN="$HF_TOKEN" \
    "$PYTHON_BIN" "${REPO_ROOT}/train.py" \
    --model "$MODEL" \
    --num_special_tokens "$NUM_SPECIAL_TOKENS" \
    --epoch "$EPOCH" \
    --aggr "$AGGR" \
    --remove_eos "$REMOVE_EOS" \
    --push_to_hub "$PUSH_TO_HUB" \
    --hf_repo_id "$HF_REPO_ID" \
    --output_dir "$OUTPUT_DIR"
)
