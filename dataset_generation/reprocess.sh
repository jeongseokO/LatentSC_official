#!/usr/bin/env bash
set -euo pipefail

# Re-run process_dataset.py using existing raw_data files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Set PYTHON_BIN or install python3." >&2
  exit 1
fi

MODEL_TYPE="${MODEL_TYPE:-llama}" # process_dataset expects: llama | qwen
SEED="${SEED:-42}"
DATASETS="${DATASETS:-mmlu,mmlu_cot}"
RAW_MMLU_PATH="${RAW_MMLU_PATH:-}"
RAW_MMLU_COT_PATH="${RAW_MMLU_COT_PATH:-}"

IFS=', ' read -r -a DATASET_LIST <<< "$DATASETS"
for ds in "${DATASET_LIST[@]}"; do
  [[ -z "$ds" ]] && continue
  expected_raw="${SCRIPT_DIR}/raw_data_${ds}_${MODEL_TYPE}.jsonl"
  if [[ ! -f "$expected_raw" ]]; then
    candidate=""
    if [[ "$ds" == "mmlu" && -n "$RAW_MMLU_PATH" ]]; then
      candidate="$RAW_MMLU_PATH"
    elif [[ "$ds" == "mmlu_cot" && -n "$RAW_MMLU_COT_PATH" ]]; then
      candidate="$RAW_MMLU_COT_PATH"
    else
      # Try to find meta-llama directory layout
      candidate="$(find "$SCRIPT_DIR" -maxdepth 2 -type f -name "raw_data_${ds}_*.jsonl" | head -n 1)"
      if [[ -z "$candidate" ]]; then
        candidate="$(find "$SCRIPT_DIR" -maxdepth 3 -type f -path "*/raw_data_${ds}_*/Llama-3.1-8B-Instruct.jsonl" | head -n 1)"
      fi
    fi
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      cp -f "$candidate" "$expected_raw"
    fi
  fi
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON_BIN" "${SCRIPT_DIR}/process_dataset.py" \
    --model "$MODEL_TYPE" \
    --dataset "$ds" \
    --seed "$SEED"
done
