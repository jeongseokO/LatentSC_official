#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Log files (.out/.err)
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_PREFIX="${LOG_PREFIX:-dataset_generation}"
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

# datasets: gsm8k | math | triviaqa | mmlu | mmlu_cot
DATASET="${DATASET:-}"
# Optional: run multiple datasets in one go (comma or space separated)
DATASETS="${DATASETS:-triviaqa,mmlu,mmlu_cot}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-llama}" # process_dataset expects: llama | qwen
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
NUM_RESPONSES="${NUM_RESPONSES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1500}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
BACKEND="${BACKEND:-vllm}" # vllm | transformers
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

cd "$SCRIPT_DIR"

# Optional vLLM flags
EXTRA_VLLM_FLAGS=()
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  EXTRA_VLLM_FLAGS+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
fi
if [[ "$VLLM_TRUST_REMOTE_CODE" == "1" || "$VLLM_TRUST_REMOTE_CODE" == "true" || "$VLLM_TRUST_REMOTE_CODE" == "True" ]]; then
  EXTRA_VLLM_FLAGS+=(--vllm_trust_remote_code)
fi

run_one_dataset() {
  local ds="$1"

  # Ensure output directory exists even when MODEL_NAME contains slashes.
  local raw_output_path="raw_data_${ds}_${MODEL_NAME}.jsonl"
  mkdir -p "$(dirname "$raw_output_path")"

  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    VLLM_ENABLE_V1_MULTIPROCESSING="$VLLM_ENABLE_V1_MULTIPROCESSING" \
    VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
    HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    HF_TOKEN="$HF_TOKEN" \
    "$PYTHON_BIN" "${SCRIPT_DIR}/dataset_generation.py" \
    --model "$MODEL_NAME" \
    --num_samples "$NUM_SAMPLES" \
    --num_responses "$NUM_RESPONSES" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --precision "$PRECISION" \
    --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --gen_batch_size "$GEN_BATCH_SIZE" \
    --dataset "$ds" \
    --backend "$BACKEND" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    "${EXTRA_VLLM_FLAGS[@]}" \
    ${DATASET_CONFIG:+--dataset_config "$DATASET_CONFIG"}

  local process_dataset="${PROCESS_DATASET:-$ds}"
  local raw_process_path="raw_data_${process_dataset}_${MODEL_TYPE}.jsonl"
  mkdir -p "$(dirname "$raw_process_path")"
  if [[ -f "$raw_output_path" ]]; then
    cp -f "$raw_output_path" "$raw_process_path"
  fi

  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
    HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    HF_TOKEN="$HF_TOKEN" \
    "$PYTHON_BIN" "${SCRIPT_DIR}/process_dataset.py" \
    --model "$MODEL_TYPE" \
    --dataset "$process_dataset" \
    --seed "$SEED"
}

if [[ -n "$DATASETS" ]]; then
  IFS=', ' read -r -a DATASET_LIST <<< "$DATASETS"
  for ds in "${DATASET_LIST[@]}"; do
    [[ -z "$ds" ]] && continue
    run_one_dataset "$ds"
  done
else
  run_one_dataset "$DATASET"
fi
