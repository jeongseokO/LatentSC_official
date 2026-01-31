#!/usr/bin/env bash
set -euo pipefail

# Mix undersampled datasets with ratio:
# gsm8k:math:triviaqa:mmlu:mmlu_cot = 1:2:2:2.5:2.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Set PYTHON_BIN or install python3." >&2
  exit 1
fi

SEED="${SEED:-42}"
OUT_PATH="${OUT_PATH:-${SCRIPT_DIR}/mixed_data.jsonl}"

GSM8K_PATH="${GSM8K_PATH:-${SCRIPT_DIR}/undersampled_gsm8k_llama.jsonl}"
MATH_PATH="${MATH_PATH:-${SCRIPT_DIR}/undersampled_math_llama.jsonl}"
TRIVIAQA_PATH="${TRIVIAQA_PATH:-${SCRIPT_DIR}/undersampled_triviaqa_llama.jsonl}"
MMLU_PATH="${MMLU_PATH:-${SCRIPT_DIR}/undersampled_mmlu_llama.jsonl}"

RATIOS="1,2,2,2.5,2.5"

GSM8K_PATH="$GSM8K_PATH" \
MATH_PATH="$MATH_PATH" \
TRIVIAQA_PATH="$TRIVIAQA_PATH" \
MMLU_PATH="$MMLU_PATH" \
RATIOS="$RATIOS" \
OUT_PATH="$OUT_PATH" \
SEED="$SEED" \
"$PYTHON_BIN" - <<'PY'
import json
import os
import random
import sys

seed = int(os.environ.get("SEED", "42"))
random.seed(seed)

paths = {
    "gsm8k": os.environ.get("GSM8K_PATH"),
    "math": os.environ.get("MATH_PATH"),
    "triviaqa": os.environ.get("TRIVIAQA_PATH"),
    "mmlu": os.environ.get("MMLU_PATH"),
    "mmlu_cot": os.environ.get("MMLU_COT_PATH"),
}

ratios = [float(x) for x in os.environ.get("RATIOS", "1,2,2,2.5,2.5").split(",")]
keys = ["gsm8k", "math", "triviaqa", "mmlu", "mmlu_cot"]

def load_jsonl(p):
    if not p or not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

datasets = {}
for k in keys:
    try:
        datasets[k] = load_jsonl(paths[k])
    except FileNotFoundError:
        missing_path = paths[k] if paths[k] else "<unset>"
        print(f"[warn] missing {k} file: {missing_path}", file=sys.stderr)
        datasets[k] = []

# If mmlu_cot file is missing but mmlu contains mixed tasks, split by task field.
if not datasets["mmlu_cot"] and datasets["mmlu"]:
    mmlu_only = []
    mmlu_cot = []
    for rec in datasets["mmlu"]:
        qid = str(rec.get("question_id", ""))
        task = rec.get("task", "")
        if task == "multiple_choice_cot" or qid.startswith("mmlu_step_"):
            mmlu_cot.append(rec)
        else:
            mmlu_only.append(rec)
    if mmlu_cot:
        datasets["mmlu"] = mmlu_only
        datasets["mmlu_cot"] = mmlu_cot
        print(f"[info] split mmlu into mmlu={len(mmlu_only)} and mmlu_cot={len(mmlu_cot)} by task/qid", file=sys.stderr)
    else:
        # Fallback: random split by target ratio if task field is missing
        ratio_mmlu = ratios[keys.index("mmlu")]
        ratio_mmlu_cot = ratios[keys.index("mmlu_cot")]
        total = ratio_mmlu + ratio_mmlu_cot
        if total > 0:
            frac_cot = ratio_mmlu_cot / total
            data = datasets["mmlu"][:]
            random.shuffle(data)
            cut = int(round(len(data) * frac_cot))
            datasets["mmlu_cot"] = data[:cut]
            datasets["mmlu"] = data[cut:]
            print(f"[info] split mmlu into mmlu={len(datasets['mmlu'])} and mmlu_cot={len(datasets['mmlu_cot'])} by ratio (no task/qid match)", file=sys.stderr)

if not datasets["gsm8k"]:
    print("[error] gsm8k dataset is empty or missing; cannot compute base ratio.", file=sys.stderr)
    sys.exit(1)

base = len(datasets["gsm8k"])
targets = {k: int(round(base * r)) for k, r in zip(keys, ratios)}

mixed = []
for k in keys:
    data = datasets[k]
    target = targets[k]
    if not data:
        print(f"[warn] skipping {k}; no data.", file=sys.stderr)
        continue
    if len(data) < target:
        print(f"[warn] {k} has {len(data)} < target {target}; using all.", file=sys.stderr)
        sample = data[:]
    else:
        sample = random.sample(data, target)
    mixed.extend(sample)

random.shuffle(mixed)

out_path = os.environ["OUT_PATH"]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for rec in mixed:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"[ok] mixed size={len(mixed)} written to {out_path}")
print("[ok] targets:", targets)
PY
