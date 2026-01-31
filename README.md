---
language: [en]
tags:
  - latent-self-consistency
  - reasoning
  - evaluation
license: other
---

# Latent Self-Consistency (LatentSC)

Official implementation for Latent Self-Consistency (LSC): a latent-space self-consistency method for reliable majority-set selection in short- and long-answer reasoning.

This repository is intended for public release. It does **not** store any API keys or private tokens. Please provide all credentials via environment variables.

---

## Quick start

```bash
# 1) Install deps
python3 -m pip install -r requirements.txt

# (Recommended) FlashAttention2 (GPU only, requires compatible CUDA + PyTorch)
python3 -m pip install flash-attn --no-build-isolation

# 2) Run inference (example)
MODEL=jeongseokoh/LatentSC_llama3.1_8b DATASET=MATH bash run.sh
```

If FlashAttention2 is not available, you can switch attention implementation:
```bash
ATTN_IMPL=sdpa MODEL=jeongseokoh/LatentSC_llama3.1_8b DATASET=MATH bash run.sh
```

---

## Datasets

Supported datasets (by default in `inference.py`):
- GSM8K
- MATH
- TriviaQA
- MMLU
- TruthfulQA (generation + MC)
- CommonsenseQA
- CNN/DailyMail

### Generate datasets

All dataset artifacts are generated locally and are **not committed** to the repo.

```bash
# Generate raw data -> processed -> undersampled
bash dataset_generation/run.sh

# Mix undersampled datasets into a single JSONL
bash dataset_generation/data_mixing.sh
```

Outputs (generated locally):
- `dataset_generation/raw_data_*.jsonl`
- `dataset_generation/proc_data_*.jsonl`
- `dataset_generation/undersampled_*.jsonl`
- `dataset_generation/mixed_data.jsonl`

---

## Training

`train.sh` defaults to **mixed_data.jsonl**. If it does not exist, run the data mixing step above.

```bash
# Basic training
bash train.sh

# Change model or training options
MODEL=llama3.1_8b EPOCH=3 AGGR=mean REMOVE_EOS=True bash train.sh
```

Push to HF Hub after training:
```bash
PUSH_TO_HUB=1 \
HF_REPO_ID=yourname/LatentSC_llama3.1_8b \
HUGGINGFACE_HUB_TOKEN=hf_xxx \
bash train.sh
```

Notes:
- If `HF_REPO_ID` is not provided, the script infers a repo name using your HF username.
- Training saves to `outputs/` and can upload the final model + tokenizer.

---

## Inference

```bash
# Use a Hugging Face repo directly
MODEL=jeongseokoh/LatentSC_llama3.1_8b DATASET=MATH bash run.sh

# Adjust knobs
NUM_PATH=10 LSC_TEMP=0.5 REMOVE_EOS=True bash run.sh
```

If the HF repo contains `config.json` with LSC fields, inference will auto-load:
- `lsc_num_special_tokens`
- `lsc_aggr`
- `lsc_remove_eos`
- `lsc_temp`

---

## Sync LSC config to HF

If you need to populate `config.json` in your HF repo:

```bash
HUGGINGFACE_HUB_TOKEN=hf_xxx \
python3 tools/sync_hf_config.py \
  --repo yourname/LatentSC_llama3.1_8b \
  --num_special_tokens 6 \
  --aggr mean \
  --remove_eos True \
  --lsc_temp 0.5
```

---

## Evaluation notes

- **EvalPlus**: use the official EvalPlus repo. You can integrate LSC by editing the HF provider and prompts.
- **MSMARCO-NLG**: use the official MSMARCO-Question-Answering repo and generate `responses.json` for evaluation.

---

## Public release hygiene

This repository ships **only public-safe files**. The following are excluded via `.gitignore` and removed from the working tree:
- `logs/`, `outputs/`
- Generated datasets (`dataset_generation/*.jsonl`, `raw_data_*`, `proc_data_*`, `undersampled_*`)
- HF cache (`.hf_home/`)

If you need to reproduce results, regenerate datasets locally with the scripts above.

---

## Environment variables

Common environment variables:
- `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` for gated models / uploads
- `MODEL`, `DATASET`, `NUM_PATH`, `LSC_TEMP`, `REMOVE_EOS`, `AGGR`, `NUM_SPECIAL_TOKENS`

---

## License

Add your license here.
