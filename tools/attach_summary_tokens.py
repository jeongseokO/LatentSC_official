#!/usr/bin/env python3
import argparse
import os
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _as_bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y")


def _make_summary_tokens(num_special_tokens: int, prefix: str) -> List[str]:
    return [f"<|{prefix}{i}|>" for i in range(1, num_special_tokens + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach trained Summary-token embeddings to a backbone model."
    )
    parser.add_argument("--backbone", required=True, help="Backbone model repo id")
    parser.add_argument("--trained", required=True, help="Trained model repo id")
    parser.add_argument("--output_dir", required=True, help="Local output dir")
    parser.add_argument("--num_special_tokens", type=int, default=6)
    parser.add_argument("--summary_prefix", type=str, default="Summary")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--push_to_hub", type=str, default="False")
    parser.add_argument("--hf_repo_id", type=str, default="")
    args = parser.parse_args()

    summary_tokens = _make_summary_tokens(args.num_special_tokens, args.summary_prefix)

    print("Loading tokenizers...")
    backbone_tok = AutoTokenizer.from_pretrained(args.backbone)
    trained_tok = AutoTokenizer.from_pretrained(args.trained)

    # Keep the original backbone vocab size for checkpoint loading.
    backbone_base_vocab = len(backbone_tok)

    # Ensure backbone tokenizer has Summary tokens (for post-load resize/copy)
    backbone_tok.add_special_tokens({"additional_special_tokens": summary_tokens})
    backbone_tok.pad_token = backbone_tok.eos_token

    print("Loading models (this may take a while)...")
    torch_dtype = getattr(torch, args.torch_dtype, None)

    backbone_cfg = AutoConfig.from_pretrained(args.backbone, trust_remote_code=True)
    if hasattr(backbone_cfg, "vocab_size") and backbone_cfg.vocab_size != backbone_base_vocab:
        backbone_cfg.vocab_size = backbone_base_vocab

    trained_cfg = AutoConfig.from_pretrained(args.trained, trust_remote_code=True)
    if hasattr(trained_cfg, "vocab_size") and trained_cfg.vocab_size != len(trained_tok):
        trained_cfg.vocab_size = len(trained_tok)

    backbone_model = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        config=backbone_cfg,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.trained,
        config=trained_cfg,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    backbone_model.resize_token_embeddings(len(backbone_tok))

    # Copy Summary token embeddings by token string (ids can differ)
    b_inp = backbone_model.get_input_embeddings().weight.data
    t_inp = trained_model.get_input_embeddings().weight.data
    b_out = None
    t_out = None
    if backbone_model.get_output_embeddings() is not None:
        b_out = backbone_model.get_output_embeddings().weight.data
    if trained_model.get_output_embeddings() is not None:
        t_out = trained_model.get_output_embeddings().weight.data

    copied = 0
    for tok in summary_tokens:
        if tok not in trained_tok.get_vocab():
            print(f"[warn] Token missing in trained tokenizer: {tok}")
            continue
        if tok not in backbone_tok.get_vocab():
            print(f"[warn] Token missing in backbone tokenizer: {tok}")
            continue
        t_id = trained_tok.convert_tokens_to_ids(tok)
        b_id = backbone_tok.convert_tokens_to_ids(tok)
        if t_id is None or b_id is None:
            print(f"[warn] Failed to resolve ids for token: {tok}")
            continue
        b_inp[b_id].copy_(t_inp[t_id])
        if b_out is not None and t_out is not None:
            b_out[b_id].copy_(t_out[t_id])
        copied += 1

    # Carry over LSC config fields when present
    tcfg = trained_model.config
    bcfg = backbone_model.config
    for key in ("lsc_num_special_tokens", "lsc_special_token_prefix", "lsc_aggr", "lsc_remove_eos", "lsc_temp"):
        if hasattr(tcfg, key):
            setattr(bcfg, key, getattr(tcfg, key))
    if not hasattr(bcfg, "lsc_num_special_tokens"):
        bcfg.lsc_num_special_tokens = args.num_special_tokens
    if not hasattr(bcfg, "lsc_special_token_prefix"):
        bcfg.lsc_special_token_prefix = args.summary_prefix

    # Tie weights if needed
    try:
        backbone_model.tie_weights()
    except Exception:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
    backbone_model.save_pretrained(args.output_dir)
    backbone_tok.save_pretrained(args.output_dir)
    print(f"Saved model/tokenizer to: {args.output_dir}")
    print(f"Copied Summary token embeddings: {copied}/{len(summary_tokens)}")

    if _as_bool(args.push_to_hub):
        if not args.hf_repo_id:
            raise ValueError("hf_repo_id is required when push_to_hub=True")
        backbone_model.push_to_hub(args.hf_repo_id)
        backbone_tok.push_to_hub(args.hf_repo_id)
        print(f"Pushed model/tokenizer to: {args.hf_repo_id}")


if __name__ == "__main__":
    main()
