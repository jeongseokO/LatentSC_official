#!/usr/bin/env python3
import argparse
import os
import tempfile

from huggingface_hub import HfApi, create_repo, snapshot_download
from transformers import AutoConfig


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ["1", "true", "yes", "y"]


def infer_base_model(repo_id: str) -> str:
    name = repo_id.split("/")[-1].lower()
    if "llama3.1_8b" in name:
        return "meta-llama/Llama-3.1-8B-Instruct"
    if "llama3_8b" in name:
        return "meta-llama/Llama-3-8B-Instruct"
    if "qwen3_8b" in name:
        return "Qwen/Qwen3-8B"
    if "llama3.3_70b" in name:
        return "meta-llama/Llama-3.3-70B-Instruct"
    return os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync config.json to HF repo and local cache.")
    parser.add_argument("--repo", default=os.environ.get("HF_REPO_ID", "jeongseokoh/LatentSC_llama3.1_8b"))
    parser.add_argument("--base_model", default=os.environ.get("BASE_MODEL"))
    parser.add_argument("--num_special_tokens", type=int, default=int(os.environ.get("NUM_SPECIAL_TOKENS", "6")))
    parser.add_argument("--aggr", default=os.environ.get("AGGR", "mean"))
    parser.add_argument("--remove_eos", default=os.environ.get("REMOVE_EOS", "True"))
    parser.add_argument("--lsc_temp", type=float, default=float(os.environ.get("LSC_TEMP", "0.5")))
    parser.add_argument("--push", default=os.environ.get("PUSH_TO_HUB", "1"))
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME", "/workspace/.hf_home"))
    args = parser.parse_args()

    repo_id = args.repo
    base_model = args.base_model or infer_base_model(repo_id)
    remove_eos = parse_bool(args.remove_eos)
    push = parse_bool(args.push)

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    os.environ.setdefault("HF_HOME", args.hf_home)

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True, token=token)
    cfg.lsc_num_special_tokens = args.num_special_tokens
    cfg.lsc_special_token_prefix = "Summary"
    cfg.lsc_aggr = args.aggr
    cfg.lsc_remove_eos = remove_eos
    cfg.lsc_temp = args.lsc_temp
    cfg.base_model = base_model

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_pretrained(tmpdir)
        config_path = os.path.join(tmpdir, "config.json")

        if push:
            if not token:
                raise SystemExit("PUSH_TO_HUB is enabled but no HF token found.")
            create_repo(repo_id, exist_ok=True, token=token)
            api = HfApi()
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.json",
                repo_id=repo_id,
                token=token,
            )

    # Refresh local cache (pull config.json)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json"],
        token=token,
    )

    print(f"[ok] synced config.json for {repo_id}")


if __name__ == "__main__":
    main()
