# train_contrastive_full.py

import os
import json
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import create_repo, HfApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,   default="llama3.1_8b")
parser.add_argument('--num_special_tokens', type=int, default=6,
                    help="Number of special Summary tokens to use")
parser.add_argument('--epoch', type=int,   default=3)
parser.add_argument('--aggr', type=str, default="last", choices=["mean", "last"],
                    help="Aggregation method for feature extraction")
parser.add_argument('--remove_eos', type=str, default="False",
                    choices=["True","False"],
                    help="If you want to attach Special tokens without EOS, set this to True")
parser.add_argument('--push_to_hub', type=str, default=os.environ.get("PUSH_TO_HUB", "1"),
                    help="Push final model/tokenizer to Hugging Face Hub (1/0/True/False)")
parser.add_argument('--hf_repo_id', type=str, default=os.environ.get("HF_REPO_ID"),
                    help="Hugging Face repo id (e.g., username/LatentSC_llama3)")
parser.add_argument('--output_dir', type=str, default=os.environ.get("OUTPUT_DIR"),
                    help="Local output directory for saving the final model/tokenizer")

args = parser.parse_args()

# Boolean 설정
remove_eos = args.remove_eos == "True"
push_to_hub = str(args.push_to_hub).lower() in ["1", "true", "yes", "y"]
# Special tokens 정의
special_tokens_only = [f"<|Summary{i}|>" for i in range(1, args.num_special_tokens + 1)]

# Model name 설정
model_name = args.model

# SPECIAL_TOKEN 구성
SPECIAL_TOKEN = ''.join(special_tokens_only)

# Repository naming (config handles aggregation/remove_eos)
# ─────────────────────────────────────────────────────────────────────────────
# 0) 설정
SEED          = 42
DATASETS      = ["gsm8k", "math", "triviaqa", "mmlu", "mmlu_cot"]

# Model selection
if model_name == "llama3.1_8b":
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
elif model_name == "qwen3_8b":
    MODEL_NAME = "Qwen/Qwen3-8B"
else:
    # 기타 모델의 경우 직접 모델 이름 지정 필요
    MODEL_NAME = model_name  # 또는 raise ValueError(f"Unknown model: {model_name}")

LR            = 5e-4
TEMPERATURE   = 0.07
NUM_EPOCHS    = args.epoch
SPLIT_RATIOS  = (0.9, 0.1)
HF_REPO_ID    = None #YOUR huggingface Repository URL ex) anonymous/LatentSC_llama3
# Allow CLI/env override for repo id
HF_REPO_ID    = args.hf_repo_id or HF_REPO_ID
# reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

family = ""
if "llama" in model_name:
    family = "llama"
elif "qwen" in model_name:
    family = "qwen"
records = []
for ds in DATASETS:
    path = Path(f"dataset_generation/undersampled_{ds}_{family}.jsonl")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

qid2recs = {}
for rec in records:
    qid2recs.setdefault(rec["question_id"], []).append(rec)
groups = list(qid2recs.values())
random.shuffle(groups)

# split
n = len(groups)
n_train = int(n * SPLIT_RATIOS[0])
n_val   = int(n * SPLIT_RATIOS[1])
train_groups = groups[:n_train]
val_groups   = groups[n_train:]

print(f"Groups: total={n}, train={len(train_groups)}, val={len(val_groups)}")

class GroupDataset(Dataset):
    def __init__(self, groups, tokenizer, special_token):
        self.groups = groups
        self.tokenizer = tokenizer
        self.special = special_token

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        prompts, labels = [], []
        for rec in group:
            if args.model == "gemma3_12b":
              messages = [
                {"role": "system",    "content": [{"type": "text", "text":rec["system"]}]},
                {"role": "user",      "content": [{"type": "text", "text": rec["user"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": rec["assistant"]}]},
            ]
            else:
              messages = [
                  {"role": "system",    "content": rec["system"]},
                  {"role": "user",      "content": rec["user"]},
                  {"role": "assistant", "content": rec["assistant"]},
              ]
            # apply_chat_template → formatted string
            if args.model == "qwen3_8b":
                chat = self.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
            else:
                chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

            # remove_eos가 True이면 EOS 토큰 제거
            if remove_eos and self.tokenizer.eos_token and chat.endswith(self.tokenizer.eos_token):
                chat = chat[:-len(self.tokenizer.eos_token)]

            chat += self.special
            prompts.append(chat)
            labels.append(rec["response_id"])
        return prompts, labels

def collate_fn(batch):
    # batch_size=1 으로 그룹 단위 처리
    prompts, labels = batch[0]
    labels = torch.as_tensor([int(l) for l in labels], dtype=torch.long)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs, labels

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Special tokens 추가
new_special_tokens = special_tokens_only.copy()
tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
print(f"토크나이저 Special Token 추가 완료: {new_special_tokens}")
print(f"SPECIAL_TOKEN string: {SPECIAL_TOKEN}")

# pad_token = eos_token 으로 지정
tokenizer.pad_token = tokenizer.eos_token
print(f"토크나이저 설정 완료: pad_token={tokenizer.pad_token}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto", torch_dtype=torch.bfloat16
)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
print(f"모델 로드 완료: {MODEL_NAME}")

model.config.pad_token_id = tokenizer.pad_token_id
print(f"모델 pad_token_id 설정 완료: {model.config.pad_token_id}")
model.config.lsc_num_special_tokens = args.num_special_tokens
model.config.lsc_special_token_prefix = "Summary"
model.config.lsc_aggr = args.aggr
model.config.lsc_remove_eos = remove_eos
model.config.lsc_temp = TEMPERATURE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for p in model.parameters():
    p.requires_grad = False
print("모델 파라미터 Freeze 완료")
inp_embed = model.get_input_embeddings()
print(f"전체 모델 파라미터 수: {model.num_parameters()}")

# 새로 추가된 토큰들만 학습 대상으로
special_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_special_tokens]

print(f"학습할 special token IDs: {special_ids}")

inp_embed.weight.requires_grad = True
num_learnable = len(special_ids) * model.config.hidden_size
print(f"학습 가능한 파라미터 수: {num_learnable} ({len(special_ids)} tokens × {model.config.hidden_size} dim)")
print(f"전체 파라미터 대비 학습 가능한 파라미터 비율: "
      f"{num_learnable / model.num_parameters():.4%}")

def _mask_grad(grad):
    # grad와 같은 shape의 zero tensor 생성
    mask = torch.zeros_like(grad)
    # special_ids 위치만 gradient 허용
    for idx in special_ids:
        mask[idx] = 1.0
    return grad * mask


def extract_features(hidden_states, input_ids, attention_mask, aggr_method="last"):
    """
    hidden_states: (batch_size, seq_len, hidden_size)
    input_ids: (batch_size, seq_len)
    attention_mask: (batch_size, seq_len)
    aggr_method: "last" or "mean"
    """
    if aggr_method == "last":
        # 기존 방식: 마지막 토큰의 hidden state 사용
        seq_lens = attention_mask.sum(dim=1) - 1
        idxs = torch.arange(len(seq_lens), device=hidden_states.device)
        feats = hidden_states[idxs, seq_lens, :]
        return feats
    
    elif aggr_method == "mean":
        # 새로운 방식: latent token들의 hidden state 평균 사용
        batch_size = hidden_states.size(0)
        features = []
        
        for i in range(batch_size):
            # 각 시퀀스에서 special token들의 위치 찾기
            seq_input_ids = input_ids[i]
            seq_attention = attention_mask[i]
            
            # attention mask가 1인 부분만 고려
            valid_length = seq_attention.sum().item()
            valid_input_ids = seq_input_ids[:valid_length]
            
            # special token들의 위치 찾기
            special_positions = []
            special_ids_list = special_ids
            for special_id in special_ids_list:
                positions = (valid_input_ids == special_id).nonzero(as_tuple=True)[0]
                special_positions.extend(positions.tolist())
            
            if len(special_positions) > 0:
                # special token 위치들의 hidden state 평균
                special_positions = torch.tensor(special_positions, device=hidden_states.device)
                special_hiddens = hidden_states[i, special_positions, :]  # (num_specials, hidden_size)
                feat = special_hiddens.mean(dim=0)  # (hidden_size,)
            else:
                # special token이 없으면 마지막 토큰 사용 (fallback)
                seq_len = seq_attention.sum() - 1
                feat = hidden_states[i, seq_len, :]
            
            features.append(feat)
        
        return torch.stack(features, dim=0)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggr_method}")


def supervised_contrastive_loss(features: torch.Tensor,
                                labels: torch.Tensor,
                                temperature: float) -> torch.Tensor:
    feats = F.normalize(features, dim=1)
    logits = torch.matmul(feats, feats.T) / temperature

    labels = labels.view(-1,1)
    mask_pos = torch.eq(labels, labels.T).to(feats.device)
    mask_self = torch.eye(len(labels), dtype=torch.bool, device=feats.device)
    mask_pos = mask_pos & ~mask_self

    exp_logits = torch.exp(logits) * (~mask_self)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    denom = mask_pos.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / denom
    loss = - mean_log_prob_pos
    return loss.mean()

params = [inp_embed.weight]



optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

train_loader = DataLoader(
    GroupDataset(train_groups, tokenizer, SPECIAL_TOKEN),
    batch_size=1, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    GroupDataset(val_groups, tokenizer, SPECIAL_TOKEN),
    batch_size=1, shuffle=False, collate_fn=collate_fn
)

accelerator = Accelerator()
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)
inp_embed.weight.register_hook(_mask_grad)

print(f"gradient hook 설정 완료: {special_ids} → 1.0")
aggr_method = model.config.lsc_aggr
print(f"Feature aggregation method: {aggr_method}")


for epoch in range(1, NUM_EPOCHS+1):
    # Train
    model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        base_outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=False,  # 꺼서 메모리 절약
            return_dict=True,
            use_cache=False
        )
        hs = base_outputs.last_hidden_state
        
        # Feature extraction using specified aggregation method
        feats = extract_features(hs, inputs["input_ids"], inputs["attention_mask"], aggr_method)

        loss = supervised_contrastive_loss(feats, labels, TEMPERATURE)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            base_outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=False,  # 꺼서 메모리 절약
                return_dict=True,
                use_cache=False, 
            )
            hs = base_outputs.last_hidden_state
            
            # Feature extraction using specified aggregation method
            feats = extract_features(hs, inputs["input_ids"], inputs["attention_mask"], aggr_method)
            
            val_loss_item = supervised_contrastive_loss(feats, labels, TEMPERATURE).item()
            val_loss += val_loss_item
    avg_val = val_loss / len(val_loader)


    print(f"[Epoch {epoch}] Avg train_loss={avg_train:.4f}  Avg val_loss={avg_val:.4f}")

# Save and (optionally) push to HF Hub
model_tag = model_name.replace("/", "_")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = args.output_dir or f"outputs/{model_tag}_{args.aggr}_e{NUM_EPOCHS}_{run_id}"

accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)

if accelerator.is_main_process:
    os.makedirs(output_dir, exist_ok=True)
    unwrapped.save_pretrained(output_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved final model/tokenizer to {output_dir}")

    if push_to_hub:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            print("[warn] PUSH_TO_HUB is enabled but no HF token found. Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN.")
        else:
            if not HF_REPO_ID:
                api = HfApi()
                who = api.whoami(token=token)
                username = who.get("name") or who.get("fullname") or "unknown-user"
                repo_name = f"LatentSC_{model_tag}"
                HF_REPO_ID = f"{username}/{repo_name}"
                print(f"[info] HF_REPO_ID not set. Using inferred repo: {HF_REPO_ID}")
            create_repo(HF_REPO_ID, exist_ok=True, token=token)
            unwrapped.push_to_hub(HF_REPO_ID, token=token)
            tokenizer.push_to_hub(HF_REPO_ID, token=token)
            print(f"Pushed model/tokenizer to HF Hub: {HF_REPO_ID}")

accelerator.wait_for_everyone()
