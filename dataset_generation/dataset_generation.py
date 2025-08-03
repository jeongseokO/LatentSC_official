import os
import argparse
import json
import random
import time
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset

# ----------------------------------------
# 1) Parser setup
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str,   default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('--dataset',        type=str,   default="openai/gsm8k")
parser.add_argument('--dataset_config', type=str,   default=None)
parser.add_argument('--num_samples',    type=int,   default=1000)
parser.add_argument('--num_responses',  type=int,   default=10)
parser.add_argument('--temperature',    type=float, default=1.0)
parser.add_argument('--max_tokens',     type=int,   default=1024)
parser.add_argument('--precision',      type=str,   default="bf16", choices=["fp32","fp16","bf16"])
parser.add_argument('--seed',           type=int,   default=99)
parser.add_argument('--batch_size',     type=int,   default=1500)  # Add batch save size
args = parser.parse_args()

# ----------------------------------------
# 2) Seed and device setup
# ----------------------------------------
set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------
# 3) Load model & tokenizer
# ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dtype = {"fp32":torch.float32, "fp16":torch.float16, "bf16":torch.bfloat16}[args.precision]
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation="flash_attention_2",
).to(device)
model.eval()
print(model)
# ----------------------------------------
# 4) Dataset preparation function (reusing existing code)
# ----------------------------------------
def prepare_dataset(dataset_name, dataset_config=None, num_samples=1000):
    """Dataset preparation and question extraction"""
    print(f"Loading dataset {dataset_name}...")
    
    # Handle specific datasets
    if "gsm8k" in dataset_name.lower():
        dataset = load_dataset("openai/gsm8k", dataset_config or "main")
        questions = [item["question"] for item in dataset["train"]][:num_samples]
        task_type = "gsm8k"
    
    elif "math" in dataset_name.lower():
        dataset = load_dataset("jeongseokoh/MATH")
        questions = [item["problem"] for item in dataset["train"]][:num_samples]
        task_type = "math"
    
    elif "mmlu" in dataset_name.lower():
        dataset = load_dataset("cais/mmlu", "all")
        # Convert data to list and shuffle
        import random
        train_items = list(dataset["auxiliary_train"])
        random.shuffle(train_items)
        
        questions = []
        for item in train_items[:num_samples]:
            question = item["question"]
            choices = [item["choices"][i] for i in range(len(item["choices"]))]
            formatted_question = f"{question}\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(65+i)}: {choice}\n"
            questions.append(formatted_question.strip())
        if dataset_name.lower() == "mmlu_cot":
            task_type = "multiple_choice_cot"
        else:
            task_type = "multiple_choice"
    
    elif "triviaqa" in dataset_name.lower():
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
        questions = [item["question"] for item in dataset["train"]][:num_samples]
        task_type = "triviaqa"
    
    else:
        # Handle general datasets
        try:
            dataset = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
            split = "train" if "train" in dataset else "validation" if "validation" in dataset else "test"
            
            # Find question field
            possible_question_fields = ["question", "query", "prompt", "text"]
            question_field = None
            for field in possible_question_fields:
                if field in dataset[split].features:
                    question_field = field
                    break
            
            if not question_field:
                raise ValueError(f"Could not find question field: {dataset[split].features.keys()}")
            
            # Shuffle and sample data
            all_items = list(dataset[split])
            random.shuffle(all_items)
            questions = [item[question_field] for item in all_items[:num_samples]]
            task_type = "general"
        except Exception as e:
            print(f"Dataset loading error: {e}")
            # Replace with sample questions
            questions = [f"Sample question {i}" for i in range(num_samples)]
            task_type = "general"
    
    print(f"Loaded {len(questions)} questions.")
    return questions, task_type

questions, task_type = prepare_dataset(args.dataset, args.dataset_config, args.num_samples)

# ----------------------------------------
# 5) System Prompt & COT example setup
#    (using original get_cot_Ex_for_data, get_messages_for_data)
# ----------------------------------------
from utils.prompts import get_cot_Ex_for_data, get_messages_for_data
cot_ex = get_cot_Ex_for_data(task_type)
if task_type == "gsm8k":
    system_prompt = ("You are a methodical mathematician, adept at solving complex mathematical problems. ")
elif task_type == "math":
    system_prompt = ("You are a methodical mathematician, adept at solving complex mathematical problems. ")
elif task_type == "multiple_choice" or task_type == "multiple_choice_cot":
    system_prompt = "You are a methodical problem solver, adept at solving complex problems."
elif task_type == "triviaqa":
    system_prompt = ("You are a methodical problem solver, adept at solving complex problems. ")

# ----------------------------------------
# 6) Response generation function (embedding logic removed)
# ----------------------------------------
@torch.inference_mode()
def generate_responses(messages, num_return, temperature, max_new_tokens):
    inp = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(device)
    
    out = model.generate(
        inp,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return,
        do_sample=True,
        temperature=temperature,
        top_p=1,
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode only after prompt length
    prompt_len = inp.shape[-1]
    texts = [ tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
              for seq in out ]
    
    return texts

# ----------------------------------------
# 7) Answer extraction utility
# ----------------------------------------
from utils.utils import extract_number, last_boxed_only_string, extract_alphabet, last_boxed_only_string_for_triviaqa
def extract_answer(response, task):
    if task == "gsm8k":
        return last_boxed_only_string(response)
    elif task == "math":
        return last_boxed_only_string(response)
    elif task == "triviaqa":
        return last_boxed_only_string_for_triviaqa(response)
    elif task == "multiple_choice" or task == "multiple_choice_cot":
        return extract_alphabet(response)
    else:
        # For general QA, return the entire last sentence
        return response.strip().split("\n")[-1]

# ----------------------------------------
# 8) Add batch save function
# ----------------------------------------
def append_batch_to_jsonl(records, jsonl_path, response2id_mapping, batch_num):
    """Save batch to JSONL file with append mode"""
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        for r in records:
            out = {
                "question_id": r["question_id"],
                "response_id": response2id_mapping[r["answer"]],
                "system":      r["system"],
                "user":        r["user"],
                "assistant":   r["assistant"],
                "answer":      r["answer"]
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"✅ Batch {batch_num} saved successfully: {len(records)} records added")

# ───────────────────────────────────────────────────────────────
# 9) Main processing loop (including batch save)
# ───────────────────────────────────────────────────────────────
all_records = []
current_batch = []
batch_num = 0
all_answers = set()  # For collecting all answers

# Set output filename in advance
jsonl_path = f"raw_data_{args.dataset}_{args.model}.jsonl"

# Delete existing file if it exists (start fresh)
if os.path.exists(jsonl_path):
    os.remove(jsonl_path)
    print(f"Deleted existing file: {jsonl_path}")

print(f"Starting processing of {len(questions)} questions (batch size: {args.batch_size})")
print(f"Save file: {jsonl_path}")

for q_idx, question in enumerate(tqdm(questions, desc="Processing questions")):
    msgs = get_messages_for_data(question, cot_ex, task_type)
    responses = generate_responses(
        msgs,
        num_return=args.num_responses,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )
    
    print(f"Question {q_idx}: {question}\n")
    print(f"Response: {responses[0]}\n")
    
    for resp in responses:
        ans = extract_answer(resp, task_type)
        # Skip and don't save if "No Match"
        if ans == "No Match":
            continue
        clean_ans = ans.replace("\\", "")
        
        record = {
            "question_id": q_idx,
            "system":      system_prompt,
            "user":        question,
            "assistant":   resp,
            "answer":      clean_ans,
            "task":        task_type
        }
        
        current_batch.append(record)
        all_records.append(record)
        all_answers.add(clean_ans)
        
        # Save when batch size is reached
        if len(current_batch) >= args.batch_size:
            # Generate response_id mapping (based on all answers so far)
            unique_answers = sorted(all_answers)
            response2id = {ans: i for i, ans in enumerate(unique_answers)}
            
            # Append batch to file
            append_batch_to_jsonl(current_batch, jsonl_path, response2id, batch_num)
            
            # Reset batch
            current_batch = []
            batch_num += 1

# ───────────────────────────────────────────────────────────────
# 10) Save last batch (if there's remaining data)
# ───────────────────────────────────────────────────────────────
if current_batch:
    # Generate final response_id mapping
    unique_answers = sorted(all_answers)
    response2id = {ans: i for i, ans in enumerate(unique_answers)}
    
    # Append last batch to file
    append_batch_to_jsonl(current_batch, jsonl_path, response2id, batch_num)

print(f"\n✅ All work completed!")
print(f"   - Total batches: {batch_num + 1}")
print(f"   - Total records: {len(all_records)}")
print(f"   - Total questions: {len(set(r['question_id'] for r in all_records))}")
print(f"   - Total response groups: {len(response2id)}")
print(f"   - Save file: {jsonl_path}")