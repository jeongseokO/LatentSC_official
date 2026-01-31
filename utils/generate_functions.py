from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from utils.prompts import get_messages, get_messages_usc
import torch
import time
import torch.nn.functional as F
import re 
import json
import gc
from typing import List
client = OpenAI(api_key="") 

def extract_boolean(response_str):
    match = re.search(r'\b(yes|no)\b', response_str, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'yes'
    else:
        return "No Match"
    
def ask_gpt_truth(
    question, gold_answer, pred_answer
) -> str:
    # 프롬프트 생성
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an assistant that determines whether a given prediction is correct based on a provided question and its correct answer."
            )
        },
        {
            "role": "user",
            "content": f"""You will be given:
- Question: {question}
- Correct Answer: {gold_answer}
- Friend A's Prediction: {pred_answer}

Please check if Friend A's prediction matches the correct answer.  
Respond with “#### yes” if it does, or “#### no” if it doesn't.
"""
        }
    ]

    # 3) GPT 호출
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=prompt,
        seed=42
    )
    content = response.choices[0].message.content.strip()

    answer = extract_boolean(content)

    return answer  # yes or no
def ask_gpt_consistency(
    responses: list[str],
    idx_a: int
) -> str:
    sentences = "\n".join(f"{i}. {resp}" for i, resp in enumerate(responses))
    num_sentences = len(responses)

    prompt = [
        {
            "role": "system",
            "content": (
                "You are an impartial referee who decides whether Friend A's selected sentence "
                "represents the most common conclusion among a list of sentences. "
                "Ignore truthfulness; focus only on meaning frequency."
            )
        },
        {
            "role": "user",
            "content": f"""Here are {num_sentences} sentences on the same topic:
{sentences}

Friend A selected sentence {idx_a}.

Steps:
1. Group the {num_sentences} sentences by the meaning of their conclusions, and count how many sentences each group contains.
2. Identify which group has the highest count.
3. Check if sentence {idx_a} belongs to that most common group:
   - If yes → output: #### yes
   - Otherwise → output: #### no

Let's think step by step."""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=prompt,
        seed=42
    )
    content = response.choices[0].message.content.strip()

    matches = re.findall(r'####\s*(yes|no)', content, re.IGNORECASE)
    if not matches:
        print(f"[Consistency parsing failed] {content!r}")
        answer = "no"
    else:
        answer = matches[-1].lower()

    return answer 

def USC_generate(
    question: str,
    responses: List[str],
    model,
    tokenizer,
    model_name: str | None = None,
    use_cot: bool = False,
):
    MB = 1024 ** 2
    device = model.device 

    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    baseline_bytes = torch.cuda.memory_allocated(device)

    t0 = time.perf_counter()

    messages = get_messages_usc(question, responses, use_cot)
    chat_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    if model_name == "qwen3_8b":
        chat_kwargs["enable_thinking"] = False
    input_ids = tokenizer.apply_chat_template(messages, **chat_kwargs).to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=False,
        return_dict_in_generate=True,
    )

    prompt_len = input_ids.shape[-1]
    usc_out = tokenizer.decode(
        outputs.sequences[0][prompt_len:], skip_special_tokens=True
    ).strip()
    m = re.search(r"Path\s*(\d+)", usc_out, re.IGNORECASE)
    if m:
        usc_index = max(int(m.group(1)) - 1, 0)
    else:
        m = re.search(r"####\s*(\d+)", usc_out)
        usc_index = int(m.group(1)) if m else 0

    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    delta_mb   = (peak_bytes - baseline_bytes) / MB
    peak_mb    = peak_bytes / MB
    base_mb    = baseline_bytes / MB
    wall       = time.perf_counter() - t0

    print(
        f"[USC Generate] base={base_mb:8.1f} MB │ "
        f"peak={peak_mb:8.1f} MB │ +Δ={delta_mb:6.3f} MB │ {wall:5.2f}s"
    )

    return usc_index
