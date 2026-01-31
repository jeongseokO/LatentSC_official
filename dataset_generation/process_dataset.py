# process_dataset.py
import argparse
import json
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

def parse_args():
    p = argparse.ArgumentParser(description="Post-process JSONL for contrastive learning")
    p.add_argument(
        "--model", required=True,
        choices=["llama", "qwen"],
        help="Model type for dataset generation"
    )
    p.add_argument(
        "--dataset", required=True,
        choices=["gsm8k", "math", "triviaqa", "mmlu", "mmlu_cot"],
        help="Dataset type"
    )
    p.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed"
    )
    p.add_argument(
        "--step", type=str, default="True",
        help="CoT step or not"
    )
    return p.parse_args()

def undersample_dataset(input_path: Path, output_path: Path, threshold: float = 0.5):
    """
    Examines response groups by question_id and applies undersampling:
      1) If all response_ids are unique → discard all
      2) If a specific response_id is overly dominant (ratio > threshold) → discard all
    Keeps remaining records and saves them to a new file (output_path).
    """
    if not input_path.exists():
        print(f"⚠️  Input file not found: {input_path}")
        return
    print(f"📋 [{input_path.name}] Undersampling threshold: {threshold:.2f}")

    # 1) Load records
    records = [json.loads(line) for line in input_path.open("r", encoding="utf-8")]

    # 2) Group by question_id
    q2recs = defaultdict(list)
    for rec in records:
        q2recs[rec["question_id"]].append(rec)

    # 3) Undersampling logic
    kept, dropped = [], 0
    for qid, recs in q2recs.items():
        total = len(recs)
        counts = Counter(r["response_id"] for r in recs)

        # (A) All unique → discard
        if len(counts) == total:
            dropped += total
            continue

        max_count = max(counts.values())
        # (B) Dominant response ratio exceeds threshold → discard
        if max_count / total > threshold:
            dropped += total
        else:
            kept.extend(recs)

    # 4) Save results
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in kept:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ '{output_path.name}' saved: kept {len(kept)} out of {len(records)} total, dropped {dropped} (questions: {len(q2recs)})")

def map_numbers_to_letters(record, is_alphabet=True):
    """
    Number → alphabet mapping used only in GSM8K/Math.
    Replaces numbers appearing in questions (user) and answers (assistant) with A, B, C...
    """
    nums_user      = re.findall(r"\d+", record["user"])
    nums_assistant = re.findall(r"\d+", record["assistant"])
    sorted_nums    = sorted(set(nums_user + nums_assistant), key=int)
    if not sorted_nums:
        return record

    letters = [chr(ord("A") + i) for i in range(len(sorted_nums))]
    num2letter = dict(zip(sorted_nums, letters))
    pattern = re.compile(r"(?<!\d)(" + "|".join(map(re.escape, sorted_nums)) + r")(?!\d)")

    def repl(m: re.Match) -> str:
        return num2letter[m.group(1)]
    if is_alphabet:
        user_rec = pattern.sub(repl, record["user"])
        assist_rec = pattern.sub(repl, record["assistant"])
    else:
        user_rec = record["user"]
        assist_rec = record["assistant"]
    return {
        "question_id": record["question_id"],
        "response_id": record["response_id"],
        "system":      record["system"],
        "user":        user_rec,
        "assistant":   assist_rec,
        "answer":      record["answer"],
    }

def unbox_assistant(record, remove_dollar: bool = False):
    """
    1) \\boxed{X} → X
    2) Remove '\' before numbers/uppercase letters (\10 → 10, \A → A)
    3) (Only when remove_dollar=True) $number$ → number
    """
    text = record["assistant"]
    text = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?=[A-Z]|\d)", "", text)
    if remove_dollar:
        text = re.sub(r"\$(\d+(?:\.\d+)?)\$", r"\1", text)

    rec = record.copy()
    rec["assistant"] = text

    # Apply same processing to answer field
    ans = rec["answer"]
    ans = re.sub(r"\\(?=\d+)", "", ans)
    if remove_dollar:
        ans = re.sub(r"\$(\d+(?:\.\d+)?)\$", r"\1", ans)
    rec["answer"] = ans

    return rec

def to_open_ended(record):
    """
    MMLU specific: 
    - Use only first line of question
    - Cut assistant response from "Therefore, the answer is" onwards
    - Return None if "Therefore, the answer is" not found (for dropping)
    """
    q_line = record["user"].split("\n")[0]
    assistant_text = record["assistant"]
    
    # Find last occurrence of "Therefore, the answer is"
    target_phrase = "Therefore, the answer is"
    last_pos = assistant_text.rfind(target_phrase)
    
    # Return None if phrase not found (mark for dropping)
    if last_pos == -1:
        return None
    
    # Cut from that phrase onwards
    assistant_new = assistant_text[:last_pos].strip()
    
    return {
        "question_id": record["question_id"],
        "response_id": record["response_id"],
        "system":    record["system"],
        "user":      q_line,
        "assistant": assistant_new,
        "answer":    record["answer"]
    }

def main():
    args = parse_args()
    random.seed(args.seed)
    is_alphabet = False # Not used in this script, but kept for consistency with the original code
    input_path  = Path(f"raw_data_{args.dataset}_{args.model}.jsonl")
    if is_alphabet:
        output_path = Path(f"proc_data_alphabet_{args.dataset}_{args.model}.jsonl")
    else:
        output_path = Path(f"proc_data_{args.dataset}_{args.model}.jsonl")

    # Load original JSONL
    raw = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))

    processed = []
    dropped_count = 0

    if args.dataset in ("gsm8k", "math"):
        # Apply number→alphabet mapping to only half of records per question
        qid2idxs = defaultdict(list)
        for idx, rec in enumerate(raw):
            qid2idxs[rec["question_id"]].append(idx)

        mapped_set = set()
        for qid, idxs in qid2idxs.items():
            k = len(idxs) // 2
            mapped_set.update(random.sample(idxs, k))

        for i, rec in enumerate(raw):
            rec2 = rec
            if i in mapped_set:
                rec2 = map_numbers_to_letters(rec2, is_alphabet)
            # Remove $...$ only for GSM8K
            rec2 = unbox_assistant(rec2, remove_dollar=(args.dataset == "gsm8k"))
            processed.append(rec2)

    elif args.dataset == "triviaqa":
        # Apply unbox to all, preserve $...$
        for rec in raw:
            rec2 = unbox_assistant(rec, remove_dollar=False)
            processed.append(rec2)

    else:  # mmlu / mmlu_cot
        for rec in raw:
            rec2 = to_open_ended(rec)
            if rec2 is not None:  # Process only non-None cases
                rec2 = unbox_assistant(rec2, remove_dollar=False)
                processed.append(rec2)
            else:
                dropped_count += 1

    # Reassign IDs based on answer
    answer2id = {}
    next_id = 0
    for rec in processed:
        ans = rec["answer"]
        if ans not in answer2id:
            answer2id[ans] = next_id
            next_id += 1
        rec["id"] = answer2id[ans]

    # Set system prompt
    if args.dataset in ("gsm8k", "math"):
        system_prompt = "You are a methodical mathematician, adept at solving complex mathematical problems. "
    elif args.dataset in ("mmlu", "mmlu_cot"):
        system_prompt = "You are a methodical problem solver, adept at solving complex problems."
    else:  # triviaqa
        system_prompt = "You are a methodical problem solver, adept at solving complex problems. "

    # Save as JSONL
    # Distinguish mmlu vs mmlu_cot in question_id prefix
    if args.dataset == "mmlu_cot":
        qid_prefix = "mmlu_step_"
    elif args.dataset == "mmlu":
        qid_prefix = "mmlu_"
    else:
        qid_prefix = f"{args.dataset}_step_"

    with output_path.open("w", encoding="utf-8") as f:
        for rec in processed:
            out = {
                "question_id":   f"{qid_prefix}{rec['question_id']}",
                "response_id":   f"{rec['response_id']}",
                "system":        system_prompt,
                "user":          rec["user"],
                "assistant":     rec["assistant"],
                "answer":        rec["answer"]
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"✅ '{args.dataset}' data preprocessing completed → {output_path} ({len(processed)} records)")
    if args.dataset == "mmlu" and dropped_count > 0:
        print(f"⚠️  Records dropped due to missing 'Therefore, the answer is': {dropped_count}")
    
    # Execute undersampling
    undersampled_path = Path(f"undersampled_{args.dataset}_{args.model}.jsonl")
    undersample_dataset(output_path, undersampled_path)

if __name__ == "__main__":
    main()
