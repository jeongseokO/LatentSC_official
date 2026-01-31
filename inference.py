#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset
from tqdm.auto import tqdm

from utils.prompts import get_cot_Ex, get_messages, get_messages_legacy
from utils.generate_functions import USC_generate, ask_gpt_consistency, ask_gpt_truth
from utils.utils import (
    last_boxed_only_string_for_triviaqa, extract_boolean, normalize_answer,
    extract_number, safe_convert_to_float, last_boxed_only_string,
    majority_vote_with_conf
)
from utils.wucs_utils import weighted_ucs_rerank, get_text_probabilities, clean_generated_sequences
import time
import random
from collections import defaultdict, Counter
import re 
import string
from utils.lsc_generate import add_enhanced_generation

import warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    ROUGE_AVAILABLE = True
    BERTSCORE_AVAILABLE = True
    print("Rouge와 BertScore 라이브러리가 정상적으로 로드되었습니다.")
except ImportError as e:
    print(f"라이브러리 로드 실패: {e}")
    print("Rouge나 BertScore 라이브러리가 없습니다.")
    ROUGE_AVAILABLE = False
    BERTSCORE_AVAILABLE = False

def calculate_rouge_scores(reference, hypothesis):
    """Calculate Rouge1, Rouge2, RougeL scores - enhanced debugging and exception handling"""
    if not ROUGE_AVAILABLE:
        print(" Rouge library not available, returning 0")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        # Add input validation
        if not reference or not hypothesis:
            print(f" Empty text input detected:")
            print(f"  Reference length: {len(reference) if reference else 0}")
            print(f"  Hypothesis length: {len(hypothesis) if hypothesis else 0}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        # Check string types
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            print(f" Non-string type input:")
            print(f"  Reference type: {type(reference)}")
            print(f"  Hypothesis type: {type(hypothesis)}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        result = {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
        
        return result
        
    except Exception as e:
        print(f" Exception during Rouge calculation: {e}")
        print(f"Reference sample: '{reference[:100]}...' (length: {len(reference)})")
        print(f"Hypothesis sample: '{hypothesis[:100]}...' (length: {len(hypothesis)})")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def calculate_bert_score(references, hypotheses):
    """Calculate BertScore (batch processing) - model caching to prevent loading every time"""
    if not BERTSCORE_AVAILABLE:
        print(" BertScore library not available, returning 0")
        return [0.0] * len(hypotheses)

    global BERTSCORE_MODEL_CACHED
    
    try:
        # Input validation
        if not references or not hypotheses:
            print(" Empty references or hypotheses")
            return [0.0] * len(hypotheses)
        
        if len(references) != len(hypotheses):
            print(f" References and hypotheses length mismatch: {len(references)} vs {len(hypotheses)}")
            return [0.0] * len(hypotheses)
        
        # Check for empty strings
        empty_pairs = []
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            if not ref or not hyp or not isinstance(ref, str) or not isinstance(hyp, str):
                empty_pairs.append(i)
        
        if empty_pairs:
            print(f" Empty text or invalid type found in {len(empty_pairs)} pairs: {empty_pairs[:5]}...")
            return [0.0] * len(hypotheses)
        
        # Load BertScore model only once (prevent loading every time)
        if not BERTSCORE_MODEL_CACHED:
            print("🔄 Loading BertScore model for the first time... (will only run once)")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"✅ BertScore model loaded successfully (device: {device}) - subsequent samples will be processed quickly")
            BERTSCORE_MODEL_CACHED = True
        
        # Calculate quickly since model is already loaded
        P, R, F1 = bert_score(
            hypotheses, 
            references, 
            lang="en", 
            verbose=False
        )
        result = F1.tolist()
        
        return result
        
    except Exception as e:
        print(f" Exception during BertScore calculation: {e}")
        print(f"References count: {len(references) if references else 0}")
        print(f"Hypotheses count: {len(hypotheses) if hypotheses else 0}")
        if references and hypotheses:
            print(f"References sample: '{references[0][:100]}...'")
            print(f"Hypotheses sample: '{hypotheses[0][:100]}...'")
        return [0.0] * len(hypotheses)

def extract_alphabet(text):
    # Prefer boxed format if present, else fallback to ####
    boxed = re.search(r"\\boxed\s*\{\s*([A-Z])\s*\}", text)
    if boxed:
        return boxed.group(1)
    pattern = r'#{4}\s*\{?([A-Z])\}?'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        if answer in string.ascii_uppercase:
            return answer
    return "No Match"

def adapt_cot_ex_for_legacy(dataset, cot_ex):
    # Legacy prompt style uses #### for some datasets
    legacy_boxed_to_hash = {"MMLU", "truthfulqa_mcqa", "commonsense_qa", "GSM8K"}
    if dataset not in legacy_boxed_to_hash:
        return cot_ex
    out = {}
    for key, item in cot_ex.items():
        ans = item.get("Answer", "")
        ans = re.sub(r"\\boxed\\s*\\{([^}]*)\\}", r"#### \\1", ans)
        out[key] = {**item, "Answer": ans}
    return out

def extract_candidate(completions, ds, gold=False):
    if gold:
        if ds=="GSM8K":
            real_ans = safe_convert_to_float(extract_number(completions).replace(',',''))
        elif ds=="MATH":
            real_ans = safe_convert_to_float(last_boxed_only_string(completions).replace(',',''))
        elif ds=="triviaqa":
            real_ans = completions['normalized_aliases']
        elif ds=="cnn_dailymail":
            real_ans = completions  # Already in string format for highlights
        else:
            real_ans = completions
        return real_ans

    ans_list = []
    if ds == "triviaqa":
        for text in completions:
            a = normalize_answer(last_boxed_only_string_for_triviaqa(text))
            ans_list.append(a)
    elif ds == "GSM8K":
        for text in completions:
            boxed = last_boxed_only_string(text)
            if boxed != "No Match":
                a = safe_convert_to_float(boxed.replace(',', '').rstrip('.'))
            else:
                a = safe_convert_to_float(extract_number(text).replace(',','').rstrip('.'))
            ans_list.append(a)
    elif ds == "MATH":
        for text in completions:
            a = safe_convert_to_float(last_boxed_only_string(text).replace(',','').rstrip('.'))
            ans_list.append(a)
    elif ds in ["truthfulqa_mcqa", "MMLU", "commonsense_qa"]:
        for text in completions:
            a = extract_alphabet(text)
            ans_list.append(a)
    elif ds == "cnn_dailymail":
        # Extract only text after "Highlights: "
        for text in completions:
            if "Highlights: " in text:
                highlights = text.split("Highlights: ", 1)[1].strip()
            else:
                highlights = text.strip()
            ans_list.append(highlights)
    else:
        ans_list = [comp.replace('\n', ' ') for comp in completions]

    return ans_list

def evaluate_function(question, pred, gold, ds, eval_cache=None):
    """Correctness evaluation function (with caching for truthfulqa)"""
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.lower() in ("true", "yes", "match")
        return False

    if ds == "triviaqa":
        is_corr = pred in gold
    elif ds == "truthfulqa":
        # Check cache first before making expensive API calls
        if eval_cache is not None:
            # Create unique key for this question-pred-gold combination
            truth_key = f"{question}|{gold}|{pred}"
            
            # Check if we already have evaluated this combination
            if truth_key in eval_cache["truth"]:
                is_corr = eval_cache["truth"][truth_key]
            else:
                # Make API call and cache the result
                result = to_bool(extract_boolean(ask_gpt_truth(question, gold, pred)))
                eval_cache["truth"][truth_key] = result
                is_corr = result
        else:
            # Legacy behavior (no caching)
            is_corr = to_bool(extract_boolean(ask_gpt_truth(question, gold, pred)))
    elif ds == "cnn_dailymail":
        # CNN DailyMail is evaluated with Rouge and BertScore, so return default value
        is_corr = True  # Score calculated separately
    else:
        is_corr = pred == gold

    return is_corr

def calculate_ranking_scores(method_results, texts, dataset):
    """
    Calculate consistency scores for each method
    Use ask_gpt_consistency only for TruthfulQA, give 1 point to all methods for others
    """
    ranking_scores = defaultdict(float)

    if dataset == "truthfulqa":
        # Use ask_gpt_consistency only for TruthfulQA
        for method_name, result in method_results.items():
            idx = result["idx"]
            if idx == -1:  # For FSC, new response was generated so consistency score is set to 1.0
                ranking_scores[method_name] = 1.0
            else:
                # Independently evaluate whether each method selected the most consistent answer
                is_most_consistent = ask_gpt_consistency(texts, idx)
                if is_most_consistent == "yes":
                    ranking_scores[method_name] = 1.0
                else:
                    ranking_scores[method_name] = 0.0
    else:
        # For other datasets, give 1 point to all methods
        for method_name in method_results.keys():
            ranking_scores[method_name] = 1.0

    return dict(ranking_scores)

def dynamic_topk_lsc(weights, ans_list, temp=0.9, decrease_threshold=0.0):
    """
    Dynamic TopK LSC: Calculate from top2 to topN each,
    find the point with the largest decrease in max_sim_score and use K right before that point
    """
    num_paths = len(ans_list)
    max_k = min(num_paths - 1, num_paths)  # Set maximum K
    
    max_scores = []
    best_indices = []
    avg_weights_list = []
    
    # Calculate from K=2 to max_k
    for k in range(2, max_k + 1):
        # Select only top K weights from each row and average
        avg_weight_topk = np.array([
            np.mean(np.sort(row)[-k:])
            for row in weights
        ])
        
        best_idx = int(np.argmax(avg_weight_topk))
        max_score = avg_weight_topk[best_idx]
        
        avg_weights_list.append(avg_weight_topk)
        max_scores.append(max_score)
        best_indices.append(best_idx)
    
    # Find point with largest decrease in max_sim_score
    optimal_k = max_k  # Default: last K
    largest_drop = 0.0
    largest_drop_idx = -1
    
    # Calculate differences between consecutive points to find largest decrease
    for i in range(1, len(max_scores)):
        score_diff = max_scores[i-1] - max_scores[i]  # Decrease from previous to current point
        if score_diff > largest_drop:
            largest_drop = score_diff
            largest_drop_idx = i
    
    # Select K right before the point with largest decrease
    if largest_drop_idx != -1 and largest_drop > 0:
        optimal_k = largest_drop_idx + 1  # largest_drop_idx is where decrease occurred, so optimal is right before
    
    # Select result corresponding to optimal_k
    k_idx = optimal_k - 2  # Convert to 0-based index (K=2 is index 0)
    if k_idx >= len(best_indices):
        k_idx = len(best_indices) - 1
    elif k_idx < 0:
        k_idx = 0
    
    selected_idx = best_indices[k_idx]
    selected_answer = ans_list[selected_idx]
    selected_conf = ans_list.count(selected_answer) / len(ans_list)
    selected_calib = avg_weights_list[k_idx][selected_idx]
    
    # Add decrease information
    score_diffs = []
    for i in range(1, len(max_scores)):
        score_diffs.append(max_scores[i-1] - max_scores[i])
    
    return {
        "answer": selected_answer,
        "optimal_k": optimal_k,
        "selected_idx": selected_idx,
        "conf": selected_conf,
        "calib": selected_calib,
        "max_scores": max_scores,
        "score_diffs": score_diffs,
        "largest_drop": largest_drop,
        "largest_drop_idx": largest_drop_idx,
        "all_k_results": list(zip(range(2, max_k + 1), max_scores, best_indices))
    }

def calculate_eos_lsc(embeddings, ans_list, temp):
    """
    Perform LSC calculation for EOS embedding (same method as existing LSC)
    """
    if embeddings is None:
        # When EOS embeddings is None (EOS token not detected in all paths)
        return {
            "answer": ans_list[0] if ans_list else None,
            "idx": 0,
            "conf": 0.0,
            "calib": 0.0,
            "weights": None
        }
    
    # Same calculation as LSC
    embs_norm = F.normalize(embeddings.float(), p=2, dim=1)
    sim_matrix = embs_norm @ embs_norm.T
    sim_np = sim_matrix.cpu().float().numpy()
    np.fill_diagonal(sim_np, 0.0)

    weights = np.exp(sim_np / temp)
    np.fill_diagonal(weights, 0.0)
    avg_weight = weights.mean(axis=1)
    best_idx = int(np.argmax(avg_weight))
    answer = ans_list[best_idx]
    conf = ans_list.count(answer) / len(ans_list)
    calib = avg_weight[best_idx]
    
    return {
        "answer": answer,
        "idx": best_idx,
        "conf": conf,
        "calib": calib,
        "weights": weights
    }

def main():
    parser = argparse.ArgumentParser(description="10-Path + <Summary> Embedding Inference with 3 EOS Methods")
    parser.add_argument(
        "--model", required=True,
        help="Model name or Hugging Face repo id (e.g., llama3.1_8b or username/LatentSC_llama3.1_8b)"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["GSM8K", "MATH", "triviaqa", "truthfulqa", "truthfulqa_mcqa", "MMLU", "commonsense_qa", "cnn_dailymail"],
        help="Choose which dataset to process"
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="How many examples to process (per dataset)"
    )
    parser.add_argument(
        "--num_special_tokens", type=int, default=6,
        help="Number of Summary special tokens to use"
    )
    parser.add_argument(
        "--aggr", type=str, default="mean",
        help="Aggregate version"
    )
    parser.add_argument(
        "--lsc_topk_list", type=str, default="5",
        help="Comma-separated list of LSC_TOPK values (e.g., '5,7')"
    )
    parser.add_argument(
        "--enable_dynamic_topk", type=str, default="True", choices=["True", "False"],
        help="Enable dynamic TopK LSC method"
    )
    parser.add_argument(
        "--LSC_TEMP", type=float, default=0.5,
        help="LSC temperature for sim matrix calculation"
    )
    parser.add_argument(
        "--remove_eos", type=str, default="True", choices=["True", "False"],
        help="Remove EOS token from generated sequences"
    )
    parser.add_argument(
        "--use_eos", type=str, default="True", choices=["True", "False"],
        help="Enable all 3 EOS methods (EOS_pred, EOS_first, EOS_special)"
    )
    parser.add_argument(
        "--use_eos_second", type=str, default="False", choices=["True", "False"],
        help="Use 2nd layer for all EOS methods (if False, use last layer)"
    )
    parser.add_argument('--num_path', type=int, default=10)
    parser.add_argument('--seed', type=int, default=77)
    args = parser.parse_args()
    
    model_arg = args.model
    enable_dynamic_topk = args.enable_dynamic_topk == "True"
    use_eos = args.use_eos == "True"
    use_eos_second = args.use_eos_second == "True"
    remove_eos = args.remove_eos == "True"

    num_special_tokens = args.num_special_tokens

    # Configuration (accept full HF repo id)
    if "/" in model_arg:
        HF_REPO = model_arg
        repo_base = model_arg.split("/")[-1]
    else:
        repo_base = f"{model_arg}_Summary{num_special_tokens}"
        HF_REPO = f"jeongseokoh/{repo_base}"

    # Prefer config values for aggregation/remove_eos/num_special_tokens if available
    try:
        cfg = AutoConfig.from_pretrained(HF_REPO, trust_remote_code=True)
        if hasattr(cfg, "lsc_num_special_tokens"):
            num_special_tokens = int(cfg.lsc_num_special_tokens)
        if hasattr(cfg, "lsc_aggr"):
            args.aggr = cfg.lsc_aggr
        if hasattr(cfg, "lsc_remove_eos"):
            remove_eos = bool(cfg.lsc_remove_eos)
        if hasattr(cfg, "lsc_temp"):
            args.LSC_TEMP = float(cfg.lsc_temp)
    except Exception:
        cfg = None
    if "/" not in model_arg:
        repo_base = f"{model_arg}_Summary{num_special_tokens}"
        HF_REPO = f"jeongseokoh/{repo_base}"
    Repo_name = repo_base
    HF_MODEL_NAME = model_arg
    is_remove_eos = remove_eos
    SPECIAL_TOKENS = [f"<|Summary{i}|>" for i in range(1, num_special_tokens + 1)]
    SPECIAL_TOKEN = SPECIAL_TOKENS[-1] if SPECIAL_TOKENS else None
    NUM_PATHS = args.num_path
    MAX_NEW_TOKENS = 2048
    TEMP = 0.9
    TOP_P = 0.95
    
    # Convert LSC_TOPK to list
    LSC_TOPK_LIST = [int(k) for k in args.lsc_topk_list.split(',')]
    
    # Define new 3 EOS methods
    eos_methods = ["EOS_pred", "EOS_first", "EOS_special"]
    
    # Define methods used in TruthfulQA and CNN DailyMail (excluding SC)
    if args.dataset in ["truthfulqa", "cnn_dailymail"]:
        all_methods = ["LSC", "WUCS", "USC", "Random"]
        if enable_dynamic_topk:
            all_methods.append("LSC_dynamic")
        if use_eos:
            all_methods.extend(eos_methods)
    else:
        # Include SC for other datasets
        all_methods = ["LSC", "SC", "WUCS", "USC", "Random"]
        if enable_dynamic_topk:
            all_methods.append("LSC_dynamic")
        if use_eos:
            all_methods.extend(eos_methods)
    
    # Print experiment configuration summary
    print("=" * 80)
    print(f"{'EXPERIMENT CONFIGURATION':^80}")
    print("=" * 80)
    print(f"• Dataset:            {args.dataset}")
    print(f"• Max Samples:        {args.max_samples}")
    print(f"• LSC_TOPK Values:    {LSC_TOPK_LIST}")
    print(f"• Dynamic TopK:       {enable_dynamic_topk}")
    print(f"• Use EOS Methods:    {use_eos}")
    if use_eos:
        print(f"• EOS Methods:        {', '.join(eos_methods)}")
        print(f"• Use EOS 2nd Layer:  {use_eos_second}")
    print(f"• Remove EOS:         {remove_eos}")
    print(f"• # Summary Tokens:   {num_special_tokens}")
    print(f"• Aggregation Style:  {args.aggr}")
    print(f"• Model:              {HF_REPO}")
    print(f"• Special Tokens:     {', '.join(SPECIAL_TOKENS)}")
    print(f"• Paths:              {NUM_PATHS}")
    print(f"• Temperature:        {TEMP}")
    print(f"• Top_P:              {TOP_P}")
    print(f"• Max New Tokens:     {MAX_NEW_TOKENS}")
    print(f"• LSC TEMP:           {args.LSC_TEMP}")
    print(f"• Seed:               {args.seed}")
    
    if args.dataset in ["truthfulqa", "cnn_dailymail"]:
        methods_str = "LSC, WUCS, USC, Random"
        if enable_dynamic_topk:
            methods_str += ", LSC_dynamic"
        if use_eos:
            methods_str += f", {', '.join(eos_methods)}"
        methods_str += f" (SC excluded for {args.dataset})"
    else:
        methods_str = ', '.join(all_methods)
    print(f"• Methods:            {methods_str}")
    print("=" * 80)

    # Baseline time & accuracy storage structure
    if args.dataset in ["truthfulqa", "cnn_dailymail"]:
        baseline_times = {
            "LSC": 0.0,
            "WUCS": 0.0,
            "USC": 0.0,
            "Random": 0.0
        }
        baseline_acc = {
            "LSC": 0.0,
            "WUCS": 0.0,
            "USC": 0.0,
            "Random": 0.0
        }
    else:
        baseline_times = {
            "LSC": 0.0,
            "WUCS": 0.0,
            "USC": 0.0,
            "SC": 0.0,
            "Random": 0.0
        }
        baseline_acc = {
            "LSC": 0.0,
            "WUCS": 0.0,
            "USC": 0.0,
            "SC": 0.0,
            "Random": 0.0
        }
    
    # Add Rouge/BertScore storage structure for CNN DailyMail
    if args.dataset == "cnn_dailymail":
        rouge_scores = {
            "LSC": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "WUCS": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "USC": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "Random": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        }
        bert_scores = {
            "LSC": 0.0,
            "WUCS": 0.0,
            "USC": 0.0,
            "Random": 0.0
        }
    
    # Add 3 EOS methods entries (only when use_eos is True)
    if use_eos:
        for eos_method in eos_methods:
            baseline_times[eos_method] = 0.0
            baseline_acc[eos_method] = 0.0
            if args.dataset == "cnn_dailymail":
                rouge_scores[eos_method] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                bert_scores[eos_method] = 0.0
    
    # Add LSC Dynamic TopK entry
    if enable_dynamic_topk:
        baseline_times["LSC_dynamic"] = 0.0
        baseline_acc["LSC_dynamic"] = 0.0
        if args.dataset == "cnn_dailymail":
            rouge_scores["LSC_dynamic"] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            bert_scores["LSC_dynamic"] = 0.0

    # Add entry for each LSC_TOPK value (only when not truthfulqa or cnn_dailymail)
    if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
        for k in LSC_TOPK_LIST:
            baseline_times[f"LSC_top{k}"] = 0.0
            baseline_acc[f"LSC_top{k}"] = 0.0

    # New ranking score storage structure (independent consistency scores)
    ranking_scores = defaultdict(float)

    # Load datasets
    DATASETS = {
        "GSM8K": ("openai/gsm8k", "main", "question", None, "answer", "test"),
        "MATH": ("jeongseokoh/MATH", None, "problem", None, "solution", "test"),
        "triviaqa": ("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", "question", None, "answer", "validation"),
        "truthfulqa": ("truthfulqa/truthful_qa", "generation", "question", None, "best_answer", "validation"),
        "truthfulqa_mcqa": ("truthfulqa/truthful_qa", "multiple_choice", "question", None, "mc1_targets", "validation"),
        "MMLU": ("cais/mmlu", "all", "question", "choices", "answer", "test"),
        "commonsense_qa": ("tau/commonsense_qa", None, "question", "choices", "answerKey", "validation"),
        "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "article", None, "highlights", "test")
    }
    hf_name, hf_cfg, qcol, ccol, acol, split = DATASETS[args.dataset]
    raw = load_dataset(hf_name, hf_cfg)[split]

    # Common initialization of method_responses
    method_responses = {method: [] for method in all_methods}
    # Add LSC_TOPK methods only when not TruthfulQA or CNN DailyMail
    if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
        for k in LSC_TOPK_LIST:
            method_responses[f"LSC_top{k}"] = []

    # Dataset branching
    if args.dataset in ["truthfulqa", "cnn_dailymail"]:
        ds = raw.select(range(min(len(raw), args.max_samples)))
    else:
        ds = raw.shuffle(seed=42).select(range(min(len(raw), args.max_samples)))

    # Initialize model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    special_tokens_for_add = [f"<|Summary{i}|>" for i in range(1, num_special_tokens + 1)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_for_add})
    tokenizer.pad_token = tokenizer.eos_token
    attn_impl = os.environ.get("ATTN_IMPL", "flash_attention_2")
    # Ensure config vocab matches tokenizer length to avoid embedding size mismatch
    # For backbone/base checkpoints, skip overriding vocab_size (set IS_BACKBONE=1).
    if cfg is None:
        cfg = AutoConfig.from_pretrained(HF_REPO, trust_remote_code=True)
    is_backbone = os.environ.get("IS_BACKBONE", "0").lower() in ("1", "true", "yes", "y")
    if (not is_backbone) and hasattr(cfg, "vocab_size") and cfg.vocab_size != len(tokenizer):
        cfg.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
            HF_REPO,
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True
        )
        
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = add_enhanced_generation(model)
    model.config.lsc_num_special_tokens = num_special_tokens
    model.config.lsc_special_token_prefix = "Summary"
    # Prefer config values when present
    if hasattr(model.config, "lsc_aggr"):
        args.aggr = model.config.lsc_aggr
    else:
        model.config.lsc_aggr = args.aggr
    if hasattr(model.config, "lsc_remove_eos"):
        remove_eos = bool(model.config.lsc_remove_eos)
    else:
        model.config.lsc_remove_eos = remove_eos
    model.eval()

    # Few-shot CoT examples
    prompt_style = os.environ.get("PROMPT_STYLE", "legacy").lower()
    cot_ex = get_cot_Ex(args.dataset)
    if prompt_style == "legacy":
        cot_ex = adapt_cot_ex_for_legacy(args.dataset, cot_ex)
    print(f"Using CoT examples for {args.dataset}:\n{cot_ex}\n")
    total = 0
    results = []
    
    # Initialize TruthfulQA evaluation cache (only for truth now, no info)
    eval_cache = {"truth": {}} if args.dataset == "truthfulqa" else None
    option_alphabets = ["A", "B", "C", "D"]
    
    # Prepare special tokens
    special_tokens = []
    for token_text in SPECIAL_TOKENS:
        try:
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            if token_ids:
                special_tokens.append(token_ids[0])
        except:
            pass
            
    generation_kwargs = {
        'max_new_tokens': MAX_NEW_TOKENS,
        'num_return_sequences': NUM_PATHS,
        'do_sample': True,
        'temperature': 0.9,
        'top_p': 0.95,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'return_dict_in_generate': True,
        "output_scores": True,
    }

    # Inference for each sample
    for qi, sample in enumerate(tqdm(ds, desc=f"Inferencing {args.dataset}")):
        # REPRODUCIBILITY SETUP
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch._dynamo.config.capture_scalar_outputs = True
        
        total += 1
        question = sample[qcol]
        gold_ans_texts = sample[acol]
        choices = None
        if ccol is not None:
            choices = sample[ccol]
        if args.dataset == "truthfulqa_mcqa":
            labels = gold_ans_texts['labels']
            gold_ans = option_alphabets[labels.index(1)]
            choices = gold_ans_texts['choices'] 
        elif args.dataset == "MMLU":
            gold_ans = option_alphabets[gold_ans_texts]
        elif args.dataset == "commonsense_qa":
            gold_ans = gold_ans_texts
            choices = sample["choices"]["text"]
        else:
            gold_ans = extract_candidate(gold_ans_texts, args.dataset, gold=True)
        
        # Example messages
        if args.dataset in ["MMLU", "truthfulqa_mcqa", "commonsense_qa"]:
            options = choices
        else:
            options = None
        if prompt_style == "legacy":
            messages = get_messages_legacy(
                question=question,
                cot_ex=cot_ex,
                model_name=args.model,
                dataset=args.dataset,
                emphasize=True,
                choices=options
            )
        else:
            messages = get_messages(
                question=question,
                cot_ex=cot_ex,
                model_name=args.model,
                dataset=args.dataset,
                emphasize=True, 
                choices=options
            )
        if "qwen" in args.model:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).to(model.device)
        else:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

        # 1) generate paths + embeddings (extract all EOS embeddings)
        with torch.inference_mode():
            out_result = model.generate_with_special_tokens(
                inputs=inputs,
                special_tokens=special_tokens,
                remove_eos=is_remove_eos,
                return_embeddings=True,
                use_eos=use_eos,
                use_eos_second=use_eos_second,
                **generation_kwargs
            )
        
        context_len = inputs.shape[1]
        embs = out_result.mean_embeddings
        sequences = out_result.sequences
        if use_eos:
            # Get 3 types of EOS embeddings
            eos_pred_embeddings = out_result.eos_pred_embeddings
            eos_first_embeddings = out_result.eos_first_embeddings
            eos_special_embeddings = out_result.eos_special_embeddings
        
        extra_time = getattr(out_result, "extra", None)
        if extra_time is None:
            extra_time = getattr(out_result, "time", 0.0)
        
        texts = []
        for i, seq in enumerate(sequences):
            seq_text = tokenizer.decode(seq[context_len:], skip_special_tokens=True)
            texts.append(seq_text)
        text_probs = get_text_probabilities(out_result)
        cleaned_sequences = clean_generated_sequences(
            out_result,
            special_tokens=special_tokens,
            eos_token_id=tokenizer.eos_token_id,
            context_len=inputs.shape[1],
            pad_token_id=tokenizer.pad_token_id
        )    
        ans_list = extract_candidate(texts, args.dataset)
        print(texts[0])
        
        # 1b) Calculate 3 EOS methods (only when use_eos is True)
        eos_results = {}
        
        if use_eos:
            # EOS_pred: embedding when predicting EOS
            s_eos_pred = time.perf_counter()
            eos_pred_result = calculate_eos_lsc(eos_pred_embeddings, ans_list, args.LSC_TEMP)
            e_eos_pred = time.perf_counter()
            eos_pred_time = (e_eos_pred - s_eos_pred) + extra_time
            eos_pred_result["time"] = eos_pred_time
            eos_results["EOS_pred"] = eos_pred_result
            baseline_times['EOS_pred'] += eos_pred_time
            
            # EOS_first: embedding when first EOS detected
            s_eos_first = time.perf_counter()
            eos_first_result = calculate_eos_lsc(eos_first_embeddings, ans_list, args.LSC_TEMP)
            e_eos_first = time.perf_counter()
            eos_first_time = (e_eos_first - s_eos_first) + extra_time
            eos_first_result["time"] = eos_first_time
            eos_results["EOS_first"] = eos_first_result
            baseline_times['EOS_first'] += eos_first_time
            
            # EOS_special: average embedding of 6 EOS
            s_eos_special = time.perf_counter()
            eos_special_result = calculate_eos_lsc(eos_special_embeddings, ans_list, args.LSC_TEMP)
            e_eos_special = time.perf_counter()
            eos_special_time = (e_eos_special - s_eos_special) + extra_time
            eos_special_result["time"] = eos_special_time
            eos_results["EOS_special"] = eos_special_result
            baseline_times['EOS_special'] += eos_special_time

        # 2) LSC calculation (overall avg)
        s_lsc = time.perf_counter()
        embs_norm = F.normalize(embs.float(), p=2, dim=1)
        sim_matrix = embs_norm @ embs_norm.T
        sim_np = sim_matrix.cpu().float().numpy()
        np.fill_diagonal(sim_np, 0.0)

        weights = np.exp(sim_np / args.LSC_TEMP)
        np.fill_diagonal(weights, 0.0)
        avg_weight = weights.mean(axis=1)
        best_idx = int(np.argmax(avg_weight))
        lsc_answer = ans_list[best_idx]
        e_lsc = time.perf_counter()
        lsc_time = (e_lsc - s_lsc) + extra_time
        lsc_conf = ans_list.count(lsc_answer) / len(ans_list)
        lsc_calib = avg_weight[best_idx]

        baseline_times['LSC'] += lsc_time

        # 2b) Calculate for each LSC_TOPK value (only when not truthfulqa or cnn_dailymail)
        lsc_topk_results = {}
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            for k in LSC_TOPK_LIST:
                s_lsc_topk = time.perf_counter()
                # Select only top K weights from each row and average
                avg_weight_topk = np.array([
                    np.mean(np.sort(row)[-k:])
                    for row in weights
                ])
                best_idx_topk = int(np.argmax(avg_weight_topk))
                lsc_topk_answer = ans_list[best_idx_topk]
                e_lsc_topk = time.perf_counter()
                lsc_topk_time = (e_lsc_topk - s_lsc_topk) + extra_time
                lsc_topk_conf = ans_list.count(lsc_topk_answer) / len(ans_list)
                lsc_topk_calib = avg_weight_topk[best_idx_topk]

                baseline_times[f"LSC_top{k}"] += lsc_topk_time
                
                lsc_topk_results[k] = {
                    "answer": lsc_topk_answer,
                    "time": lsc_topk_time,
                    "conf": lsc_topk_conf,
                    "calib": lsc_topk_calib,
                    "idx": best_idx_topk,
                    "is_corr": None,
                    "is_major": None
                }

        # 2c) Dynamic TopK calculation
        dynamic_result = None
        if enable_dynamic_topk:
            s_dynamic = time.perf_counter()
            dynamic_result = dynamic_topk_lsc(
                weights, ans_list, temp=args.LSC_TEMP
            )
            e_dynamic = time.perf_counter()
            dynamic_time = (e_dynamic - s_dynamic) + extra_time
            dynamic_result["time"] = dynamic_time
            baseline_times["LSC_dynamic"] += dynamic_time

        # SC calculation (only when not truthfulqa or cnn_dailymail)
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            s_sc = time.perf_counter()
            sc_answer, sc_conf, sc_indices = majority_vote_with_conf(ans_list)
            e_sc = time.perf_counter()
            sc_time = e_sc - s_sc
            baseline_times['SC'] += sc_time

        # WUCS calculation
        s_wucs = time.perf_counter()
        wucs_idx, wucs_calib = weighted_ucs_rerank(texts, cleaned_sequences, text_probs, use_consensus=True)
        wucs_answer = ans_list[wucs_idx]
        e_wucs = time.perf_counter()
        wucs_time = e_wucs - s_wucs
        baseline_times['WUCS'] += wucs_time

        # USC calculation
        s_usc = time.perf_counter()
        usc_idx = USC_generate(question, texts, model, tokenizer, model_name=args.model, use_cot=False)
        usc_idx = usc_idx if 0 <= usc_idx < len(ans_list) else 0
        usc_answer = ans_list[usc_idx]
        e_usc = time.perf_counter()
        usc_time = e_usc - s_usc
        baseline_times['USC'] += usc_time

        # Random calculation
        s_rand = time.perf_counter()
        rand_idx = random.randrange(NUM_PATHS)
        rand_answer = ans_list[rand_idx]
        e_rand = time.perf_counter()
        rand_time = e_rand - s_rand
        baseline_times['Random'] += rand_time

        # ===============================
        # CNN DailyMail evaluation logic - enhanced debugging
        # ===============================
        if args.dataset == "cnn_dailymail":
            print(f"\n=== CNN DailyMail evaluation start - Sample {qi+1} ===")
            print(f"Gold answer: '{gold_ans[:200]}...' (length: {len(gold_ans)})")
            print(f"Number of generated answers: {len(ans_list)}")
            
            # Output basic information for each answer
            for i, ans in enumerate(ans_list[:3]):  # Only output first 3
                print(f"Answer {i+1}: '{ans[:100]}...' (length: {len(ans)})")
                if len(ans) == 0:
                    print(f"  ⚠️ Answer {i+1} is empty string!")
            
            # Calculate Rouge scores for all completions
            all_rouge_scores = []
            print("\nCalculating Rouge scores...")
            for i, ans in enumerate(ans_list):
                rouge_result = calculate_rouge_scores(gold_ans, ans)
                all_rouge_scores.append(rouge_result)
                if i == 0:  # Output detailed log only for first one
                    print(f"Rouge scores for first answer: {rouge_result}")

            # Calculate BertScore for all completions (batch processing)
            print("Calculating BertScore...")
            all_bert_scores = calculate_bert_score([gold_ans] * len(ans_list), ans_list)
            if all_bert_scores and len(all_bert_scores) > 0:
                print(f"BertScore for first answer: {all_bert_scores[0]:.4f}")
            else:
                print(" BertScore calculation failed")

            # Check if all scores are 0
            if all_rouge_scores:
                all_rouge_zero = all(
                    score["rouge1"] == 0 and score["rouge2"] == 0 and score["rougeL"] == 0 
                    for score in all_rouge_scores
                )
                if all_rouge_zero:
                    print("⚠️ All Rouge scores are 0!")
                    print("Possible causes:")
                    print("  1. Generated text is empty or too short")
                    print("  2. Gold answer and generated answer are completely different")
                    print("  3. Rouge library issue")
            
            if all_bert_scores:
                all_bert_zero = all(score == 0 for score in all_bert_scores)
                if all_bert_zero:
                    print("⚠️ All BertScores are 0!")
                    print("Possible causes:")
                    print("  1. Generated text is empty")
                    print("  2. BertScore library issue")
                    print("  3. GPU memory shortage")

            # Store scores for answers selected by each method
            methods_data = {
                "LSC": {"idx": best_idx, "answer": lsc_answer},
                "WUCS": {"idx": wucs_idx, "answer": wucs_answer},
                "USC": {"idx": usc_idx, "answer": usc_answer},
                "Random": {"idx": rand_idx, "answer": rand_answer}
            }

            # Add 3 EOS methods (only when use_eos is True)
            if use_eos:
                for eos_method in eos_methods:
                    eos_result = eos_results[eos_method]
                    methods_data[eos_method] = {"idx": eos_result["idx"], "answer": eos_result["answer"]}

            # Add LSC Dynamic
            if enable_dynamic_topk and dynamic_result is not None:
                methods_data["LSC_dynamic"] = {"idx": dynamic_result["selected_idx"], "answer": dynamic_result["answer"]}

            for method, data in methods_data.items():
                idx = data["idx"]
                answer = data["answer"]
                method_responses[method].append(answer)

                # Accumulate Rouge scores (only when valid scores exist)
                if all_rouge_scores and idx < len(all_rouge_scores):
                    rouge_scores[method]["rouge1"] += all_rouge_scores[idx]["rouge1"]
                    rouge_scores[method]["rouge2"] += all_rouge_scores[idx]["rouge2"]
                    rouge_scores[method]["rougeL"] += all_rouge_scores[idx]["rougeL"]
                else:
                    print(f"⚠️ Failed to accumulate Rouge scores for {method} (idx: {idx}, score count: {len(all_rouge_scores)})")

                # Accumulate BertScore (only when valid scores exist)
                if all_bert_scores and idx < len(all_bert_scores):
                    bert_scores[method] += all_bert_scores[idx]
                else:
                    print(f"⚠️ Failed to accumulate BertScore for {method} (idx: {idx}, score count: {len(all_bert_scores)})")

            # Store all completion results in sample results
            sample_all_rouge_scores = all_rouge_scores
            sample_all_bert_scores = all_bert_scores
            print("=== CNN DailyMail evaluation complete ===\n")

        # ===============================
        # Use independent consistency evaluation system only for TruthfulQA
        # ===============================
        sample_ranking_scores = {}
        if args.dataset == "truthfulqa":
            method_results = {
                "LSC": {"answer": lsc_answer, "idx": best_idx, "time": lsc_time, "conf": lsc_conf, "calib": lsc_calib},
                "WUCS": {"answer": wucs_answer, "idx": wucs_idx, "time": wucs_time, 'wucs_calib': wucs_calib},
                "USC": {"answer": usc_answer, "idx": usc_idx, "time": usc_time},
                "Random": {"answer": rand_answer, "idx": rand_idx, "time": rand_time}
            }
            
            # Add 3 EOS methods (only when use_eos is True)
            if use_eos:
                for eos_method in eos_methods:
                    eos_result = eos_results[eos_method]
                    method_results[eos_method] = {
                        "answer": eos_result["answer"], 
                        "idx": eos_result["idx"], 
                        "time": eos_result["time"],
                        "conf": eos_result["conf"],
                        "calib": eos_result["calib"]
                    }
            
            # Add LSC Dynamic
            if enable_dynamic_topk and dynamic_result is not None:
                method_results["LSC_dynamic"] = {
                    "answer": dynamic_result["answer"], 
                    "idx": dynamic_result["selected_idx"], 
                    "time": dynamic_result["time"],
                    "conf": dynamic_result["conf"],
                    "calib": dynamic_result["calib"]
                }
            
            # Calculate independent consistency scores
            sample_ranking_scores = calculate_ranking_scores(method_results, texts, args.dataset)
            
            # Accumulate to overall consistency scores
            for method, score in sample_ranking_scores.items():
                ranking_scores[method] += score
        
        # Store responses (CNN DailyMail and TruthfulQA)
        if args.dataset in ["truthfulqa", "cnn_dailymail"]:
            method_responses["LSC"].append(lsc_answer)
            method_responses["WUCS"].append(wucs_answer)
            method_responses["USC"].append(usc_answer)
            method_responses["Random"].append(rand_answer)
            if use_eos:
                for eos_method in eos_methods:
                    method_responses[eos_method].append(eos_results[eos_method]["answer"])
            if enable_dynamic_topk and dynamic_result is not None:
                method_responses["LSC_dynamic"].append(dynamic_result["answer"])
        else:
            if enable_dynamic_topk:
                method_responses["LSC_dynamic"].append(dynamic_result["answer"])
            
            for k, result in lsc_topk_results.items():
                method_responses[f"LSC_top{k}"].append(result["answer"])
            
            if use_eos:
                for eos_method in eos_methods:
                    method_responses[eos_method].append(eos_results[eos_method]["answer"])
            
        print(f"\n[{args.dataset}] Sample {qi+1}/{len(ds)}")
        print("Q:", question)
        if args.dataset in ["truthfulqa", "cnn_dailymail"]:
            for i, ans in enumerate(ans_list):
                print(f"Path {i+1}: {ans}")
        else:
            print("Candidates:", ans_list)
        print("AVG Weight:", np.round(avg_weight, 3).tolist())
        
        # Perform evaluation
        if args.dataset == "truthfulqa":
            # TruthfulQA: Evaluate only selected answers
            selected_answers = {
                "LSC": lsc_answer,
                "WUCS": wucs_answer,
                "USC": usc_answer,
                "Random": rand_answer
            }
            
            # Add 3 EOS methods (only when use_eos is True)
            if use_eos:
                for eos_method in eos_methods:
                    selected_answers[eos_method] = eos_results[eos_method]["answer"]

            if enable_dynamic_topk and dynamic_result is not None:
                selected_answers["LSC_dynamic"] = dynamic_result["answer"]
            
            evaluation_results = {}
            unique_selected_answers = set(selected_answers.values())
            for answer in unique_selected_answers:
                is_corr = evaluate_function(question, answer, gold_ans, args.dataset, eval_cache)
                evaluation_results[answer] = {"is_corr": is_corr}
            
            # Update accuracy
            for method, answer in selected_answers.items():
                is_corr = evaluation_results[answer]["is_corr"]
                baseline_acc[method] += int(is_corr)
                
            # Get evaluation results for each method
            lsc_is_corr = evaluation_results[lsc_answer]["is_corr"]
            wucs_is_corr = evaluation_results[wucs_answer]["is_corr"]
            usc_is_corr = evaluation_results[usc_answer]["is_corr"]
            rand_is_corr = evaluation_results[rand_answer]["is_corr"]
            
            # 3 EOS methods evaluation results (only when use_eos is True)
            eos_evaluation_results = {}
            if use_eos:
                for eos_method in eos_methods:
                    answer = eos_results[eos_method]["answer"]
                    eos_evaluation_results[eos_method] = evaluation_results[answer]

            # LSC Dynamic evaluation results
            if enable_dynamic_topk and dynamic_result is not None:
                lsc_dynamic_is_corr = evaluation_results[dynamic_result["answer"]]["is_corr"]
            
            print(f"TruthfulQA evaluation cache statistics: truth evaluations {len(eval_cache['truth'])} items")
            
            if sample_ranking_scores:
                print("\n===== Independent consistency scores for this sample =====")
                for method in all_methods:
                    score = sample_ranking_scores.get(method, 0.0)
                    print(f"{method:12s}: {score:.3f}")
                print("=" * 40)
            
            print()
        elif args.dataset == "cnn_dailymail":
            # CNN DailyMail is evaluated with Rouge/BertScore, so set is_corr to True
            lsc_is_corr = True
            wucs_is_corr = True
            usc_is_corr = True
            rand_is_corr = True

            # Set 3 EOS methods to True as well
            eos_evaluation_results = {}
            if use_eos:
                for eos_method in eos_methods:
                    eos_evaluation_results[eos_method] = {"is_corr": True}

            # Set LSC Dynamic to True as well
            if enable_dynamic_topk and dynamic_result is not None:
                lsc_dynamic_is_corr = True

            # Output Rouge and BertScore
            print(f"\n===== Rouge & BertScore Results for Sample {qi+1} =====")
            print(f"Gold Answer: {gold_ans[:100]}...")
            if sample_all_rouge_scores and sample_all_bert_scores:
                print(f"LSC Answer (idx={best_idx+1}): {lsc_answer[:100]}...")
                print(f"  Rouge1: {sample_all_rouge_scores[best_idx]['rouge1']:.4f}, Rouge2: {sample_all_rouge_scores[best_idx]['rouge2']:.4f}, RougeL: {sample_all_rouge_scores[best_idx]['rougeL']:.4f}")
                print(f"  BertScore: {sample_all_bert_scores[best_idx]:.4f}")
                    
                print(f"WUCS Answer (idx={wucs_idx+1}): {wucs_answer[:100]}...")
                print(f"  Rouge1: {sample_all_rouge_scores[wucs_idx]['rouge1']:.4f}, Rouge2: {sample_all_rouge_scores[wucs_idx]['rouge2']:.4f}, RougeL: {sample_all_rouge_scores[wucs_idx]['rougeL']:.4f}")
                print(f"  BertScore: {sample_all_bert_scores[wucs_idx]:.4f}")
                    
                print(f"USC Answer (idx={usc_idx+1}): {usc_answer[:100]}...")
                print(f"  Rouge1: {sample_all_rouge_scores[usc_idx]['rouge1']:.4f}, Rouge2: {sample_all_rouge_scores[usc_idx]['rouge2']:.4f}, RougeL: {sample_all_rouge_scores[usc_idx]['rougeL']:.4f}")
                print(f"  BertScore: {sample_all_bert_scores[usc_idx]:.4f}")
                    
                print(f"Random Answer (idx={rand_idx+1}): {rand_answer[:100]}...")
                print(f"  Rouge1: {sample_all_rouge_scores[rand_idx]['rouge1']:.4f}, Rouge2: {sample_all_rouge_scores[rand_idx]['rouge2']:.4f}, RougeL: {sample_all_rouge_scores[rand_idx]['rougeL']:.4f}")
                print(f"  BertScore: {sample_all_bert_scores[rand_idx]:.4f}")
                
                # Output 3 EOS methods results (only when use_eos is True)
                if use_eos:
                    for eos_method in eos_methods:
                        eos_result = eos_results[eos_method]
                        idx = eos_result["idx"]
                        print(f"{eos_method} Answer (idx={idx+1}): {eos_result['answer'][:100]}...")
                        print(f"  Rouge1: {sample_all_rouge_scores[idx]['rouge1']:.4f}, Rouge2: {sample_all_rouge_scores[idx]['rouge2']:.4f}, RougeL: {sample_all_rouge_scores[idx]['rougeL']:.4f}")
                        print(f"  BertScore: {sample_all_bert_scores[idx]:.4f}")
                
                # Output LSC Dynamic results
                if enable_dynamic_topk and dynamic_result is not None:
                    idx = dynamic_result["selected_idx"]
                    print(f"LSC_dynamic Answer (idx={idx+1}): {dynamic_result['answer'][:100]}...")
                    print(f"  Rouge1: {sample_all_rouge_scores[idx]['rouge1']:.4f}, Rouge2: {sample_all_rouge_scores[idx]['rouge2']:.4f}, RougeL: {sample_all_rouge_scores[idx]['rougeL']:.4f}")
                    print(f"  BertScore: {sample_all_bert_scores[idx]:.4f}")
            else:
                print(" Rouge/BertScore calculation failed.")
            print("=" * 60)
        else:
            # Existing evaluation method (other datasets)
            lsc_is_corr = evaluate_function(question, lsc_answer, gold_ans, args.dataset)
            sc_is_corr = evaluate_function(question, sc_answer, gold_ans, args.dataset)
            wucs_is_corr = evaluate_function(question, wucs_answer, gold_ans, args.dataset)
            usc_is_corr = evaluate_function(question, usc_answer, gold_ans, args.dataset)
            rand_is_corr = evaluate_function(question, rand_answer, gold_ans, args.dataset)
            
            # 3 EOS methods evaluation (only when use_eos is True)
            eos_evaluation_results = {}
            if use_eos:
                for eos_method in eos_methods:
                    is_corr = evaluate_function(question, eos_results[eos_method]["answer"], gold_ans, args.dataset)
                    eos_evaluation_results[eos_method] = {"is_corr": is_corr}
                    baseline_acc[eos_method] += int(is_corr)
            
            # Dynamic TopK evaluation
            if enable_dynamic_topk and dynamic_result is not None:
                dynamic_is_corr = evaluate_function(question, dynamic_result["answer"], gold_ans, args.dataset)
                dynamic_result["is_corr"] = dynamic_is_corr
                dynamic_result["is_major"] = (dynamic_result["conf"] == sc_conf)
                baseline_acc["LSC_dynamic"] += int(dynamic_is_corr)
            
            # Evaluate each LSC_TOPK value
            for k, result in lsc_topk_results.items():
                result["is_corr"] = evaluate_function(question, result["answer"], gold_ans, args.dataset)
                result["is_major"] = (result["conf"] == sc_conf)
                baseline_acc[f'LSC_top{k}'] += int(result["is_corr"])
            
            # Update accuracy
            baseline_acc['LSC'] += int(lsc_is_corr)
            baseline_acc['SC'] += int(sc_is_corr)
            baseline_acc['WUCS'] += int(wucs_is_corr)
            baseline_acc['USC'] += int(usc_is_corr)
            baseline_acc['Random'] += int(rand_is_corr)

        # Output results
        print(f"▶ LSC selection (idx={best_idx+1}, weight={lsc_calib:.3f}): {lsc_answer}")
        print(f"\t\t\t• LSC time: {lsc_time:.4f} sec  | Correct: {lsc_is_corr}")
        
        # Output 3 EOS methods results (only when use_eos is True)
        if use_eos:
            for eos_method in eos_methods:
                eos_result = eos_results[eos_method]
                eos_eval = eos_evaluation_results[eos_method]
                layer_info = "2nd layer" if use_eos_second else "last layer"
                print(f"▶ {eos_method} selection (idx={eos_result['idx']+1}, {layer_info}, weight={eos_result['calib']:.3f}): {eos_result['answer']}")
                print(f"\t\t\t• {eos_method} time: {eos_result['time']:.4f} sec  | Correct: {eos_eval['is_corr']}")
        
        # Output Dynamic TopK results
        if enable_dynamic_topk and dynamic_result is not None:
            print(f"▶ LSC_dynamic selection (idx={dynamic_result['selected_idx']+1}, optimal_k={dynamic_result['optimal_k']}, weight={dynamic_result['calib']:.3f}): {dynamic_result['answer']}")
            if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
                print(f"\t\t\t• LSC_dynamic time: {dynamic_result['time']:.4f} sec  | Correct: {dynamic_result['is_corr']}")
            else:
                print(f"\t\t\t• LSC_dynamic time: {dynamic_result['time']:.4f} sec  | Correct: {lsc_dynamic_is_corr}")
        
        # Output results for each LSC_TOPK value (only when not truthfulqa or cnn_dailymail)
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            for k, result in lsc_topk_results.items():
                print(f"▶ LSC_top{k} selection (idx={result['idx']+1}, weight_top{k}={result['calib']:.3f}): {result['answer']}")
                print(f"\t\t\t• LSC_top{k} time: {result['time']:.4f} sec  | Correct: {result['is_corr']}")
        print("")
        
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            print(f"▶ SC selection: {sc_answer}  | conf={sc_conf:.3f}  | time={sc_time:.4f} sec  | Correct={sc_is_corr}")
        print(f"▶ WUCS selection: {wucs_answer}  | time={wucs_time:.4f} sec  | Correct={wucs_is_corr}")
        print(f"▶ USC selection: {usc_answer}  | time={usc_time:.4f} sec  | Correct={usc_is_corr}")
        print(f"▶ Random selection (idx={rand_idx+1}): {rand_answer}  | time={rand_time:.4f} sec  | Correct={rand_is_corr}")
        
        # Average time & accuracy so far
        print("Baseline avg times up to now:")
        print(f"  LSC:       {baseline_times['LSC']/(qi+1):.6f} sec")
        if use_eos:
            for eos_method in eos_methods:
                print(f"  {eos_method}: {baseline_times[eos_method]/(qi+1):.6f} sec")
        
        if enable_dynamic_topk:
            print(f"  LSC_dynamic: {baseline_times['LSC_dynamic']/(qi+1):.6f} sec")
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            for k in LSC_TOPK_LIST:
                print(f"  LSC_top{k}: {baseline_times[f'LSC_top{k}']/(qi+1):.6f} sec")
        print(f"  WUCS:      {baseline_times['WUCS']/(qi+1):.6f} sec  | USC:       {baseline_times['USC']/(qi+1):.6f} sec")
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            print(f"  SC:        {baseline_times['SC']/(qi+1):.6f} sec  | Random:    {baseline_times['Random']/(qi+1):.6f} sec")
        else:
            print(f"  Random:    {baseline_times['Random']/(qi+1):.6f} sec")
        print()
        print()

        if args.dataset == "cnn_dailymail":
            # For CNN DailyMail, output Rouge/BertScore averages
            print("Baseline avg Rouge & BertScore up to now:")
            for method in all_methods:
                if method in rouge_scores and method in bert_scores:
                    avg_rouge1 = rouge_scores[method]["rouge1"] / (qi+1)
                    avg_rouge2 = rouge_scores[method]["rouge2"] / (qi+1)
                    avg_rougeL = rouge_scores[method]["rougeL"] / (qi+1)
                    avg_bert = bert_scores[method] / (qi+1)
                    print(f"  {method:8s}: R1={avg_rouge1:.4f}, R2={avg_rouge2:.4f}, RL={avg_rougeL:.4f}, BERT={avg_bert:.4f}")
        else:
            print("Baseline avg ACC up to now:")
            print(f"  LSC:       {baseline_acc['LSC']/(qi+1):.4f}")
            if use_eos:
                for eos_method in eos_methods:
                    print(f"  {eos_method}: {baseline_acc[eos_method]/(qi+1):.4f}")
            if enable_dynamic_topk:
                print(f"  LSC_dynamic: {baseline_acc['LSC_dynamic']/(qi+1):.4f}")
            if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
                for k in LSC_TOPK_LIST:
                    print(f"  LSC_top{k}: {baseline_acc[f'LSC_top{k}']/(qi+1):.4f}")
            print(f"  WUCS:      {baseline_acc['WUCS']/(qi+1):.4f}    | USC:       {baseline_acc['USC']/(qi+1):.4f}")
            if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
                print(f"  SC:        {baseline_acc['SC']/(qi+1):.4f}    | Random:    {baseline_acc['Random']/(qi+1):.4f}")
            else:
                print(f"  Random:    {baseline_acc['Random']/(qi+1):.4f}")
        
        # Average consistency scores so far (only for TruthfulQA)
        if args.dataset == "truthfulqa":
            print("\n===== Average consistency scores so far (consistent selection ratio) =====")
            # Output only methods used in TruthfulQA
            truthfulqa_methods = ["LSC", "WUCS", "USC", "Random"]
            if enable_dynamic_topk:
                truthfulqa_methods.append("LSC_dynamic")
            if use_eos:
                truthfulqa_methods.extend(eos_methods)
                
            for method in truthfulqa_methods:
                avg_consistency_score = ranking_scores.get(method, 0) / (qi + 1)
                print(f"  {method:12s}: {avg_consistency_score:.4f}")
            print("=" * 50)

        # LSC majority prediction success/failure (only when not truthfulqa or cnn_dailymail)
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            is_lsc_major = (lsc_conf == sc_conf)
            print("LSC majority prediction success" if is_lsc_major else "LSC majority prediction failure")
            if use_eos:
                for eos_method in eos_methods:
                    eos_result = eos_results[eos_method]
                    is_eos_major = (eos_result["conf"] == sc_conf)
                    print(f"{eos_method} majority prediction success" if is_eos_major else f"{eos_method} majority prediction failure")
            if enable_dynamic_topk and dynamic_result is not None:
                print(f"LSC_dynamic majority prediction success" if dynamic_result["is_major"] else f"LSC_dynamic majority prediction failure")
            for k, result in lsc_topk_results.items():
                print(f"LSC_top{k} majority prediction success" if result["is_major"] else f"LSC_top{k} majority prediction failure")
        else:
            is_lsc_major = None

        # Organize one sample result as dict
        sample_result = {
            "question": question,
            "gold_ans": gold_ans,
            "ans_list": json.dumps(ans_list, ensure_ascii=False),
            "lsc_answer": lsc_answer,
            "lsc_correct": lsc_is_corr,
            "lsc_time": lsc_time,
            "lsc_conf": lsc_conf,
            "lsc_calib": lsc_calib,
            "is_lsc_major": is_lsc_major,
            "wucs_answer": wucs_answer,
            "wucs_correct": wucs_is_corr,
            "wucs_time": wucs_time,
            "wucs_calib": wucs_calib,
            "usc_answer": usc_answer,
            "usc_correct": usc_is_corr,
            "usc_time": usc_time,
            "random_answer": rand_answer,
            "random_correct": rand_is_corr,
            "random_time": rand_time,
        }
        
        # Add 3 EOS methods results (only when use_eos is True)
        if use_eos:
            for eos_method in eos_methods:
                eos_result = eos_results[eos_method]
                eos_eval = eos_evaluation_results[eos_method]
                sample_result.update({
                    f"{eos_method.lower()}_answer": eos_result["answer"],
                    f"{eos_method.lower()}_correct": eos_eval["is_corr"],
                    f"{eos_method.lower()}_time": eos_result["time"],
                    f"{eos_method.lower()}_conf": eos_result["conf"],
                    f"{eos_method.lower()}_calib": eos_result["calib"],
                })
                if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
                    is_eos_major = (eos_result["conf"] == sc_conf)
                    sample_result[f"is_{eos_method.lower()}_major"] = is_eos_major
        
        # Save SC only when not truthfulqa or cnn_dailymail
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            sample_result.update({
                "sc_answer": sc_answer,
                "sc_correct": sc_is_corr,
                "sc_time": sc_time,
                "sc_conf": sc_conf,
            })
        
        # Add Dynamic TopK results
        if enable_dynamic_topk and dynamic_result is not None:
            sample_result.update({
                "lsc_dynamic_answer": dynamic_result["answer"],
                "lsc_dynamic_correct": dynamic_result["is_corr"] if args.dataset not in ["truthfulqa", "cnn_dailymail"] else lsc_dynamic_is_corr,
                "lsc_dynamic_time": dynamic_result["time"],
                "lsc_dynamic_conf": dynamic_result["conf"],
                "lsc_dynamic_calib": float(dynamic_result["calib"]),
                "lsc_dynamic_optimal_k": dynamic_result["optimal_k"],
                "lsc_dynamic_selected_idx": dynamic_result["selected_idx"],
                "lsc_dynamic_max_scores": json.dumps([float(x) for x in dynamic_result["max_scores"]]),
                "lsc_dynamic_score_diffs": json.dumps([float(x) for x in dynamic_result["score_diffs"]]),
                "lsc_dynamic_largest_drop": float(dynamic_result["largest_drop"]),
                "lsc_dynamic_largest_drop_idx": dynamic_result["largest_drop_idx"],
                "lsc_dynamic_all_k_results": json.dumps([(int(k), float(score), int(idx)) for k, score, idx in dynamic_result["all_k_results"]]),
                "is_lsc_dynamic_major": dynamic_result["is_major"] if args.dataset not in ["truthfulqa", "cnn_dailymail"] else None,
            })
        
        # Add results for each LSC_TOPK value (only when not truthfulqa or cnn_dailymail)
        if args.dataset not in ["truthfulqa", "cnn_dailymail"]:
            for k, result in lsc_topk_results.items():
                sample_result.update({
                    f"lsc_top{k}_answer": result["answer"],
                    f"lsc_top{k}_correct": result["is_corr"],
                    f"lsc_top{k}_time": result["time"],
                    f"lsc_top{k}_conf": result["conf"],
                    f"lsc_top{k}_calib": result["calib"],
                    f"is_lsc_top{k}_major": result["is_major"],
                })
        
        # Add Rouge/BertScore results for CNN DailyMail
        if args.dataset == "cnn_dailymail":
            # Save Rouge/BertScore for each method (only when scores are valid)
            if sample_all_rouge_scores and sample_all_bert_scores:
                sample_result.update({
                    "lsc_rouge1": sample_all_rouge_scores[best_idx]["rouge1"],
                    "lsc_rouge2": sample_all_rouge_scores[best_idx]["rouge2"],
                    "lsc_rougeL": sample_all_rouge_scores[best_idx]["rougeL"],
                    "lsc_bertscore": sample_all_bert_scores[best_idx],
                    "wucs_rouge1": sample_all_rouge_scores[wucs_idx]["rouge1"],
                    "wucs_rouge2": sample_all_rouge_scores[wucs_idx]["rouge2"],
                    "wucs_rougeL": sample_all_rouge_scores[wucs_idx]["rougeL"],
                    "wucs_bertscore": sample_all_bert_scores[wucs_idx],
                    "usc_rouge1": sample_all_rouge_scores[usc_idx]["rouge1"],
                    "usc_rouge2": sample_all_rouge_scores[usc_idx]["rouge2"],
                    "usc_rougeL": sample_all_rouge_scores[usc_idx]["rougeL"],
                    "usc_bertscore": sample_all_bert_scores[usc_idx],
                    "random_rouge1": sample_all_rouge_scores[rand_idx]["rouge1"],
                    "random_rouge2": sample_all_rouge_scores[rand_idx]["rouge2"],
                    "random_rougeL": sample_all_rouge_scores[rand_idx]["rougeL"],
                    "random_bertscore": sample_all_bert_scores[rand_idx],
                    "all_rouge_scores": json.dumps(sample_all_rouge_scores, ensure_ascii=False),
                    "all_bert_scores": json.dumps(sample_all_bert_scores, ensure_ascii=False),
                })

                # Add 3 EOS methods Rouge/BertScore (only when use_eos is True)
                if use_eos:
                    for eos_method in eos_methods:
                        eos_result = eos_results[eos_method]
                        idx = eos_result["idx"]
                        sample_result.update({
                            f"{eos_method.lower()}_rouge1": sample_all_rouge_scores[idx]["rouge1"],
                            f"{eos_method.lower()}_rouge2": sample_all_rouge_scores[idx]["rouge2"],
                            f"{eos_method.lower()}_rougeL": sample_all_rouge_scores[idx]["rougeL"],
                            f"{eos_method.lower()}_bertscore": sample_all_bert_scores[idx],
                        })

                # Add LSC Dynamic Rouge/BertScore
                if enable_dynamic_topk and dynamic_result is not None:
                    idx = dynamic_result["selected_idx"]
                    sample_result.update({
                        "lsc_dynamic_rouge1": sample_all_rouge_scores[idx]["rouge1"],
                        "lsc_dynamic_rouge2": sample_all_rouge_scores[idx]["rouge2"],
                        "lsc_dynamic_rougeL": sample_all_rouge_scores[idx]["rougeL"],
                        "lsc_dynamic_bertscore": sample_all_bert_scores[idx],
                    })
            else:
                # Set to 0 when score calculation failed
                print("⚠️ Rouge/BertScore calculation failed, setting to 0.")
                sample_result.update({
                    "lsc_rouge1": 0.0, "lsc_rouge2": 0.0, "lsc_rougeL": 0.0, "lsc_bertscore": 0.0,
                    "wucs_rouge1": 0.0, "wucs_rouge2": 0.0, "wucs_rougeL": 0.0, "wucs_bertscore": 0.0,
                    "usc_rouge1": 0.0, "usc_rouge2": 0.0, "usc_rougeL": 0.0, "usc_bertscore": 0.0,
                    "random_rouge1": 0.0, "random_rouge2": 0.0, "random_rougeL": 0.0, "random_bertscore": 0.0,
                    "all_rouge_scores": "[]",
                    "all_bert_scores": "[]",
                })
                
                if use_eos:
                    for eos_method in eos_methods:
                        sample_result.update({
                            f"{eos_method.lower()}_rouge1": 0.0,
                            f"{eos_method.lower()}_rouge2": 0.0,
                            f"{eos_method.lower()}_rougeL": 0.0,
                            f"{eos_method.lower()}_bertscore": 0.0,
                        })
                
                if enable_dynamic_topk and dynamic_result is not None:
                    sample_result.update({
                        "lsc_dynamic_rouge1": 0.0,
                        "lsc_dynamic_rouge2": 0.0,
                        "lsc_dynamic_rougeL": 0.0,
                        "lsc_dynamic_bertscore": 0.0,
                    })
        
        # Add independent consistency scores only for TruthfulQA
        if args.dataset == "truthfulqa":
            # Save only for methods used in TruthfulQA
            truthfulqa_methods = ["LSC", "WUCS", "USC", "Random"]
            if enable_dynamic_topk:
                truthfulqa_methods.append("LSC_dynamic")
            if use_eos:
                truthfulqa_methods.extend(eos_methods)
                
            for method in truthfulqa_methods:
                sample_result[f"consistency_score_{method}"] = sample_ranking_scores.get(method, 0.0)
        
        results.append(sample_result)

    # Save all results as DataFrame → CSV
    df_out = pd.DataFrame(results)
    filename_suffix = ""
    if enable_dynamic_topk:
        filename_suffix += "_with_dynamic"
    if use_eos:
        layer_suffix = "2nd" if use_eos_second else "last"
        filename_suffix += f"_with_3eos_{layer_suffix}"
    if args.dataset == "truthfulqa":
        filename_suffix += "_with_consistency"
    elif args.dataset == "cnn_dailymail":
        filename_suffix += "_with_rouge_bert"
    
    df_out.to_csv(f"results/{args.dataset}_evaluation_results{Repo_name}{filename_suffix}_{args.num_path}Paths_{args.seed}.csv", index=False)
    print(f"\n✅ Saved all results to {args.dataset}_evaluation_results{Repo_name}{filename_suffix}_{args.num_path}Paths_{args.seed}.csv (samples: {total})\n")

    # Save responses for all methods as individual CSV files
    for method, responses in method_responses.items():
        df = pd.DataFrame({method: responses})
        filename = f"responses/{method}_{args.dataset}_responses_{Repo_name}{filename_suffix}_{args.num_path}Paths_{args.seed}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {method} responses to {filename}")

    # Summary output
    print("===== Summary =====")
    if args.dataset == "cnn_dailymail":
        print(f"{'Method':15s} | {'Rouge1':8s} | {'Rouge2':8s} | {'RougeL':8s} | {'BertScore':10s} | {'Avg Time':10s}")
        print("-" * 80)
        
        for method in all_methods:
            if method in rouge_scores and method in bert_scores:
                avg_time = baseline_times[method] / total
                avg_rouge1 = rouge_scores[method]["rouge1"] / total
                avg_rouge2 = rouge_scores[method]["rouge2"] / total
                avg_rougeL = rouge_scores[method]["rougeL"] / total
                avg_bert = bert_scores[method] / total
                print(f"{method:15s} | {avg_rouge1:.6f} | {avg_rouge2:.6f} | {avg_rougeL:.6f} | {avg_bert:.8f} | {avg_time:.4f} sec")
    elif args.dataset == "truthfulqa":
        print(f"{'Method':15s} | {'Accuracy':10s} | {'Avg Time':10s} | {'Consistency':10s}")
        print("-" * 65)
        
        # Output only methods used in TruthfulQA
        truthfulqa_methods = ["LSC", "WUCS", "USC", "Random"]
        if enable_dynamic_topk:
            truthfulqa_methods.append("LSC_dynamic")
        if use_eos:
            truthfulqa_methods.extend(eos_methods)
            
        for method in truthfulqa_methods:
            avg_time = baseline_times[method] / total
            acc_rate = baseline_acc[method] / total
            consistency_rate = ranking_scores[method] / total
            print(f"{method:15s} | {acc_rate:.4f} | {avg_time:.4f} sec | {consistency_rate:.4f}")
    else:
        print(f"{'Method':15s} | {'Accuracy':10s} | {'Avg Time':10s}")
        print("-" * 45)
        
        # Output all methods for other datasets (including SC)
        for method in all_methods:
            avg_time = baseline_times[method] / total
            acc_rate = baseline_acc[method] / total
            print(f"{method:15s} | {acc_rate:.4f} | {avg_time:.4f} sec")
        
        # Also output summary for each LSC_TOPK value
        for k in LSC_TOPK_LIST:
            method = f"LSC_top{k}"
            avg_time = baseline_times[method] / total
            acc_rate = baseline_acc[method] / total
            print(f"{method:15s} | {acc_rate:.4f} | {avg_time:.4f} sec")

    # Output independent consistency results only for TruthfulQA
    if args.dataset == "truthfulqa":
        print("\n===== Independent Consistency Results (Consistent Selection Ratio) =====")
        # Sort only methods used in TruthfulQA
        truthfulqa_methods = ["LSC", "WUCS", "USC", "Random"]
        if enable_dynamic_topk:
            truthfulqa_methods.append("LSC_dynamic")
        if use_eos:
            truthfulqa_methods.extend(eos_methods)
            
        sorted_methods = sorted(truthfulqa_methods, key=lambda x: ranking_scores.get(x, 0), reverse=True)
        print(f"{'Rank':^4} | {'Method':^12} | {'Consistency Ratio':^16} | {'Total Score':^12}")
        print("-" * 52)
        for rank, method in enumerate(sorted_methods, 1):
            consistency_rate = ranking_scores.get(method, 0) / total
            total_score = ranking_scores.get(method, 0)
            print(f"{rank:^4} | {method:^12} | {consistency_rate:^16.4f} | {total_score:^12.2f}")
        print("=" * 52)

    # Output TruthfulQA evaluation cache statistics (if exists)
    if eval_cache is not None:
        print(f"\nTruthfulQA evaluation cache final statistics:")
        print(f"  - Truth evaluation (ask_gpt_truth) calls: {len(eval_cache['truth'])}")

        # Calculate cache efficiency (API calls that would have been made without caching vs actual API calls)
        total_samples = total
        # Each sample has methods_count * 1 API call for truth evaluation (no info evaluation)
        methods_count = len(all_methods)
        without_cache_calls = total_samples * 1 * methods_count
        actual_calls = len(eval_cache["truth"])
        saved_calls = without_cache_calls - actual_calls
        saved_percentage = (saved_calls / without_cache_calls) * 100 if without_cache_calls > 0 else 0

        print(f"  - API calls that would have been needed without caching: {without_cache_calls}")
        print(f"  - Actual API calls made: {actual_calls}")
        print(f"  - Saved API calls: {saved_calls} ({saved_percentage:.2f}%)")

    print("🎉 Experiment completed!")

if __name__ == "__main__":
    main()
