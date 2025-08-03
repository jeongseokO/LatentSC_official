import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple


def weighted_ucs_rerank(
    texts: List[str],
    token_ids_list: List[List[int]],
    token_probs_list: List[List[float]],
    use_consensus: bool = True
) -> Tuple[int, str, float, List[float]]:
    """
    Reranker supporting both Weighted Unigram Consistency Score 
    and Consensus-WUCS
    """
    # 1) Build vocabulary
    all_token_ids = set(tok for seq in token_ids_list for tok in seq)
    V = list(all_token_ids)
    V_size = len(V)
    token_to_idx = {tok: i for i, tok in enumerate(V)}

    # 2) WUCS vectorization
    M = len(token_ids_list)
    vectors = np.zeros((M, V_size), dtype=float)
    for i, (toks, probs) in enumerate(zip(token_ids_list, token_probs_list)):
        positions = {}
        for pos, t in enumerate(toks):
            positions.setdefault(t, []).append(pos)
        for t, pos_list in positions.items():
            j = token_to_idx[t]
            valid_ps = [p for p in pos_list if 0 <= p < len(probs)]
            if valid_ps:
                vectors[i, j] = np.mean([probs[p] for p in valid_ps])
            else:
                vectors[i, j] = 0.0

    # 3) sim_matrix = (v·vᵀ)/|V|
    sim_matrix = (vectors @ vectors.T) / V_size

    # 4) WUCS
    wucs_scores = []
    for i in range(M):
        if M > 1:
            score = (sim_matrix[i].sum() - sim_matrix[i, i]) / (M - 1)
        else:
            score = sim_matrix[i, i]
        wucs_scores.append(float(score))

    # 5) Additional weighting for Consensus-WUCS
    consensus_scores = []
    for wucs, probs in zip(wucs_scores, token_probs_list):
        arr = np.array(probs, dtype=float)
        avg_logprob = np.mean(np.log(arr + 1e-12))
        consensus_scores.append(wucs * np.exp(avg_logprob))

    # 6) Final selection
    all_scores = consensus_scores if use_consensus else wucs_scores
    best_idx = int(np.argmax(all_scores))
    best_text = texts[best_idx]
    best_score = all_scores[best_idx]

    return best_idx, best_score


def get_text_probabilities(
    result,       # Object returned by generate_with_special_tokens
) -> List[List[float]]:
    """
    result.prob_list: list of torch.Tensor (length = gen_steps), each shape=(batch,)
    Output: list of batch count, each list stores probabilities (float) in order
    """
    # (steps, batch) -> stack as tensor
    probs = torch.stack(result.prob_list, dim=0)    # shape=(steps, batch)
    probs = probs.permute(1, 0)                     # shape=(batch, steps)
    return [probs[i].tolist() for i in range(probs.size(0))]

def clean_generated_sequences(
    result,                # Object returned by generate_with_special_tokens
    special_tokens: List[int],
    eos_token_id: int,
    context_len: int,
    pad_token_id: int
) -> List[List[int]]:
    """
    result.sequences: Tensor shape=(batch, ctx_len+gen_len)
    Return: batch count lists, each list contains clean token_ids
    """
    seqs = result.sequences[:, context_len:]  # Only tokens after context
    cleaned = []
    for seq in seqs.tolist():
        out = []
        for t in seq:
            if t == eos_token_id:
                break
            if t in special_tokens or t == pad_token_id:
                continue
            out.append(t)
        cleaned.append(out)
    return cleaned