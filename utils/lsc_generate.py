import time
from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
def dynamic_topk_lsc(weights, ans_list, temp=0.5):
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
def generate_with_special_tokens( 
    self,
    inputs: torch.Tensor,
    special_tokens: List[int],
    remove_eos: bool = False,
    return_embeddings: bool = True,
    use_eos: bool = False,
    use_eos_second: bool = False,
    verbose: bool = False,
    **kwargs,
):
    # ──────────────────────────────────────────────────────────
    # Basic setup
    # ──────────────────────────────────────────────────────────
    self.eval()
    base_device = inputs.device              
    cpu_device = torch.device("cpu")

    # Decoding parameters --------------------------------------------------
    num_return_sequences = kwargs.get("num_return_sequences", 1)
    do_sample      = kwargs.get("do_sample", True)
    temperature    = kwargs.get("temperature", 1.0)
    top_p          = kwargs.get("top_p", 1.0)
    max_new_tokens = kwargs.get("max_new_tokens", 100)
    max_length     = kwargs.get("max_length")             
    pad_token_id   = kwargs.get("pad_token_id",  self.config.pad_token_id)
    eos_token_id   = kwargs.get("eos_token_id",  self.config.eos_token_id)
    output_scores  = kwargs.get("output_scores",  True)
    return_dict    = kwargs.get("return_dict_in_generate", True)

    # Batch expansion ---------------------------------------------------------
    orig_bs, ctx_len = inputs.shape
    if num_return_sequences > 1:
        inputs = (
            inputs.unsqueeze(1)
            .expand(orig_bs, num_return_sequences, ctx_len)
            .reshape(-1, ctx_len)
        )
    batch_size = inputs.size(0)

    # State variables ---------------------------------------------------------
    input_ids      = inputs
    attention_mask = inputs.ne(pad_token_id).to(torch.bool)   # bool mask
    final_max_len  = ctx_len + max_new_tokens if max_length is None else max_length

    past_key_values = None
    unfinished = torch.ones(batch_size, dtype=torch.long, device=base_device)

    # Special tokens -------------------------------------------------------
    use_special = len(special_tokens) > 0
    if use_special:
        st_tensor  = torch.tensor(special_tokens, dtype=torch.long, device=base_device)
        st_len     = st_tensor.size(0)
        st_idx     = torch.zeros(batch_size, dtype=torch.long,  device=base_device)
        st_mode    = torch.zeros(batch_size, dtype=torch.bool,  device=base_device)
        special_embs      = [[] for _ in range(batch_size)]
        prev_was_special  = torch.zeros(batch_size, dtype=torch.bool, device=base_device)

    # EOS tracker --------------------------------------------------------
    eos_pred_list, eos_first_list, eos_special_embs = None, None, None
    if use_eos and return_embeddings:
        eos_pred_list    = [None] * batch_size
        eos_first_list   = [None] * batch_size
        eos_special_embs = [[] for _ in range(batch_size)]
    first_eos_seen   = torch.zeros(batch_size, dtype=torch.bool, device=base_device)
    eos_special_mode = torch.zeros(batch_size, dtype=torch.bool, device=base_device)
    eos_count        = torch.zeros(batch_size, dtype=torch.long, device=base_device)

    need_final_eos   = torch.zeros(batch_size, dtype=torch.bool, device=base_device)

    # Others --------------------------------------------------------------
    prob_list  = [] if (output_scores and do_sample) else None
    hidden_dim: Optional[int] = None
    cur_len, start_t = ctx_len, time.perf_counter()

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────
    while True:

        # Exit loop if all sequences are finished
        if unfinished.max() == 0 or cur_len >= final_max_len:
            break

        # ─ A. Model forward pass ───────────────────────────────────────
        if past_key_values is None:   # First step
            model_inputs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:                         # Subsequent steps (last token only)
            model_inputs = dict(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask,   # pass full mask
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        with torch.inference_mode():
            outputs = self(**model_inputs)

        past_key_values = outputs.past_key_values
        h       = outputs.hidden_states
        last_h  = h[-1][:, -1, :].detach().clone()           # hidden on shard
        second_h = h[1][:, -1, :].detach() if use_eos_second else None
        eos_h    = second_h if use_eos_second else last_h
        hidden_dim = hidden_dim or last_h.size(-1)
        del h, outputs

        # ─ B. Save previous special token embeddings ─────────────────────
        if use_special and return_embeddings:
            idxs = torch.nonzero(prev_was_special, as_tuple=True)[0]
            if idxs.numel():
                sel = last_h.index_select(0, idxs.to(last_h.device)).to(cpu_device)
                for j, i in enumerate(idxs):
                    special_embs[i].append(sel[j])
            prev_was_special.zero_()

        # ─ C. logits & sampling ─────────────────────────────────────
        logits = self.lm_head(last_h)                # lm_head on GPU shard
        if temperature != 1.0:
            logits = logits / temperature

        if do_sample:
            if top_p < 1.0:
                s_l, s_idx = torch.sort(logits, dim=-1, descending=True)
                probs_sorted = torch.softmax(s_l, dim=-1, dtype=torch.float32)
                cum = torch.cumsum(probs_sorted, dim=-1)
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1]
                mask[..., 0]  = False
                logits.masked_fill_(mask.scatter(-1, s_idx, mask), float("-inf"))
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            next_tokens = torch.multinomial(probs, 1).squeeze(1)
        else:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            next_tokens = torch.argmax(logits, dim=-1)

        if output_scores and cur_len > ctx_len:
            prob_list.append(probs.gather(1, next_tokens[:, None]).squeeze(1).cpu())

        # Unify tensors to base_device
        next_tokens = next_tokens.to(base_device)

        # ─ D. Force queued EOS ───────────────────────────────────
        if need_final_eos.any():
            idxs = need_final_eos.nonzero(as_tuple=True)[0]
            next_tokens[idxs] = eos_token_id
            unfinished[idxs]  = 0
            need_final_eos[idxs] = False

        # ─ E. EOS method 1 (prediction embedding) ─────────────────────────
        if use_eos and return_embeddings:
            will_eos = next_tokens == eos_token_id
            for i in will_eos.nonzero(as_tuple=True)[0]:
                if eos_pred_list[i] is None:
                    eos_pred_list[i] = eos_h[i].to(cpu_device)

        # ─ F. Special token / EOS logic ────────────────────────────
        if use_special:
            will_output_special = torch.zeros(batch_size, dtype=torch.bool, device=base_device)

            # (i) st_mode state: emit scheduled special tokens
            dispense = st_mode & (unfinished == 1)
            if dispense.any():
                # First check if we just output the last ST
                done_disp = dispense & (st_idx >= st_len - 1)
                if done_disp.any():
                    st_mode[done_disp] = False
                    need_final_eos[done_disp] = True      # Reserve EOS
                
                # Only assign tokens if there are still STs to output
                valid_dispense = dispense & (st_idx < st_len)
                if valid_dispense.any():
                    next_tokens[valid_dispense] = st_tensor[st_idx[valid_dispense]]
                    will_output_special[valid_dispense] = True
                    st_idx[valid_dispense] += 1

            # (ii) Newly detected EOS
            is_eos  = next_tokens == eos_token_id
            new_eos = is_eos & ~st_mode & (unfinished == 1)

            if new_eos.any():
                # (a) First EOS embedding
                if use_eos and return_embeddings:
                    first_mask = new_eos & ~first_eos_seen
                    first_eos_seen[first_mask] = True
                    for i in first_mask.nonzero(as_tuple=True)[0]:
                        eos_first_list[i] = eos_h[i].to(cpu_device)

                # (b) EOS-special mode (max 6 times)
                first_time = new_eos & ~eos_special_mode
                if not (remove_eos and st_len > 0):
                    if first_time.any():
                        eos_special_mode[first_time] = True
                        eos_count[first_time] += 1
                        if return_embeddings:
                            for i in first_time.nonzero(as_tuple=True)[0]:
                                eos_special_embs[i].append(eos_h[i].to(cpu_device))

                # (c) Replace EOS with ST
                if remove_eos:
                    st_mode[new_eos] = True
                    repl = new_eos & (st_len > 0)
                    if repl.any():
                        next_tokens[repl] = st_tensor[0]
                        will_output_special[repl] = True
                        st_idx[repl] = 1
                    unfinished[new_eos & (st_len == 0)] = 0
                else:
                    st_mode[new_eos] = True

            # (iii) EOS-special mode progression
            eos_disp = eos_special_mode & (unfinished == 1) & ~st_mode
            if eos_disp.any():
                next_tokens[eos_disp] = eos_token_id
                eos_count[eos_disp] += 1
                if return_embeddings:
                    for i in eos_disp.nonzero(as_tuple=True)[0]:
                        eos_special_embs[i].append(eos_h[i].to(cpu_device))
                done = eos_disp & (eos_count >= 6)
                eos_special_mode[done] = False
                unfinished[done] = 0

            # Sequence termination → replace with pad token
            if (unfinished == 0).any():
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

            prev_was_special.copy_(will_output_special)

        else:
            unfinished[next_tokens == eos_token_id] = 0
            if (unfinished == 0).any():
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

        # ─ G. Concatenate tokens / masks ───────────────────────────
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # Mask for new tokens → **always True** to prevent FA2 query length 0
        new_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=base_device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        cur_len += 1

        if verbose:
            used = torch.cuda.memory_allocated(base_device) / 1e9
            print(f"[{cur_len-ctx_len:03}] tok={next_tokens.tolist()} │ GPU={used:.2f} GB")

    # ──────────────────────────────────────────────────────────
    # Post-processing (embedding aggregation)
    # ──────────────────────────────────────────────────────────
    time = time.perf_counter() - start_t

    def _stack_cpu(seq_list):
        if not seq_list or hidden_dim is None:
            return None
        zero = torch.zeros(hidden_dim, device=cpu_device)
        return torch.stack(
            [torch.stack(s).mean(0) if s else zero for s in seq_list], dim=0
        )

    def _pack_cpu(lst):
        if lst is None or hidden_dim is None:
            return None
        zero = torch.zeros(hidden_dim, device=cpu_device)
        return torch.stack([e if e is not None else zero for e in lst], dim=0)

    mean_emb    = _stack_cpu(special_embs)      if return_embeddings and use_special else None
    eos_pred    = _pack_cpu(eos_pred_list)      if eos_pred_list  else None
    eos_first   = _pack_cpu(eos_first_list)     if eos_first_list else None
    eos_special = _stack_cpu(eos_special_embs)  if eos_special_embs else None

    if return_dict:
        class GenOutput:
            def __init__(self, **kw): self.__dict__.update(kw)

        return GenOutput(
            sequences=input_ids.to(cpu_device),
            mean_embeddings=mean_emb,
            eos_pred_embeddings=eos_pred,
            eos_first_embeddings=eos_first,
            eos_special_embeddings=eos_special,
            time=time,
            prob_list=prob_list,
        )

    return input_ids.to(cpu_device)


# Inject method into model ----------------------------------------------------
def add_enhanced_generation(model):
    """
    >>> model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")
    >>> model = add_enhanced_generation(model)
    >>> out = model.generate_with_special_tokens(...)
    """
    model.generate_with_special_tokens = generate_with_special_tokens.__get__(
        model, model.__class__
    )
    return model