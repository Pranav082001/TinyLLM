# relpron_eval_gpt2_checkpoint.py
# Zero-shot RELPRON evaluation for GPT-style causal LMs (your own checkpoint).
#
# Scores term+property sentences via length-normalized NLL (lower is better),
# ranks candidate properties per term, and reports MRR (+ optional Recall@K).
#
# Author: Houda
# Date: <fill>

import os
import re
import math
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config


# ---------------------------------------------------------------------
# Model loading (same idea as your CommonsenseQA/ARC script)
# ---------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    """
    Load a GPT-2 small architecture model from a state_dict checkpoint (.pth).
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------
# RELPRON parsing
# ---------------------------------------------------------------------
# RELPRON files typically contain:
#   term<TAB>property
# Example (approx):
#   accordion    musical instrument that is played by squeezing
#
# We'll parse robustly: split on tab first; if missing, fall back to multiple spaces.

def parse_relpron_file(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                term, prop = line.split("\t", 1)
            else:
                # fallback: first token is term, rest is property
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    continue
                term, prop = parts[0], parts[1]

            term = term.strip()
            prop = prop.strip()
            if term and prop:
                pairs.append((term, prop))
    return pairs


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def make_sentence(term: str, prop: str) -> str:
    """
    RELPRON-style cloze as a simple declarative sentence.
    You can tweak this template if you want.
    """
    # Many properties already start like "person that ..." / "tool that ..."
    # "term is a {prop}." usually works well.
    return f"{term} is a {prop}."


def length_normalized_nll(model, tokenizer, text: str, device: str) -> float:
    """
    Returns average NLL per token for the full sequence (lower is better).
    This is exactly what HF loss gives you for labels=input_ids.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return float(out.loss.item())


# Optional: if you want *conditional* scoring of the continuation only,
# you can do that too. But RELPRON is commonly done with full-sentence PPL-ish scoring.
# Keeping it simple + stable here.


# ---------------------------------------------------------------------
# Evaluation: MRR / Recall@K
# ---------------------------------------------------------------------

def evaluate_relpron(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    device: str,
    output_csv: Optional[str] = None,
    recall_ks: Tuple[int, ...] = (1, 5, 10),
    max_candidates: Optional[int] = None,
) -> Dict[str, float]:
    """
    For each (term, gold_property), rank all candidate properties (by NLL),
    compute reciprocal rank; average => MRR.

    Candidate set = all properties in the split by default (standard for RELPRON).
    If max_candidates is set, we subsample candidates per item (mainly for quick debugging).
    """
    # Candidate properties (global set)
    all_props = [p for _, p in pairs]
    unique_props = list(dict.fromkeys(all_props))  # stable unique

    rows = []
    reciprocal_ranks = []
    recall_hits = {k: 0 for k in recall_ks}

    for term, gold_prop in tqdm(pairs, desc="Evaluating RELPRON"):
        candidates = unique_props

        if max_candidates is not None and max_candidates < len(candidates):
            # Always include gold in the candidate list
            # and sample the rest deterministically-ish.
            # (This is for speed only; not “standard”.)
            # We'll take the first N-1 non-gold + gold.
            sampled = [p for p in candidates if p != gold_prop][: max_candidates - 1]
            candidates = sampled + [gold_prop]

        scored = []
        for prop in candidates:
            sent = make_sentence(term, prop)
            nll = length_normalized_nll(model, tokenizer, sent, device)
            scored.append((prop, nll, sent))

        # Rank by increasing NLL (lower loss = better)
        scored.sort(key=lambda x: x[1])
        ranked_props = [p for p, _, _ in scored]

        rank = ranked_props.index(gold_prop) + 1
        rr = 1.0 / rank
        reciprocal_ranks.append(rr)

        for k in recall_ks:
            if rank <= k:
                recall_hits[k] += 1

        # Store a compact view for CSV
        rows.append({
            "term": term,
            "gold_property": gold_prop,
            "rank": rank,
            "reciprocal_rank": rr,
            "top1_property": scored[0][0],
            "top1_nll": scored[0][1],
            "top5_properties": [p for p, _, _ in scored[:5]],
            "top5_nll": [nll for _, nll, _ in scored[:5]],
        })

    mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
    metrics = {"MRR": mrr}

    n = len(pairs)
    for k in recall_ks:
        metrics[f"Recall@{k}"] = recall_hits[k] / max(1, n)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot RELPRON eval for GPT-style checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth state_dict checkpoint")
    parser.add_argument("--split_path", type=str, required=True, help="Path to relpron.{train,dev,test}.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, default="results/relpron_predictions.csv")
    parser.add_argument("--max_candidates", type=int, default=None, help="Optional: subsample candidate properties for speed")
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    print(f"Using device: {args.device}")
    model, tokenizer = load_model(args.checkpoint, args.device)

    print(f"Loading RELPRON split from: {args.split_path}")
    pairs = parse_relpron_file(args.split_path)
    print(f"Loaded {len(pairs)} (term, property) pairs")

    metrics = evaluate_relpron(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=args.device,
        output_csv=args.output_csv,
        recall_ks=(1, 5, 10),
        max_candidates=args.max_candidates,
    )

    print("\n=== RELPRON Zero-shot Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"\nSaved per-item results to: {args.output_csv}")


if __name__ == "__main__":
    main()
