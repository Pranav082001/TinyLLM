"""
Zero-shot multiple-choice QA evaluation for GPT-style causal LMs.

This script evaluates a a TinyLLm GPT2-like–architecture model
on:
  - CommonsenseQA
  - ARC Challenge

Evaluation is done via length-normalized log-likelihood scoring
(no fine-tuning, no gradient updates).

Author: Houda
Date: <18 Dec 2025>
"""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import argparse
import os


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    """
    Load a GPT-2 small architecture model from a state_dict checkpoint.
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


def option_loglikelihood(model, tokenizer, question: str, option: str, device: str) -> float:
    """
    Compute length-normalized log-likelihood of an answer option
    conditioned on the question.

    Score = - average negative log-likelihood per token
    (higher is better)
    """
    prompt = question.strip() + "\nAnswer: " + option.strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        nll = outputs.loss.item()  # average NLL per token

    return -nll


# ---------------------------------------------------------------------
# Evaluation: CommonsenseQA
# ---------------------------------------------------------------------

def evaluate_commonsenseqa(model, tokenizer, device: str, output_csv: str):
    dataset = load_dataset("commonsense_qa")
    data = dataset["validation"]

    correct = 0
    results = []

    for item in tqdm(data, desc="Evaluating CommonsenseQA"):
        question = item["question"]
        options = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold = item["answerKey"]

        scores = [
            option_loglikelihood(model, tokenizer, question, opt, device)
            for opt in options
        ]

        pred_idx = scores.index(max(scores))
        pred = labels[pred_idx]

        if pred == gold:
            correct += 1

        results.append({
            "question": question,
            "gold": gold,
            "prediction": pred,
            "scores": scores,
            "options": options
        })

    accuracy = correct / len(data)
    print(f"CommonsenseQA accuracy (zero-shot): {accuracy * 100:.2f}%")

    pd.DataFrame(results).to_csv(output_csv, index=False)


# ---------------------------------------------------------------------
# Evaluation: ARC Challenge
# ---------------------------------------------------------------------

def evaluate_arc_challenge(model, tokenizer, device: str, output_csv: str):
    dataset = load_dataset("ai2_arc", "ARC-Challenge")
    data = dataset["validation"]

    correct = 0
    results = []

    for item in tqdm(data, desc="Evaluating ARC Challenge"):
        question = item["question"]
        options = item["choices"]
        gold = item["answerKey"]

        labels = ["A", "B", "C", "D"][:len(options)]

        scores = [
            option_loglikelihood(model, tokenizer, question, opt, device)
            for opt in options
        ]

        pred_idx = scores.index(max(scores))
        pred = labels[pred_idx]

        if pred == gold:
            correct += 1

        results.append({
            "question": question,
            "gold": gold,
            "prediction": pred,
            "scores": scores,
            "options": options
        })

    accuracy = correct / len(data)
    print(f"ARC Challenge accuracy (zero-shot): {accuracy * 100:.2f}%")

    pd.DataFrame(results).to_csv(output_csv, index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot QA evaluation for GPT-style causal LMs"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth state_dict)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to store CSV prediction files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cuda or cpu)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.checkpoint}")
    print(f"Using device: {args.device}")

    model, tokenizer = load_model(args.checkpoint, args.device)

    evaluate_commonsenseqa(
        model,
        tokenizer,
        args.device,
        os.path.join(args.output_dir, "commonsenseqa_predictions.csv")
    )

    evaluate_arc_challenge(
        model,
        tokenizer,
        args.device,
        os.path.join(args.output_dir, "arc_challenge_predictions.csv")
    )


if __name__ == "__main__":
    main()
