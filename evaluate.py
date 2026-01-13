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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, AutoTokenizer
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
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model=torch.load(checkpoint_path,map_location=torch.device("cpu"),weights_only=False)
    model=model["model"]
    model.to(device)

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
        outputs = model(inputs["input_ids"], inputs["input_ids"])
        nll = outputs[1].item()  # average NLL per token

    return -nll


# ---------------------------------------------------------------------
# Evaluation: CommonsenseQA
# ---------------------------------------------------------------------

def evaluate_commonsenseqa(model, tokenizer,device: str):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
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

    return accuracy


# ---------------------------------------------------------------------
# Evaluation: ARC Challenge
# ---------------------------------------------------------------------

def evaluate_arc_challenge(model, tokenizer, device: str):

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

    return accuracy


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def evaluate(model_path ,device):

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")

    model, tokenizer = load_model(model_path, device)

    commonsense_acc=evaluate_commonsenseqa(
        model,
        tokenizer,
        device,
    )

    arc_challenge_acc=evaluate_arc_challenge(
        model,
        tokenizer,
        device,
    )

    return commonsense_acc,arc_challenge_acc

if __name__ == "__main__":
    evaluate()
