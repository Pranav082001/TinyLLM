# TinyLLM

Code repository for the UdS Pretraining LLM Software Project.

This project explores training and evaluating a lightweight GPT-2–style language model, with a focus on **zero-shot evaluation** on multiple-choice question answering benchmarks.

---

## Zero-Shot QA Evaluation

The trained model is evaluated **without any fine-tuning** using likelihood-based multiple-choice scoring on:

* **CommonsenseQA**
* **ARC Challenge**

For each question, the model scores all answer options using the **length-normalized log-likelihood** under a causal language model.
The option with the highest score is selected as the prediction.

---

## Run Evaluation

```bash
python evaluate_qa_zero_shot.py \
  --checkpoint checkpoint_epoch_1_step_25000_FIXED.pth \
  --output_dir qa_results \
  --device cuda
```

Generated files:

```text
qa_results/
├── commonsenseqa_predictions.csv
└── arc_challenge_predictions.csv
```

---

## TruthfulQA Evaluation

The trained model is also evaluated **without any fine-tuning** on the **TruthfulQA multiple-choice benchmark**.

For each question, the model scores:

* one best **true** answer
* several additional **true** answers
* several **false** answers

Only answer tokens are scored under the causal language model, while question tokens are masked during scoring.
This ensures that only the answer likelihood contributes to the final score.

### Metrics

The evaluation reports three standard TruthfulQA multiple-choice metrics:

* **MC1** — whether the best true answer is ranked above all false answers
* **MC2** — normalized probability mass assigned to all true answers
* **MC3** — fraction of true answers that score above the strongest false answer

These metrics measure how consistently the model prefers truthful answers over plausible but misleading alternatives.

### Run TruthfulQA Evaluation

```bash
python TinyLLM_TruthfulQA_eval.py \
  --checkpoint checkpoint_epoch_1_step_25000_FIXED.pth \
  --input_csv TruthfulQA.csv \
  --output_csv truthfulqa_results.csv \
  --device cuda
```

Generated file:

```text
truthfulqa_results.csv
```

---

## Physical Interaction Question Answering (PIQA)

The trained model is also evaluated **without any fine-tuning** on the **PIQA benchmark**, which tests whether a language model can distinguish between physically plausible and implausible solutions in everyday situations.

For each example, the model receives:

* one practical goal
* two candidate solutions

The model scores both candidate solutions and selects the one with the higher answer likelihood.

Only answer tokens are scored under the causal language model, while prompt tokens are masked during scoring.

### Metric

The evaluation reports **accuracy** on the validation split:

* **Accuracy** — proportion of examples where the higher-scoring solution matches the correct physical solution

This measures whether the model prefers solutions that are physically reasonable in real-world situations.

### Run PIQA Evaluation

Install the required dataset version first:

```bash
pip install datasets==3.6.0
```
Then run:

```bash
python TinyLLM_PIQA_eval.py \
  --checkpoint checkpoint_epoch_1_step_25000_FIXED.pth \
  --output_csv piqa_results.csv \
  --device cuda
```

Generated file:

```text
piqa_results.csv
```
