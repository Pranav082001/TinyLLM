# TinyLLM

Code repository for the UdS Pretraining LLM Software Project.

This project explores training and evaluating a lightweight GPT-2–style language model, with a focus on **zero-shot evaluation** on multiple-choice question answering benchmarks.

---

## Zero-Shot QA Evaluation

The trained model is evaluated **without any fine-tuning** using likelihood-based multiple-choice scoring on:

- **CommonsenseQA**
- **ARC Challenge**

For each question, the model scores all answer options using the **length-normalized log-likelihood** under a causal language model.  
The option with the highest score is selected as the prediction.

---

## Run Evaluation

```bash
python evaluate_qa_zero_shot.py \
  --checkpoint checkpoint_epoch_1_step_25000_FIXED.pth \
  --output_dir qa_results \
  --device cuda


qa_results/
├── commonsenseqa_predictions.csv
└── arc_challenge_predictions.csv


_______________________________________________

## TruthfulQA Evaluation

The trained model is also evaluated **without any fine-tuning** on the **TruthfulQA multiple-choice benchmark**, using likelihood-based scoring over truthful and false candidate answers.

For each question, the model scores:

* one best true answer
* several additional true answers
* several false answers

Only the answer tokens are scored under the causal language model, so the question itself does not influence the final likelihood score.

### Metrics

The evaluation reports three standard TruthfulQA multiple-choice metrics:

* **MC1** — whether the best true answer is ranked above all false answers
* **MC2** — normalized probability mass assigned to all true answers
* **MC3** — fraction of true answers that score above the strongest false answer

These metrics measure how consistently the model prefers truthful answers over plausible but misleading alternatives.

---

## Run Evaluation

```bash
python TinyLLM_TruthfulQA_eval.py \
  --checkpoint checkpoint_epoch_1_step_25000_FIXED.pth \
  --input_csv TruthfulQA.csv \
  --output_csv truthfulqa_results.csv \
  --device cuda
```

---

## Output

```text
truthfulqa_results.csv
```

The output CSV contains:

* question
* best gold answer
* best true answer preferred by the model
* best incorrect answer preferred by the model
* MC1 / MC2 / MC3 scores
* true and false answer scores

This makes it possible to inspect both aggregate benchmark performance and individual failure cases.

```
```
