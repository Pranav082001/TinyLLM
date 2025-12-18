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
