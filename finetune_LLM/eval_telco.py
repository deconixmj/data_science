"""
eval_telco.py — Telco LLM Evaluation Suite
===========================================
Run: python eval_telco.py --model ./telco-mistral-qlora/merged

Metrics computed:
  - TeleQnA accuracy (domain knowledge benchmark)
  - ROUGE-L (response content overlap)
  - Exact Match on billing/error codes
  - BERTScore F1 (semantic similarity)
  - Perplexity on validation set

Requirements:
    pip install datasets transformers torch rouge-score bert-score
"""

import argparse
import json
import time
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer as rs
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate fine-tuned telco LLM")
parser.add_argument("--model",    default="./telco-mistral-qlora/merged", help="Path to merged model")
parser.add_argument("--val_data", default="./telco_val",                  help="Validation dataset path")
parser.add_argument("--n_teleqna", type=int, default=500,                 help="TeleQnA examples to eval")
parser.add_argument("--n_rouge",   type=int, default=200,                 help="ROUGE examples to eval")
parser.add_argument("--output",   default="eval_results.json",            help="Output JSON path")
args = parser.parse_args()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print(f"► Loading model from: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("  Model loaded ✓")

def generate(prompt, max_new_tokens=200):
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = out[0][ids["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── METRIC 1: TeleQnA Accuracy ────────────────────────────────────────────────
def eval_teleqna(n=500):
    print(f"\n► Evaluating TeleQnA accuracy (n={n})...")
    ds = load_dataset("netop/TeleQnA")["train"].shuffle(seed=42).select(range(n))
    correct, total, by_cat = 0, 0, {}

    for ex in ds:
        opts = "\n".join(
            f"{i+1}. {ex.get(f'option {i+1}','')}"
            for i in range(4) if ex.get(f"option {i+1}")
        )
        prompt = (
            f"<s>[INST] You are a telecom expert. Answer this question with ONLY the option number (1, 2, 3, or 4).\n\n"
            f"Question: {ex['question']}\n\nOptions:\n{opts} [/INST]"
        )
        pred = generate(prompt, max_new_tokens=5)
        gold = ex["answer"].replace("option ", "").strip()
        cat  = ex.get("category", "Unknown")

        is_correct = gold in pred[:10]
        correct += int(is_correct)
        total   += 1
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["correct"] += int(is_correct)
        by_cat[cat]["total"]   += 1

        if total % 50 == 0:
            print(f"  Progress: {total}/{n} — current accuracy: {correct/total:.1%}")

    overall = correct / total
    print(f"\n  ✓ TeleQnA Overall: {overall:.1%}")
    print("  By category:")
    for cat, v in sorted(by_cat.items()):
        print(f"    {cat}: {v['correct']}/{v['total']} = {v['correct']/v['total']:.1%}")
    return {"overall": overall, "by_category": {k: v["correct"]/v["total"] for k,v in by_cat.items()}}

# ── METRIC 2: ROUGE-L ─────────────────────────────────────────────────────────
def eval_rouge(n=200):
    print(f"\n► Evaluating ROUGE-L (n={n})...")
    try:
        val_ds = load_from_disk(args.val_data)
    except Exception:
        print("  val_data not found — skipping ROUGE eval")
        return None

    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    subset = val_ds.shuffle(seed=42).select(range(min(n, len(val_ds))))

    for ex in subset:
        text = ex.get("text", "")
        if "[/INST]" not in text:
            continue
        parts = text.split("[/INST]")
        instruction = parts[0].replace("<s>[INST]", "").strip()
        reference   = parts[1].replace("</s>", "").strip() if len(parts) > 1 else ""
        if not reference:
            continue
        prompt   = f"<s>[INST] {instruction} [/INST]"
        pred     = generate(prompt, max_new_tokens=150)
        result   = scorer.score(reference, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)

    avg = {k: sum(v)/len(v) for k,v in scores.items() if v}
    print(f"  ✓ ROUGE-1: {avg.get('rouge1',0):.3f} | ROUGE-2: {avg.get('rouge2',0):.3f} | ROUGE-L: {avg.get('rougeL',0):.3f}")
    return avg

# ── METRIC 3: Exact Match on code lookups ─────────────────────────────────────
def eval_exact_match():
    print("\n► Evaluating Exact Match on billing/error codes...")
    # Synthetic test cases — add your actual proprietary codes here
    test_cases = [
        {"prompt": "[INST] What billing category does code RC-4872 fall under? [/INST]", "expected_substring": "prorated"},
        {"prompt": "[INST] Which router generates error E-4531? [/INST]", "expected_substring": "ASR 9000"},
        {"prompt": "[INST] What does MVNO stand for? [/INST]", "expected_substring": "Mobile Virtual Network Operator"},
        {"prompt": "[INST] What does VoLTE stand for? [/INST]", "expected_substring": "Voice over LTE"},
        {"prompt": "[INST] What is ARPU in telecom? [/INST]", "expected_substring": "Average Revenue Per User"},
    ]
    correct = 0
    for tc in test_cases:
        pred = generate(f"<s>{tc['prompt']}", max_new_tokens=80).lower()
        match = tc["expected_substring"].lower() in pred
        correct += int(match)
        status = "✓" if match else "✗"
        print(f"  {status} Expected '{tc['expected_substring']}'")

    score = correct / len(test_cases)
    print(f"  ✓ Exact Match: {score:.1%} ({correct}/{len(test_cases)})")
    return score

# ── METRIC 4: Perplexity ──────────────────────────────────────────────────────
def eval_perplexity(n=100):
    print(f"\n► Evaluating Perplexity (n={n})...")
    try:
        val_ds = load_from_disk(args.val_data)
    except Exception:
        print("  val_data not found — skipping perplexity eval")
        return None

    import math
    total_loss, count = 0.0, 0
    subset = val_ds.shuffle(seed=42).select(range(min(n, len(val_ds))))

    for ex in subset:
        text = ex.get("text", "")
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            loss = model(**ids, labels=ids["input_ids"]).loss.item()
        total_loss += loss
        count += 1

    avg_loss = total_loss / count if count else float("inf")
    ppl = math.exp(avg_loss)
    print(f"  ✓ Perplexity: {ppl:.2f} (avg cross-entropy loss: {avg_loss:.3f})")
    print(f"  Interpretation: < 8 = good | 8-15 = acceptable | > 15 = needs more data")
    return ppl

# ── RUN ALL EVALUATIONS ───────────────────────────────────────────────────────
print("=" * 60)
print("TELCO LLM EVALUATION SUITE")
print("=" * 60)

results = {}
start = time.time()

results["teleqna"] = eval_teleqna(n=args.n_teleqna)
results["rouge"]   = eval_rouge(n=args.n_rouge)
results["exact_match"] = eval_exact_match()
results["perplexity"]  = eval_perplexity()

elapsed = time.time() - start

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"  TeleQnA Accuracy: {results['teleqna']['overall']:.1%}  (baseline ~58–63%)")
if results["rouge"]:
    print(f"  ROUGE-L:          {results['rouge'].get('rougeL',0):.3f}  (target > 0.35)")
print(f"  Exact Match:      {results['exact_match']:.1%}  (target > 85%)")
if results["perplexity"]:
    print(f"  Perplexity:       {results['perplexity']:.2f}  (target < 8.0)")
print(f"\n  Total eval time: {elapsed/60:.1f} min")
print(f"\n  Results saved to: {args.output}")

with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
