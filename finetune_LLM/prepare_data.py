"""
prepare_data.py — Telco LLM Fine-Tuning: Data Engineering Pipeline
==================================================================
Run: python prepare_data.py
Output: ./telco_train/, ./telco_val/  (HuggingFace datasets on disk)

Requirements:
    pip install datasets transformers presidio-analyzer presidio-anonymizer
"""

import json, re
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
PLACEHOLDER_SUBS = {
    "{{WEBSITE_URL}}":           "support.mytelco.com",
    "{{INVOICE_SECTION}}":       "My Bills",
    "{{DISPUTE_INVOICE_OPTION}}":"Report an Issue",
    "{{DAYS_NUMBER}}":           "3-5 business days",
    "{{ROAMING_RATE}}":          "$0.25/MB",
    "{{PLAN_NAME}}":             "Unlimited Pro",
    "{{CONTACT_NUMBER}}":        "1-800-MY-TELCO",
    "{{SERVICE_NAME}}":          "MyTelco Connect",
    "{{NETWORK_AREA}}":          "your registered service area",
}
MAX_PER_CATEGORY = 4000
MIN_RESPONSE_WORDS = 25
MAX_RESPONSE_WORDS = 450
TRAIN_SPLIT = 0.9
SEED = 42

# ── STEP 1: Load datasets ────────────────────────────────────────────────────
print("► Loading Bitext Telco dataset...")
bitext = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset")["train"]
print(f"  Bitext: {len(bitext)} rows, columns: {bitext.column_names}")

print("► Loading TeleQnA dataset...")
teleqna_raw = load_dataset("netop/TeleQnA")["train"]
print(f"  TeleQnA: {len(teleqna_raw)} rows")

# ── STEP 2: Convert TeleQnA MCQ → SFT pairs ──────────────────────────────────
def teleqna_to_sft(ex):
    """Convert multiple-choice question to instruction-response pair."""
    answer_key = ex.get("answer", "option 1")
    answer_text = ex.get(answer_key, "")
    options = "\n".join(
        f"{i+1}. {ex.get(f'option {i+1}','')}"
        for i in range(4) if ex.get(f"option {i+1}")
    )
    instruction = f"{ex['question']}\n\nOptions:\n{options}"
    response = (
        f"The correct answer is: {answer_text}. "
        f"This falls under the {ex.get('category','telecom')} domain of telecommunications knowledge."
    )
    return {
        "instruction": instruction,
        "response": response,
        "category": ex.get("category", "GENERAL"),
        "intent": "knowledge_qa",
        "tags": "B",
    }

print("► Converting TeleQnA MCQ to SFT format...")
teleqna_sft = teleqna_raw.map(teleqna_to_sft, remove_columns=teleqna_raw.column_names)
# Sample 2000 for balance (TeleQnA is an eval set primarily)
teleqna_sft = teleqna_sft.shuffle(seed=SEED).select(range(min(2000, len(teleqna_sft))))

# ── STEP 3: Placeholder substitution ─────────────────────────────────────────
def substitute_placeholders(text):
    for k, v in PLACEHOLDER_SUBS.items():
        text = text.replace(k, v)
    return text

# ── STEP 4: Quality filtering ─────────────────────────────────────────────────
def quality_filter(ex):
    resp = ex.get("response", "")
    instr = ex.get("instruction", "")
    # Remove residual placeholders
    if "{{" in resp or "}}" in resp:
        return False
    # Length checks
    word_count = len(resp.split())
    if word_count < MIN_RESPONSE_WORDS or word_count > MAX_RESPONSE_WORDS:
        return False
    # Must have a non-empty instruction
    if len(instr.strip()) < 10:
        return False
    return True

# ── STEP 5: Format to ChatML ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are TelcoBot, an expert telecom support agent. "
    "You help customers with billing queries, network issues, device support, "
    "and technical troubleshooting. Always be professional, concise, and accurate."
)

def format_example(ex):
    instruction = ex.get("instruction", "").strip()
    response = substitute_placeholders(ex.get("response", "").strip())
    category = ex.get("category", "GENERAL")
    # ChatML format for Mistral / Llama 3.1
    text = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"[Category: {category}] {instruction} [/INST] {response}</s>"
    )
    return {"text": text, "category": category}

# ── STEP 6: Combine and process ───────────────────────────────────────────────
print("► Processing Bitext dataset...")
bitext_processed = bitext.map(
    lambda ex: {**ex, "response": substitute_placeholders(ex.get("response", ""))}
)
bitext_filtered = bitext_processed.filter(quality_filter)
print(f"  After filtering: {len(bitext_filtered)} / {len(bitext)} rows kept")

print("► Combining datasets...")
combined = concatenate_datasets([bitext_filtered, teleqna_sft])
print(f"  Combined: {len(combined)} total rows")

# ── STEP 7: Deduplication ─────────────────────────────────────────────────────
print("► Deduplicating on instruction text...")
seen = set()
def dedup_filter(ex):
    key = ex.get("instruction", "").lower().strip()[:200]
    if key in seen:
        return False
    seen.add(key)
    return True
deduped = combined.filter(dedup_filter)
print(f"  After dedup: {len(deduped)} rows")

# ── STEP 8: Category balancing ────────────────────────────────────────────────
print("► Balancing categories...")
counts = defaultdict(int)
def balance_filter(ex):
    cat = ex.get("category", "OTHER")
    if counts[cat] >= MAX_PER_CATEGORY:
        return False
    counts[cat] += 1
    return True
balanced = deduped.filter(balance_filter)
print(f"  After balancing: {len(balanced)} rows")
print("  Category distribution:")
cat_counts = defaultdict(int)
for ex in balanced:
    cat_counts[ex.get("category", "OTHER")] += 1
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"    {cat}: {cnt}")

# ── STEP 9: Format + split ────────────────────────────────────────────────────
print("► Formatting to ChatML...")
formatted = balanced.map(format_example)
split = formatted.train_test_split(test_size=1 - TRAIN_SPLIT, seed=SEED)

print("► Saving to disk...")
split["train"].save_to_disk("./telco_train")
split["test"].save_to_disk("./telco_val")

print(f"\n✓ Pipeline complete!")
print(f"  Train: {len(split['train'])} examples → ./telco_train/")
print(f"  Val:   {len(split['test'])} examples  → ./telco_val/")
print(f"\nNext step: python train_telco_qlora.py")
