"""
train_telco_qlora.py — Telco LLM QLoRA Fine-Tuning Script
==========================================================
Run: python train_telco_qlora.py
Prereqs: python prepare_data.py must have been run first.

Requirements:
    pip install trl transformers peft bitsandbytes accelerate wandb datasets

GPU: Requires CUDA GPU with >= 6GB VRAM (RTX 3090 / A100 recommended)
Time: ~4-8 hours on A100 40GB for 17K examples, 3 epochs
"""

import os
import torch
import wandb
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── CONFIG ────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Model selection
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # Uncomment for Llama 3.1:
    # model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Output paths
    output_dir: str = "./telco-mistral-qlora"
    train_data_path: str = "./telco_train"
    val_data_path: str = "./telco_val"

    # LoRA hyperparameters
    lora_r: int = 16             # Rank — increase to 32 for Llama 3.1
    lora_alpha: int = 32         # Scaling = 2 * rank
    lora_dropout: float = 0.05
    # Mistral target modules:
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    # For Llama 3.1, add: "gate_proj", "up_proj", "down_proj"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4     # Effective batch = 4 * 4 = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"

    # Evaluation
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10

    # W&B project name
    wandb_project: str = "telco-llm-hackathon"

cfg = TrainingConfig()

# ── QUANTIZATION CONFIG ───────────────────────────────────────────────────────
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,     # Extra ~0.5GB savings
    )

# ── LORA CONFIG ───────────────────────────────────────────────────────────────
def get_lora_config():
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

# ── TRAINING ARGUMENTS ────────────────────────────────────────────────────────
def get_training_args():
    return TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        fp16=True,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"telco-{cfg.model_id.split('/')[-1]}",
        dataloader_num_workers=4,
        group_by_length=True,   # Speeds up training by batching similar-length examples
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"► Loading model: {cfg.model_id}")
    print(f"  VRAM check: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB available")

    # Initialize W&B
    wandb.init(project=cfg.wandb_project, config=cfg.__dict__)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Important for causal LM training

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False         # Required for gradient checkpointing
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)  # Prepare for QLoRA

    # Apply LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected: trainable params ~21M / 7.2B total = ~0.3%

    # Load datasets
    print("► Loading datasets from disk...")
    train_ds = load_from_disk(cfg.train_data_path)
    val_ds   = load_from_disk(cfg.val_data_path)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
        args=get_training_args(),
        packing=False,   # Set True for 10-15% speed boost (may mix contexts)
    )

    print("► Starting training...")
    print(f"  Model: {cfg.model_id}")
    print(f"  LoRA rank: {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    print(f"  Epochs: {cfg.num_epochs}, LR: {cfg.learning_rate}")
    print(f"  Expected time: 4-8h on A100 40GB")

    trainer.train()

    # Save adapter weights
    adapter_path = f"{cfg.output_dir}/adapter"
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"✓ Adapter saved to: {adapter_path}")

    # Merge adapter into base model for deployment
    print("► Merging adapter into base model...")
    merged_path = f"{cfg.output_dir}/merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"✓ Merged model saved to: {merged_path}")
    print(f"\nNext step: python eval_telco.py --model {merged_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
