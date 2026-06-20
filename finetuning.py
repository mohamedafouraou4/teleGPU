#!/usr/bin/env python
# coding: utf-8
"""
ATL Model Checking - LLM Fine-tuning (Local Machine)
=====================================================
Adapted from Google Colab notebook.

Requirements:
    pip install torch transformers peft trl accelerate bitsandbytes datasets

Hardware:
    - Needs a CUDA-capable GPU (≥8 GB VRAM recommended for 4-bit Qwen2-1.5B)
    - If you don't have a GPU, set USE_GPU = False below (very slow)

Usage:
    python finetune_local.py
    python finetune_local.py --resume   # resume from last checkpoint
"""

import os
import gc
import json
import argparse

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in outputs/")
parser.add_argument("--dataset", default="dataset_augmented.json", help="Path to dataset JSON")
parser.add_argument("--output-dir", default="outputs", help="Where to save checkpoints and final model")
args = parser.parse_args()

# ── Env tweaks (must happen before torch import) ───────────────────────────────
os.environ["UNSLOTH_DISABLE_FUSED_CE_LOSS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Imports ───────────────────────────────────────────────────────────────────
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2-1.5B"
OUTPUT_DIR      = args.output_dir
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
MAX_SEQ_LENGTH  = 512
DATASET_PATH    = args.dataset

# ── GPU check ─────────────────────────────────────────────────────────────────
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU found. Training will be extremely slow.")

torch.cuda.empty_cache()
gc.collect()

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"\nLoading dataset from {DATASET_PATH} …")
with open(DATASET_PATH, "r") as f:
    file = json.load(f)
print(f"Loaded {len(file)} examples. Sample:\n{file[1]}\n")


def format_prompt(item):
    inp = item["input"]
    return (
        f"States: {', '.join(inp['states'])}\n"
        f"Agents: {', '.join(inp['agents'])}\n"
        f"Actions: {' | '.join(f'{a}: {chr(44).join(acts)}' for a, acts in inp['actions'].items())}\n"
        f"Transitions: {' | '.join(f\"{t['from']} --[{', '.join(t['joint'])}]--> {t['to']}\" for t in inp['transitions'])}\n"
        f"Initial state: {inp['initial_state']}\n"
        f"Labeling: {' | '.join(f\"{s}: {', '.join(labels)}\" for s, labels in inp['labeling'].items())}\n"
        f"Coalition: {', '.join(inp['coalition'])}\n"
        f"Formula: {inp['formula_ATL']}\n"
        f"Notes: {item['metadata']['notes']}\n"
    )


formatted_data = []
for i, item in enumerate(file):
    try:
        formatted_data.append(format_prompt(item))
    except Exception as e:
        print(f"Skipping item {i}: {e}")

dataset = Dataset.from_dict({"text": formatted_data})
print(f"Dataset ready: {len(dataset)} examples\n")

# ── Load tokenizer ────────────────────────────────────────────────────────────
print(f"Loading tokenizer: {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── Load model in 4-bit ───────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} in 4-bit …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

# ── Trainer setup ─────────────────────────────────────────────────────────────
torch.cuda.empty_cache()
gc.collect()

# Detect whether a checkpoint already exists so we can auto-resume
resume_from_checkpoint = None
if args.resume and os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    if checkpoints:
        # Pick the latest checkpoint by step number
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        resume_from_checkpoint = os.path.join(OUTPUT_DIR, latest)
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        print("--resume specified but no checkpoints found. Starting fresh.")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_num_proc=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        # Save a checkpoint every epoch AND every 200 steps as a safety net
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,           # keep the 3 most recent checkpoints
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        report_to="none",
    ),
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training …")
trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
print(f"\nTraining complete. Stats:\n{trainer_stats}")

# ── Save final model ──────────────────────────────────────────────────────────
print(f"\nSaving final model to {FINAL_MODEL_DIR} …")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print("Model saved.")

# ── Quick inference test ──────────────────────────────────────────────────────
print("\nRunning inference test …")
model.eval()

test_input = {
    "states": ["s0", "s1"],
    "agents": ["h", "i"],
    "actions": {"h": ["lock", "idle"], "i": ["push", "idle"]},
    "transitions": [
        {"from": "s0", "joint": ["lock", "push"],  "to": "s1"},
        {"from": "s0", "joint": ["lock", "idle"],  "to": "s1"},
        {"from": "s0", "joint": ["idle", "push"],  "to": "s0"},
        {"from": "s0", "joint": ["idle", "idle"],  "to": "s0"},
    ],
    "initial_state": "s0",
    "labeling": {"s0": ["open"], "s1": ["locked", "secure"]},
    "coalition": ["h"],
    "formula_ATL": "<h> X locked",
}

actions_str  = " | ".join(f"{a}: {', '.join(acts)}" for a, acts in test_input["actions"].items())
trans_str    = " | ".join(f"{t['from']} --[{', '.join(t['joint'])}]--> {t['to']}" for t in test_input["transitions"])
labeling_str = " | ".join(f"{s}: {', '.join(labels)}" for s, labels in test_input["labeling"].items())

test_prompt = (
    f"States: {', '.join(test_input['states'])}\n"
    f"Agents: {', '.join(test_input['agents'])}\n"
    f"Actions: {actions_str}\n"
    f"Transitions: {trans_str}\n"
    f"Initial state: {test_input['initial_state']}\n"
    f"Labeling: {labeling_str}\n"
    f"Coalition: {', '.join(test_input['coalition'])}\n"
    f"Formula: {test_input['formula_ATL']}\n"
)

inputs = tokenizer(
    test_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_SEQ_LENGTH,
).to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Model output ===")
print(response)

# ── Reload instructions (for later use) ──────────────────────────────────────
print("""
=== To reload the model later ===

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B", quantization_config=bnb_config, device_map="auto"
    )
    model     = PeftModel.from_pretrained(base_model, "outputs/final_model")
    tokenizer = AutoTokenizer.from_pretrained("outputs/final_model")
""")
