# -*- coding: utf-8 -*-

import os
os.environ["UNSLOTH_DISABLE_FUSED_CE_LOSS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import gc

with open("dataset_augmented.json", "r") as f:
    file = json.load(f)
print(file[1])

# ── GPU check ─────────────────────────────────────────────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Load model ────────────────────────────────────────────────────────────────
torch.cuda.empty_cache()
gc.collect()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0, bias="none",
    task_type="CAUSAL_LM",
))

# ── Format dataset ────────────────────────────────────────────────────────────
from datasets import Dataset

def format_prompt(item):
    inp = item['input']
    return f"""States: {', '.join(inp['states'])}
Agents: {', '.join(inp['agents'])}
Actions: {' | '.join(f"{a}: {', '.join(acts)}" for a, acts in inp['actions'].items())}
Transitions: {' | '.join(f"{t['from']} --[{', '.join(t['joint'])}]--> {t['to']}" for t in inp['transitions'])}
Initial state: {inp['initial_state']}
Labeling: {' | '.join(f"{s}: {', '.join(labels)}" for s, labels in inp['labeling'].items())}
Coalition: {', '.join(inp['coalition'])}
Formula: {inp['formula_ATL']}
Notes: {item['metadata']['notes']}
"""

formatted_data = []
for item in file:
    try:
        formatted_data.append(format_prompt(item))
    except Exception as e:
        print(f"error in {item}: {e}")

dataset = Dataset.from_dict({"text": formatted_data})

# ── Trainer setup ─────────────────────────────────────────────────────────────
import trl
print(trl.__version__)

from trl.trainer.sft_trainer import SFTTrainer
from trl import SFTConfig

torch.cuda.empty_cache()
gc.collect()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=512,
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
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        report_to="none",
    ),
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer_stats = trainer.train()
print(trainer_stats)

# ── Merge and save (required for Ollama / standalone use) ─────────────────────
merged = model.merge_and_unload()
merged.save_pretrained("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")
print("Modèle sauvegardé.")

# ── Copy to Google Drive ───────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree("outputs/final_model", "/content/drive/MyDrive/final_model")
print("Copié sur Drive.")

# ── Inference test ────────────────────────────────────────────────────────────
model.eval()

test_input = {
    "states": ["s0", "s1"],
    "agents": ["h", "i"],
    "actions": {"h": ["lock", "idle"], "i": ["push", "idle"]},
    "transitions": [
        {"from": "s0", "joint": ["lock", "push"], "to": "s1"},
        {"from": "s0", "joint": ["lock", "idle"], "to": "s1"},
        {"from": "s0", "joint": ["idle", "push"], "to": "s0"},
        {"from": "s0", "joint": ["idle", "idle"], "to": "s0"},
    ],
    "initial_state": "s0",
    "labeling": {"s0": ["open"], "s1": ["locked", "secure"]},
    "coalition": ["h"],
    "formula_ATL": "<h> X locked"
}

messages = [
    {"role": "system", "content": "You are a formal verification assistant for ATL model checking."},
    {"role": "user", "content": f"""States: {', '.join(test_input['states'])}
Agents: {', '.join(test_input['agents'])}
Actions: {' | '.join(f"{a}: {', '.join(acts)}" for a, acts in test_input['actions'].items())}
Transitions: {' | '.join(f"{t['from']} --[{', '.join(t['joint'])}]--> {t['to']}" for t in test_input['transitions'])}
Initial state: {test_input['initial_state']}
Labeling: {' | '.join(f"{s}: {', '.join(labels)}" for s, labels in test_input['labeling'].items())}
Coalition: {', '.join(test_input['coalition'])}
Formula: {test_input['formula_ATL']}"""},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)

# ── Reload from Drive (run separately when needed) ────────────────────────────
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
#
# base_model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2-1.5B",
#     quantization_config=bnb_config,
#     device_map="auto",
# )
# model = PeftModel.from_pretrained(base_model, "/content/drive/MyDrive/final_model")
# tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/final_model")
# print("Modèle rechargé.")