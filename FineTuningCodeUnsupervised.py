#!/usr/bin/env python
# coding: utf-8

import json

file = json.load(open("dataset_augmented.json", "r"))
print(file[1])

# pip install unsloth trl peft accelerate bitsandbytes

# ── GPU check ─────────────────────────────────────────────────────────────────
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Memory config ─────────────────────────────────────────────────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Load model ────────────────────────────────────────────────────────────────
import gc
from unsloth import FastLanguageModel

torch.cuda.empty_cache()
gc.collect()

model_name = "unsloth/Qwen3-4B"
max_seq_length = 512
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

# ── Format dataset ────────────────────────────────────────────────────────────
from datasets import Dataset

def format_prompt(item):
    inp = item['input']
    actions_str = ' | '.join(a + ': ' + ', '.join(acts) for a, acts in inp['actions'].items())
    transitions_str = ' | '.join(t['from'] + ' --[' + ', '.join(t['joint']) + ']--> ' + t['to'] for t in inp['transitions'])
    labeling_str = ' | '.join(s + ': ' + ', '.join(labels) for s, labels in inp['labeling'].items())
    return (
        'States: ' + ', '.join(inp['states']) + '\n'
        'Agents: ' + ', '.join(inp['agents']) + '\n'
        'Actions: ' + actions_str + '\n'
        'Transitions: ' + transitions_str + '\n'
        'Initial state: ' + inp['initial_state'] + '\n'
        'Labeling: ' + labeling_str + '\n'
        'Coalition: ' + ', '.join(inp['coalition']) + '\n'
        'Formula: ' + inp['formula_atl'] + '\n'
        'Notes: ' + item['metadata']['notes'] + '\n'
    )

formatted_data = []
for item in file:
    try:
        text = format_prompt(item)
        if isinstance(text, str) and text.strip():
            formatted_data.append(text)
    except Exception as e:
        print('Skipping item due to error: ' + str(e))

print('Total valid samples: ' + str(len(formatted_data)))

dataset = Dataset.from_dict({"text": formatted_data})

# ── Add LoRA adapters ─────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ── Pre-tokenize dataset ──────────────────────────────────────────────────────
# Pre-tokenizing before passing to SFTTrainer avoids the dataset probing issue
# in Unsloth's compiled SFTTrainer (next(iter(train_dataset)) crash).
# With dataset_text_field omitted, SFTTrainer treats the dataset as already
# tokenized and never tries to inspect or convert text columns.

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
    num_proc=2,
)
tokenized_dataset.set_format("torch")

# ── Trainer setup ─────────────────────────────────────────────────────────────
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

torch.cuda.empty_cache()
gc.collect()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,
    args=TrainingArguments(
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

# ── Inference test ────────────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)

test_input = {
    "states": ["s0", "s1"],
    "agents": ["h", "i"],
    "actions": {"h": ["lock", "idle"], "i": ["push", "idle"]},
    "transitions": [
        {"from": "s0", "joint": ["lock", "push"],   "to": "s1"},
        {"from": "s0", "joint": ["lock", "idle"],   "to": "s1"},
        {"from": "s0", "joint": ["idle", "push"],   "to": "s0"},
        {"from": "s0", "joint": ["idle", "idle"],   "to": "s0"},
    ],
    "initial_state": "s0",
    "labeling": {"s0": ["open"], "s1": ["locked", "secure"]},
    "coalition": ["h"],
    "formula_ATL": "<h> X locked",
}

actions_str  = ' | '.join(a + ': ' + ', '.join(acts) for a, acts in test_input['actions'].items())
trans_str    = ' | '.join(t['from'] + ' --[' + ', '.join(t['joint']) + ']--> ' + t['to'] for t in test_input['transitions'])
labeling_str = ' | '.join(s + ': ' + ', '.join(labels) for s, labels in test_input['labeling'].items())

user_content = (
    'States: '       + ', '.join(test_input['states'])    + '\n'
    'Agents: '       + ', '.join(test_input['agents'])    + '\n'
    'Actions: '      + actions_str                        + '\n'
    'Transitions: '  + trans_str                          + '\n'
    'Initial state: '+ test_input['initial_state']        + '\n'
    'Labeling: '     + labeling_str                       + '\n'
    'Coalition: '    + ', '.join(test_input['coalition']) + '\n'
    'Formula: '      + test_input['formula_ATL']
)

messages = [
    {
        "role": "system",
        "content": "You are a formal verification assistant for ATL model checking.",
    },
    {
        "role": "user",
        "content": user_content,
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

response = tokenizer.batch_decode(outputs)[0]
print(response)