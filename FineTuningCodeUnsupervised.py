#!usr/bin/env python
# coding: utf-8

# <h1>
# DO NOT RUN IT ON YOUR OWN MACHINE, Use something like <i>Google Colab</i>
# </h1>
# 
# 

# In[1]:


import json

file = json.load(open("dataset_augmented.json", "r"))
print(file[1])


# In[2]:


#get_ipython().system('pip install unsloth trl peft accelerate bitsandbytes')


# In[ ]:


# For GPU check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


# In[ ]:


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# In[ ]:


from unsloth import FastLanguageModel
import torch, gc

torch.cuda.empty_cache()
gc.collect()

model_name = "unsloth/Qwen3-4B"

max_seq_length = 512  # ← was 2048; biggest single memory saving (attention is O(n²))
dtype = None

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)


# <b>No need for format markers in the unsupervised finetuning</b>

# In[ ]:


from datasets import Dataset
#here we format a specific string to give to the LLM
def format_prompt(model):
    inp = model['input']
    return f"""States: {', '.join(inp['states'])}
Agents: {', '.join(inp['agents'])}
Actions: {' | '.join(f"{a}: {', '.join(acts)}" for a, acts in inp['actions'].items())}
Transitions: {' | '.join(f"{t['from']} --[{', '.join(t['joint'])}]--> {t['to']}" for t in inp['transitions'])}
Initial state: {inp['initial_state']}
Labeling: {' | '.join(f"{s}: {', '.join(labels)}" for s, labels in inp['labeling'].items())}
Coalition: {', '.join(inp['coalition'])}
Formula: {inp['formula_atl']}
Notes: {model['metadata']['notes']}
"""

formatted_data = []
for item in file:
    try:
        formatted_data.append(format_prompt(item))
    except Exception as e:
        print(f"error in {item}: {e}")

dataset = Dataset.from_dict({"text": formatted_data})


# In[ ]:


# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=4,               # ← was 8; halves adapter memory
    target_modules=["q_proj", "v_proj"],
    lora_alpha=8,      # ← was 16; keep 2x rank
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# In[ ]:


from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # mlm off → causal LM / predict next token
)

# Clear cache right before trainer init
torch.cuda.empty_cache()
gc.collect()

# ── Patch: bypass Unsloth/TRL dataset probing that crashes on text-only datasets ──
# Unsloth's compiled trainer probes the dataset in two ways on __init__:
#   - line 897:  next(iter(dataset))          → fine, Dataset supports this
#   - line 1166: dataset[field][0]            → KeyError: 0 (dict-style access on a str)
# The TRL base also probes dataset_text_field. Both are patched below.

from trl.trainer.sft_trainer import SFTTrainer as _TRLSFTTrainer

_original_prepare = _TRLSFTTrainer._prepare_dataset

def _patched_prepare_dataset(self, dataset, tokenizer, packing, dataset_text_field, *args, **kwargs):
    if dataset_text_field and dataset_text_field in dataset.column_names:
        sample = dataset[dataset_text_field]
        if sample and not isinstance(sample[0], str):
            raise ValueError(
                f"Column '{dataset_text_field}' contains {type(sample[0])} instead of str. "
                "Check your format_prompt function is appending strings."
            )
    return _original_prepare(self, dataset, tokenizer, packing, dataset_text_field, *args, **kwargs)

_TRLSFTTrainer._prepare_dataset = _patched_prepare_dataset

# Patch Unsloth's own __init__ which wraps TRL's and adds its own dataset probe
import unsloth.trainer as _unsloth_trainer
_orig_unsloth_init = _unsloth_trainer.SFTTrainer.__init__

def _patched_unsloth_init(self, *args, **kwargs):
    if "train_dataset" in kwargs:
        _ds = kwargs["train_dataset"]

        class _ProbeSafeDataset:
            def __iter__(self):
                return iter(_ds)
            def __len__(self):
                return len(_ds)
            def __getitem__(self, key):
                return _ds[key]
            @property
            def column_names(self):
                return _ds.column_names
            @property
            def features(self):
                return _ds.features
            def map(self, *a, **kw):
                return _ds.map(*a, **kw)
            def select(self, *a, **kw):
                return _ds.select(*a, **kw)

        kwargs["train_dataset"] = _ProbeSafeDataset()

    _orig_unsloth_init(self, *args, **kwargs)

_unsloth_trainer.SFTTrainer.__init__ = _patched_unsloth_init
# ─────────────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=1,      # ← was 2; halves activation memory
        gradient_accumulation_steps=8,      # ← was 4; keeps effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,                          # ← hard-coded; P100 has no bfloat16
        bf16=False,                         # ← hard-coded off
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,           # ← added; kills extra worker memory copies
        gradient_checkpointing=True,        # ← added; trades recompute for ~30% memory saving
        report_to="none",
    ),
)


# In[ ]:


# Train the model
trainer_stats = trainer.train()


# In[ ]:


# Test the fine-tuned model
FastLanguageModel.for_inference(model)

# Test prompt
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

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

# Decode and print
response = tokenizer.batch_decode(outputs)[0]
print(response)