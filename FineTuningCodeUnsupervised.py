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
import torch,gc

torch.cuda.empty_cache()
gc.collect()

model_name = "unsloth/Qwen3-4B"

max_seq_length = 2048  # Choose sequence length
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
Formula: {inp['formula_ATL']}
Notes: {model['metadata']['notes']}
"""

#try:
 #   formatted_data = [format_prompt(item) for item in file]
#except:
#    print(f"error in {item}")
formatted_data = []
for item in file:
    try : 
        formatted_data.append(format_prompt(item))
    except: 
        print(f"error in {item}")

dataset = Dataset.from_dict({"text": formatted_data})


# In[ ]:


# Add LoRA adapters ??hidden parameters??
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "v_proj"],  # reduce from 7 to 2 modules
    lora_alpha=16,                         # should be 2x rank, not 128
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
    mlm=False # mlm is set off to perform an inference of gpt style, ie we try to predict the next word?
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
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
