import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback,
    logging,
)
import os
import tqdm as tqdm

# --- Configuration ---
model_name = "hssawhney/Reasoning-Model"
output_dir = "output/mcq-model"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training arguments
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "adamw_torch"
learning_rate = 2e-5
fp16 = False
bf16 = True
weight_decay = 0.01
max_grad_norm = 0.3
num_train_epochs = 1
warmup_ratio = 0.03
lr_scheduler_type = "cosine"
logging_steps = 5
save_steps = 500
eval_steps = 500
max_seq_length = 256  # Maximum sequence length for training
gradient_checkpointing = True
gradient_checkpointing_kwargs = {"use_reentrant": False}
report_to = "tensorboard"
disable_tqdm = False

# Early Stopping parameters
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
greater_is_better = False
early_stopping_patience = 3
early_stopping_threshold = 0.01  # Minimum change to qualify as an improvement


# --- Data Formatting Function ---
def format_example(example):
    prompt = f"{example['question']}\n"
    response = f"{example['answer']}"

    return {
        "text": prompt + response,
        "prompt_length_chars": len(prompt),
    }


# --- Utility Function for Dataset Analysis ---
def find_median_length(dataset):
    lengths = [len(example["text"]) for example in dataset]
    lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
    median_len = int(lengths_tensor.median().item())
    percentile_90_len = int(lengths_tensor.quantile(0.9).item())
    return median_len, percentile_90_len


# --- Load and Format Dataset ---
print("Loading and formatting dataset...")
dataset = load_dataset("hssawhney/MCQ-Dataset")
ds_train = dataset["train"]
ds_eval = dataset["test"]

ds_train_formatted = [
    format_example(example)
    for example in tqdm.tqdm(ds_train, desc="Formatting Training Data")
]
ds_eval_formatted = [
    format_example(example)
    for example in tqdm.tqdm(ds_eval, desc="Formatting Evaluation Data")
]

ds_train = Dataset.from_list(ds_train_formatted)
ds_eval = Dataset.from_list(ds_eval_formatted)

median_length, percentile_length = find_median_length(ds_train)
print(f"Median character length of combined data: {median_length}")
print(f"90th percentile character length of combined data: {percentile_length}")

print(f"Training examples: {len(ds_train)}")
print(f"Evaluation examples: {len(ds_eval)}")
print(f"Example formatted training data (first entry): {ds_train[0]}")

# --- Load Tokenizer and Model ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# NOTE: When resuming, the Trainer will load the model weights from the checkpoint.
# So, it's generally fine to load the base model here, and the Trainer will overwrite
# the weights with the checkpoint's weights.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False


# --- MODIFIED Tokenization and Loss Masking Function ---
def tokenize_and_add_labels(example):
    """
    Tokenizes the full text and creates labels for loss masking.
    Labels corresponding to the prompt tokens and padding tokens are set to -100.
    """
    # Tokenize the full combined text, NOW WITH padding="max_length"
    tokenized_full = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",  # <--- CRUCIAL CHANGE: Pad to max_seq_length
        return_offsets_mapping=True,  # Still needed to find prompt/response boundary
    )

    # Create labels: initially identical to input_ids
    labels = list(tokenized_full["input_ids"])

    # Find the token index where the 'response' part begins.
    prompt_char_length = example["prompt_length_chars"]

    response_start_token_idx = 0
    for i, (start_char, end_char) in enumerate(tokenized_full["offset_mapping"]):
        if start_char >= prompt_char_length:
            response_start_token_idx = i
            break

    # Mask the prompt part AND any padding tokens
    for i in range(len(labels)):
        # Mask prompt tokens
        if i < response_start_token_idx:
            labels[i] = -100
        # Mask any padding tokens (if the sequence was shorter than max_seq_length)
        # Check against the pad_token_id, which is likely tokenizer.eos_token_id for Qwen
        if tokenized_full["input_ids"][i] == tokenizer.pad_token_id:
            labels[i] = -100

    return {
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,  # The masked labels
    }


print("Tokenizing datasets and applying loss masking...")

if os.path.exists(os.path.join(output_dir, "train_dataset")):
    print("Loading existing tokenized training dataset from disk...")
    ds_train = Dataset.load_from_disk(os.path.join(output_dir, "train_dataset"))
    ds_eval = Dataset.load_from_disk(os.path.join(output_dir, "eval_dataset"))
else:
    print("No existing tokenized datasets found. Proceeding with tokenization...")
    # Apply the tokenization and labeling function to the datasets
    ds_train = ds_train.map(
        tokenize_and_add_labels,
        batched=False,
        remove_columns=["text", "prompt_length_chars"],
    )
    ds_eval = ds_eval.map(
        tokenize_and_add_labels,
        batched=False,
        remove_columns=["text", "prompt_length_chars"],
    )

    # Save the tokenized datasets to disk for future use
    ds_train.save_to_disk(os.path.join(output_dir, "train_dataset"))
    ds_eval.save_to_disk(os.path.join(output_dir, "eval_dataset"))

# Remove the temporary 'offset_mapping' column added during tokenization (if it was added by map)
# This was causing an error before because it wasn't returned, but it's good practice
# to ensure it's removed if it somehow gets through, though it shouldn't now.
# To be absolutely safe, let's include it conditionally or just let it pass if it's not there.
# Given the previous error, it's definitively NOT there, so these lines remain commented.
# ds_train = ds_train.remove_columns(["offset_mapping"])
# ds_eval = ds_eval.remove_columns(["offset_mapping"])

print(f"Tokenized training example (first entry):\n{ds_train[0]}")

# --- Setup Trainer ---
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience,
    early_stopping_threshold=early_stopping_threshold,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    weight_decay=weight_decay,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    # group_by_length=True, # You can try keeping this, but with max_length padding, it's less critical
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    gradient_checkpointing=gradient_checkpointing,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    disable_tqdm=disable_tqdm,
)

# DataCollatorForLanguageModeling is still used. It will now receive uniformly
# padded sequences and just convert them to tensors, respecting the -100 labels.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
)

# --- Train and Save ---
print("Starting training...")

last_checkpoint = None
if os.path.isdir(training_arguments.output_dir):
    checkpoints = [
        d
        for d in os.listdir(training_arguments.output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(training_arguments.output_dir, d))
    ]
    if checkpoints:
        # Sort to find the latest checkpoint numerically
        last_checkpoint = os.path.join(
            training_arguments.output_dir,
            max(checkpoints, key=lambda x: int(x.split("-")[1])),
        )
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoints found. Starting training from scratch.")

# Pass the identified checkpoint path to trainer.train()
trainer.train(resume_from_checkpoint=last_checkpoint)


print("Training complete. Saving model and tokenizer...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(
    output_dir
)  # Always save the tokenizer, it's small and important
print(f"Model and tokenizer saved to {output_dir}")
