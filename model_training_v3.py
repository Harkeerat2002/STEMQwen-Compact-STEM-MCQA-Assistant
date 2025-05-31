import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk, ClassLabel
import transformers 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
import torch
import os
import random


# --- Data Formatting Function ---
def format_cot_example_with_variations(example):
    """
    Formaats a single example (dictionary row from a Dataset) into a text string
    suitable for fine-tuning, applying Chain-of-Thought (CoT) prompting for
    'reasoning' types and random prompt variations for robustness.
    """
    question = example["question"]
    answer = example["answer"]
    q_type = example["type"]

    question_prefixes = [
        "Question: ",
        "Solve: ",
        "Problem: ",
        "Here's a multiple-choice question: ",
    ]
    cot_instructions = [
        "Think step-by-step: ",
        "Reasoning process: ",
        "Let's break this down: ",
        "Solution path: ",
    ]
    answer_suffixes = [
        "Answer: ",
        "Final Answer: ",
        "The correct option is: ",
        "Option: ",
    ]

    chosen_q_prefix = random.choice(question_prefixes)
    chosen_ans_suffix = random.choice(answer_suffixes)

    if q_type == "reasoning":
        chosen_cot_instruction = random.choice(cot_instructions)
        formatted_text = f"{chosen_q_prefix}{question}\n{chosen_cot_instruction}{answer}"
    else:
        formatted_text = f"{chosen_q_prefix}{question}\n{chosen_ans_suffix}{answer}"

    return {"text": formatted_text}


# --- Main Fine-Tuning Script ---
def main():
    print(f"Transformers version: {transformers.__version__}")
    print(f"Transformers module path: {transformers.__file__}")
    print(f"TrainingArguments module: {TrainingArguments.__module__}")
    try:
        print(f"TrainingArguments file path: {TrainingArguments.__init__.__globals__['__file__']}")
    except KeyError:
        print("Could not determine TrainingArguments file path directly from __init__.__globals__.")
        # Fallback for more complex class structures or if __init__ is not Python-defined
        import inspect
        try:
            print(f"TrainingArguments file path (inspect): {inspect.getsourcefile(TrainingArguments)}")
        except TypeError:
            print("Could not determine TrainingArguments file path using inspect.getsourcefile.")

    # Define paths for saving/loading preprocessed datasets
    train_path = "train_dataset_formatted"
    val_path = "val_dataset_formatted"

    # 1. Load and Prepare Data
    print("1. Loading and preparing data...")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Found preprocessed datasets. Loading from disk...")
        train_dataset = load_from_disk(train_path)
        val_dataset = load_from_disk(val_path)
        print(f"Loaded training data rows: {len(train_dataset)}")
        print(f"Loaded validation data rows: {len(val_dataset)}")
    else:
        print(
            "Preprocessed datasets not found. Loading raw dataset from Hugging Face Hub..."
        )
        try:
            ds_full = load_dataset("zay25/full-mcqa-stem", split="train")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(
                "Please ensure the dataset 'zay25/full-mcqa-stem' exists and is accessible."
            )
            return

        print(
            f"Raw dataset loaded with {len(ds_full)} rows. Preparing for stratified split..."
        )

        # Convert 'type' column to ClassLabel for stratified split
        unique_types = list(set(ds_full["type"]))
        type_class_label = ClassLabel(names=unique_types)
        ds_full = ds_full.cast_column("type", type_class_label)

        # Perform train/validation split directly on the Dataset object
        split_datasets = ds_full.train_test_split(
            test_size=0.1, stratify_by_column="type", seed=42
        )
        train_dataset_raw = split_datasets["train"]
        val_dataset_raw = split_datasets["test"]

        print(
            f"Train set size: {len(train_dataset_raw)}, Validation set size: {len(val_dataset_raw)}"
        )

        print("Applying formatting with Chain-of-Thought (CoT) variations...")
        train_dataset = train_dataset_raw.map(
            format_cot_example_with_variations,
            remove_columns=[
                col
                for col in train_dataset_raw.column_names
                if col not in ["question", "answer", "type"]
            ],
            desc="Formatting train set",
            num_proc=os.cpu_count(),
        )
        val_dataset = val_dataset_raw.map(
            format_cot_example_with_variations,
            remove_columns=[
                col
                for col in val_dataset_raw.column_names
                if col not in ["question", "answer", "type"]
            ],
            desc="Formatting validation set",
            num_proc=os.cpu_count(),
        )
        train_dataset = train_dataset.remove_columns(["question", "answer", "type"])
        val_dataset = val_dataset.remove_columns(["question", "answer", "type"])

        print(f"Formatted training data rows: {len(train_dataset)}")
        print(f"Formatted validation data rows: {len(val_dataset)}")
        print("Data preparation complete. No oversampling applied.")

        train_dataset.save_to_disk(train_path)
        val_dataset.save_to_disk(val_path)
        print("Formatted datasets saved to disk.")

    # 2. Load Model and Tokenizer
    print("2. Loading model and tokenizer...")
    model_id = "Qwen/Qwen3-0.6B-Base"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print("Model and tokenizer loaded.")

    # 3. Tokenize Datasets
    print("3. Tokenizing datasets...")
    max_sequence_length = 512

    tokenized_train_path = "tokenized_train_dataset_v2"
    tokenized_val_path = "tokenized_val_dataset_v2"

    def tokenize_function(examples):
        """Tokenizes text examples and ensures proper truncation and padding."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_sequence_length,
            padding="max_length",
        )

    def prepare_labels(examples):
        """Prepares labels for Causal LM by cloning input_ids."""
        examples["labels"] = examples["input_ids"].clone()
        return examples

    if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_val_path):
        print("Found tokenized datasets. Loading from disk...")
        tokenized_train_dataset = load_from_disk(tokenized_train_path)
        tokenized_val_dataset = load_from_disk(tokenized_val_path)
    else:
        print("Tokenized datasets not found. Tokenizing now...")
        tokenized_train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=["text"],
            desc="Tokenizing train set",
        )
        tokenized_val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=["text"],
            desc="Tokenizing validation set",
        )

        tokenized_train_dataset.set_format("torch")
        tokenized_val_dataset.set_format("torch")

        tokenized_train_dataset = tokenized_train_dataset.map(
            prepare_labels, desc="Preparing labels for train set"
        )
        tokenized_val_dataset = tokenized_val_dataset.map(
            prepare_labels, desc="Preparing labels for validation set"
        )

        tokenized_train_dataset.save_to_disk(tokenized_train_path)
        tokenized_val_dataset.save_to_disk(tokenized_val_path)
        print("Tokenized datasets saved to disk.")

    print("Datasets tokenized and formatted for training.")

    # 4. Configure Training Arguments and Trainer
    print("4. Configuring training arguments...")
    output_dir = "./qwen3_epfl_full_sft_model_output"

    # Early stopping callback
    early_stopping_patience = 3
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.0,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=0.001,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.05,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        eval_strategy="steps", 
        save_steps=10000,
        eval_steps=10000,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        push_to_hub=False,
        disable_tqdm=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
    )
    print("Training configuration complete.")

    # 5. Start Training
    print("5. Starting training...")
    trainer.train()
    print("Training finished.")

    # 6. Save Model Locally
    print(f"6. Saving the fine-tuned model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Model and tokenizer saved successfully locally.")


if __name__ == "__main__":
    main()