from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import trl
from trl import SFTTrainer, SFTConfig 
from datasets import Dataset, load_dataset
import json
import os
import inspect
import sys

print(f"Python executable being used: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path (where Python looks for modules):")
for p in sys.path:
    print(f"  - {p}")

try:
    print(f"trl module actually loaded from: {trl.__file__}")
except AttributeError:
    print("Could not determine source of trl module.")


print("\n--- SFTTrainer __init__ parameters (from the *currently loaded* trl module) ---")
try:
    print(inspect.signature(SFTTrainer.__init__))
except Exception as e:
    print(f"Error inspecting SFTTrainer signature: {e}")
print("------------------------------------------------------------------\n")

# Dataset Loading
data = load_dataset("zay25/test_dataset")
df = pd.DataFrame(data['train'])
print(df.head())

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR = "./qwen3_0_6b_stem_mcq_finetuned"
FINAL_MODEL_DIR = "./qwen3_0_6b_stem_mcq_merged_model" # New directory for the merged model
DATASET_PATH = "synthetic_stem_mcq_dataset.jsonl" # Path to your dataset file
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Adjust based on your GPU VRAM
GRADIENT_ACCUMULATION_STEPS = 4 # Effectively batch size becomes 2 * 4 = 8
MAX_SEQ_LENGTH = 1024 # Adjust based on typical length of your questions/answers
# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# --- 2. Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4", # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Set pad_token_id to eos_token_id for generation, crucial for Qwen
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right" # Qwen models prefer right padding
tokenizer.model_max_length = MAX_SEQ_LENGTH # Set max length for tokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto", # Automatically assigns model to available devices
    torch_dtype=torch.bfloat16, # Model will be loaded in bfloat16 for better precision
    trust_remote_code=True, # Required for Qwen models
)

# Prepare model for k-bit training (important for PEFT with quantization)
model = prepare_model_for_kbit_training(model)

# Set model config for fine-tuning
model.config.use_cache = False
model.config.pretraining_tp = 1 # Important for Qwen models

print("Model and Tokenizer loaded successfully.")

# --- 3. Prepare Dataset for SFT ---
print(f"Loading dataset from {DATASET_PATH}...")
dataset = df
print(f"Dataset loaded with {len(dataset)} examples.")

def format_for_qwen(example):
    """
    Formats the input and output into the Qwen3 chat template for the new dataset structure.
    Uses 'prompt' as the user message and 'target' as the assistant's response.
    """
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["target"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

print("Applying Qwen chat template to dataset...")
processed_texts = dataset.apply(format_for_qwen, axis=1).tolist()
processed_df = pd.DataFrame(processed_texts)  # Ensures a DataFrame with 'text' column
processed_dataset = Dataset.from_pandas(processed_df)
print("Dataset formatted.")
print("Example of processed data entry:")
print(processed_dataset[0])

# --- 4. Configure LoRA (Parameter-Efficient Fine-Tuning) ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Modules to apply LoRA to for Qwen3
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Get the PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA configured.")

# --- 5. Set up Training Arguments and SFTTrainer ---
print("Setting up TrainingArguments and SFTTrainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=LEARNING_RATE,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    logging_steps=25,
    save_steps=100,
    eval_strategy="steps", # Changed from evaluation_strategy
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
 
)
# For a small synthetic dataset, we'll use a shard for evaluation.
# In a real scenario, you'd use a dedicated `validation_split` from your dataset.
eval_dataset = processed_dataset.shard(num_shards=10, index=0) # Take first 1/10th for eval




trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    eval_dataset=eval_dataset, # Use the small shard for evaluation
    peft_config=lora_config,
    args=training_args,
    max_seq_length=1024
)
print("SFTTrainer initialized.")

# --- 6. Train the Model ---
print("Starting training...")
trainer.train()
print("Training complete.")

# --- MODIFIED: Saving the Model for HuggingFace compatibility ---
print(f"Saving fine-tuned LoRA adapters to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR) # Saves only the LoRA adapters
tokenizer.save_pretrained(OUTPUT_DIR) # Saves the tokenizer

print("\n--- Merging LoRA adapters into the base model and saving as a full HuggingFace model ---")


base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # Or torch.float16 or torch.float32
    device_map="cpu", # Load to CPU first to free up GPU memory
    trust_remote_code=True,
)


lora_model = PeftModel.from_pretrained(base_model_for_merge, OUTPUT_DIR)


print("Merging LoRA adapters...")
merged_model = lora_model.merge_and_unload() # This function returns a new, merged model
print("LoRA adapters merged.")

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
print(f"Saving merged model to {FINAL_MODEL_DIR}...")
merged_model.save_pretrained(FINAL_MODEL_DIR)

# Save the tokenizer again in the new merged model directory
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print(f"Merged model saved successfully to {FINAL_MODEL_DIR}.")
print(f"You can now load this model using: AutoModelForCausalLM.from_pretrained('{FINAL_MODEL_DIR}')")
print(f"And its tokenizer using: AutoTokenizer.from_pretrained('{FINAL_MODEL_DIR}')")
