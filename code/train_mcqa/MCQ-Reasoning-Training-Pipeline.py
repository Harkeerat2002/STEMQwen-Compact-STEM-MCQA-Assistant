import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback,
)
import os
import tqdm as tqdm
import re

# --- Configuration ---
model_name = "Qwen/Qwen3-0.6B-Base"
output_dir = "./train_mcqa/model_output"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token as it is None.")
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Training Arguments (Now fixed, no longer overridden by Optuna) ---
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "adamw_torch"
learning_rate = 1e-5
fp16 = False
bf16 = True
weight_decay = 0.1
max_grad_norm = 0.3
num_train_epochs = 1
warmup_ratio = 0.03
lr_scheduler_type = "cosine"
logging_steps = 10
save_steps = 100
eval_steps = 100
max_seq_length = 1024
gradient_checkpointing = True
gradient_checkpointing_kwargs = {"use_reentrant": False}
report_to = "tensorboard"
disable_tqdm = False

# Early Stopping parameters
load_best_model_at_end = True
metric_for_best_model = "eval_mcqa_accuracy"
greater_is_better = True
early_stopping_patience = 3
early_stopping_threshold = 0.001

LETTER_INDICES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def format_example(example):
    original_question_text = example["question"].strip()
    if example["reasoning"] is None:
        support = ""
    else:
        support = example["reasoning"].strip()

    choice_pattern = r"([A-E]):\s*(.*?)(?=\n[A-E]:|\n[a-z]|\n\d|\s*$|\Z)"
    matches = list(re.finditer(choice_pattern, original_question_text, re.DOTALL))

    extracted_choices = []
    question_body = original_question_text

    if matches:
        first_choice_start = matches[0].start()
        question_body = original_question_text[:first_choice_start].strip()

        for match in matches:
            letter = match.group(1)
            choice_text = match.group(2).strip()
            extracted_choices.append(f"{letter}. {choice_text}")
    else:
        question_body = original_question_text.strip()

    prompt = question_body + "\n"
    if extracted_choices:
        prompt += "\n".join(extracted_choices)
    prompt += "\nAnswer:"

    match = re.match(r"([A-Z])\.?", example["answer"].strip())
    if match:
        gold_answer_letter = match.group(1)
    else:
        gold_answer_letter = example["answer"].strip().upper()
        if len(gold_answer_letter) > 1 and gold_answer_letter.endswith("."):
            gold_answer_letter = gold_answer_letter[0]

    if support == "":
        response = f" {gold_answer_letter}{tokenizer.eos_token}"
    else:
        response = f" {gold_answer_letter}\nReasoning: {support}{tokenizer.eos_token}"

    return {
        "text": prompt + response,
        "prompt_length_chars": len(prompt),
    }


def find_median_length(dataset):
    lengths = [len(example["text"]) for example in dataset]
    lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
    median_len = int(lengths_tensor.median().item())
    percentile_90_len = int(lengths_tensor.quantile(0.9).item())
    max_len = int(lengths_tensor.max().item())
    return median_len, percentile_90_len, max_len


print("Loading and formatting dataset...")
dataset = load_dataset("hssawhney/MNLP_M3_mcqa_dataset")
ds_train = dataset["train"]
ds_eval = dataset["test"]

ds_train_formatted = []
for example in tqdm.tqdm(ds_train, desc="Formatting Training Data"):
    formatted_example = format_example(example)
    if formatted_example:
        ds_train_formatted.append(formatted_example)

ds_eval_formatted = []
for example in tqdm.tqdm(ds_eval, desc="Formatting Evaluation Data"):
    formatted_example = format_example(example)
    if formatted_example:
        ds_eval_formatted.append(formatted_example)

ds_train = Dataset.from_list(ds_train_formatted)
ds_eval = Dataset.from_list(ds_eval_formatted)

median_length, percentile_length, max_len = find_median_length(ds_train)
print(f"Median character length of combined data: {median_length}")
print(f"90th percentile character length of combined data: {percentile_length}")
print(f"Maximum character length of combined data: {max_len}")

print(f"Training examples (pre-filtering): {len(ds_train)}")
print(f"Evaluation examples (pre-filtering): {len(ds_eval)}")

print("Loading tokenizer and model...")

model_for_loading = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model_for_loading.config.use_cache = False


def tokenize_and_add_labels(example):
    """
    Tokenizes the full text and creates labels for loss masking.
    Labels corresponding to the prompt tokens and padding tokens are set to -100.
    """
    tokenized_full = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_attention_mask=True,
    )

    labels = list(tokenized_full["input_ids"])

    prompt_char_length = example["prompt_length_chars"]

    response_start_token_idx = 0
    for i, (start_char, end_char) in enumerate(tokenized_full["offset_mapping"]):
        if start_char >= prompt_char_length:
            response_start_token_idx = i
            break

    for i in range(len(labels)):
        # Mask prompt tokens
        if i < response_start_token_idx:
            labels[i] = -100
        # Mask padding tokens
        if tokenized_full["input_ids"][i] == tokenizer.pad_token_id:
            labels[i] = -100

    return {
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }


print("Tokenizing datasets and applying loss masking...")

tokenized_train_path = os.path.join(
    output_dir, "train_dataset_tokenized"
)  # Changed name to avoid conflict
tokenized_eval_path = os.path.join(
    output_dir, "eval_dataset_tokenized"
)  # Changed name to avoid conflict

if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_eval_path):
    print("Loading pre-tokenized training and evaluation datasets from disk...")
    ds_train = Dataset.load_from_disk(tokenized_train_path)
    ds_eval = Dataset.load_from_disk(tokenized_eval_path)
else:
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

    # --- Data Filtering for max_seq_length (by token length) ---
    print("Filtering tokenized datasets for valid labels (response present)...")
    initial_train_count = len(ds_train)
    initial_eval_count = len(ds_eval)

    # Ensure there's at least one non-masked label
    ds_train = ds_train.filter(lambda x: any(label != -100 for label in x["labels"]))
    ds_eval = ds_eval.filter(lambda x: any(label != -100 for label in x["labels"]))

    print(
        f"Filtered out {initial_train_count - len(ds_train)} training examples where no response tokens were present."
    )
    print(
        f"Filtered out {initial_eval_count - len(ds_eval)} evaluation examples where no response tokens were present."
    )

    # Remove offset_mapping before saving as it's not needed for training
    if "offset_mapping" in ds_train.column_names:
        ds_train = ds_train.remove_columns(["offset_mapping"])
    if "offset_mapping" in ds_eval.column_names:
        ds_eval = ds_eval.remove_columns(["offset_mapping"])

    ds_train.save_to_disk(tokenized_train_path)
    ds_eval.save_to_disk(tokenized_eval_path)


print(f"Filtered training examples (by token length/labels): {len(ds_train)}")
print(f"Filtered evaluation examples (by token length/labels): {len(ds_eval)}")
print(f"Tokenized training example (first entry):\n{ds_train[0]}")


early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience,
    early_stopping_threshold=early_stopping_threshold,
)


class MCQALoglikelihoodEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        dataset_name="hssawhney/Evaluation-Dataset",
        split="test",
        eval_topic="knowledge and skills in advanced master-level STEM courses",
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.LETTER_INDICES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        self.eval_ds = load_dataset(self.dataset_name, split=self.split)
        self.eval_topic = eval_topic

    def compute_loglikelihood_for_choices(self, model, prompt, choices):
        # Ensure F is imported here if it's not global
        import torch.nn.functional as F

        model.eval()
        device = next(model.parameters()).device
        prompt_inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_length
        ).to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        logprobs = []
        for choice in tqdm.tqdm(
            choices, desc="Computing log-likelihoods for choices", leave=False
        ):
            full_input = prompt + choice
            inputs = self.tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(device)
            choice_token_ids = inputs["input_ids"][0][prompt_len:]

            if choice_token_ids.numel() == 0:
                logprobs.append(float("-inf"))
                continue

            with torch.no_grad():
                outputs = model(
                    **{k: v for k, v in inputs.items() if k != "token_type_ids"}
                )
                logits = outputs.logits[
                    0, prompt_len - 1 : inputs["input_ids"].shape[1] - 1
                ]

                if logits.shape[0] != choice_token_ids.shape[0]:
                    print(
                        f"Warning: Logits length ({logits.shape[0]}) does not match choice_token_ids length ({choice_token_ids.shape[0]}) for choice '{choice}'. Assigning -inf logprob."
                    )
                    logprobs.append(float("-inf"))
                    continue

                log_probs = F.log_softmax(logits, dim=-1)
                choice_logprobs = log_probs.gather(
                    1, choice_token_ids.unsqueeze(1)
                ).squeeze(1)
                total_logprob = choice_logprobs.sum().item()
                logprobs.append(total_logprob)
        return logprobs

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics,
        **kwargs,
    ):

        model = kwargs["model"]
        total = 0
        correct = 0
        correct_norm = 0

        print(f"\n--- Entering on_evaluate at step {state.global_step} ---")  # DEBUG

        for example in tqdm.tqdm(self.eval_ds, desc="Evaluating MCQA Log-Likelihood"):
            base_prompt_prefix = f"The following are multiple choice questions (with answers) about {self.eval_topic}.\n\n"
            prompt = base_prompt_prefix
            prompt += example["question"].strip() + "\n"

            evaluation_choices_text = []
            for idx, choice_text in enumerate(example["choices"]):
                if idx < len(self.LETTER_INDICES):
                    prompt += f"{self.LETTER_INDICES[idx]}. {choice_text.strip()}\n"
                    evaluation_choices_text.append(f" {self.LETTER_INDICES[idx]}")
                else:
                    print(
                        f"Warning: Example has more choices ({len(example['choices'])}) than available LETTER_INDICES ({len(self.LETTER_INDICES)}). Skipping extra choices."
                    )
                    break
            prompt += "Answer:"

            gold = example["answer"].strip().upper()
            if gold.endswith("."):
                gold = gold[:-1]

            logprobs = self.compute_loglikelihood_for_choices(
                model, prompt, evaluation_choices_text
            )

            if not logprobs:
                pred_letter = "NONE"
            else:
                pred_idx = int(torch.tensor(logprobs).argmax())
                pred_letter = self.LETTER_INDICES[pred_idx]

            total += 1
            if pred_letter == gold:
                correct += 1
            if pred_letter.strip() == gold.strip():
                correct_norm += 1

        accuracy = correct / total if total > 0 else 0.0
        accuracy_norm = correct_norm / total if total > 0 else 0.0
        print(f"\n[MCQA Log-Likelihood Eval on {self.dataset_name}]")
        print(f"Total: {total} | Correct: {correct} | Accuracy: {accuracy:.4f}")
        print(
            f"Correct (norm): {correct_norm} | Accuracy (norm): {accuracy_norm:.4f}\n"
        )

        metrics["eval_mcqa_accuracy"] = accuracy
        print(
            f"--- Added 'eval_mcqa_accuracy' with value: {accuracy:.4f} to metrics. ---"
        )  #

        print(f"--- Exiting on_evaluate ---")  # DEBUG

        return control


# --- Training Arguments ---
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
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    gradient_checkpointing=gradient_checkpointing,
    load_best_model_at_end=load_best_model_at_end,
    # This must match the key logged in the callback
    metric_for_best_model="eval_mcqa_accuracy",
    greater_is_better=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    disable_tqdm=False,
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

mcqa_eval_callback = MCQALoglikelihoodEvalCallback(tokenizer=tokenizer)

# Trainer instance is created *after* model loading
trainer = Trainer(
    model=model_for_loading,
    args=training_arguments,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback, mcqa_eval_callback],
)

# --- Train and Save ---
print("Starting training with fixed hyperparameters...")

last_checkpoint = None
if os.path.isdir(training_arguments.output_dir):
    checkpoints = [
        d
        for d in os.listdir(training_arguments.output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(training_arguments.output_dir, d))
    ]
    if checkpoints:
        last_checkpoint = os.path.join(
            training_arguments.output_dir,
            max(checkpoints, key=lambda x: int(x.split("-")[1])),
        )
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoints found. Starting training from scratch.")

trainer.train(resume_from_checkpoint=last_checkpoint)

print("Training complete. Saving model and tokenizer...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# --- Use the model to generate a response (for quick testing) ---
print("\n--- Testing the trained model ---")

# Reload the model and tokenizer from the saved output_dir
model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    output_dir,
    trust_remote_code=True,
)

test_example_prompt = """The line $y=k x+4$, where $k$ is a constant, is graphed in the $x y$-plane. If the line contains the point $(c, d)$, where $c \\neq 0$ and $d \\neq 0$, what is the slope of the line in terms of $c$ and $d$ ?
A: $\\frac{d-4}{c}$
B: $\\frac{c-4}{d}$
C: $\\frac{4-d}{c}$
D: $\\frac{4-c}{d}$
Answer:"""

inputs = tokenizer(test_example_prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=20,  # Increased for potential reasoning output in test
    do_sample=True,
    temperature=0.01,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Prompt:\n{test_example_prompt}")
print(
    f"Generated response (should be the answer letter and reasoning):\n'{response.strip()}'"
)
