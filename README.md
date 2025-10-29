# MNLP-Model — STEMQwen (MCQ-focused pipeline)

Short summary

- STEMQwen is a reproducible research pipeline that adapts a compact pretrained LLM (Qwen3-0.6B-Base) into a resource-efficient, STEM-focused assistant. It integrates four interoperable components: MCQA supervised fine-tuning, Direct Preference Optimization (DPO) alignment, Retrieval-Augmented Generation (RAG), and multiple quantization/compression strategies to enable high-quality answers and efficient inference on consumer hardware.

This repository contains the code, configs and datasets used to develop the MCQ (multiple-choice question answering) branch of the STEMQwen project. The full project report with experiments, ablations and ethical considerations is included alongside the code.

Why this project

- Large LLMs often fail to provide fast, reliable, structured STEM answers on limited hardware. STEMQwen demonstrates how careful, modular adaptations of a small base model (0.6B parameters) — including alignment, MCQ-focused SFT, retrieval, and quantization — can produce compact, accurate, and interpretable assistants suitable for educational settings.

What I implemented (MCQ branch)

- Supervised fine-tuning pipeline for MCQA: formatting, tokenization, training loop and evaluation tailored to multiple-choice STEM questions.
- Rationale-augmented training examples: prompt + options + correct answer + step-by-step reasoning (when available) are concatenated into a single sequence to stabilize the model.
- Custom discriminative evaluation (log-likelihood per choice): instead of free-form generation, the MCQA evaluation computes the log-probability of each option conditioned on the prompt and selects the highest-scoring choice — this gives a stable, discriminative metric.
- Data curation scripts and a notebook to assemble MCQ datasets from diverse STEM sources.
- A simple LoRA alternative and configuration for lighter-weight adaptation.
- Hooks for quantization and RAG experiments (config YAMLs, training wrappers) so the MCQA branch fits into the larger STEMQwen pipeline.

Key findings (from the experiments)

- DPO alignment yields strong gains in aligning model outputs to human preferences (reward accuracy often > 80% in the best runs).
- MCQA training that includes rationale or CoT-style supervision substantially improves accuracy compared to answer-only objectives. Careful loss masking (to avoid penalizing prompt reproduction) stabilizes training.
- Dataset curation matters more than raw scale: smaller, cleaner, and more diverse datasets outperform noisy huge mixtures for MCQA tasks.
- Quantization trade-offs: AWQ and SmoothQuant+GPTQ provide a good accuracy–efficiency balance (AWQ is especially stable), while BitsAndBytes can be smallest but may lose pedagogical clarity.
- RAG (LoRA + retrieval) further boosts accuracy on several benchmarks, but is sensitive to retrieval quality and domain mismatch.

Repository layout (relevant files)

- `code/train_mcqa/` — MCQ training scripts and notebook for dataset creation and experiments.
  - `MCQ-Training-Pipeline.py` — main MCQA training pipeline and evaluation utilities.
  - `Lora-Fine-Tuning.py` — LoRA variant for lightweight adaptation.
  - `MCQ-Dataset-Creation.ipynb` — notebook used to curate and preview the MCQ datasets.
- `code/train_mcqa.sh` — convenience wrapper that runs the MCQ training pipeline with the default configs.
- `model_configs/mcqa_model.yaml` — MCQA training hyperparameters and model selection.
- `model_configs/dpo_model.yaml`, `model_configs/rag_model.yaml`, `model_configs/quantized_model.yaml` — configurations for other branches in the pipeline (alignment, RAG, quantization).
- `data/data_repo.json` — data manifest used by the scripts to locate and load training data.
- `report/` — full experimental write-up, tables and appendix used to produce the results summarized above.

Quick start — run the MCQA training (local / research setup)

1. Create and activate a Python environment, then install requirements. These commands use zsh (default macOS shell):

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r code/train_mcqa/requirements.txt
```

2. Prepare or point to your data

- Edit `data/data_repo.json` (or the dataset paths referenced in `code/train_mcqa/MCQ-Training-Pipeline.py`) so your input files are available. The dataset notebook `code/train_mcqa/MCQ-Dataset-Creation.ipynb` shows expected formats.

3. Run the MCQA training pipeline

- Option A — run the pipeline script directly (recommended for debugging / customization):

```zsh
python code/train_mcqa/MCQ-Training-Pipeline.py \
	--config model_configs/mcqa_model.yaml \
	--output_dir ./outputs/mcqa
```

- Option B — use the wrapper script (runs with default args):

```zsh
bash code/train_mcqa.sh
```

Notes on common flags

- Use `--bf16` or `--fp16` depending on your hardware and the training script's support. The YAML configs include default batch sizes and gradient accumulation steps — lower them if you have less memory.
- Training scripts save tokenized/cached datasets; re-running will often reuse caches unless you change preprocessing options.

Evaluating MCQA (discriminative metric)

- The training pipeline includes an evaluation mode that computes the log-likelihood for each choice and reports accuracy for the selected highest-scoring option. This avoids brittle text-generation evaluation and produces a stable discriminative score.

Tips and troubleshooting

- If you run out of GPU memory: lower per-device batch size, increase gradient-accumulation steps, or use LoRA instead of full SFT.
- For quantized experiments, follow the configs under `model_configs/quantized_model.yaml`. These experiments require additional toolchains (bitsandbytes, AWQ/LLMCompressor) depending on the method.
- If using RAG, ensure the retriever embeddings and retrieval corpus are prepared as described in the `report/` and `model_configs/rag_model.yaml`.

Ethics and disclaimers

- The system was designed for research and educational assistance. Quantized or compressed models can degrade on out-of-distribution inputs. Training and alignment datasets can contain biases; evaluate thoroughly on your target user population before any deployment. See `report/` for a detailed discussion of ethical considerations and suggested deployment controls.

Contact and next steps

- The work and experimental results are documented in `report/`. For further development, recommended next steps include: multilingual extensions, joint retrieval–generation training, and classroom pilots with human-in-the-loop evaluation.

License

- Check repository root or `LICENSE` (if present) for licensing. If none exists, add the appropriate license before public release.

