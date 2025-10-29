#!/bin/bash
echo "▶ Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "▶ Installing dependencies..."
pip install -r ./train_mcqa/requirements.txt
echo "▶ Upgrading pip..."
pip install torch transformers datasets accelerate tensorboardX
echo "▶ Upgrading transformers..."
pip install --upgrade transformers

echo "▶ Running MCQA training script..."
python ./train_mcqa/MCQ-Reasoning-Training-Pipeline.py
echo "▶ MCQA training completed."