#!/bin/bash
#SBATCH -J hw5_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# HW5: Self-Supervised Learning with Transformers
# Run all three tasks sequentially.

echo "Starting HW5 training on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd "$(dirname "$0")"

echo "=== Task 0: End-to-End ViT ==="
uv run python main.py --task t0_endtoend

echo "=== Task 1a: Mini-DINO Pretraining (full dataset) ==="
uv run python main.py --task t1_dino

echo "=== Task 1b: Mini-DINO Pretraining (single image) ==="
uv run python main.py --task t1_dino_single

echo "=== Task 2: Transfer Evaluation ==="
uv run python main.py --task t2_transfer

echo "All tasks complete at $(date)"
