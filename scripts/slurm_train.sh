#!/bin/bash
#SBATCH --job-name=deep3d_ft
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/deep3d_runs/deep3d}"
cd "$ROOT_DIR"

mkdir -p logs checkpoints

# Activate environment if available.
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate deep3d || true
fi

echo "[INFO] host=$(hostname)"
echo "[INFO] pwd=$PWD"
echo "[INFO] date=$(date)"
python -V || true
nvidia-smi || true

python -u train_finetune.py \
  --data-left Deep3D/data/left \
  --data-right Deep3D/data/right \
  --data-ground Deep3D/data/ground \
  --pretrained Deep3D/export/deep3d_v1.0_640x360_cpu.pt \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --frame-stride 2 \
  --ground-loss-weight 0.5 \
  --parallax-loss-weight 0.1 \
  --augment \
  --save-dir checkpoints \
  --resume auto
