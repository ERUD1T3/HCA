#!/bin/bash

#SBATCH --job-name=sep_hca_aip         # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --mem=64GB                     # Memory per node
#SBATCH --time=infinite               # Time limit
#SBATCH --partition=gpu1              # Partition
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --output=./logs/%x.%J.out      # Output file
#SBATCH --error=./logs/%x.%J.err       # Error file

echo "Starting SEP HCA training at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Configuration for HCA (Hierarchical Classification) ---
SEEDS="456789 42 123 0 9999"
DATASET="sep"
BATCH_SIZE=200
EPOCHS=5602
MLP_HIDDENS="512 32 256 32 128 32 64 32"
MLP_EMBED_DIM=32
MLP_DROPOUT=0.5

LR=5e-4
WEIGHT_DECAY=1

# HCA specific settings - replace ConR parameters
HEAD_NUM=6               # Number of hierarchical classification heads
BUCKET_NUM=100          # Number of buckets for hierarchical classification  
FC_LNUM=1               # Number of FC layers in hierarchical heads (1 or 2)
S2FC_LNUM=2             # Number of FC layers in adjustment head (1, 2, or 3)
HEAD_DETACH=true        # Whether to detach gradients for hierarchical heads


DATA_DIR="/home1/jmoukpe2016/BalancedMSE/neurips2025/data"
UPPER_THRESHOLD=2.30258509299

# --- Run Training ---
echo "Starting HCA training for dataset: ${DATASET}, seeds: ${SEEDS}"
echo "Using Hierarchical Classification with ${HEAD_NUM} heads"
echo "FC layers in heads: ${FC_LNUM}, Adjustment head layers: ${S2FC_LNUM}"
echo "=================================="

srun python main.py \
    --seeds ${SEEDS} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCHS} \
    --mlp_hiddens ${MLP_HIDDENS} \
    --mlp_embed_dim ${MLP_EMBED_DIM} \
    --mlp_dropout ${MLP_DROPOUT} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --upper_threshold ${UPPER_THRESHOLD} \
    --head_num ${HEAD_NUM} \
    --bucket_num ${BUCKET_NUM} \
    --fc_lnum ${FC_LNUM} \
    --s2fc_lnum ${S2FC_LNUM} \
    --head_detach

if [ $? -eq 0 ]; then
    echo "Successfully completed SEP HCA training"
else
    echo "Error in SEP HCA training"
    exit 1
fi

echo "Training finished for seeds: ${SEEDS} at date $(date)"

# Usage:
# 1. Run: sbatch aip_scripts/run_sep_hca_aip.sh 