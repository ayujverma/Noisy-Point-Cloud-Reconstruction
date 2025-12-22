#!/bin/bash
#SBATCH -J save_latents               # Job name
#SBATCH -o logs/latentes_%j.out        # Stdout output file
#SBATCH -e logs/latents_%j.err        # Stderr error file
#SBATCH -p gpu-a100                  # GPU partition (A100)
#SBATCH -N 1                         # Total number of nodes
#SBATCH -n 1                         # Total number of tasks
#SBATCH -t 04:00:00                  # Walltime (hh:mm:ss)
#SBATCH -A ASC25079                  # Project/Allocation name
#SBATCH --mail-type=ALL              # Send email at begin and end
#SBATCH --mail-user=maadhav_kothuri@utexas.edu  # Email for notifications

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2
module load gcc/9.4.0

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/09634/maadhavk631/miniconda3/etc/profile.d/conda.sh
conda activate PointFlow
export PYTHONUNBUFFERED=1

# ------------------------------
# Change to Repo Root
# ------------------------------
cd /work/09634/maadhavk631/Noisy-Point-Cloud-Reconstruction

# ------------------------------
# Configuration
# ------------------------------
START_DIR="/work/09634/maadhavk631/Noisy-Point-Cloud-Reconstruction"
CKPT_PATH=$START_DIR/models/checkpoint-latest.pt
DATA_PATH=$START_DIR/third_party/pointflow/data/ShapeNetCore.v2.PC15k/02691156/train/
SAVE_PATH=$START_DIR

# ------------------------------
# Run Sampling
# ------------------------------
echo "Encoding training examples and saving latents"
python save_latents.py \
    --chkpt $CKPT_PATH \
    --dataset $DATA_PATH \
    --savepath $SAVE_PATH

echo "Encoding finished."

