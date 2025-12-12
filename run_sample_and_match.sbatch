#!/bin/bash
#SBATCH -J poc_sample                # Job name
#SBATCH -o logs/sample_%j.out        # Stdout output file
#SBATCH -e logs/sample_%j.err        # Stderr error file
#SBATCH -p gpu-a100                  # GPU partition (A100)
#SBATCH -N 1                         # Total number of nodes
#SBATCH -n 1                         # Total number of tasks
#SBATCH -t 00:30:00                  # Walltime (hh:mm:ss)
#SBATCH -A ASC25079                  # Project/Allocation name
#SBATCH --mail-type=ALL              # Send email at begin and end
#SBATCH --mail-user=ayuj@utexas.edu  # Email for notifications

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
CKPT_PATH="third_party/pointflow/checkpoints/ae/shapenet15k-catechair-a100/checkpoint-latest.pt"
NUM_SAMPLES=1000

# ------------------------------
# Run Sampling
# ------------------------------
echo "Starting sampling..."
python sample_and_decode.py \
    --ckpt "$CKPT_PATH" \
    --num_samples $NUM_SAMPLES \
    --batch_size 50

echo "Sampling finished."

echo "Step 2: Matching and Refining..."
# Note: Using --generate_fake_real for demo. 
# If you have real data, replace with: --real_data_path "path/to/real/plys"
python match_and_refine.py \
    --ckpt "$CKPT_PATH" \
    --generate_fake_real \
    --num_real $NUM_REAL \
    --steps $REFINE_STEPS
