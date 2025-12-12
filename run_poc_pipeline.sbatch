#!/bin/bash
#SBATCH -J poc_pipeline              # Job name
#SBATCH -o logs/poc_%j.out           # Stdout output file
#SBATCH -e logs/poc_%j.err           # Stderr error file
#SBATCH -p gpu-a100                  # GPU partition (A100)
#SBATCH -N 1                         # Total number of nodes
#SBATCH -n 1                         # Total number of tasks
#SBATCH -t 04:00:00                  # Walltime (hh:mm:ss)
#SBATCH -A ASC25079                  # Project/Allocation name
#SBATCH --mail-type=ALL              # Send email at begin and end
#SBATCH --mail-user=ayuj@utexas.edu  # Email for notifications

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2                # Load CUDA 12.2
module load gcc/9.4.0                # Load GCC 9.4.0

# ------------------------------
# Activate conda environment
# ------------------------------
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate PointFlow
export PYTHONUNBUFFERED=1

# ------------------------------
# Print job info
# ------------------------------
echo "============================================================================"
echo "PoC PIPELINE JOB"
echo "============================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
python - <<'EOF'
import torch
print("PyTorch sees GPU:", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
echo "============================================================================"

# ------------------------------
# Change to Repo Root
# ------------------------------
# Adjust this path if your repo root is different on TACC
cd /work/10692/ayuj/Noisy-Point-Cloud-Reconstruction

# ------------------------------
# Create output directories
# ------------------------------
mkdir -p logs results

# ------------------------------
# Configuration
# ------------------------------
CKPT_PATH="third_party/pointflow/checkpoints/ae/shapenet15k-catechair-a100/checkpoint-latest.pt"
NUM_SAMPLES=1000
NUM_REAL=100
REFINE_STEPS=300

# ------------------------------
# Run Pipeline
# ------------------------------

echo "Step 1: Sampling and Decoding..."
python sample_and_decode.py \
    --ckpt "$CKPT_PATH" \
    --num_samples $NUM_SAMPLES \
    --batch_size 50

echo "Step 2: Matching and Refining..."
# Note: Using --generate_fake_real for demo. 
# If you have real data, replace with: --real_data_path "path/to/real/plys"
python match_and_refine.py \
    --ckpt "$CKPT_PATH" \
    --generate_fake_real \
    --num_real $NUM_REAL \
    --steps $REFINE_STEPS

echo "Step 3: Tracking Correspondences..."
python track_correspondences.py \
    --ckpt "$CKPT_PATH" \
    --interp_steps 25

echo "Step 4: Generating Visualizations and Metrics..."
python visualize_and_metrics.py

# ------------------------------
# Job finished
# ------------------------------
echo "Job finished at: $(date)"
