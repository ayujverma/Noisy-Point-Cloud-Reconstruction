#!/bin/bash
#SBATCH -J airplane_ae_h100             # Job name
#SBATCH -o logs/slurm_h100_%j.out    # Stdout output file
#SBATCH -e logs/slurm_h100_%j.err    # Stderr error file
#SBATCH -p gpu-h100                  # GPU partition (H100) - Adjust if needed (e.g. 'gh' on Vista)
#SBATCH -N 1                         # Total number of nodes
#SBATCH -n 1                         # Total number of tasks
#SBATCH -t 48:00:00                  # Walltime (hh:mm:ss)
#SBATCH -A ASC25079                  # Project/Allocation name
#SBATCH --mail-type=ALL              # Send email at begin and end
#SBATCH --mail-user=ayuj@utexas.edu  # Email for notifications

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2                # Load CUDA 12.2 (Compatible with PyTorch CUDA 12.1)
module load gcc/9.4.0                # Load GCC 9.4.0 (Required for C++17 runtime compatibility)

# ------------------------------
# Activate conda environment
# ------------------------------
# Adjust path to your miniconda if different
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate PointFlow
export PYTHONUNBUFFERED=1

# ------------------------------
# Print job info
# ------------------------------
echo "============================================================================"
echo "CHAIR AE TRAINING JOB (H100 - 1 GPU)"
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
# Change to PointFlow repo
# ------------------------------
# Ensure this path is correct for your setup
cd /work/10692/ayuj/Noisy-Point-Cloud-Reconstruction/third_party/pointflow

# ------------------------------
# Create output directories
# ------------------------------
mkdir -p logs output checkpoints

# ------------------------------
# Run Autoencoder training script
# ------------------------------
bash ./scripts/shapenet_airplane_ae_dist.sh h100

# ------------------------------
# Job finished
# ------------------------------
echo "Job finished at: $(date)"
