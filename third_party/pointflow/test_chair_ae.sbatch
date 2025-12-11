#!/bin/bash
#SBATCH -J test_chair_ae_a100     # Job name
#SBATCH -o logs/slurm_a100_test_%j.out    # Stdout output file
#SBATCH -e logs/slurm_a100_test_%j.err    # Stderr error file
#SBATCH -p gpu-a100                  # GPU partition (A100)
#SBATCH -N 1                         # Total number of nodes
#SBATCH -n 1                         # Total number of tasks
#SBATCH -t 04:00:00                  # Walltime (hh:mm:ss) - Testing is faster than training
#SBATCH -A ASC25079                  # Project/Allocation name
#SBATCH --mail-type=ALL              # Send email at begin and end
#SBATCH --mail-user=ayuj@utexas.edu  # Email for notifications

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2                # Load CUDA 12.2
module load gcc/9.4.0                # Load GCC 9.4.0
module load python_cacher            # Cache python modules

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
echo "TESTING CHAIR AE (A100 - 3 GPU)"
echo "============================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "============================================================================"

# ------------------------------
# Change to PointFlow repo
# ------------------------------
cd /work/10692/ayuj/Noisy-Point-Cloud-Reconstruction/third_party/pointflow

# ------------------------------
# I/O Optimization: Copy Dataset to /tmp
# ------------------------------
echo "Setting up local scratch data..."
TMP_DATA_DIR="/tmp/$SLURM_JOB_ID/data"
mkdir -p $TMP_DATA_DIR

ZIP_FILE="data/ShapeNetCore.v2.PC15k.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "Found archive $ZIP_FILE. Copying and extracting to local scratch..."
    cp $ZIP_FILE $TMP_DATA_DIR/
    
    # Save current directory
    PUSHD_DIR=$PWD
    cd $TMP_DATA_DIR
    
    echo "Extracting zip file..."
    unzip -q ShapeNetCore.v2.PC15k.zip
    
    # Return to original directory
    cd $PUSHD_DIR
else
    echo "Archive $ZIP_FILE not found. Falling back to recursive copy (slower)..."
    echo "Copying data from data/ShapeNetCore.v2.PC15k to $TMP_DATA_DIR..."
    cp -r data/ShapeNetCore.v2.PC15k $TMP_DATA_DIR/
fi

echo "Data copy complete."

# ------------------------------
# Run Testing
# ------------------------------
LOG_NAME="ae/shapenet15k-catechair-a100"
CHECKPOINT_DIR="checkpoints/${LOG_NAME}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint-latest.pt"
MEAN_PATH="${CHECKPOINT_DIR}/train_set_mean.npy"
STD_PATH="${CHECKPOINT_DIR}/train_set_std.npy"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Testing checkpoint: $CHECKPOINT_PATH"

# Point to the local copy of the data
DATA_DIR="$TMP_DATA_DIR/ShapeNetCore.v2.PC15k"

python test.py \
    --cates chair \
    --resume_checkpoint ${CHECKPOINT_PATH} \
    --resume_dataset_mean ${MEAN_PATH} \
    --resume_dataset_std ${STD_PATH} \
    --dims 512-512-512 \
    --use_deterministic_encoder \
    --evaluate_recon \
    --data_dir ${DATA_DIR} \
    --batch_size 256

# Cleanup /tmp
echo "Cleaning up local data..."
rm -rf $TMP_DATA_DIR

echo "Job finished at: $(date)"
