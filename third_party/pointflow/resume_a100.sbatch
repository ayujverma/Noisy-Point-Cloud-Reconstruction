#!/bin/bash
#SBATCH -J resume_chair_ae_a100      # Job name
#SBATCH -o logs/slurm_a100_resume_%j.out    # Stdout output file
#SBATCH -e logs/slurm_a100_resume_%j.err    # Stderr error file
#SBATCH -p gpu-a100                  # GPU partition (A100)
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
module load python_cacher            # Cache python modules to local disk to reduce I/O

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
echo "RESUMING CHAIR AE TRAINING JOB (A100 - 3 GPUs)"
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
# As per TACC best practices, we copy the dataset to the local node's /tmp directory
# to reduce load on the global $WORK file system and improve I/O performance.
echo "Setting up local scratch data..."
TMP_DATA_DIR="/tmp/$SLURM_JOB_ID/data"
mkdir -p $TMP_DATA_DIR
echo "Copying data from data/ShapeNetCore.v2.PC15k to $TMP_DATA_DIR..."
cp -r data/ShapeNetCore.v2.PC15k $TMP_DATA_DIR/
echo "Data copy complete."

# ------------------------------
# Resume Training
# ------------------------------
LOG_NAME="ae/shapenet15k-catechair-a100"
CHECKPOINT_PATH="checkpoints/${LOG_NAME}/checkpoint-latest.pt"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found checkpoint: $CHECKPOINT_PATH"
    echo "Resuming training..."
    
    cate="chair"
    dims="512-512-512"
    latent_dims="256-256"
    num_blocks=1
    latent_num_blocks=1
    zdim=128
    batch_size=256
    lr=2e-3
    epochs=4000
    ds=shapenet15k
    # Point to the local copy of the data
    data_dir="$TMP_DATA_DIR/ShapeNetCore.v2.PC15k"

    python -u train.py \
        --log_name ${LOG_NAME} \
        --lr ${lr} \
        --dataset_type ${ds} \
        --data_dir ${data_dir} \
        --cates ${cate} \
        --dims ${dims} \
        --latent_dims ${latent_dims} \
        --num_blocks ${num_blocks} \
        --latent_num_blocks ${latent_num_blocks} \
        --batch_size ${batch_size} \
        --zdim ${zdim} \
        --epochs ${epochs} \
        --save_freq 20 \
        --viz_freq 100 \
        --log_freq 10 \
        --val_freq 100 \
        --max_validate_shapes 100 \
        --distributed \
        --use_deterministic_encoder \
        --prior_weight 0 \
        --entropy_weight 0 \
        --resume_checkpoint ${CHECKPOINT_PATH} \
        --resume_optimizer

    # Cleanup /tmp is usually handled by the system, but we can be polite
    echo "Cleaning up local data..."
    rm -rf $TMP_DATA_DIR

else
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Job finished at: $(date)"
