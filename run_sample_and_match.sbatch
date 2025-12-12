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
NUM_REAL=10
REFINE_STEPS=500

echo "Setting up local scratch data..."
TMP_DATA_DIR="/tmp/$SLURM_JOB_ID/data"
mkdir -p $TMP_DATA_DIR

# Path to the zip file in the repo
ZIP_FILE="third_party/pointflow/data/ShapeNetCore.v2.PC15k.zip"

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
    # Fallback path if zip is missing
    DATA_SRC="third_party/pointflow/data/ShapeNetCore.v2.PC15k"
    if [ -d "$DATA_SRC" ]; then
        echo "Copying data from $DATA_SRC to $TMP_DATA_DIR..."
        cp -r $DATA_SRC $TMP_DATA_DIR/
    else
        echo "Error: Data source not found at $ZIP_FILE or $DATA_SRC"
        exit 1
    fi
fi

# Chair category ID: 03001627
# Using test set for real data simulation
REAL_DATA_PATH="$TMP_DATA_DIR/ShapeNetCore.v2.PC15k/03001627/test"

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
    --real_data_path "$REAL_DATA_PATH" \
    --num_real $NUM_REAL \
    --steps $REFINE_STEPS
