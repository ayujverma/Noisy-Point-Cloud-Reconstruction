#!/bin/bash
#SBATCH -J plane_ae_a100_multi       # Job name
#SBATCH -o logs/slurm_a100_multi/%j.out
#SBATCH -e logs/slurm_a100_multi/%j.err
#SBATCH -p gpu-a100                  # GPU partition
#SBATCH -N 8                         # Total number of nodes (Adjust as needed)
#SBATCH -n 8                         # Total number of tasks (1 per node)
#SBATCH -t 48:00:00                  # Walltime
#SBATCH -A ASC25079                  # Project/Allocation name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayuj@utexas.edu

# ------------------------------
# Load system modules
# ------------------------------
module purge
module load cuda/12.2
module load gcc/9.4.0

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
echo "CHAIR AE TRAINING JOB (A100 Multi-Node)"
echo "============================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

# ------------------------------
# Change to PointFlow repo
# ------------------------------
cd /work/10692/ayuj/Noisy-Point-Cloud-Reconstruction/third_party/pointflow

# ------------------------------
# Create output directories
# ------------------------------
mkdir -p logs output checkpoints

# ------------------------------
# Distributed Setup
# ------------------------------
# Get the list of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get IP address of head node
# Note: On some clusters, you might need to use a specific interface (e.g., ib0)
master_addr=$head_node
master_port=29500

echo "Master node: $head_node"
echo "Master Addr: $master_addr"
echo "Master Port: $master_port"
echo "World Size (Nodes): $SLURM_NNODES"

# ------------------------------
# Run Training Script on Each Node
# ------------------------------
# We use one srun command. It runs the script on ALL nodes at once.
# We pass 'auto' as the rank argument, so the script knows to look for SLURM_PROCID.
srun bash ./scripts/shapenet_airplane_ae_multinode.sh \
    a100-multi \
    auto \
    $SLURM_NNODES \
    $master_addr \
    $master_port

wait

echo "Job finished at: $(date)"
