#! /bin/bash

# Arguments:
# $1: suffix
# $2: node_rank
# $3: num_nodes (world_size)
# $4: master_addr
# $5: master_port

SUFFIX=$1
NODE_RANK_ARG=$2 # Renamed variable
NUM_NODES=$3
MASTER_ADDR=$4
MASTER_PORT=$5

# --- AUTO DETECT RANK ---
if [ "$NODE_RANK_ARG" == "auto" ]; then
    NODE_RANK=$SLURM_PROCID
else
    NODE_RANK=$NODE_RANK_ARG
fi
# ------------------------

if [ -z "$SUFFIX" ]; then
  SUFFIX=""
else
  SUFFIX="-$SUFFIX"
fi

cate="airplane"
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=8
lr=1e-3
epochs=4000
ds=shapenet15k
log_name="ae/${ds}-cate${cate}${SUFFIX}"

# ------------------------------
# I/O Optimization: Copy Dataset to /tmp
# ------------------------------
echo "Setting up local scratch data on node $(hostname)..."
# Use SLURM_JOB_ID if available, otherwise use a default or random
JOB_ID=${SLURM_JOB_ID:-"local_run"}
TMP_DATA_DIR="/tmp/pointflow_data_${JOB_ID}"
mkdir -p $TMP_DATA_DIR

ZIP_FILE="data/ShapeNetCore.v2.PC15k.zip"
DATA_SRC_DIR="data/ShapeNetCore.v2.PC15k"

if [ -f "$ZIP_FILE" ]; then
    echo "Found archive $ZIP_FILE. Copying and extracting to local scratch..."
    cp $ZIP_FILE $TMP_DATA_DIR/
    
    PUSHD_DIR=$PWD
    cd $TMP_DATA_DIR
    unzip -q ShapeNetCore.v2.PC15k.zip
    cd $PUSHD_DIR
    
    data_dir="$TMP_DATA_DIR/ShapeNetCore.v2.PC15k"
elif [ -d "$DATA_SRC_DIR" ]; then
    echo "Archive $ZIP_FILE not found. Falling back to recursive copy..."
    cp -r $DATA_SRC_DIR $TMP_DATA_DIR/
    data_dir="$TMP_DATA_DIR/ShapeNetCore.v2.PC15k"
else
    echo "Warning: Could not find data source. Using default shared path."
    data_dir="data/ShapeNetCore.v2.PC15k"
fi

# Construct dist_url
DIST_URL="tcp://${MASTER_ADDR}:${MASTER_PORT}"

echo "Running on node rank ${NODE_RANK} of ${NUM_NODES} nodes. Master: ${DIST_URL}"

python -u train.py \
    --log_name ${log_name} \
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
    --save_freq 100 \
    --viz_freq 100 \
    --log_freq 10 \
    --val_freq 100 \
    --max_validate_shapes 100 \
    --distributed \
    --world_size ${NUM_NODES} \
    --rank ${NODE_RANK} \
    --dist_url ${DIST_URL} \
    --use_deterministic_encoder \
    --prior_weight 0 \
    --entropy_weight 0 \

echo "Done node ${NODE_RANK}"
exit 0
