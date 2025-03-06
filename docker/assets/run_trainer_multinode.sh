#!/usr/bin/env bash

TEST_ARTIFACTS_PATH="/shared/czhenguo/Projects/fruitstand/run_artifacts/$POD_UID/"
mkdir -p "$TEST_ARTIFACTS_PATH"
NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump/$OMPI_COMM_WORLD_RANK
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump

source ./flag_list.sh $NEURON_DUMP_PATH

# Manually specify CCOM interface, often required on EKS
export CCOM_SOCKET_IFNAME=eth0
 
# Neuron env vars for distributed training
nodes=`/neuron/scripts/nodelist_helper.py`
devices_per_node=$((128/$NEURON_RT_VIRTUAL_CORE_SIZE))
export COORDINATOR_ADDRESS=$(echo "$nodes" | head -n 1):64272
export NEURON_RT_ROOT_COMM_ID=$(echo "$nodes" | head -n 1):5552
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $OMPI_COMM_WORLD_SIZE | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$OMPI_COMM_WORLD_RANK
unset OMPI_MCA_orte_hnp_uri

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

# Use tcmalloc - this is required
LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
if [ -n "$LIBTCMALLOC" ]; then
    # Create a symbolic link to the found libtcmalloc version
    ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
    echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
    # Export LD_PRELOAD
    export LD_PRELOAD=/usr/lib/libtcmalloc.so
    echo "LD_PRELOAD set to: $LD_PRELOAD"
else
    echo "Error: libtcmalloc.so not found"
    exit 1
fi

PYINSTRUMENT_OUTPUT_PATH="${TEST_ARTIFACT_PATH}/recovery.pyisession"

# Sync changes
git config --global --add safe.directory /shared/czhenguo/Projects/fruitstand/axlearn
cd /shared/czhenguo/Projects/fruitstand/axlearn
git diff czhenguo/scale-out --no-color > changes.patch
echo "patch generated"
cd /neuron/axlearn
echo "applying patch"
git apply /shared/czhenguo/Projects/fruitstand/axlearn/changes.patch
cd /neuron

# show env vars in logs
set

# Run the training script
if [ $NEURON_PJRT_PROCESS_INDEX == 0 ]; then
    pyinstrument -o $PYINSTRUMENT_OUTPUT_PATH --hide-regex ".*traceback_util\.py" -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash \
        --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
        --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$COORDINATOR_ADDRESS \
        --num_processes=$OMPI_COMM_WORLD_SIZE \
        --process_id=$OMPI_COMM_WORLD_RANK 2>&1 | tee ${OUTPUT_DIR}/${PMIX_HOSTNAME}.log
else
    pyinstrument -o $PYINSTRUMENT_OUTPUT_PATH --hide-regex ".*traceback_util\.py" -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash \
        --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
        --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$COORDINATOR_ADDRESS \
        --num_processes=$OMPI_COMM_WORLD_SIZE \
        --process_id=$OMPI_COMM_WORLD_RANK 2>&1 | tee ${OUTPUT_DIR}/${PMIX_HOSTNAME}.log
fi
