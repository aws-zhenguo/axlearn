#!/usr/bin/env bash

MASTER_ADDR="ip-10-0-2-209"
MASTER_PORT=41000
# Added
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
sudo dpkg -i /fsx/apoorvgu/aws-neuronx-runtime-lib-2.x.19993.0-1bf746e12.deb
sudo dpkg -i /fsx/apoorvgu/aws-neuronx-collectives-2.x.21370.0-8cbb4877b.deb
sudo dpkg -i /fsx/apoorvgu/axlearn/aws-neuronx-dkms_2.x.3951.0_amd64.deb

JOB_ID=2025010901
ARTIFACTS_PATH="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/"
TEST_ARTIFACTS_PATH="/fsx/czhenguo/Projects/fruitstand/axlearn/runs/artifacts/${JOB_ID}"
NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
NEURON_RT_CORE_DUMP_DIRECTORY=${TEST_ARTIFACTS_PATH}/neuron_core_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

# Added
export NEURON_GRAD_ACC_COUNT=1

# export TF_CPP_VMODULE='neuron_token_threading=5,neuron_fsdp_all_gather_split=5,neuron_hierarchical_collectives=5,neuron_all_gather_combiner=5,neuron_reduce_scatter_combiner=5'
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
# Added
export NEURON_WHILE_LOOP_UNROLL=1
# Added
export TRN2=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1
# Added
export NEURON_ALL_REDUCE_UPCASTER=1
# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1
# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2" # --distribution-strategy=llm-training"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
export NEURON_FSDP=1
# export NEURON_HIERARCHICAL_INNER_SIZE=512
export LNC=2
# Added
export ENABLE_NEW_UNSHARDED_ATTN_KERNEL=1
# export TF_CPP_MIN_LOG_LEVEL=0 # Enable verbose logging - 0 means most verbose
# export TF_CPP_MAX_VLOG_LEVEL=3 # Required with the above, but in reverse


# LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
#  
# if [ -n "$LIBTCMALLOC" ]; then
#     # Create a symbolic link to the found libtcmalloc version
#     sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
#     echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
#  
#     # Export LD_PRELOAD
#     export LD_PRELOAD=/usr/lib/libtcmalloc.so
#     echo "LD_PRELOAD set to: $LD_PRELOAD"
# else
#     echo "Error: libtcmalloc.so not found"
#     exit 1
# fi
OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# export NEURON_RT_LOG_LEVEL=DEBUG
# export NEURON_RT_LOG_LOCATION=CONSOLE
# export NCCL_DEBUG=INFO

export DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# python axlearn_inference.py --jax_backend=neuron --module="" --config="" --trainer_dir=""
python axlearn_inference.py --jax_backend=neuron --module=text.gpt.c4_trainer --config=fuji-7Bfsdp16tp4-v2 --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --mesh_selector=neuron-trn2.48xlarge-64
# python axlearn_inference.py
# python axlearn_train.py --jax_backend=neuron --module=text.gpt.c4_trainer --config=fuji-7Bfsdp16tp4-v2 --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --mesh_selector=neuron-trn2.48xlarge-64
# python -m axlearn.common.launch_trainer_main \
#     --module=text.gpt.c4_trainer --config=fuji-7Bfsdp16tp4-v2 \
#     --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
#     --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64
