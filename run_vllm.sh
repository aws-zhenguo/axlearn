#!/bin/bash
# start vllm server with bash ./start_server.sh /opt/dlami/nvme/czhenguo/llm_eval/Meta-Llama-3.1-70B-Instruct/ 22794 4 2048 32 8 8
model_id=${1}
tokenizer=${2}
port=${3:-8000}
cores=${4:-8}
max_seq_len=${5:-2048}
cont_batch_size=${6:-32}
tp_size=${7:-8}
n_threads=${8:-8}

# Shift positional arguments out of the way before parsing named arguments
shift 8

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quantization) quantization="$2"; shift ;;
        --chat-template) chat_template="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;  # Handle unknown parameters
    esac
    shift  # Move to the next argument
done

echo "Starting VLLM Server for model: ${model_id}"
echo "Using tokenizer: ${tokenizer}"

export vLLM_MAX_LEN=${max_seq_len} #sequence length
export vLLM_MODEL_ID=${model_id}
export vLLM_CONT_BATCH_SIZE=${cont_batch_size} #continuous batch size for transformers-neuronx
export vLLM_TENSOR_PARALLEL_SIZE=${tp_size} #Tensor parallel degree for sharding the model
export NEURON_RT_VISIBLE_CORES=${cores} #Neuron cores on which the model needs to be deployed
export MASTER_PORT=12355
export OMP_NUM_THREADS=${n_threads}

# Build base command arguments
cmd_args=(
    --model "${model_id}"
    # --tokenizer "/fsx/czhenguo/Projects/fruitstand/axlearn/axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.model.v1"
    --tokenizer "${tokenizer}"
    --tensor-parallel-size "${tp_size}"
    --max-num-seqs "${cont_batch_size}"
    --max-model-len "${max_seq_len}"
    --port "${port}"
    --device "cuda"
    --trust-remote-code 
    --tokenizer-mode "mistral"
)

# Conditionally set the environment variable and add the fixed argument if --quantization is provided
[ -n "$quantization" ] && {
    export NEURON_QUANT_DTYPE="$quantization"
    echo "Setting NEURON_QUANT_DTYPE to: $NEURON_QUANT_DTYPE"
    cmd_args+=(--quantization "neuron_quant")
}

[ -n "$chat_template" ] && cmd_args+=(--chat-template "${chat_template}")

# Print the final command that will be executed
echo "Executing command:"
echo "python3 -m vllm.entrypoints.openai.api_server ${cmd_args[*]}"
echo "----------------------------------------"

# Execute the command with all arguments
python3 -m vllm.entrypoints.openai.api_server "${cmd_args[@]}"

