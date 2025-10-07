#!/bin/bash
set -e
set -o pipefail

model=$1
dataset=$2

if [ -z "$model" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 <model> <dataset>"
    echo ""
    echo "Arguments:"
    echo "  model    - Model name (e.g., Kimi/kimi-k2-turbo-preview, models/Qwen3-4B-Instruct-2507)"
    echo "  dataset  - Dataset name (musique, multihop-rag)"
    echo ""
    echo "Examples:"
    echo "  $0 Kimi/kimi-k2-turbo-preview musique"
    echo "  $0 models/Qwen3-4B-Instruct-2507 multihop-rag"
    exit 1
fi

# Fixed configurations
prompt_name="MultiHop-RAG-NoThink"

# Generate model name for directory
model_name=$(echo "$model" | tr '/:' '-')

# Determine dataset path and output directory based on dataset name
case "$dataset" in
    musique)
        dataset_path="./data/musique/musique_dev_200.jsonl"
        output_dir="output/musique/${prompt_name}/${model_name}"
        ;;
    multihop-rag)
        dataset_path="./data/MultiHop-RAG/_data/val.jsonl"
        output_dir="output/MultiHop-RAG/${prompt_name}/${model_name}"
        ;;
    hotpotqa)
        dataset_path="./data/hotpotqa/hotpotqa_dev_200.jsonl"
        output_dir="output/hotpotqa/${prompt_name}/${model_name}"
        ;;
    2wikimultihopqa)
        dataset_path="./data/2wikimultihopqa/2wikimultihopqa_dev_200.jsonl"
        output_dir="output/2wikimultihopqa/${prompt_name}/${model_name}"
        ;;
    *)
        echo "Error: Unknown dataset '$dataset'"
        echo "Supported datasets: musique, multihop-rag, hotpotqa, 2wikimultihopqa"
        exit 1
        ;;
esac

# Check if dataset file exists
if [ ! -f "$dataset_path" ]; then
    echo "Error: Dataset file not found: $dataset_path"
    exit 1
fi

# Detect if it's a local model (starts with "models/" or doesn't contain "/")
if [[ "$model" == models/* ]] || [[ "$model" != */* ]]; then
    # Local model deployment
    echo "Using local model: $model"
    base_url="http://localhost:8000/v1"
    api_key="EMPTY"

    python deepsearch_agent.py run \
        --base_url "$base_url" \
        --api_key "$api_key" \
        --prompt-name "$prompt_name" \
        --dataset "$dataset_path" \
        --do_eval \
        --model "$model" \
        --output_dir "$output_dir"
else
    # Remote model (e.g., Kimi/xxx, OpenAI/xxx)
    echo "Using remote model: $model"

    python deepsearch_agent.py run \
        --prompt-name "$prompt_name" \
        --dataset "$dataset_path" \
        --do_eval \
        --model "$model" \
        --output_dir "$output_dir"
fi

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

# Run trajectory analysis
echo ""
echo "Running trajectory analysis..."
python analyze_trajectory.py --output_dir "$output_dir" --with_eval

if [ $? -ne 0 ]; then
    echo "Warning: Trajectory analysis failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Model: $model"
echo "Dataset: $dataset"
echo "Results saved to: $output_dir"
echo "=========================================="
