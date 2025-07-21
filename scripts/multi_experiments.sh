#!/bin/bash

log_dir="group_outputs/logs/single_exp"
config_paths=(
    "configs/experiments/qwen-0.6B/exp0001/config.yaml"
    "configs/experiments/qwen-1.7B/exp0001/config.yaml"

)

mkdir -p "$log_dir"

run_experiment() {
    local config_path=$1
    local model_name=$(basename $(dirname $(dirname "$config_path")))  # Extracts qwen-0.6B
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${log_dir}/${model_name}_${timestamp}.log"
    
    echo "Starting $model_name with config $config_path"
    echo "Logging to: $log_file"
    
    python main.py --config "$config_path" &> "$log_file" &
}

for config_path in "${config_paths[@]}"; do
    run_experiment "$config_path"
done

wait
echo "All experiments completed"