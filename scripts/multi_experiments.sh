#!/bin/bash

log_dir="group_outputs/logs/single_exp"
config_paths=(
    # "configs/experiments/unsloth-qwen3-0.6B/exp0002/config.yaml"
    # "configs/experiments/unsloth-qwen3-1.7B/exp0002/config.yaml"
"configs/experiments/unsloth-qwen3-4B/exp0002/config.yaml"
)

mkdir -p "$log_dir"
for config_path in "${config_paths[@]}"; do
    python main.py --config "$config_path"  # No &, runs sequentially
done

# Starts a single experiment with the given config file
#
# Args:
#     config_path: path to the config file
#
# Logs the experiment to a file named <model_name>_<timestamp>.log in the
# log directory, e.g. qwen-0.6B_20230315_140000.log
# run_experiment() {
#     local config_path=$1
#     local model_name=$(basename $(dirname $(dirname "$config_path")))  # Extracts qwen-0.6B
#     local timestamp=$(date +"%Y%m%d_%H%M%S")
#     local log_file="${log_dir}/${model_name}_${timestamp}.log"
    
#     echo "Starting $model_name with config $config_path"
#     echo "Logging to: $log_file"
    
#     python main.py --config "$config_path" &> "$log_file" &
# }

# for config_path in "${config_paths[@]}"; do
#     run_experiment "$config_path"
# done

# wait
# echo "All experiments completed"