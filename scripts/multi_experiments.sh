#!/bin/bash

log_dir="group_outputs/logs/single_exp"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")  
log_dir="${log_dir}/${timestamp}"
mkdir -p "$log_dir"

config_paths=(
    "configs/experiments/unsloth-qwen3-0.6B/exp0002/config.yaml"
    "configs/experiments/unsloth-qwen3-1.7B/exp0002/config.yaml" 
    "configs/experiments/unsloth-qwen3-4B/exp0002/config.yaml"
)

run_experiment() {
    local config_path=$1
    local model_name=$(basename $(dirname $(dirname "$config_path")))
    local experiment_name=$(basename $(dirname "$config_path"))
    local log_file="${log_dir}/${model_name}_${experiment_name}.log"
    
    echo "==================================================================" | tee "$log_file"
    echo "Experiment started at: $(date)" | tee -a "$log_file"
    echo "Model: $model_name" | tee -a "$log_file"
    echo "Experiment: $experiment_name" | tee -a "$log_file"
    echo "Config: $config_path" | tee -a "$log_file"
    echo "Log file: $log_file" | tee -a "$log_file"
    echo "==================================================================" | tee -a "$log_file"
    
    # Run the experiment and log output
    python main.py --config "$config_path" 2>&1 | tee -a "$log_file"
    
    # Check exit status
    local exit_status=${PIPESTATUS[0]}
    echo "==================================================================" | tee -a "$log_file"
    echo "Experiment finished at: $(date)" | tee -a "$log_file"
    
    if [ $exit_status -ne 0 ]; then
        echo "ERROR: Experiment failed - Exit code: $exit_status" | tee -a "$log_file"
        return 1
    else
        echo "SUCCESS: Experiment completed successfully" | tee -a "$log_file"
        return 0
    fi
}

# Run all experiments
for config_path in "${config_paths[@]}"; do
    run_experiment "$config_path"
done

# Final summary
echo "=================================================================="
echo "All experiments completed."
echo "Logs directory: $log_dir"
echo "Contents:"
ls -lh "$log_dir"
echo "=================================================================="

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