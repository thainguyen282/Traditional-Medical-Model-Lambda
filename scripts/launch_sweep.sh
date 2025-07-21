#!/bin/bash

# ===== CONFIGURATION =====
sweep_path="configs/experiments/qwen-0.6B/exp0001/sweep.yaml"
project_name="VeneraMedicalM1"  
entity="nguyenduc150202"    
num_agents=1
count=1
log_dir="group_outputs/logs/sweeps"

# ===== ARGUMENT PARSING =====
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -n|--num-agents)
      if [[ "$2" =~ ^[0-9]+$ ]] && [[ "$2" -gt 0 ]]; then
        num_agents="$2"
        shift
      else
        echo "Error: --num-agents requires a positive integer" >&2
        usage
        exit 1
      fi
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    -h|--help)
      show_help=true
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

# Show help if requested
if [ "$show_help" = true ]; then
  usage
  exit 0
fi
# ===== INITIALIZATION =====
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_dir/sweep_${timestamp}.log"
mkdir -p "$log_dir"

{
echo "=== SWEEP LAUNCHER ==="
date
echo "Sweep config: $sweep_path"
echo "Project: $project_name"
echo "Entity: $entity"
echo "Agents: $num_agents"
echo "Runs per agent: $count"
echo "Dry run: $DRY_RUN"

# ===== VALIDATION =====
echo -n "Validating sweep config... "

# Validate YAML syntax
if ! python3 -c "import yaml; yaml.safe_load(open('$sweep_path'))" >/dev/null 2>&1; then
    echo "❌ Invalid YAML syntax"
    echo "YAML validation error:"
    python3 -c "import yaml; yaml.safe_load(open('$sweep_path'))" 2>&1
    exit 1
fi

# Then validate with wandb (using temporary project)
if [[ "$DRY_RUN" = true ]]; then
    temp_project="tmp-validation-$(date +%s)"
    echo -n "(using temporary project $temp_project)... "
    sweep_output=$(wandb sweep --project "$temp_project" --entity "$entity" "$sweep_path" 2>&1)
    validation_status=$?
    
    if [ "$validation_status" -eq 0 ]; then
        SWEEP_ID=$(echo "$sweep_output" | grep -o 'ID: .*' | cut -d' ' -f2)
        echo "✅ Valid (ID: $SWEEP_ID)"
        echo "Cleaning up temporary sweep..."
        wandb sweep --delete "$SWEEP_ID" --project "$temp_project" --entity "$entity"
        exit 0
    else
        echo "❌ Invalid sweep configuration"
        echo "Error details:"
        echo "$sweep_output"
        exit 1
    fi
else
    echo "✅ YAML valid (use --dry-run for full validation)"
fi

# ===== SWEEP CREATION =====
echo -n "Creating sweep... "
sweep_output=$(wandb sweep --project "$project_name" --entity "$entity" "$sweep_path" 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ Failed: $sweep_output"
    exit 1
fi

SWEEP_ID=$(echo "$sweep_output" | grep -o 'ID: .*' | cut -d' ' -f2)
[[ -z "$SWEEP_ID" ]] && { echo "❌ Failed to extract sweep ID"; exit 1; }
echo "✅ Created (ID: $SWEEP_ID)"

# ===== AGENT LAUNCH =====
for i in $(seq 1 "$num_agents"); do
    agent_log="$log_dir/agent_${i}_${timestamp}.log"
    echo "🚀 Launching agent $i/$num_agents (log: $agent_log)..."
    
    WANDB_DEBUG=1 wandb agent --project "$project_name" --entity "$entity" "$SWEEP_ID" --count "$count" > "$agent_log" 2>&1 &
    sleep 2
done

wait
echo "🎉 All agents completed. Please check the agent logs to make sure things does not break"
} | tee -a "$log_file"