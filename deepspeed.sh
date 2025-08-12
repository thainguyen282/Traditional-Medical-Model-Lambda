#!/usr/bin/bash
root_dir=/root/VeneraMedicalM1
#number of params
nps=(
    4
    8
    14
    32
)
#quant type
qtypes=(
    "4"
    "f"
)
#num gpus
ngs=(
    "1"
    "2"
    "4"
    # "8"
)
#release by (who optimized it)
optims=(
    "unsloth"
    "Qwen"
)
#dir for stress test config
config_dir="$root_dir"/configs/experiments/hw
mkdir -p $config_dir

train_samples=1000

# ----------- FUNCTIONS--------------

export_config() {
    mkdir -p $config_dir/$optim
    if [[ $optim == "unsloth" ]]
    then
        deepspeed_path=/root/VeneraMedicalM1/dz2_config.jsonl
    else
        deepspeed_path=/root/VeneraMedicalM1/dz3_config.jsonl
    fi
    echo """
name: $optim/$model_name-$ng

model:
  name: $optim/$model_name
  max_seq_length: 2048

datasets:
    # currently multidataset is broken
    # - path: "configs/dataset/finetome.yaml"
    #   train_samples: 1000
    #   train_val_split: 0.7

    - tag: "venera_med_m1"
      path: "configs/dataset/venera_med_m1.yaml"
      train_samples: $train_samples
      train_val_split: 0.2

benchmarks:
  - path: "$root_dir/configs/evals/green.yaml"

unsloth:
    load_in_4bit: true
    attn_implementation: "sdpa"

training:
    output_dir: "group_outputs/outputs"
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 16
    num_train_epochs: 2
    weight_decay: 0.01
    warmup_steps: 100
    lr_scheduler_type: "cosine"
    optim: "adamw_8bit"
    eval_strategy: "steps"
    eval_steps: $train_samples
    save_strategy: "steps"
    save_steps: $train_samples
    logging_steps: 10
    seed: 3407
    deepspeed: $deepspeed_path

lora:
    lora_dropout: 0.1
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    bias: "none"
    use_rslora: false

monitoring:
  project: "VeneraMedicalM1"
  tags: $tags

    """ > $config_dir/$optim/$model_name-$qtype-$ng.yaml
}

run_deepspeed() {
    ROOT=$root_dir deepspeed --num_gpus=$ng \
    main.py --config $config_dir/$optim/$model_name-$qtype-$ng.yaml \
    --deepspeed
    sleep 15s #wait for dealloc
}

#--------------- MAIN ----------------------
for np in ${nps[@]}
do
    for ng in ${ngs[@]}
    do
        for optim in ${optims[@]}
        do
            case $optim in
                "unsloth")
                    for qtype in ${qtypes[@]}
                    do
                        case $qtype in
                            4)
                                model_name=Qwen3-"$np"B-unsloth-bnb-4bit
                                tags="""[\"lora\", \"$optim\",\"$model_name\",\""$ng"gpus\"]"""
                            ;;
                            f)
                                model_name=Qwen3-"$np"B
                                tags="""[\"lora\", \"$optim\",\"$model_name\",\""$ng"gpus\"]"""
                            ;;
                        esac
                        export_config
                        run_deepspeed
                    done
                ;;
                "Qwen")
                    model_name=Qwen3-"$np"B
                    tags="""[\"lora\", \"$optim\",\"$model_name\",\""$ng"gpus\"]"""
                    echo  "Not supported: $optim/$model_name"
                    # export_config
                    # run_deepspeed
                ;;
            esac
        done
    done
done