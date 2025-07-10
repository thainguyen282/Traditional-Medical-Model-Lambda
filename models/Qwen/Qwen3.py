import os

os.environ["TRITON_DISABLE_COMPILE"] = "1"
os.environ["TRITON_USE_PYTORCH_ALLOCATOR"] = "1"
os.environ["TRITON_DISABLE"] = "1"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
import triton
print(f"Triton: {triton.__version__}")  # Should be ≥2.1.0
# print(f"Flash Attention: {flash_attn.__version__}")  # Should be ≥2.5.0

fourbit_models = [
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-14B",
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_4bit = True,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    #  attn_implementation="flash_attention_2"  #
    attn_implementation="sdpa",
    #    torch_dtype = torch.float16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

def convert_conversation(conv):
    return [
        {"role": "user" if msg["from"] == "human" else "assistant", "content": msg["value"]}
        for msg in conv
    ]

def formatting_func(examples):
    if "conversations" in examples:
        conversations = examples["conversations"]
        
        # If it's batched, conversations will be a list of lists
        if isinstance(conversations, list) and len(conversations) > 0:
            if isinstance(conversations[0], list):
                # Batched case
                texts = []
                for conversation in conversations:
                    messages = [
                        {"role": "user" if msg["from"] == "human" else "assistant", "content": msg["value"]}
                        for msg in conversation
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                return texts
            else:
                # Single example case
                messages = [
                    {"role": "user" if msg["from"] == "human" else "assistant", "content": msg["value"]}
                    for msg in conversations
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return [text]
    
    # Fallback
    return [""]

def formatting_funcs(examples):
    return [formatting_func(example) for example in examples]

n_samples=10
sampled_data = [non_reasoning_dataset[i] for i in range(min(n_samples, len(non_reasoning_dataset)))]
combined_dataset = Dataset.from_list(sampled_data).shuffle(seed=3407)
example=combined_dataset[0]
print("combined dataset[0]",combined_dataset[0])
print("format func",formatting_func(example))
print("len combine dataset",len(combined_dataset))

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, 
    formatting_func=formatting_func, 
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        max_steps=4,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none" # Use this for WandB etc
    )
)
trainer_stats = trainer.train()

messages = [
    {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 256, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
