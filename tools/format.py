from functools import partial

def formatting_func(examples, tokenizer=None):
    output = []
    if "conversations" in examples:
        conversations = examples["conversations"]
        if isinstance(conversations, list) and len(conversations) > 0:
            if isinstance(conversations[0], list):  
                for conversation in conversations:
                    messages = []
                    for msg in conversation:
                        if msg["role"] == "system":
                            messages.append({
                                "role": "system",
                                "content": msg["content"]
                            })
                        else:
                            messages.append({
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            })
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    output.append(text)
            else:  # Single conversation
                messages = []
                for msg in conversations:
                    if msg["role"] == "system":
                        messages.append({
                            "role": "system",
                            "content": msg["content"]
                        })
                    else:
                        messages.append({
                            "role": "user" if msg["role"] == "user" else "assistant",
                            "content": msg["content"]
                        })
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                 
                )
                output.append(text)
    return output

if __name__ == "__main__":
    import os 
    import sys 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import AutoTokenizer
    from data.dataset.finetome import load_fine_tome_ds
    from data.dataset.med_sample import load_med_sample
    config = {
        "name": "mlabonne/FineTome-100k",
        "train": {"n_samples": 30},
        "val": {"percent": 0.2}
    }
    tokenizer=AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B")
    bound_formatting_func = partial(formatting_func, tokenizer=tokenizer)
    
    # train_set, val_set = load_fine_tome_ds(config, dataset_id="fine_tome")
    train_set=load_med_sample(config)
    for i, sample in enumerate(train_set):
        print("-------------------------------")
        print("Before formatting:", sample)
        format_sample = bound_formatting_func(sample)
        print("-------------------------------")
        print(f"Sample {i + 1}: {format_sample}")