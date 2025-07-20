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
                else:  
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