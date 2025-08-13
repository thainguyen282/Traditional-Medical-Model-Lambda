import torch
from unsloth import FastLanguageModel

def single_inference(check_point_path:str,message:list):
    
  
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path, 
        # max_seq_length=2048, 
        load_in_4bit=True,  
        attn_implementation="sdpa",  
    )
  

    text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
    enable_thinking = False, 
    )
 
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    request= {
        "max_new_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20
    }
    with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request["max_new_tokens"],
                temperature=request["temperature"],
                top_p=request["top_p"],
                top_k=request["top_k"],
                do_sample=True,
            )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )

    print("Generated text:", generated_text)


def batch_inference(check_point_path:str,batch_messages:list):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    
    batch_texts = []
    for messages in batch_messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False, 
        )
        batch_texts.append(text)
    
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  
    ).to("cuda")
    
    generation_config = {
        "max_new_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
    }
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_config,
        )
    
    generated_texts = []
    for i in range(len(outputs)):
        output_sequence = outputs[i][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(
            output_sequence,
            skip_special_tokens=True
        )
        generated_texts.append(generated_text)
    
    return generated_texts


if __name__=="__main__":
    batch_messages = [
        [{"role": "user", "content": "I have headache and fever. What should I do?"}],
        [{"role": "user", "content": "Solve (x + 2)^2 = 0."}],
        [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    ]
    check_point_path=checkpoint_path="/root/weights/VeneraMedicalM1/outputs/p4nhb9kq/model/checkpoint-1126"
    # checkpoint_path="/root/weights/VeneraMedicalM1/outputs/p4nhb9kq/model/checkpoint-1126"
    # checkpoint_path="/root/weights/VeneraMedicalM1/outputs/nltzx8a6/model/checkpoint-1126"
    #  or some thing like  unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit
    output=batch_inference(check_point_path,batch_messages)
    print("output",output)
    # messages = [
    # {"role" : "user", "content" : "I have headache and fever. What should I do?"}
    # ]