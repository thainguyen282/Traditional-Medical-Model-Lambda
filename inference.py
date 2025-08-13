import torch
from unsloth import FastLanguageModel
# checkpoint_path="/root/weights/VeneraMedicalM1/outputs/p4nhb9kq/model/checkpoint-1126"
checkpoint_path="/root/weights/VeneraMedicalM1/outputs/nltzx8a6/model/checkpoint-1126"
#  or some thing like  unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path, 
    # max_seq_length=2048, 
    load_in_4bit=True,  
    attn_implementation="sdpa",  
)
messages = [
# {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
{"role" : "user", "content" : "I have headache and fever. What should I do?"}
]

text = tokenizer.apply_chat_template(
messages,
tokenize = False,
add_generation_prompt = True,
enable_thinking = False, 
)

# _ = model.generate(
# **tokenizer(text, return_tensors = "pt").to("cuda"),
# max_new_tokens = 255, 
# temperature = 0.7, top_p = 0.8, top_k = 20, 
# streamer = TextStreamer(tokenizer, skip_prompt = True),
# )
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