from unsloth import FastLanguageModel
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModel
checkpoint_path="outputs/checkpoint-9"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path, 
    max_seq_length=2048, 
    load_in_4bit=True,  
    attn_implementation="sdpa",  
)
messages = [
{"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
]

text = tokenizer.apply_chat_template(
messages,
tokenize = False,
add_generation_prompt = True,
enable_thinking = False, 
)

_ = model.generate(
**tokenizer(text, return_tensors = "pt").to("cuda"),
max_new_tokens = 256, 
temperature = 0.7, top_p = 0.8, top_k = 20, 
streamer = TextStreamer(tokenizer, skip_prompt = True),
)