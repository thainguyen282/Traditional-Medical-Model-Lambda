 # messages = [
        #     {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
        # ]

        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize = False,
        #     add_generation_prompt = True, # Must add for generation
        #     enable_thinking = False, # Disable thinking
        # )

        # from transformers import TextStreamer
        # _ = model.generate(
        #     **tokenizer(text, return_tensors = "pt").to("cuda"),
        #     max_new_tokens = 256, # Increase for longer outputs!
        #     temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
        #     streamer = TextStreamer(tokenizer, skip_prompt = True),
        # )
        # print("Logged config:", run.config)