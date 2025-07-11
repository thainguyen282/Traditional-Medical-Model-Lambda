def convert_conversation(conv):
            return [
                {"role": "user" if msg["from"] == "human" else "assistant", "content": msg["value"]}
                for msg in conv
            ]

        # def formatting_funcs(examples):
        #     return [formatting_func(example) for example in examples]
