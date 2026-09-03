from typing import List, Dict

MCQ_FORMAT_INSTRUCTION = (
    "Hãy giải thích lập luận, sau đó kết thúc câu trả lời bằng định dạng \"### X\", "
    "trong đó X là chữ cái của phương án đúng (ví dụ: A, B, C hoặc D)."
)


def conversations_to_prompt_completion(example: Dict, tokenizer) -> Dict:
    """
    Convert a conversations-format example into prompt/completion format
    for Unsloth's completion_only_loss path.

    Split at the last assistant turn:
    - prompt: all messages before the last assistant message
    - completion: the final assistant message content as a string
    """
    convs = example.get("conversations") or example.get("messages") or []

    # Find last assistant message index
    last_assistant_idx = None
    for i in range(len(convs) - 1, -1, -1):
        if convs[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        # No assistant message — use all as prompt, empty completion
        return {
            "prompt": tokenizer.apply_chat_template(
                convs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "completion": "",
        }

    prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in convs[:last_assistant_idx]]
    if example.get("answer_letter"):
        for index in range(len(prompt_msgs) - 1, -1, -1):
            if prompt_msgs[index].get("role") == "user":
                prompt_msgs[index] = {
                    **prompt_msgs[index],
                    "content": prompt_msgs[index]["content"].rstrip()
                    + "\n\n" + MCQ_FORMAT_INSTRUCTION,
                }
                break
    prompt = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    completion = convs[last_assistant_idx]["content"]

    return {"prompt": prompt, "completion": completion}