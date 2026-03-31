from typing import List, Dict


def conversations_to_prompt_completion(example: Dict) -> Dict:
    """
    Convert a conversations-format example into prompt/completion format
    for Unsloth's completion_only_loss path.

    Split at the last assistant turn:
    - prompt: all messages before the last assistant message
    - completion: the last assistant message (as a list for conversational format)
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
            "prompt": [{"role": m["role"], "content": m["content"]} for m in convs],
            "completion": [{"role": "assistant", "content": ""}],
        }

    prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in convs[:last_assistant_idx]]
    completion_msgs = [{"role": m["role"], "content": m["content"]} for m in convs[last_assistant_idx:]]

    return {"prompt": prompt_msgs, "completion": completion_msgs}