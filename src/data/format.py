from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

def validate_msg(msg: Dict, require_tools: bool = False) -> Optional[Message]:
    """Validate message, optionally requiring tool fields"""
    try:
        if not all(k in msg for k in ["role", "content"]):
            return None
        if require_tools and "tool_calls" not in msg:
            return None
        return Message(
            role=msg["role"],
            content=msg["content"],
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id")
        )
    except Exception:
        return None

def format_conversations(batch: Dict, tokenizer, require_tools=False) -> List[str]:
    """Format validated conversations"""
    outputs = []
    for conv in batch.get("conversations", []):
        messages = []
        for msg in (conv if isinstance(conv, list) else [conv]):
            if valid := validate_msg(msg, require_tools):
                messages.append({
                    k: v for k, v in vars(valid).items() 
                    if v is not None
                })
        if messages:
            outputs.append(tokenizer.apply_chat_template(
                messages, tokenize=False,
                
    add_generation_prompt=True,
    enable_thinking=False 
            ))
    return outputs