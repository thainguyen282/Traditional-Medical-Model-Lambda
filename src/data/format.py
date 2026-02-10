from dataclasses import dataclass
from typing import Optional, List, Dict, Union

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

def format_conversations(
    examples: Union[Dict, List[Dict]], 
    tokenizer, 
    require_tools: bool = False
) -> List[str]:
    """
    Format conversations for Unsloth SFTTrainer.
    
    Unsloth calls this function in BATCHED mode by default, meaning:
    - examples is a dict where each value is a LIST
    - We need to return a LIST of formatted strings (one per example)
    
    Args:
        examples: Either a batch dict (batched=True) or single example dict
        tokenizer: Tokenizer with chat template
        require_tools: Whether to require tool fields
        
    Returns:
        List[str]: List of formatted conversation strings
    """
    # Detect if this is batched mode or single example mode
    # In batched mode, 'conversation' will be a list of conversations
    # In single mode, 'conversation' will be a single conversation (list of messages)
    
    conversation_data = examples.get("conversation") or examples.get("conversations")
    
    if conversation_data is None:
        return []
    
    # Check if batched (list of lists) or single (list of dicts)
    is_batched = (
        isinstance(conversation_data, list) and 
        len(conversation_data) > 0 and 
        isinstance(conversation_data[0], list)
    )
    
    # Convert to batched format if single example
    if not is_batched:
        # Single example: conversation_data is [{'role': ..., 'content': ...}, ...]
        conversation_data = [conversation_data]
    
    # Now process each conversation in the batch
    formatted_texts = []
    
    for conversation in conversation_data:
        # Validate and clean messages for this conversation
        messages = []
        for msg in conversation:
            if not isinstance(msg, dict):
                continue
            
            validated = validate_msg(msg, require_tools)
            if validated:
                messages.append({
                    k: v for k, v in vars(validated).items() 
                    if v is not None
                })
        
        # Format this conversation
        if messages:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                formatted_texts.append(text)
            except Exception as e:
                print(f"Error formatting conversation: {e}")
                formatted_texts.append("")  # Add empty string to maintain batch size
        else:
            formatted_texts.append("")  # Empty conversation
    
    return formatted_texts