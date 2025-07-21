from src.data.schema import TOOL_CALL_SCHEMA, BASIC_SCHEMA
from datasets import Dataset
import warnings
from datasets.features import Features, Sequence, Value

class ToolCallStandardizer:
    @staticmethod
    def validate(sample):
        """Validate tool call structure"""
        for msg in sample["conversations"]:
            if "tool_calls" in msg:
                if not isinstance(msg["tool_calls"], list):
                    return False
                for call in msg["tool_calls"]:
                    if not all(k in call.get("function", {}) for k in ["arguments", "name"]):
                        return False
        return True

    @staticmethod
    def standardize(sample):
        """Standardize tool call messages"""
        standardized = []
        for msg in sample["conversations"]:
            new_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "tool_call_id": msg.get("tool_call_id", "")
            }
            
            if "tool_calls" in msg:
                new_msg["tool_calls"] = [
                    {
                        "function": {
                            "arguments": call["function"]["arguments"],
                            "name": call["function"]["name"]
                        },
                        "id": call["id"],
                        "type": call["type"]
                    }
                    for call in msg["tool_calls"] 
                    if isinstance(call, dict)
                ]
            
            standardized.append(new_msg)
        return {"conversations": standardized}

class BasicStandardizer:
    @staticmethod
    def get_output_features():
        """Define the exact output schema we expect"""
        return Features({
            "conversations": Sequence({
                "role": Value("string"),
                "content": Value("string")
            })
        })
    @staticmethod
    def validate(sample):
        """Validate basic message structure (works for both single and batch)"""
        conversations = sample["conversations"]
        if not isinstance(conversations, list):
            return False
            
        first_item = conversations[0] if conversations else None
        is_batch = isinstance(first_item, list) if first_item else False
        
        if is_batch:
            return all(
                isinstance(msg.get("role"), str) and 
                isinstance(msg.get("content"), str)
                for conv in conversations
                for msg in conv
            )
        else:
            return all(
                isinstance(msg.get("role"), str) and 
                isinstance(msg.get("content"), str)
                for msg in conversations
            )

    @staticmethod
    def standardize(sample):
        """Standardize messages (handles both single and batch processing)"""
        conversations = sample["conversations"]
        
        if conversations and isinstance(conversations[0], list):
            return {
                "conversations": [
                    [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in conv
                    ]
                    for conv in conversations
                ]
            }
        else:
            return {
                "conversations": [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in conversations
                ]
            }

def get_standardizer(dataset: Dataset):
    """Detect and return appropriate standardizer"""
    sample = dataset[0]
    for msg in sample["conversations"]:
        if "tool_calls" in msg and msg["tool_calls"]:
            if ToolCallStandardizer.validate(sample):
                return ToolCallStandardizer, TOOL_CALL_SCHEMA
            break
    return BasicStandardizer, BASIC_SCHEMA