from schema import TOOL_CALL_SCHEMA, BASIC_SCHEMA
from datasets import Dataset
import warnings

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
    def validate(sample):
        """Validate basic message structure"""
        return all(
            isinstance(msg.get("role"), str) and 
            isinstance(msg.get("content"), str)
            for msg in sample["conversations"]
        )

    @staticmethod
    def standardize(sample):
        """Standardize basic messages"""
        return {
            "conversations": [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                for msg in sample["conversations"]
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