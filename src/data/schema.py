from datasets import Features, Sequence, Value

# Base message features
BASE_MESSAGE = {
    "role": Value("string"),
    "content": Value("string")
}

# Schema for conversations WITH tool calls
TOOL_CALL_SCHEMA = Features({
    "conversations": Sequence({
        **BASE_MESSAGE,
        "tool_call_id": Value("string"),
        "tool_calls": Sequence({
            "function": {
                "arguments": Value("string"),
                "name": Value("string")
            },
            "id": Value("string"),
            "type": Value("string")
        })
    })
})

# Schema for basic conversations WITHOUT tool calls
BASIC_SCHEMA = Features({
    "conversations": Sequence({
        **BASE_MESSAGE
    })
})

# Schema detection mapping
SCHEMA_MAP = {
    "tool": TOOL_CALL_SCHEMA,
    "basic": BASIC_SCHEMA
}