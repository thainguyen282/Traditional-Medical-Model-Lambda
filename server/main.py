from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Optional, List
from dotenv import load_dotenv
import os
import json
import torch
from unsloth import FastLanguageModel

load_dotenv()
MODEL_WEIGHT_DIR = os.getenv("MODEL_WEIGHT_DIR", "/root/weights/outputs")

# Load model catalog and list
model_catalog_path = "model_catalog.txt"
model_list_path = "model_list.json"

with open(model_catalog_path, "r") as f:
    model_catalog = [line.strip() for line in f if line.strip()]

with open(model_list_path, "r") as f:
    model_list = json.load(f)

app = FastAPI()

def load_model_from_path(path: str):
    """Load model and tokenizer from given path"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    return model, tokenizer

class ModelLoader:
    _model_books: List[Dict] = model_list
    _loaded_models: Dict[str, tuple] = {}

    @classmethod
    def load_model(cls, model_slug: str):
        if model_slug in cls._loaded_models:
            return cls._loaded_models[model_slug]
            
        model_info = next((m for m in cls._model_books if m.get("id") == model_slug), None)
        print("model_info",model_info)
        if not model_info:
            print(f"Model not found in catalog: {model_slug}")
            return None, None
            
        try:
            path = model_info.get("path")
            if not path:
                raise ValueError(f"No path specified for model {model_slug}")
                
            full_path = os.path.join(MODEL_WEIGHT_DIR, path)
            print("Loading model from path:", full_path)
            model, tokenizer = load_model_from_path(full_path)
            cls._loaded_models[model_slug] = (model, tokenizer)
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model {model_slug}: {str(e)}")
            return None, None

class ModelRequest(BaseModel):
    model_slug: str
    input_data: Optional[dict] = None
    max_new_tokens: int = 250
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

class ModelInfoResponse(BaseModel):
    model_slug: str
    status: str
    description: Optional[str] = None

@app.get("/")
async def root():
    return {"available_models": model_catalog}

@app.get("/model/info/{model_slug}", response_model=ModelInfoResponse)
async def get_model_info(model_slug: str):
    model_info = next((m for m in ModelLoader._model_books if m.get("id") == model_slug), None)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_slug} not found")
    return {
        "model_slug": model_slug,
        "status": "available",
        "description": model_info.get("description", "No description available"),
    }


@app.post("/model/predict")
async def predict(model_request: ModelRequest = Body(...)):
    try:
        model, tokenizer = ModelLoader.load_model(model_request.model_slug)
        if model is None or tokenizer is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {model_request.model_slug} not found or failed to load"
            )

        # Prepare input
        input_content = model_request.input_data.get("content", "No input provided")
        print("Input content:", input_content)
        messages = [{"role": "user", "content": input_content}]
        
        # Convert to prompt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=model_request.max_new_tokens,
                temperature=model_request.temperature,
                top_p=model_request.top_p,
                top_k=model_request.top_k,
                do_sample=True,
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )

        return {
            "model": model_request.model_slug,
            "input": model_request.input_data,
            "output": generated_text,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)