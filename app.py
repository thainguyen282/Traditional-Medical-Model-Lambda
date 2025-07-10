from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from typing import Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lightweight LLM API",
    description="FastAPI server for lightweight Hugging Face models",
    version="1.0.0"
)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class GenerateResponse(BaseModel):
    response: str
    processing_time: float

# Global model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    
    # Using DistilGPT-2 as a lightweight model (82M parameters)
    model_name = "distilgpt2"
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM API is running",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(next(model.parameters()).device) if model else "N/A"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(
            request.prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to same device as model
        inputs = inputs.to(next(model.parameters()).device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        response_text = generated_text[len(request.prompt):].strip()
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated response in {processing_time:.2f}s")
        
        return GenerateResponse(
            response=response_text,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat(request: GenerateRequest):
    """Chat endpoint with conversation formatting"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt for conversation
    formatted_prompt = f"Human: {request.prompt}\nAssistant:"
    
    # Create new request with formatted prompt
    chat_request = GenerateRequest(
        prompt=formatted_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample
    )
    
    return await generate_text(chat_request)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )