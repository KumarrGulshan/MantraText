from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.generate import generate
from src.config.model_config import Config

app = FastAPI(title="MindCore LLM API")

# Request body schema
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50

# Response schema
class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str

# GPT generation endpoint
@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    output = generate(
        prompt=req.prompt,
        tokenizer_path=Config.TOKENIZER_PATH,
        model_path=Config.MODEL_PATH,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        device=Config.device
    )
    return GenerateResponse(prompt=req.prompt, generated_text=output)

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}
