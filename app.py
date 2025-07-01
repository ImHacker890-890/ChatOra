from fastapi import FastAPI
from core_model import NeuralChat

app = FastAPI()
model = NeuralChat()

@app.post("/chat")
async def chat(message: str):
    return {"response": model.respond(message)}
