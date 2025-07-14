import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# --- Configuration ---
APP_VERSION = "1.3" # On incrémente encore pour être sûr
OOBABOOGA_API_URL = "http://192.168.199.78:5000/v1"
LISA_SYSTEM_PROMPT = """Tu es Lisa, une intelligence artificielle de gestion de HomeLab, conçue pour être efficace, précise et légèrement formelle. Tu es l'assistante principale de ton administrateur. Ton rôle est de répondre à ses questions, d'exécuter ses ordres, et de mémoriser les informations importantes."""

app = FastAPI()

# --- Configuration CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Structures de Données ---
class UserInput(BaseModel):
    message: str
    history: list = []

# --- Événements et Endpoints ---

# Cette fonction est maintenant au bon niveau d'indentation (niveau 0)
@app.on_event("startup")
async def startup_event():
    print(f"--- Intent Router - Version {APP_VERSION} ---")

# Cette fonction est aussi au niveau 0
@app.post("/chat")
async def handle_chat(user_input: UserInput):
    conversation_history = [
        {"role": "system", "content": LISA_SYSTEM_PROMPT}
    ]
    conversation_history.extend(user_input.history)
    conversation_history.append({"role": "user", "content": user_input.message})

    oobabooga_payload = {
        "model": "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf",
        "messages": conversation_history,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=oobabooga_payload)
            response.raise_for_status()
            ai_response = response.json()
            lisa_message = ai_response["choices"][0]["message"]["content"]
            return {"reply": lisa_message}
    except httpx.RequestError as exc:
        error_details = f"Erreur de communication avec Oobabooga: {exc}"
        print(error_details)
        raise HTTPException(status_code=502, detail=error_details)
    except Exception as exc:
        error_details = f"Erreur interne inattendue: {type(exc).__name__} - {exc}"
        print(error_details)
        raise HTTPException(status_code=500, detail=error_details)
