import os
import httpx
import traceback
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =================================================================================
# CONFIGURATION
# =================================================================================
APP_VERSION = "6.0"
LLM_BACKEND = os.getenv("LLM_BACKEND", "oobabooga")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# -- Configuration des Backends --
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LISA_SYSTEM_PROMPT = "Tu es Lisa, une IA de gestion de HomeLab..." # Votre prompt complet ici

app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    message: str
    history: list = []

# =================================================================================
# FONCTIONS SPÉCIFIQUES À CHAQUE BACKEND
# =================================================================================

async def get_reply_from_oobabooga(user_input: UserInput, history: list):
    payload = {
        "model": "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf",
        "messages": history,
        "max_tokens": 500, "temperature": 0.7, "stream": False
    }
    logging.debug(f"Payload Oobabooga: {payload}")
    timeout = httpx.Timeout(300.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=payload)
        response.raise_for_status()
        ai_response = response.json()
        return ai_response.get("choices", [{}])[0].get("message", {}).get("content", "Erreur: Réponse Oobabooga malformée.")

async def get_reply_from_gemini(user_input: UserInput, history: list):
    gemini_history = []
    for item in user_input.history:
        role = "model" if item["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": item["content"]}]})
    gemini_history.append({"role": "user", "parts": [{"text": user_input.message}]})
    
    payload = {"contents": gemini_history, "systemInstruction": {"parts": [{"text": LISA_SYSTEM_PROMPT}]}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    logging.debug(f"Payload Gemini: {payload}")
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        ai_response = response.json()
        return ai_response["candidates"][0]["content"]["parts"][0]["text"]

# =================================================================================
# DÉMARRAGE ET ENDPOINT PRINCIPAL
# =================================================================================

@app.on_event("startup")
async def startup_event():
    logging.info(f"--- Intent Router - Version {APP_VERSION} ---")
    logging.info(f"Backend LLM sélectionné : {LLM_BACKEND}")
    if LLM_BACKEND == "oobabooga" and not OOBABOOGA_API_URL:
        logging.error("AVERTISSEMENT: OOBABOOGA_API_URL n'est pas définie !")
    if LLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        logging.error("AVERTISSEMENT: GEMINI_API_KEY n'est pas définie !")
    logging.info("---------------------------------------------")

@app.post("/chat")
async def handle_chat(user_input: UserInput):
    logging.info(f"Requête reçue pour /chat.")
    
    # Construction de l'historique de base
    full_history = [{"role": "system", "content": LISA_SYSTEM_PROMPT}]
    full_history.extend(user_input.history)
    full_history.append({"role": "user", "content": user_input.message})

    try:
        if LLM_BACKEND == "oobabooga":
            reply_text = await get_reply_from_oobabooga(user_input, full_history)
        elif LLM_BACKEND == "gemini":
            # Gemini gère le prompt système différemment, on ne lui passe que l'historique user/model
            reply_text = await get_reply_from_gemini(user_input, user_input.history)
        else:
            raise HTTPException(status_code=500, detail="LLM_BACKEND non configuré correctement.")
        
        logging.info("Réponse générée avec succès.")
        return {"reply": reply_text}
    except Exception as exc:
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
        logging.error(f"Une erreur critique est survenue: {error_details}")
        logging.debug(f"Traceback complet : {full_traceback}")
        raise HTTPException(status_code=502, detail={"error": "Erreur de communication avec le backend IA.", "details": error_details})