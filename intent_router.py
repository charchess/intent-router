import os
import httpx
import traceback
import logging
from fastapi import FastAPI, HTTPException
# ... (le reste des imports)

# --- CONFIGURATION ---
APP_VERSION = "4.0" # Version multi-backend
LLM_BACKEND = os.getenv("LLM_BACKEND", "oobabooga") # "oobabooga" ou "gemini"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# -- Config Oobabooga --
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")

# -- Config Gemini --
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# ... (le reste de la config du logging et de FastAPI) ...

# --- ÉVÉNEMENTS DE DÉMARRAGE ---
@app.on_event("startup")
async def startup_event():
    logging.info(f"--- Intent Router - Version {APP_VERSION} ---")
    logging.info(f"Backend LLM sélectionné : {LLM_BACKEND}")
    if LLM_BACKEND == "oobabooga" and not OOBABOOGA_API_URL:
        logging.error("ERREUR FATALE: Le backend est 'oobabooga' mais OOBABOOGA_API_URL n'est pas définie !")
    if LLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        logging.error("ERREUR FATALE: Le backend est 'gemini' mais GEMINI_API_KEY n'est pas définie !")
    logging.info("---------------------")

# --- ENDPOINT PRINCIPAL (/chat) ---
@app.post("/chat")
async def handle_chat(user_input: UserInput):
    logging.info(f"Requête reçue pour /chat avec le message : '{user_input.message}'")

    if LLM_BACKEND == "oobabooga":
        return await get_reply_from_oobabooga(user_input)
    elif LLM_BACKEND == "gemini":
        return await get_reply_from_gemini(user_input)
    else:
        raise HTTPException(status_code=500, detail="LLM_BACKEND non configuré correctement.")

# --- LOGIQUE SPÉCIFIQUE À CHAQUE BACKEND ---

async def get_reply_from_oobabooga(user_input: UserInput):
    # ... (Le code de cette fonction est le même que notre ancien handle_chat)
    # Il prépare le payload pour Oobabooga et fait l'appel httpx
    # ...
    
async def get_reply_from_gemini(user_input: UserInput):
    # Gemini a un format de payload et de réponse différent.
    gemini_history = []
    for item in user_input.history:
        # Gemini alterne les rôles 'user' et 'model'
        if item["role"] == "assistant":
            gemini_history.append({"role": "model", "parts": [{"text": item["content"]}]})
        else:
            gemini_history.append({"role": "user", "parts": [{"text": item["content"]}]})

    # On ajoute le message actuel de l'utilisateur
    gemini_history.append({"role": "user", "parts": [{"text": user_input.message}]})
    
    gemini_payload = {
        "contents": gemini_history,
        # On peut aussi ajouter "systemInstruction" si besoin
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GEMINI_API_URL, json=gemini_payload)
            response.raise_for_status()
            ai_response = response.json()
            lisa_message = ai_response["candidates"][0]["content"]["parts"][0]["text"]
            return {"reply": lisa_message}
    except Exception as exc:
        # ... (La gestion d'erreur reste la même) ...