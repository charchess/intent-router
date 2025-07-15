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
APP_VERSION = "7.0"
LLM_BACKEND = os.getenv("LLM_BACKEND", "oobabooga")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# -- Config Oobabooga --
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
OOBABOOGA_MODEL_NAME = os.getenv("OOBABOOGA_MODEL_NAME", "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf")

# -- Config Gemini --
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

LISA_SYSTEM_PROMPT = """Tu es Lisa, une intelligence artificielle de gestion de HomeLab, conçue pour être efficace, précise et légèrement formelle. Tu es l'assistante principale du "Roi", ton administrateur. Ton rôle est de répondre à ses questions, d'exécuter ses ordres, et de mémoriser les informations importantes.

Règles d'action :
Quand une action doit être effectuée (contrôler un appareil, mémoriser une information), tu ne dois PAS la décrire dans ta réponse. À la place, tu dois répondre UNIQUEMENT et EXACTEMENT avec un bloc d'action spécial, comme suit :
<|ACTION|>
{
  "tool": "outil_a_utiliser",
  ...autres paramètres...
}
<|/ACTION|>

Voici les outils disponibles :

1.  **Outil `home_assistant` :** Pour contrôler les appareils de la maison.
    -   **Paramètres requis :** `domain`, `service`, `entity_id`.
    -   **Exemple :** <|ACTION|>{"tool": "home_assistant", "domain": "light", "service": "turn_on", "entity_id": "light.salon"}<|/ACTION|>

2.  **Outil `memory_store` :** Pour enregistrer une information importante.
    -   **Paramètre requis :** `text`.
    -   **Exemple :** <|ACTION|>{"tool": "memory_store", "text": "Le mot de passe du Wi-Fi invité est 'BienvenueChezMoi'."}<|/ACTION|>

Pour toutes les autres conversations, réponds normalement en langage naturel.
"""

app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    message: str
    history: list = []

# =================================================================================
# FONCTIONS SPÉCIFIQUES AUX BACKENDS
# =================================================================================

async def get_reply_from_oobabooga(user_input: UserInput, history: list):
    # Prépare et envoie la requête à Oobabooga.
    if not OOBABOOGA_API_URL:
        raise HTTPException(status_code=500, detail="OOBABOOGA_API_URL n'est pas configuré.")

    payload = {
        "model": OOBABOOGA_MODEL_NAME, # Utilise la variable d'environnement
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
    # Prépare et envoie la requête à l'API Gemini.
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY n'est pas configuré.")

    gemini_history = []
    for item in user_input.history:
        role = "model" if item["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": item["content"]}]})
    gemini_history.append({"role": "user", "parts": [{"text": user_input.message}]})
    
    payload = {"contents": gemini_history, "systemInstruction": {"parts": [{"text": LISA_SYSTEM_PROMPT}]}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
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
    if LLM_BACKEND == "oobabooga":
        if not OOBABOOGA_API_URL: logging.error("AVERTISSEMENT: OOBABOOGA_API_URL n'est pas définie !")
        logging.info(f"Modèle Oobabooga : {OOBABOOGA_MODEL_NAME}")
    if LLM_BACKEND == "gemini":
        if not GEMINI_API_KEY: logging.error("AVERTISSEMENT: GEMINI_API_KEY n'est pas définie !")
        logging.info(f"Modèle Gemini : {GEMINI_MODEL_NAME}")
    logging.info("---------------------------------------------")

@app.post("/chat")
async def handle_chat(user_input: UserInput):
    logging.info(f"Requête reçue pour /chat.")
    
    full_history = [{"role": "system", "content": LISA_SYSTEM_PROMPT}]
    full_history.extend(user_input.history)
    full_history.append({"role": "user", "content": user_input.message})

    try:
        if LLM_BACKEND == "oobabooga":
            reply_text = await get_reply_from_oobabooga(user_input, full_history)
        elif LLM_BACKEND == "gemini":
            reply_text = await get_reply_from_gemini(user_input, user_input.history)
        else:
            raise HTTPException(status_code=500, detail="LLM_BACKEND non configuré correctement. Choisir 'oobabooga' ou 'gemini'.")
        
        logging.info("Réponse générée avec succès.")
        return {"reply": reply_text}
    except Exception as exc:
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
        logging.error(f"Une erreur critique est survenue: {error_details}")
        logging.debug(f"Traceback complet : {full_traceback}")
        raise HTTPException(status_code=502, detail={"error": "Erreur de communication avec le backend IA.", "details": error_details})