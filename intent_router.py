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
APP_VERSION = "5.0" # Nouvelle version majeure, nouvelle approche
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
LISA_SYSTEM_PROMPT = """Tu es Lisa, une intelligence artificielle de gestion de HomeLab, conçue pour être efficace, précise et légèrement formelle. Tu es l'assistante principale de ton administrateur. Ton rôle est de répondre à ses questions, d'exécuter ses ordres, et de mémoriser les informations importantes."""

app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    message: str
    history: list = []

# =================================================================================
# DÉMARRAGE ET ENDPOINT PRINCIPAL
# =================================================================================

@app.on_event("startup")
async def startup_event():
    """Vérifie la configuration au démarrage."""
    logging.info(f"--- Intent Router - Version {APP_VERSION} ---")
    if not OOBABOOGA_API_URL:
        logging.error("ERREUR FATALE: La variable d'environnement OOBABOOGA_API_URL n'est pas définie !")
    else:
        logging.info(f"Backend Oobabooga configuré à l'adresse : {OOBABOOGA_API_URL}")
    logging.info("---------------------------------------------")

@app.post("/chat")
async def handle_chat(user_input: UserInput):
    """Reçoit une requête, l'envoie à Oobabooga, et renvoie la réponse."""
    logging.info(f"Requête reçue pour /chat : '{user_input.message}'")

    if not OOBABOOGA_API_URL:
        raise HTTPException(status_code=500, detail="Le backend Oobabooga n'est pas configuré.")

    # Préparation du payload pour Oobabooga
    conversation_history = [{"role": "system", "content": LISA_SYSTEM_PROMPT}]
    conversation_history.extend(user_input.history)
    conversation_history.append({"role": "user", "content": user_input.message})

    payload = {
        "model": "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf",
        "messages": conversation_history,
        "max_tokens": 500, "temperature": 0.7, "stream": False
    }
    logging.debug(f"Payload Oobabooga: {payload}")
    
    # Appel à l'API Oobabooga
    try:
        timeout = httpx.Timeout(300.0, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=payload)
            response.raise_for_status()
            ai_response = response.json()
            reply_text = ai_response.get("choices", [{}])[0].get("message", {}).get("content", "Erreur: Réponse Oobabooga malformée.")
            logging.info("Réponse générée avec succès.")
            return {"reply": reply_text}
            
    except Exception as exc:
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
        logging.error(f"Une erreur critique est survenue: {error_details}")
        logging.debug(f"Traceback complet : {full_traceback}")
        raise HTTPException(status_code=502, detail={"error": "Erreur de communication avec le backend IA.", "details": error_details})