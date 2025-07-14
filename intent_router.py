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

# Version de l'application
APP_VERSION = "3.0"

# Configuration du logging
# On lit la variable d'environnement 'VERBOSE'. Si elle est à 'true', on active les logs de débogage.
IS_VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if IS_VERBOSE else logging.INFO

# On configure le format des logs pour inclure la date, l'heure et le niveau
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Configuration des services externes via les variables d'environnement
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
LISA_SYSTEM_PROMPT = """Tu es Lisa, une intelligence artificielle de gestion de HomeLab, conçue pour être efficace, précise et légèrement formelle. Tu es l'assistante principale de ton administrateur. Ton rôle est de répondre à ses questions, d'exécuter ses ordres, et de mémoriser les informations importantes."""

# Initialisation de l'application FastAPI
app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)

# =================================================================================
# MIDDLEWARE (CORS)
# =================================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================================
# STRUCTURES DE DONNÉES
# =================================================================================
class UserInput(BaseModel):
    message: str
    history: list = []

# =================================================================================
# ÉVÉNEMENTS DE DÉMARRAGE
# =================================================================================
@app.on_event("startup")
async def startup_event():
    """Vérifie la configuration au démarrage et affiche les informations."""
    logging.info("--- Intent Router ---")
    logging.info(f"Version: {APP_VERSION}")
    logging.info(f"Mode verbeux : {'Activé' if IS_VERBOSE else 'Désactivé'}")
    if not OOBABOOGA_API_URL:
        logging.error("ERREUR FATALE: La variable d'environnement OOBABOOGA_API_URL n'est pas définie !")
    else:
        logging.info(f"Backend Oobabooga configuré à l'adresse : {OOBABOOGA_API_URL}")
    logging.info("---------------------")

# =================================================================================
# ENDPOINT PRINCIPAL (/chat)
# =================================================================================
@app.post("/chat")
async def handle_chat(user_input: UserInput):
    logging.info(f"Requête reçue pour /chat avec le message : '{user_input.message}'")

    # Préparation du payload
    conversation_history = [{"role": "system", "content": LISA_SYSTEM_PROMPT}]
    conversation_history.extend(user_input.history)
    conversation_history.append({"role": "user", "content": user_input.message})

    oobabooga_payload = {
        "model": "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf",
        "messages": conversation_history,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False
    }
    logging.debug(f"Payload préparé pour Oobabooga : {oobabooga_payload}")

    # Appel à l'API d'Oobabooga
    try:
        timeout = httpx.Timeout(300.0, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logging.debug(f"Envoi de la requête POST à {OOBABOOGA_API_URL}/chat/completions")
            response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=oobabooga_payload)
            logging.debug(f"Réponse reçue d'Oobabooga. Statut : {response.status_code}")
            response.raise_for_status()
            
            ai_response = response.json()
            lisa_message = ai_response.get("choices", [{}])[0].get("message", {}).get("content", "Désolée, je n'ai pas pu générer de réponse.")
            
            logging.info(f"Réponse de Lisa générée avec succès.")
            return {"reply": lisa_message}

    except Exception as exc:
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
        logging.error("Une erreur critique est survenue lors de la communication avec Oobabooga.")
        logging.debug(f"Traceback complet : {full_traceback}")
        raise HTTPException(status_code=502, detail={"error": "Erreur de communication avec le backend IA.", "details": error_details})