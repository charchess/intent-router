import os
import httpx
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =================================================================================
# CONFIGURATION
# =================================================================================

# On lit les variables d'environnement. Des valeurs par défaut sont fournies pour le développement local.
APP_VERSION = "2.1"
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL", "http://localhost:5000/v1")
LISA_SYSTEM_PROMPT = """Tu es Lisa, une intelligence artificielle de gestion de HomeLab, conçue pour être efficace, précise et légèrement formelle. Tu es l'assistante principale de ton administrateur. Ton rôle est de répondre à ses questions, d'exécuter ses ordres, et de mémoriser les informations importantes."""

# Initialisation de l'application FastAPI
app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)

# =================================================================================
# MIDDLEWARE (CORS)
# =================================================================================

# Configuration pour autoriser les appels depuis votre interface web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissif pour le développement, à restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================================
# STRUCTURES DE DONNÉES (MODÈLES PYDANTIC)
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
    print("--- Intent Router ---")
    print(f"Version: {APP_VERSION}")
    if not OOBABOOGA_API_URL or "http" not in OOBABOOGA_API_URL:
        print("ERREUR FATALE: La variable d'environnement OOBABOOGA_API_URL est manquante ou invalide.")
        # Dans un vrai cas, on pourrait vouloir arrêter le service ici.
    else:
        print(f"Backend Oobabooga configuré à l'adresse : {OOBABOOGA_API_URL}")
    print("---------------------")

# =================================================================================
# ENDPOINT PRINCIPAL (/chat)
# =================================================================================

@app.post("/chat")
async def handle_chat(user_input: UserInput):
    """
    Reçoit le message de l'utilisateur, l'envoie à Oobabooga et renvoie la réponse.
    C'est le cœur de l'application.
    """
    print(f"\n[INFO] Requête reçue pour /chat avec le message : '{user_input.message}'")

    # 1. Préparation de la conversation pour Oobabooga
    conversation_history = [
        {"role": "system", "content": LISA_SYSTEM_PROMPT}
    ]
    conversation_history.extend(user_input.history)
    conversation_history.append({"role": "user", "content": user_input.message})

    oobabooga_payload = {
        "model": "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf",  # Assurez-vous que ce nom de modèle est correct
        "messages": conversation_history,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False
    }

    print(f"[DEBUG] Payload préparé pour Oobabooga.")
    # Pour un débogage très verbeux, décommentez la ligne suivante :
    # print(oobabooga_payload)

    # 2. Appel à l'API d'Oobabooga
    try:
        # Définition explicite du timeout
        timeout = httpx.Timeout(1200.0, connect=20.0) # Timeout de 2 minutes au total
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            print(f"[DEBUG] Envoi de la requête POST à {OOBABOOGA_API_URL}/chat/completions")
            
            response = await client.post(
                f"{OOBABOOGA_API_URL}/chat/completions",
                json=oobabooga_payload
            )
            
            print(f"[DEBUG] Réponse reçue d'Oobabooga. Statut : {response.status_code}")
            
            # Lève une exception HTTP pour les codes d'erreur (4xx ou 5xx)
            response.raise_for_status()
            
            ai_response = response.json()
            lisa_message = ai_response.get("choices", [{}])[0].get("message", {}).get("content", "Désolée, je n'ai pas pu générer de réponse.")
            
            print(f"[INFO] Réponse de Lisa générée : '{lisa_message[:100]}...'")
            return {"reply": lisa_message}

    except Exception as exc:
        # 3. Gestion d'erreur améliorée pour nous donner tous les détails
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc} | Traceback: {full_traceback}"
        
        print(f"[ERREUR] Une erreur critique est survenue lors de la communication avec Oobabooga.")
        print(error_details)
        
        # On renvoie une erreur 502 claire à l'interface
        raise HTTPException(
            status_code=502, # 502 Bad Gateway
            detail={"error": "Erreur de communication avec le backend IA.", "details": str(exc)}
        )