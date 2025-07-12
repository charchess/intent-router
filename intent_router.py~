import os
from fastapi import FastAPI, HTTPException
import httpx

# --- Configuration chargée depuis les variables d'environnement ---
# On utilise os.getenv pour lire les variables.
# Le deuxième argument est une valeur par défaut si la variable n'est pas trouvée.
MEMORY_WEBHOOK_URL = os.getenv("N8N_MEMORY_WEBHOOK_URL")
ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL")

# Dictionnaire qui mappe les "outils" aux webhooks
# Ce dictionnaire est maintenant construit dynamiquement au démarrage.
WEBHOOKS = {
    "memory_store": MEMORY_WEBHOOK_URL,
    "home_assistant": ACTION_WEBHOOK_URL
}

app = FastAPI()

# --- Vérification au démarrage ---
@app.on_event("startup")
async def startup_event():
    # On vérifie que les URLs ont bien été configurées.
    # Si une URL manque, le service ne démarrera pas et affichera une erreur claire.
    if not MEMORY_WEBHOOK_URL:
        raise ValueError("FATAL: N8N_MEMORY_WEBHOOK_URL environment variable is not set.")
    if not ACTION_WEBHOOK_URL:
        raise ValueError("FATAL: N8N_ACTION_WEBHOOK_URL environment variable is not set.")
    print("Intent Router started successfully. Webhooks are configured.")


@app.post("/route")
async def route_intent(intent: dict):
    tool = intent.get("tool")
    webhook_url = WEBHOOKS.get(tool)

    if webhook_url:
        # On utilise httpx pour appeler le webhook de manière asynchrone
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(webhook_url, json=intent)
                response.raise_for_status() # Lève une exception pour les codes 4xx/5xx
                return {"status": "success", "tool": tool, "n8n_response_code": response.status_code}
            except httpx.RequestError as exc:
                raise HTTPException(status_code=503, detail=f"Error calling n8n webhook: {exc}")
    else:
        # Si l'outil n'est pas connu, on renvoie une erreur 404
        raise HTTPException(status_code=404, detail=f"Tool '{tool}' not found.")