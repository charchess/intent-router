import os
import httpx
import traceback
import logging
import re
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =================================================================================
# CONFIGURATION
# =================================================================================
APP_VERSION = "9.0"
LLM_BACKEND = os.getenv("LLM_BACKEND", "oobabooga")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# -- Configs des Backends et Webhooks --
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
OOBABOOGA_MODEL_NAME = os.getenv("OOBABOOGA_MODEL_NAME", "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
N8N_MEMORY_WEBHOOK_URL = os.getenv("N8N_MEMORY_WEBHOOK_URL")
N8N_ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL")

WEBHOOKS = {"memory_store": N8N_MEMORY_WEBHOOK_URL, "home_assistant": N8N_ACTION_WEBHOOK_URL}
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
# FONCTIONS
# =================================================================================

async def get_reply_from_oobabooga(history: list):
    if not OOBABOOGA_API_URL:
        raise HTTPException(status_code=500, detail="OOBABOOGA_API_URL n'est pas configuré.")
    payload = {
        "model": OOBABOOGA_MODEL_NAME,
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

async def get_reply_from_gemini(user_input: UserInput):
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
        return ai_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Erreur: Réponse Gemini malformée.")

async def execute_tool(payload: dict):
    tool = payload.get("tool")
    webhook_url = WEBHOOKS.get(tool)
    if not webhook_url:
        logging.error(f"Outil inconnu demandé : '{tool}'")
        return
    logging.info(f"Exécution de l'outil '{tool}' en appelant le webhook...")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=30.0)
        logging.info(f"Webhook pour l'outil '{tool}' appelé avec succès.")
    except Exception as exc:
        logging.error(f"Erreur lors de l'appel du webhook pour l'outil '{tool}': {exc}")

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
async def handle_chat(user_input: UserInput, background_tasks: BackgroundTasks):
    logging.info(f"Requête reçue, backend: {LLM_BACKEND}, message: '{user_input.message}'")
    
    try:
        raw_reply = ""
        if LLM_BACKEND == "oobabooga":
            history_for_llm = [{"role": "system", "content": LISA_SYSTEM_PROMPT}]
            history_for_llm.extend(user_input.history)
            history_for_llm.append({"role": "user", "content": user_input.message})
            raw_reply = await get_reply_from_oobabooga(history_for_llm)
        elif LLM_BACKEND == "gemini":
            raw_reply = await get_reply_from_gemini(user_input)
        else:
            raise HTTPException(status_code=500, detail="LLM_BACKEND non configuré correctement.")

        action_regex = re.compile(r"<\|ACTION\|>([\s\S]*?)<\|\/ACTION\|>", re.DOTALL)
        match = action_regex.search(raw_reply)
        final_reply_text = raw_reply

        if match:
            json_content = match.group(1).strip()
            logging.info(f"Bloc d'action détecté: {json_content}")
            final_reply_text = action_regex.sub("", raw_reply).strip() or "Bien sûr, mon Roi. C'est en cours."
            try:
                action_payload = json.loads(json_content)
                background_tasks.add_task(execute_tool, action_payload)
            except json.JSONDecodeError:
                logging.error("Erreur de parsing JSON dans le bloc d'action.")
                final_reply_text = "J'ai tenté une action, mais son format était invalide."
        
        logging.info(f"Réponse finale envoyée: '{final_reply_text}'")
        return {"reply": final_reply_text}
    except Exception as exc:
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
        logging.error(f"Une erreur critique est survenue: {error_details}")
        logging.debug(f"Traceback complet : {full_traceback}")
        raise HTTPException(status_code=502, detail={"error": "Erreur de communication avec le backend IA.", "details": error_details})