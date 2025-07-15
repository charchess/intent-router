import os
import httpx
import traceback
import logging
import re
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIGURATION ---
APP_VERSION = "8.1.1"
LLM_BACKEND = os.getenv("LLM_BACKEND", "oobabooga")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
N8N_MEMORY_WEBHOOK_URL = os.getenv("N8N_MEMORY_WEBHOOK_URL")
N8N_ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL")
WEBHOOKS = {"memory_store": N8N_MEMORY_WEBHOOK_URL, "home_assistant": N8N_ACTION_WEBHOOK_URL}
LISA_SYSTEM_PROMPT = """Tu es Lisa... [votre prompt complet ici]..."""

app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    message: str
    history: list = []

# --- Fonctions ---
async def get_reply_from_oobabooga(history: list):
    payload = {"model": "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf", "messages": history, "max_tokens": 500, "temperature": 0.7, "stream": False}
    timeout = httpx.Timeout(300.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=payload)
        response.raise_for_status()
        ai_response = response.json()
        return ai_response.get("choices", [{}])[0].get("message", {}).get("content", "Erreur.")

async def execute_tool(payload: dict):
    tool = payload.get("tool")
    webhook_url = WEBHOOKS.get(tool)
    if not webhook_url: return
    logging.info(f"Exécution de l'outil '{tool}'...")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=30.0)
        logging.info(f"Webhook pour '{tool}' appelé avec succès.")
    except Exception as exc:
        logging.error(f"Erreur lors de l'appel du webhook pour l'outil '{tool}': {exc}")

# --- Endpoint ---
@app.post("/chat")
async def handle_chat(user_input: UserInput, background_tasks: BackgroundTasks):
    logging.info(f"Requête reçue: '{user_input.message}'")
    full_history = [{"role": "system", "content": LISA_SYSTEM_PROMPT}]
    full_history.extend(user_input.history)
    full_history.append({"role": "user", "content": user_input.message})

    try:
        raw_reply = await get_reply_from_oobabooga(full_history)
        
        action_regex = re.compile(r"<\|ACTION\|>([\s\S]*?)<\|\/ACTION\|>", re.DOTALL)
        match = action_regex.search(raw_reply)
        
        final_reply_text = raw_reply

        if match:
            json_content = match.group(1).strip()
            logging.info(f"Bloc d'action détecté: {json_content}")
            
            final_reply_text = action_regex.sub("", raw_reply).strip()
            if not final_reply_text:
                final_reply_text = "Bien sûr, mon Roi. C'est en cours."

            try:
                action_payload = json.loads(json_content)
                background_tasks.add_task(execute_tool, action_payload)
            except json.JSONDecodeError:
                logging.error("Erreur de parsing JSON dans le bloc d'action.")
                final_reply_text = "J'ai tenté une action, mais son format était invalide."
        
        logging.info(f"Réponse finale envoyée à l'utilisateur: '{final_reply_text}'")
        return {"reply": final_reply_text}

    except Exception as exc:
        logging.error(f"Erreur critique dans handle_chat: {exc}")
        raise HTTPException(status_code=502, detail="Erreur lors du traitement de la requête.")