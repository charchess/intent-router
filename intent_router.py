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
APP_VERSION = "11.0" # Version avec mémoire proactive
LLM_BACKEND = os.getenv("LLM_BACKEND", "oobabooga")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE else logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
OOBABOOGA_MODEL_NAME = os.getenv("OOBABOOGA_MODEL_NAME", "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
N8N_MEMORY_WEBHOOK_URL = os.getenv("N8N_MEMORY_WEBHOOK_URL")
N8N_ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL")
N8N_RETRIEVAL_WEBHOOK_URL = os.getenv("N8N_RETRIEVAL_WEBHOOK_URL")

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

# NOUVEAU PROMPT pour l'analyseur de mémoire
MEMORY_ANALYZER_PROMPT = """Analyse la dernière phrase de l'utilisateur. Détermine si elle contient une information factuelle, une préférence, ou une donnée importante qui doit être mémorisée. Si oui, extrais cette information sous une forme déclarative claire.
Exemples:
- Utilisateur: "mon projet secret s'appelle Phénix" -> Fait: "Le projet secret de l'utilisateur s'appelle Phénix."
- Utilisateur: "je déteste les oignons" -> Fait: "L'utilisateur déteste les oignons."
- Utilisateur: "le code du garage est 1234" -> Fait: "Le code du garage est 1234."
- Utilisateur: "quelle heure est-il ?" -> Fait: null

Réponds UNIQUEMENT en JSON avec la structure {"fact_to_memorize": "Le fait extrait" ou null}.
"""


app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    message: str
    history: list = []

# =================================================================================
# FONCTIONS
# =================================================================================

async def analyze_and_memorize(user_message: str, background_tasks: BackgroundTasks):
    """Analyse le message et lance la mémorisation si nécessaire."""
    if not N8N_MEMORY_WEBHOOK_URL: return

    # On utilise Gemini pour cette tâche car il est rapide
    if not GEMINI_API_KEY: return
    
    # On construit un payload simple pour l'analyse
    payload = { "contents": [{"parts": [{"text": f"{MEMORY_ANALYZER_PROMPT}\n\nUtilisateur: \"{user_message}\""}]}] }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=20.0)
            response.raise_for_status()
            ai_response = response.json()
            
            # On parse la réponse JSON de l'analyseur
            analysis_text = ai_response["candidates"][0]["content"]["parts"][0]["text"]

            # On nettoie la réponse de Gemini pour extraire le JSON
            match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if not match:
                logging.warning(f"Aucun JSON trouvé dans la réponse de l'analyseur : {analysis_text}")
                return
                
            cleaned_text = match.group(0)
            analysis_json = json.loads(cleaned_text)

            fact = analysis_json.get("fact_to_memorize")
            if fact:
                logging.info(f"Fait détecté pour mémorisation : '{fact}'")
                # On lance l'écriture en tâche de fond
                action_payload = {"tool": "memory_store", "text": fact}
                background_tasks.add_task(execute_tool, action_payload)
    except Exception as exc:
        logging.error(f"Erreur durant l'analyse de mémoire : {exc}")

async def get_relevant_memories(query: str) -> str:
    # Appelle le workflow n8n pour récupérer les souvenirs pertinents.
    if not N8N_RETRIEVAL_WEBHOOK_URL:
        logging.warning("Récupération de mémoire désactivée (URL non configurée).")
        return ""

    logging.info(f"Récupération de la mémoire pour la question : '{query}'")
    try:
        payload = {"query": query}
        async with httpx.AsyncClient() as client:
            response = await client.post(N8N_RETRIEVAL_WEBHOOK_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            documents = data.get("documents", [[]])[0]
            if not documents:
                logging.info("Aucun souvenir pertinent trouvé.")
                return ""

            context_header = "Contexte pertinent de ta mémoire (utilise ces informations pour formuler ta réponse) :\n"
            formatted_memories = "\n".join([f"- {doc}" for doc in documents])
            logging.info(f"Souvenirs trouvés : {formatted_memories}")
            return context_header + formatted_memories
    except Exception as exc:
        logging.error(f"Erreur lors de la récupération de la mémoire : {exc}")
        return ""
    
async def get_relevant_memories(query: str) -> str:
    """Appelle le workflow n8n pour récupérer les souvenirs pertinents."""
    if not N8N_RETRIEVAL_WEBHOOK_URL:
        logging.warning("Récupération de mémoire désactivée (URL non configurée).")
        return ""

    logging.info(f"Récupération de la mémoire pour la question : '{query}'")
    try:
        payload = {"query": query}
        async with httpx.AsyncClient() as client:
            response = await client.post(N8N_RETRIEVAL_WEBHOOK_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            documents = data.get("documents", [[]])[0]
            if not documents:
                logging.info("Aucun souvenir pertinent trouvé.")
                return ""

            context_header = "Contexte pertinent de ta mémoire (utilise ces informations pour formuler ta réponse) :\n"
            formatted_memories = "\n".join([f"- {doc}" for doc in documents])
            logging.info(f"Souvenirs trouvés : {formatted_memories}")
            return context_header + formatted_memories
    except Exception as exc:
        logging.error(f"Erreur lors de la récupération de la mémoire : {exc}")
        return ""

async def get_reply_from_oobabooga(history_with_context: list):
    payload = {"model": OOBABOOGA_MODEL_NAME, "messages": history_with_context, "max_tokens": 500, "temperature": 0.7, "stream": False}
    timeout = httpx.Timeout(300.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=payload)
        response.raise_for_status()
        ai_response = response.json()
        return ai_response.get("choices", [{}])[0].get("message", {}).get("content", "Erreur: Réponse Oobabooga malformée.")

async def get_reply_from_gemini(user_input: UserInput, system_prompt_with_context: str):
    gemini_history = []
    for item in user_input.history:
        role = "model" if item["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": item["content"]}]})
    gemini_history.append({"role": "user", "parts": [{"text": user_input.message}]})
    
    payload = {"contents": gemini_history, "systemInstruction": {"parts": [{"text": system_prompt_with_context}]}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        ai_response = response.json()
        return ai_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Erreur: Réponse Gemini malformée.")

async def execute_tool(payload: dict):
    tool = payload.get("tool")
    webhook_url = WEBHOOKS.get(tool)
    if not webhook_url: return
    logging.info(f"Exécution de l'outil '{tool}'...")
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
    # ... (code de startup identique, ne change pas)
    pass

# =================================================================================
# ENDPOINT PRINCIPAL (/chat) - MIS À JOUR
# =================================================================================

@app.post("/chat")
async def handle_chat(user_input: UserInput, background_tasks: BackgroundTasks):
    logging.info(f"Requête reçue pour /chat. Message: '{user_input.message}'")

    # 1. On lance l'analyse de mémorisation en tâche de fond.
    #    Cela n'attendra pas la fin de l'analyse pour continuer.
    background_tasks.add_task(analyze_and_memorize, user_input.message, background_tasks)

    try:
        # 2. On récupère les souvenirs pertinents (RAG) comme avant.
        retrieved_context = await get_relevant_memories(user_input.message)
        
        # 3. On construit le prompt final pour l'IA qui va répondre.
        final_system_prompt = f"{LISA_SYSTEM_PROMPT}\n\n{retrieved_context}".strip()
        
        # 4. On appelle le LLM choisi pour générer la réponse à l'utilisateur.
        raw_reply = ""
        if LLM_BACKEND == "oobabooga":
            history_for_llm = [{"role": "system", "content": final_system_prompt}]
            history_for_llm.extend(user_input.history)
            history_for_llm.append({"role": "user", "content": user_input.message})
            raw_reply = await get_reply_from_oobabooga(history_for_llm)
        elif LLM_BACKEND == "gemini":
            raw_reply = await get_reply_from_gemini(user_input, final_system_prompt)
        else:
            raise HTTPException(status_code=500, detail="LLM_BACKEND non configuré correctement.")
            
        # 5. On analyse la réponse de l'IA pour d'éventuelles actions explicites (<|ACTION|>).
        action_regex = re.compile(r"<\|ACTION\|>([\s\S]*?)<\|\/ACTION\|>", re.DOTALL)
        match = action_regex.search(raw_reply)
        final_reply_text = raw_reply

        if match:
            json_content = match.group(1).strip()
            logging.info(f"Bloc d'action explicite détecté: {json_content}")
            final_reply_text = action_regex.sub("", raw_reply).strip() or "Action en cours, mon Roi."
            try:
                action_payload = json.loads(json_content)
                background_tasks.add_task(execute_tool, action_payload)
            except json.JSONDecodeError:
                logging.error("Erreur de parsing JSON dans le bloc d'action explicite.")
                final_reply_text = "J'ai tenté une action, mais son format était invalide."
        
        logging.info(f"Réponse finale envoyée à l'utilisateur: '{final_reply_text}'")
        return {"reply": final_reply_text}

    except Exception as exc:
        full_traceback = traceback.format_exc()
        error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
        logging.error(f"Une erreur critique est survenue dans handle_chat: {error_details}")
        logging.debug(f"Traceback complet : {full_traceback}")
        raise HTTPException(status_code=502, detail={"error": "Erreur lors du traitement de la requête.", "details": error_details})
    