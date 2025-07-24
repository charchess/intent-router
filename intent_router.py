import os
import httpx
import traceback
import logging
import re
import json
import uuid
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions

# =================================================================================
# CONFIGURATION
# =================================================================================
APP_VERSION = "13.11.2"
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if (VERBOSE or DEBUG) else logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# # # APIs Externes & Webhooks
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
OOBABOOGA_MODEL_NAME = os.getenv("OOBABOOGA_MODEL_NAME", "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf")
N8N_ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL")
WEBHOOKS = {"home_assistant": N8N_ACTION_WEBHOOK_URL, "memory_store_error": N8N_ACTION_WEBHOOK_URL}

# # # Bases de Données
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# # # Prompts
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

Pour toutes les autres conversations, réponds normalement en langage naturel.
"""

MEMORY_ANALYZER_PROMPT = """Analyse la dernière phrase de l'utilisateur. Détermine si elle contient une information factuelle, une préférence, ou une donnée importante qui doit être mémorisée pour une recherche textuelle future. Si oui, extrais cette information sous une forme déclarative claire.
Exemples:
- Utilisateur: "mon projet secret s'appelle Phénix" -> Fait: "Le projet secret de l'utilisateur s'appelle Phénix."
- Utilisateur: "je déteste les oignons" -> Fait: "L'utilisateur déteste les oignons."
- Utilisateur: "le code du garage est 1234" -> Fait: "Le code du garage est 1234."
- Utilisateur: "quelle heure est-il ?" -> Fait: null

Réponds UNIQUEMENT en JSON avec la structure {"fact_to_memorize": "Le fait extrait" ou null}.
"""

GRAPH_EXTRACTOR_PROMPT = """Ta tâche est d'analyser le texte de l'utilisateur pour en extraire et déduire toutes les relations factuelles sous forme de triplets (sujet, relation, objet). Ignore les questions, ordres et phrases sans information.

**Règles d'Extraction :**
1.  La relation doit être un verbe ou une propriété en MAJUSCULES_SNAKE_CASE.
2.  Le sujet doit être aussi précis que possible.

**Règles d'Inférence et de Déduction :**
1.  **Possession :** Si l'utilisateur dit "mon/ma/mes" ou "j'ai", la relation doit être `APPARTIENT_A` et l'objet doit être "Roi".
2.  **Décomposition :** Si une phrase contient plusieurs informations sur plusieurs sujets, crée un triplet pour chaque fait distinct.
3.  **Inférence de Type :** Si on parle de "chattes", déduis que chaque sujet mentionné `EST_UN` "chat" et `A_POUR_GENRE` "féminin". Si on parle de "serveur", il `EST_UN` "serveur".

**Format de Réponse :**
Réponds UNIQUEMENT avec un objet JSON contenant une liste de triplets. Si aucun n'est trouvé, la liste doit être vide.

**Exemples :**
- Utilisateur: "Praline et Vanille sont mes chattes."
  -> {
       "triplets": [
         {"sujet": "Praline", "relation": "EST_UN", "objet": "chat"},
         {"sujet": "Praline", "relation": "A_POUR_GENRE", "objet": "féminin"},
         {"sujet": "Praline", "relation": "APPARTIENT_A", "objet": "Roi"},
         {"sujet": "Vanille", "relation": "EST_UN", "objet": "chat"},
         {"sujet": "Vanille", "relation": "A_POUR_GENRE", "objet": "féminin"},
         {"sujet": "Vanille", "relation": "APPARTIENT_A", "objet": "Roi"}
       ]
     }
- Utilisateur: "Le serveur umi a l'IP 192.168.1.10"
  -> {
       "triplets": [
         {"sujet": "serveur umi", "relation": "EST_UN", "objet": "serveur"},
         {"sujet": "serveur umi", "relation": "A_POUR_IP", "objet": "192.168.1.10"}
       ]
     }
- Utilisateur: "Allume la lumière s'il te plaît"
  -> {"triplets": []}
"""

# =================================================================================
# ADAPTER POUR LA MÉMOIRE SÉMANTIQUE (VECTORIELLE)
# =================================================================================
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

class VectorStoreAdapter:
    # # L'interface qui définit le contrat.
    def add(self, text: str, collection_name: str, metadata: dict = None):
        raise NotImplementedError

    def query(self, query_text: str, collection_name: str, n_results: int = 3) -> list:
        raise NotImplementedError

class ChromaDBAdapter(VectorStoreAdapter):
    # # L'implémentation spécifique pour ChromaDB.
    def __init__(self, host: str, port: int):
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            logging.info(f"Connecté à ChromaDB à l'adresse {host}:{port}")
        except Exception as e:
            logging.error(f"Impossible de se connecter à ChromaDB: {e}")
            self.client = None

    def add(self, text: str, collection_name: str, metadata: dict = None):
        if not self.client:
            logging.error("Client ChromaDB non initialisé. Annulation de l'ajout.")
            return
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            doc_id = str(uuid.uuid4())
            collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            logging.info(f"Texte ajouté à la collection ChromaDB '{collection_name}'")
        except Exception as e:
            logging.error(f"Erreur lors de l'ajout à ChromaDB: {e}")

    def query(self, query_text: str, collection_name: str, n_results: int = 3) -> list:
        if not self.client:
            logging.error("Client ChromaDB non initialisé. Annulation de la recherche.")
            return []
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results.get('documents', [[]])[0]
        except Exception as e:
            logging.error(f"Erreur lors de la recherche dans ChromaDB: {e}")
            return []

# # Instanciation de notre adaptateur. Pour passer à Qdrant, on changera juste cette ligne.
vector_store = ChromaDBAdapter(host=CHROMA_HOST, port=CHROMA_PORT)

# =================================================================================
# INITIALISATION DE L'APPLICATION FASTAPI
# =================================================================================
app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lisa.truxonline.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UserInput(BaseModel):
    message: str
    session_id: str | None = None

@app.on_event("startup")
async def startup_event():
    logging.info(f"--- Intent Router - Version {APP_VERSION} ---")
    logging.info(f"Backend LLM sélectionné : {LLM_BACKEND}")
    if LLM_BACKEND == "oobabooga" and not OOBABOOGA_API_URL:
        logging.error("AVERTISSEMENT: OOBABOOGA_API_URL n'est pas définie !")
    if LLM_BACKEND == "gemini" and not GEMINI_API_KEY:
        logging.error("AVERTISSEMENT: GEMINI_API_KEY n'est pas définie !")
    logging.info("---------------------------------------------")

# =================================================================================
# FONCTIONS DE GESTION DES MÉMOIRES
# =================================================================================
async def extract_and_store_graph_data(user_message: str, max_retries=3, retry_delay=1):
    # # Extrait les relations du message et les stocke dans Neo4j.
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY]):
        logging.warning("Extraction graphe désactivée (configuration manquante).")
        return False
    # ... (le reste du code de cette fonction est inchangé)

async def analyze_and_memorize(user_message: str):
    # # Analyse le message pour un fait et le stocke dans la mémoire sémantique via l'adapter.
    if not GEMINI_API_KEY:
        return

    payload = {"contents": [{"parts": [{"text": f"{MEMORY_ANALYZER_PROMPT}\n\nUtilisateur: \"{user_message}\""}]}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=20.0)
            response.raise_for_status()
            ai_response = response.json()
            analysis_text = ai_response["candidates"][0]["content"]["parts"][0]["text"]

            match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if not match:
                logging.info("Aucun JSON trouvé dans la réponse de l'analyseur.")
                return

            cleaned_text = match.group(0)
            analysis_json = json.loads(cleaned_text)
            fact = analysis_json.get("fact_to_memorize")
            
            if fact:
                logging.info(f"Fait détecté pour mémorisation RAG: '{fact}'")
                metadata = {"source": "conversation", "timestamp": str(datetime.now())}
                vector_store.add(text=fact, collection_name="global_facts", metadata=metadata)

    except Exception as exc:
        logging.error(f"Erreur durant l'analyse de mémoire RAG: {exc}")

async def get_relevant_memories(query: str) -> str:
    # # Interroge la mémoire sémantique via l'adapter pour trouver un contexte pertinent.
    logging.info(f"Récupération de la mémoire sémantique pour: '{query}'")
    documents = vector_store.query(query_text=query, collection_name="global_facts", n_results=3)

    if not documents:
        return ""

    context_header = "Contexte pertinent de ta mémoire (utilise ces informations pour formuler ta réponse) :\n"
    formatted_memories = "\n".join([f"- {doc}" for doc in documents])
    logging.info(f"Souvenirs RAG trouvés: {formatted_memories}")
    return context_header + formatted_memories

async def execute_tool(payload: dict):
    # # Exécute une action via un webhook n8n (gardé pour les actions externes comme Home Assistant).
    tool = payload.get("tool")
    webhook_url = WEBHOOKS.get(tool)
    if not webhook_url:
        logging.warning(f"Aucun webhook trouvé pour l'outil '{tool}'")
        return
        
    logging.info(f"Exécution de l'outil '{tool}' via webhook...")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=30.0)
        logging.info(f"Webhook pour l'outil '{tool}' appelé avec succès.")
    except Exception as exc:
        logging.error(f"Erreur lors de l'appel du webhook pour l'outil '{tool}': {exc}")

# =================================================================================
# ENDPOINT PRINCIPAL (/chat)
# =================================================================================
@app.post("/chat")
async def handle_chat(user_input: UserInput, background_tasks: BackgroundTasks):
    logging.info(f"Requête reçue. Message: '{user_input.message}'")
    session_id = user_input.session_id or str(uuid.uuid4())

    # # Étape 1: Traitement des commandes internes
    if user_input.message.lower().startswith("/version"):
        final_reply_text = f"Version de l'application : {APP_VERSION}"
        logging.info(f"Réponse à la commande : '{final_reply_text}'")
        return {"reply": final_reply_text, "session_id": session_id}

    elif user_input.message.lower().startswith("/debug"):
        try:
            parts = user_input.message.split("=")
            debug_level = int(parts[1]) if len(parts) > 1 else 1
            logging.info(f"Activation du débogage (niveau {debug_level})...")
            final_reply_text = f"Mode Débogage activé (niveau {debug_level})."
            logging.info(f"Réponse à la commande : '{final_reply_text}'")
            return {"reply": final_reply_text, "session_id": session_id}
        except Exception as e:
            logging.error(f"Erreur lors de l'activation du mode debug : {e}")
            final_reply_text = "Erreur lors de l'activation du mode débogage."
            return {"reply": final_reply_text, "session_id": session_id}
            
    # # Étape 2: Traitement normal du chat
    else:
        try:
            # # Lancement des routines de mémorisation.
            await analyze_and_memorize(user_input.message)
            background_tasks.add_task(extract_and_store_graph_data, user_input.message)

            # # Récupération du contexte RAG pour la réponse immédiate
            retrieved_context = await get_relevant_memories(user_input.message)
            
            # # Construction du prompt final
            final_system_prompt = f"{LISA_SYSTEM_PROMPT}\n\n{retrieved_context}".strip()
            
            # # Appel au LLM pour la génération de la réponse
            raw_reply = ""
            if LLM_BACKEND == "gemini":
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": user_input.message}]}],
                    "systemInstruction": {"parts": [{"text": final_system_prompt}]}
                }
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    ai_response = response.json()
                    raw_reply = ai_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            else: # # oobabooga
                payload = {
                    "model": OOBABOOGA_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": final_system_prompt},
                        {"role": "user", "content": user_input.message}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(f"{OOBABOOGA_API_URL}/chat/completions", json=payload)
                    response.raise_for_status()
                    ai_response = response.json()
                    raw_reply = ai_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # # Analyse de la réponse pour les actions <|ACTION|>
            action_regex = re.compile(r"<\|ACTION\|>([\s\S]*?)<\|\/ACTION\|>", re.DOTALL)
            match = action_regex.search(raw_reply)
            final_reply_text = raw_reply.strip()

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
            return {"reply": final_reply_text, "session_id": session_id}

        except Exception as exc:
            full_traceback = traceback.format_exc()
            error_details = f"Exception: {type(exc).__name__} | Message: {exc}"
            logging.error(f"Une erreur critique est survenue dans handle_chat: {error_details}")
            logging.debug(f"Traceback complet : {full_traceback}")
            raise HTTPException(status_code=502, detail={"error": "Erreur lors du traitement de la requête.", "details": error_details})
        
