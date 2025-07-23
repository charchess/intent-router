import os
import httpx
import traceback
import logging
import re
import json
import uuid
import time
import shlex
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase

# =================================================================================
# CONFIGURATION
# =================================================================================
APP_VERSION = "13.9.2"  # Version avec logging de démarrage restauré
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if (VERBOSE or DEBUG) else logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL")
OOBABOOGA_MODEL_NAME = os.getenv("OOBABOOGA_MODEL_NAME", "L3.3-70B-Magnum-Diamond-Q5_K_S.gguf")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
N8N_MEMORY_WEBHOOK_URL = os.getenv("N8N_MEMORY_WEBHOOK_URL")
N8N_ACTION_WEBHOOK_URL = os.getenv("N8N_ACTION_WEBHOOK_URL")
N8N_RETRIEVAL_WEBHOOK_URL = os.getenv("N8N_RETRIEVAL_WEBHOOK_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

WEBHOOKS = {"memory_store": N8N_MEMORY_WEBHOOK_URL, "home_assistant": N8N_ACTION_WEBHOOK_URL, "memory_store_error": N8N_ACTION_WEBHOOK_URL}
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

2.  **Outil `memory_store` :** Pour enregistrer une information importante dans la mémoire sémantique (RAG).
    -   **Paramètre requis :** `text`.
    -   **Exemple :** <|ACTION|>{"tool": "memory_store", "text": "Le mot de passe du Wi-Fi invité est 'BienvenueChezMoi'."}<|/ACTION|>

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

GRAPH_EXTRACTOR_PROMPT = """Ta tâche est d'analyser le texte de l'utilisateur et d'en extraire toutes les relations factuelles sous forme de triplets (sujet, relation, objet). Ignore les questions, les ordres et les phrases sans information factuelle. La relation doit être un verbe ou une propriété en MAJUSCULES_SNAKE_CASE. Réponds UNIQUEMENT avec un objet JSON contenant une liste de triplets. Si aucun triplet n'est trouvé, réponds avec une liste vide.

Exemples:
- Utilisateur: "mon chat s'appelle Midas et il aime les croquettes"
  -> {"triplets": [{"sujet": "chat", "relation": "A_POUR_NOM", "objet": "Midas"}, {"sujet": "Midas", "relation": "AIME", "objet": "croquettes"}]}
- Utilisateur: "Le serveur umi a l'IP 192.168.1.10"
  -> {"triplets": [{"sujet": "serveur umi", "relation": "A_POUR_IP", "objet": "192.168.1.10"}]}
- Utilisateur: "Je n'aime pas les lundis."
  -> {"triplets": [{"sujet": "Roi", "relation": "N_AIME_PAS", "objet": "lundis"}]}
- Utilisateur: "Allume la lumière s'il te plaît"
  -> {"triplets": []}
"""

app = FastAPI(title="HomeLab Intent Router", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class UserInput(BaseModel):
    message: str
    session_id: str | None = None

# =================================================================================
# DÉMARRAGE
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

# =================================================================================
# FONCTIONS
# =================================================================================
async def extract_and_store_graph_data(user_message: str, max_retries=3, retry_delay=1):
    """Extrait les relations du message et les stocke dans Neo4j, avec tentatives."""
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY]):
        logging.warning("Extraction graphe désactivée (configuration manquante).")
        return False

    logging.info(f"Début de l'extraction de graphe pour: '{user_message}'")

    # 1. Extraire les triplets avec Gemini
    payload = {"contents": [{"parts": [{"text": f"{GRAPH_EXTRACTOR_PROMPT}\n\nUtilisateur: \"{user_message}\""}]}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=20.0)
            response.raise_for_status()
            ai_response = response.json()
            analysis_text = ai_response["candidates"][0]["content"]["parts"][0]["text"]
            match = re.search(r'\{.*\}', analysis_text, re.DOTALL)

            if not match:
                logging.info("Aucun triplet trouvé par l'extracteur de graphe.")
                return False
            extracted_data = json.loads(match.group(0))
            triplets = extracted_data.get("triplets", [])

            if DEBUG:
                logging.debug(f"Triplets extraits : {triplets}")

            if not triplets:
                logging.info("La liste des triplets est vide.")
                return False

    except Exception as exc:
        logging.error(f"Erreur lors de l'extraction des triplets: {exc}")
        return False

    # 2. Se connecter à Neo4j et mettre à jour le graphe (avec tentatives)
    for attempt in range(max_retries):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                for triplet in triplets:
                    sujet = triplet.get('sujet')
                    relation = triplet.get('relation')
                    objet = triplet.get('objet')

                    if not all([sujet, relation, objet]):
                        if DEBUG:
                            logging.warning(f"Triplet incomplet ignoré: {triplet}")
                        continue

                    # Sécurisation de la relation
                    relation_safe = re.sub(r'[^A-Z0-9_]', '', relation.upper())
                    if not relation_safe:
                        if DEBUG:
                            logging.warning(f"Relation invalide ignorée: {relation}")
                        continue

                    logging.info(f"Ajout au graphe (tentative {attempt + 1}/{max_retries}): ({sujet})-[{relation_safe}]->({objet})")
                    query = (
                        f"MERGE (s:Entite {{nom: $sujet}}) "
                        f"MERGE (o:Entite {{nom: $objet}}) "
                        f"MERGE (s)-[r:{relation_safe}]->(o) "
                        "ON CREATE SET r.poids = 1, r.cree_le = datetime(), r.derniere_utilisation = datetime(), r.mis_a_jour_le = datetime() " # Ajouter le timestamp de creation
                        "ON MATCH SET r.poids = r.poids + 1, r.derniere_utilisation = datetime(), r.mis_a_jour_le = datetime()"
                    )
                    session.run(query, sujet=sujet, objet=objet)
            driver.close()
            return True  # Si tout se passe bien, on sort de la boucle
        except Exception as exc:
            logging.error(f"Erreur lors de la connexion ou de l'écriture dans Neo4j (tentative {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:  # Ne pas attendre à la dernière tentative
                logging.info(f"Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Échec persistant de l'écriture dans Neo4j après {max_retries} tentatives.")
                # Ici, vous pouvez ajouter du code pour informer l'utilisateur (via l'interface web, par exemple)
                # On renvoie un signal d'erreur à l'interface via un webhook n8n
                # La fonction execute_tool peut aussi gérer les erreurs de webhook
                action_payload = {
                    "tool": "memory_store_error",
                    "message": f"Impossible d'ajouter l'information au graphe (erreur Neo4j). Réessayez plus tard."
                }
                # On va ajouter un try except ici aussi
                try:
                    await execute_tool(action_payload)
                except Exception as exc:
                    logging.error(f"Erreur lors de l'envoi de l'erreur via le webhook: {exc}")

                return False  # Indique l'échec de l'opération


async def analyze_and_memorize(user_message: str, background_tasks: BackgroundTasks):
    if not N8N_MEMORY_WEBHOOK_URL:
        return

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
                action_payload = {"tool": "memory_store", "text": fact}
                background_tasks.add_task(execute_tool, action_payload)
    except Exception as exc:
        logging.error(f"Erreur durant l'analyse de mémoire RAG: {exc}")


async def get_relevant_memories(query: str) -> str:
    if not N8N_RETRIEVAL_WEBHOOK_URL:
        return ""

    logging.info(f"Récupération de la mémoire RAG pour: '{query}'")
    try:
        payload = {"query": query}
        async with httpx.AsyncClient() as client:
            response = await client.post(N8N_RETRIEVAL_WEBHOOK_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            documents = data.get("documents", [])
            if not documents:
                return ""

            context_header = "Contexte pertinent de ta mémoire (utilise ces informations pour formuler ta réponse) :\n"
            formatted_memories = "\n".join([f"- {doc}" for doc in documents])
            logging.info(f"Souvenirs RAG trouvés: {formatted_memories}")
            return context_header + formatted_memories
    except Exception as exc:
        logging.error(f"Erreur lors de la récupération de la mémoire RAG: {exc}")
        return ""


async def execute_tool(payload: dict):
    tool = payload.get("tool")
    webhook_url = WEBHOOKS.get(tool)
    if not webhook_url:
        return
    logging.info(f"Exécution de l'outil '{tool}'...")
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

    # Étape 1: Traitement des commandes internes
    if user_input.message.startswith("/cmd"):
        try:
            parts = shlex.split(user_input.message)
            command = parts[1]
            args = {k: v for k, v in (p.split('=', 1) for p in parts[2:] if '=' in p)}

            if command == "debug":
                debug_level = int(args.get("level", 1))
                logging.info(f"Activation du débogage (niveau {debug_level})...")
                # Ici, on pourrait ajuster le niveau de log global si nécessaire
                # logging.getLogger().setLevel(logging.DEBUG)
                final_reply_text = f"Mode Débogage activé (niveau {debug_level})."
            
            elif command == "version":
                final_reply_text = f"Version de l'application : {APP_VERSION}"
            
            else:
                final_reply_text = f"Commande inconnue : '{command}'"
        
        except Exception as e:
            final_reply_text = f"Erreur lors de l'exécution de la commande : {e}"

        logging.info(f"Réponse à la commande : '{final_reply_text}'")
        return {"reply": final_reply_text, "session_id": session_id}

    # Étape 2: Traitement normal du chat (si ce n'est pas une commande)
    try:
        # Lancement des routines de mémorisation en tâche de fond
        background_tasks.add_task(analyze_and_memorize, user_input.message, background_tasks)
        background_tasks.add_task(extract_and_store_graph_data, user_input.message)

        # Récupération du contexte RAG pour la réponse immédiate
        retrieved_context = await get_relevant_memories(user_input.message)
        
        # Construction du prompt final
        final_system_prompt = f"{LISA_SYSTEM_PROMPT}\n\n{retrieved_context}".strip()
        
        # Appel au LLM pour la génération de la réponse
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
        else:
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

        # Analyse de la réponse pour les actions <|ACTION|>
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