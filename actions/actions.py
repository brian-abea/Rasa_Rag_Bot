import os
import logging
import csv
import re
import datetime
import requests
import psycopg2
from typing import Any, Text, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# RAG specific imports
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect # New library for language detection

# Rasa SDK imports
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted

# Load environment variables
load_dotenv()

# ====================================================================================
# Configuration
# ====================================================================================

# === General Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Database Credentials ===
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", None)
DB_PORT = os.getenv("DB_PORT", "5432")

# === RAG & LLM Configuration ===
SIMILARITY_THRESHOLD = 0.2
TOP_K = 10
MAX_CONTEXT_CHARS = 2000
CHROMA_PATH = "./chroma_persist"
COLLECTION_NAME = "rag_docs"
LOCAL_LLM_URL = "http://localhost:11434/api/generate"
MISSED_QUERIES_LOG = "missed_queries.csv"
SYNONYM_MAPPING = {
    "headquarters": ["office location", "main office", "HQ"],
    "location": ["address", "office"],
}

# === RAG Model Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(device)

# ====================================================================================
# Helper Functions
# ====================================================================================

def get_db_connection(max_retries: int = 3) -> Tuple[Optional[psycopg2.extensions.connection], Optional[psycopg2.extensions.cursor]]:
    """Establishes a connection to the PostgreSQL database with retries."""
    if not DB_PASSWORD:
        logger.warning("DB_PASSWORD is not set. Database actions will fail.")
        return None, None
    attempt = 0
    while attempt < max_retries:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            cursor = conn.cursor()
            logger.debug("Database connection established successfully.")
            return conn, cursor
        except Exception as e:
            attempt += 1
            logger.error(f"Database connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logger.error("Max retries reached. Could not connect to database.")
                return None, None
    return None, None

def get_user_language(text: str) -> str:
    """Detects the language of the user's input using the langdetect library."""
    try:
        return detect(text)
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "en"

def log_missed_query(query: str):
    """Logs a query to a CSV file if no results are found."""
    try:
        file_exists = os.path.isfile(MISSED_QUERIES_LOG)
        with open(MISSED_QUERIES_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "query"])
            writer.writerow([datetime.now().isoformat(), query])
        logger.info(f"Logged missed query: '{query}' to {MISSED_QUERIES_LOG}")
    except Exception as e:
        logger.error(f"Failed to log missed query: {e}")

def get_llm_prompt(question: str, context: str, user_lang: str) -> str:
    """Creates a dynamic prompt for the LLM based on user's language and tone."""
    if user_lang == "sw":
        tone_instruction = "Jibu kwa Kiswahili. Kuwa mfupi na mchangamfu."
    else:
        tone_instruction = "Answer in a friendly, concise, and direct manner."
    
    prompt = f"""
    You are a helpful assistant who provides brief but informative answers.
    Your goal is to answer the user's question accurately and concisely,
    using only the provided context. If the answer is not in the context,
    state that you don't have the information.

    Instructions:
    1. {tone_instruction}
    2. Start the response with the main point.
    3. Keep the answer to a maximum of 2-3 sentences.
    4. If there's relevant related information in the context,
        add it briefly after the main answer.
    5. Do not invent information not present in the context.

    Question: {question}
    Context: {context}

    Answer:
    """
    return prompt

def rephrase_with_local_llm(question: str, context: str, user_lang: str) -> str:
    """Send question + retrieved chunks to local LLM for rephrasing."""
    prompt = get_llm_prompt(question, context, user_lang)

    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "").strip()
        
        if re.search(r"i don't have the information|not in the context", response_text, re.IGNORECASE):
            return "🤔 I couldn't find a direct answer based on my knowledge."

        return response_text
    except Exception as e:
        logger.error(f"LLM rephrasing failed: {e}")
        return "⚠️ I had a problem processing your request."

def expand_query_with_synonyms(query: str) -> str:
    """Expands the query with synonyms from a predefined mapping."""
    expanded_query = query
    for key, synonyms in SYNONYM_MAPPING.items():
        if key in query.lower():
            expanded_query += " " + " ".join(synonyms)
    return expanded_query

def get_rerank_scores(query: str, documents: List[str]) -> List[Dict[str, Any]]:
    """Calculates re-ranking scores for documents based on a query."""
    if not documents:
        return []
    
    reranker_pairs = [[query, doc] for doc in documents]
    features = reranker_tokenizer(reranker_pairs, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        scores = reranker_model(**features).logits
    
    scores = scores.squeeze(dim=1).cpu().tolist()
    
    reranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return reranked_docs
    
def _normalize_phone_number(phone: str) -> str:
    """Normalizes phone number to match database format (starts with '25', no '+')."""
    cleaned_phone = re.sub(r'[\s+-]', '', phone)
    if cleaned_phone.startswith("0"):
        cleaned_phone = "254" + cleaned_phone[1:]
    elif not cleaned_phone.startswith("25"):
        cleaned_phone = "254" + cleaned_phone
    return cleaned_phone

# ====================================================================================
# Custom Rasa Actions
# ====================================================================================

class ActionRAGSearch(Action):
    def name(self) -> Text:
        # NOTE: You used two different names, "action_semantic_retrieval" and
        # "action_rag_search". I am keeping "action_rag_search" as it is more descriptive
        # and you should update your rules.yml and stories.yml to use this name.
        return "action_rag_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get("text")
        user_lang = get_user_language(user_query)

        try:
            # === Connect to ChromaDB ===
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function
            )
            
            # === Hybrid Search (Keyword + Semantic) ===
            expanded_query = expand_query_with_synonyms(user_query)
            
            keyword_results = collection.query(
                query_texts=[user_query],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )
            keyword_docs = keyword_results["documents"][0] if keyword_results["documents"] else []
            
            query_embedding = embedder.encode([expanded_query])[0].tolist()
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )
            semantic_docs = semantic_results["documents"][0] if semantic_results["documents"] else []
            
            all_documents = list(set(keyword_docs + semantic_docs))
            
            if not all_documents:
                dispatcher.utter_message("🤔 I couldn't find anything useful. Try asking another way.")
                log_missed_query(user_query)
                return []
            
            # === Contextual Re-ranking ===
            reranked_docs = get_rerank_scores(user_query, all_documents)
            reranked_filtered = [doc for doc, score in reranked_docs if score > -10]
            
            if not reranked_filtered:
                dispatcher.utter_message("🤔 I couldn't find anything useful.")
                log_missed_query(user_query)
                return []

            # === Gather top chunks for LLM ===
            context_chunks = []
            total_chars = 0
            for doc, score in reranked_docs:
                if total_chars + len(doc) > MAX_CONTEXT_CHARS:
                    break
                context_chunks.append(doc.strip())
                total_chars += len(doc)
            
            raw_context = "\n\n".join(context_chunks)

            # === Rephrase with Local LLM ===
            final_answer = rephrase_with_local_llm(user_query, raw_context, user_lang)

            # === Respond to user ===
            dispatcher.utter_message(text=final_answer)
            logger.info(f"✅ Retrieved chunks: {len(context_chunks)}")
            logger.info(f"Top Reranked Score: {reranked_docs[0][1]:.2f}")

        except Exception as e:
            logger.error(f"RAG Error: {e}")
            dispatcher.utter_message("⚠️ Sorry, I had trouble accessing my knowledge base.")
            log_missed_query(user_query)

        return []

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Handles session start by resetting slots and triggering welcome action."""
        return [
            SlotSet("phone_verified", False),
            SlotSet("mobile_no", None),
            SlotSet("first_name", None),
            SessionStarted(),
            ActionExecuted("action_welcome")
        ]

class ActionWelcome(Action):
    def name(self) -> Text:
        return "action_welcome"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Sends a welcome message with current time and language context."""
        language = tracker.get_slot("language") or "en"
        current_time = datetime.now().strftime("%H:%M %p")
        if language == "sw":
            dispatcher.utter_message(text=f"Karibu! Ni {current_time}. Jinsi gani ninaweza kukusaidia leo?")
        else:
            dispatcher.utter_message(text=f"Hello! It's {current_time}. These are your phone details?")
        return []

class ActionHelp(Action):
    def name(self) -> Text:
        return "action_help"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Provides a help message with available actions."""
        language = tracker.get_slot("language") or "en"
        if language == "sw":
            dispatcher.utter_message(text="Ninaweza kukusaidia na: \n- Kuangalia hali ya mali \n- Kulipa malipo \n- Maswali kuhusu simu iliyofungwa \n- Maelezo ya Javan")
        else:
            dispatcher.utter_message(response="utter_help")
        return []

class ActionFallback(Action):
    def name(self) -> Text:
        return "action_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Handles generic fallback for unrecognized inputs."""
        language = tracker.get_slot("language") or "en"
        if language == "sw":
            dispatcher.utter_message(text="Samahani, sijasikia hiyo. Tafadhali jaribu tena.")
        else:
            dispatcher.utter_message(text="Sorry, I didn't understand that. Can you please rephrase?")
        return []

class ActionMultilingualFallback(Action):
    def name(self) -> Text:
        return "action_multilingual_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Handles fallback with language detection attempt."""
        latest_message = tracker.latest_message.get("text", "").lower()
        language = tracker.get_slot("language") or "en"
        swahili_keywords = ["karibu", "samahani", "tafadhali", "sawa"]
        if any(keyword in latest_message for keyword in swahili_keywords) and language != "sw":
            dispatcher.utter_message(text="Tafadhali jaribu tena kwa Kiswahili au Kiingereza.")
            return [SlotSet("language", "sw")]
        else:
            dispatcher.utter_message(text="Sorry, I didn't understand. Please try again in a supported language (English or Swahili).")
        return []

class ActionSetLanguage(Action):
    def name(self) -> Text:
        return "action_set_language"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Sets the user's preferred language."""
        latest_message = tracker.latest_message.get("text", "").lower()
        if "swahili" in latest_message or "kiswahili" in latest_message:
            dispatcher.utter_message(text="Sawa, nimebadilisha lugha kuwa Kiswahili.")
            return [SlotSet("language", "sw")]
        elif "english" in latest_message:
            dispatcher.utter_message(text="Okay, I have switched the language to English.")
            return [SlotSet("language", "en")]
        else:
            dispatcher.utter_message(text="Please specify a language (e.g., English or Swahili).")
        return []

class ActionVerifyEmployee(Action):
    def name(self) -> Text:
        return "action_verify_employee"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Verifies if the user is a Watu employee."""
        employee_id = tracker.get_slot("employee_id") or tracker.latest_message.get("text")
        if not employee_id:
            dispatcher.utter_message(text="Please provide your employee ID for verification.")
            return [SlotSet("employee_id", None)]
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            dispatcher.utter_message(text="Sorry, there was an error verifying your employee status. Database connection failed.")
            return []
        try:
            query = "SELECT employee_name FROM watu_employees WHERE employee_id = %s"
            cursor.execute(query, (employee_id,))
            employee = cursor.fetchone()
            if employee:
                employee_name = employee[0]
                dispatcher.utter_message(text=f"Welcome, {employee_name}! You're verified as a Watu employee.")
                return [SlotSet("employee_verified", True), SlotSet("employee_name", employee_name)]
            else:
                dispatcher.utter_message(response="utter_not_employee")
                return [SlotSet("employee_verified", False)]
        except Exception as e:
            logger.error(f"Employee verification error: {e}")
            dispatcher.utter_message(text="Sorry, there was an error verifying your employee status.")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

class ActionVerifyPhone(Action):
    def name(self) -> Text:
        return "action_verify_phone"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Verifies the phone number from the 'mobile_no' slot."""
        phone_number = tracker.get_slot("mobile_no")
        logger.debug(f"Debug: phone_number from 'mobile_no' slot = {phone_number}")
        entities = tracker.latest_message.get("entities", [])
        mobile_no_entity = next((e["value"] for e in entities if e["entity"] == "mobile_no"), None)
        logger.debug(f"Debug: mobile_no entity extracted = {mobile_no_entity}")
        if not phone_number:
            if mobile_no_entity:
                phone_number = mobile_no_entity
                logger.debug(f"Debug: Using mobile_no entity as phone_number = {phone_number}")
            else:
                logger.debug("Debug: No phone number provided in slot or entity")
                dispatcher.utter_message(response="utter_invalid_phone_number")
                return [SlotSet("phone_verified", False)]
        normalized_phone = _normalize_phone_number(phone_number)
        logger.debug(f"Debug: Normalized phone number = {normalized_phone}")
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            dispatcher.utter_message(text="Sorry, there was an error verifying your phone number. Database connection failed.")
            return []
        try:
            query = "SELECT first_name FROM simu_active_loans WHERE mobile_no = %s"
            logger.debug(f"Debug: Running query with phone number: {normalized_phone}")
            cursor.execute(query, (normalized_phone,))
            user = cursor.fetchone()
            if user:
                first_name = user[0]
                logger.debug(f"Debug: Account found: {first_name}")
                dispatcher.utter_message(text=f"You are verified, {first_name}! How can I assist you?")
                return [SlotSet("phone_verified", True), SlotSet("first_name", first_name), SlotSet("mobile_no", normalized_phone)]
            else:
                logger.debug(f"Debug: No user found for phone number {normalized_phone}")
                dispatcher.utter_message(text="The phone number is not registered. Please contact Admin.")
                return [SlotSet("phone_verified", False)]
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            dispatcher.utter_message(text="Sorry, there was an error verifying your phone number.")
            return [SlotSet("phone_verified", False)]
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

class ActionCheckAssetStatus(Action):
    def name(self) -> Text:
        return "action_check_asset_status"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Checks asset status for a verified phone number."""
        if not tracker.get_slot("phone_verified"):
            dispatcher.utter_message(text="You need to verify your phone number first.")
            return []
        mobile_no = tracker.get_slot("mobile_no")
        logger.debug(f"Debug: Checking asset details for phone number: {mobile_no}")
        if not mobile_no:
            dispatcher.utter_message(response="utter_invalid_phone_number")
            return []
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            dispatcher.utter_message(text="Sorry, there was an error fetching asset details. Database connection failed.")
            return []
        try:
            query = """
                SELECT mobile_no, gender, country, imei
                FROM simu_active_loans WHERE mobile_no = %s
            """
            cursor.execute(query, (mobile_no,))
            asset = cursor.fetchone()
            if asset:
                language = tracker.get_slot("language") or "en"
                if language == "sw":
                    response = (
                        "📱 **Maelezo ya Mali:**\n"
                        f"- Namba ya Simu: {asset[0] or 'Hakuna'}\n"
                        f"- Jinsia: {asset[1] or 'Hakuna'}\n"
                        f"- Nchi: {asset[2] or 'Hakuna'}\n"
                        f"- IMEI: {asset[3] or 'Hakuna'}"
                    )
                else:
                    response = (
                        "📱 **Asset Details:**\n"
                        f"- Mobile Number: {asset[0] or 'N/A'}\n"
                        f"- Gender: {asset[1] or 'N/A'}\n"
                        f"- Country: {asset[2] or 'N/A'}\n"
                        f"- IMEI: {asset[3] or 'N/A'}"
                    )
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(text="No asset details found for the provided phone number.")
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            dispatcher.utter_message(text="Sorry, there was an error fetching asset details.")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return []

class ActionInquireSmartphoneFinancing(Action):
    def name(self) -> Text:
        return "action_inquire_smartphone_financing"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Fetches available smartphones from the database."""
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            dispatcher.utter_message(text="Sorry, there was an error fetching smartphone financing details. Database connection failed.")
            return []
        try:
            query = "SELECT model, storage, ram FROM available_smartphones"
            cursor.execute(query)
            phones = cursor.fetchall()
            if phones:
                language = tracker.get_slot("language") or "en"
                if language == "sw":
                    response = "📱 **Simu Zinazopatikana Kupitia Watu Simu:**\n"
                    for phone in phones:
                        response += f"- {phone[0]} ({phone[1]}/RAM {phone[2]})\n"
                else:
                    response = "📱 **Available Phones Through Watu Simu:**\n"
                    for phone in phones:
                        response += f"- {phone[0]} ({phone[1]}/RAM {phone[2]})\n"
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(response="utter_inquire_smartphone_financing")
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            dispatcher.utter_message(response="utter_inquire_smartphone_financing")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return []