import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
import os
import datetime
from pprint import pprint
import requests

try:
    from extract import AirlineQueryParser
except ImportError:
    print("ERROR: 'extract.py' not found. Please make sure it's in the same directory.")
    exit()

TRANSACTIONAL_INTENTS = {
    'search_flight', 'book_flight', 'check_flight_reservation', 'cancel_flight', 
    'change_flight', 'get_refund', 'check_in', 'get_boarding_pass', 'change_seat', 
    'check_flight_status', 'purchase_flight_insurance', 'book_trip', 'cancel_trip', 
    'change_trip', 'check_arrival_time', 'check_departure_time', 'check_flight_offers', 
    'check_flight_prices', 'check_trip_details', 'check_trip_offers', 'check_trip_plan', 
    'check_trip_prices', 'choose_seat', 'print_boarding_pass', 'purchase_trip_insurance', 'search_trip'
}

INFORMATIONAL_INTENTS = {'ask_policy'}

CONVERSATIONAL_INTENTS = {'inform', 'greeting', 'irrelevant', 'end_conversation'}

# Loads an intent model and a LLM entity parser to process queries
class NLU_Service:
    def __init__(self, model_path, gemini_api_key):
        print(f"Initialising NLU Service...")
        print(f"Loading NLU model from {model_path}...")
        try:
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Model directory not found at {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            print(f"Intent Model loaded successfully onto device: {self.device}")
        except Exception as e:
            print(f"ERROR loading Intent model: {e}")
            self.model = None

        print("\nInitialising Entity Parser...")
        try:
            if not gemini_api_key:
                raise ValueError("Gemini API key is required for Entity Parser.")
            self.parser = AirlineQueryParser(api_key=gemini_api_key)
            print("Entity Parser (Gemini) initialised successfully.")
        except Exception as e:
            print(f"ERROR initializing Entity Parser: {e}")
            self.parser = None

    def process_query(self, query, threshold = 0.5):
        if not self.model or not self.parser:
            return {
                "error_message": "NLU service not fully initialized.",
                "transactional_intents": [],
                "informational_intents": [],
                "conversational_intents": []
            }
        
        # Intent Classification
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).squeeze()
        
        predicted_intents = []
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                predicted_intents.append(self.model.config.id2label[i])
        
        categorized_intents = {
            "transactional_intents": [],
            "informational_intents": [],
            "conversational_intents": []
        }
        
        if not predicted_intents:
             categorized_intents["conversational_intents"].append('irrelevant')
        else:
            for intent in predicted_intents:
                if intent in TRANSACTIONAL_INTENTS:
                    categorized_intents["transactional_intents"].append(intent)
                elif intent in INFORMATIONAL_INTENTS:
                    categorized_intents["informational_intents"].append(intent)
                elif intent in CONVERSATIONAL_INTENTS:
                    categorized_intents["conversational_intents"].append(intent)

        # 3. Handle default/fallback cases
        if not categorized_intents["transactional_intents"] and \
           not categorized_intents["informational_intents"] and \
           not categorized_intents["conversational_intents"]:
             categorized_intents["conversational_intents"].append('irrelevant')
             
        return categorized_intents

def call_rag_service(query: str, rag_api_url: str) -> Dict[str, Any]:
    """
    Calls the external RAG service API.
    """
    print(f"--- Calling RAG Service at {rag_api_url} ---")
    payload = {
        "user_id": "local_user",
        "session_id": "local_session",
        "message": query
    }
    try:
        response = requests.post(rag_api_url, json=payload, timeout=10) # 10-second timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx, 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"🚨 RAG API HTTP error: {http_err}")
        return {"error": f"HTTP error: {http_err.response.text}"}
    except requests.exceptions.ConnectionError as conn_err:
        print(f"🚨 RAG API Connection error: {conn_err}")
        return {"error": "Could not connect to RAG service."}
    except Exception as e:
        print(f"🚨 RAG API general error: {e}")
        return {"error": str(e)}

def handle_nlu_output(nlu: NLU_Service, nlu_result: Dict[str, Any], query: str, rag_api_url: str):
    """
    This function acts as the Orchestrator.
    It receives the categorized intents and decides which agent(s) to call.
    """
    print(f"--- \nNLU Output: {nlu_result}")
    
    transactional_intents = nlu_result.get('transactional_intents', [])
    informational_intents = nlu_result.get('informational_intents', [])
    conversational_intents = nlu_result.get('conversational_intents', [])
    
    # Flag to track if any agent was called
    did_action = False

    # --- 1. Transactional Agent ---
    if transactional_intents:
        print(f"\nOrchestrator Action: Routing to Transactional Agent for intent(s): {transactional_intents}")
        print("...Calling Entity Parser (Gemini)...")
        entities = nlu.parser.extract_details(query)
        print("Entity Output:")
        pprint(entities)
        # In a real app, you would now pass these entities to the agent
        did_action = True

    # --- 2. Informational Agent (RAG) ---
    if informational_intents:
        print(f"\nOrchestrator Action: Routing to Informational Agent (RAG) for intent(s): {informational_intents}")
        rag_response = call_rag_service(query, rag_api_url)
        print("RAG Service Output:")
        pprint(rag_response)
        did_action = True
    
    # --- 3. Conversational Agent ---
    if did_action:
        # If we took an action, we don't also need to handle simple conversation
        # (e.g., "hi, check my flight" -> we check the flight, we don't just say "hi")
        if 'end_conversation' in conversational_intents:
             print("\nOrchestrator Action: (Also detected end_conversation, closing session.)")
        return

    # If no other agents were called, handle simple conversation
    if 'greeting' in conversational_intents:
        print("\nOrchestrator Action: Greet user back and ask for their query.")
    elif 'end_conversation' in conversational_intents:
        print("\nOrchestrator Action: Say goodbye and end the session.")
    elif 'inform' in conversational_intents:
        print("\nOrchestrator Action: Calling Entity Parser (Gemini) for 'inform' intent...")
        entities = nlu.parser.extract_details(query)
        print("Entity Output (from 'inform'):")
        pprint(entities)
        print("Orchestrator Action: Populating session state and asking a clarifying question.")
    elif 'irrelevant' in conversational_intents:
        print("\nOrchestrator Action: Inform user of capabilities and ask for a new query.")
    else:
        print("\nOrchestrator Action: No strong intent detected. Asking for clarification.")