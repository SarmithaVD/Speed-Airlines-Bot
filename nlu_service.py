import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
import os
import datetime
from pprint import pprint

try:
    from extract import AirlineQueryParser
except ImportError:
    print("ERROR: 'extract.py' not found. Please make sure it's in the same directory.")
    exit()

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
            return {"intents": ["error"], "entities": {}, "response_type": "error", "error_message": "NLU service not fully initialized."}

        # Intent Classification
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).squeeze()
        
        predicted_intents = []
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                predicted_intents.append(self.model.config.id2label[i])
        
        conversational_intents = ['inform', 'greeting', 'irrelevant', 'end_conversation']
        actionable_intents = [i for i in predicted_intents if i not in conversational_intents]

        intents = []

        if actionable_intents:
            response_type = 'relevant'
            intents.extend(actionable_intents)
        if 'inform' in predicted_intents:
            if len(intents) >= 1:
                intents.append('inform')
            else:
                response_type = 'inform'
                intents = ['inform']
        elif 'greeting' in predicted_intents:
            response_type = 'greeting'
            intents = ['greeting']
        elif 'end_conversation' in predicted_intents:
            response_type = 'end_conversation' 
            intents = ['end_conversation']
        else:
            response_type = 'irrelevant'
            intents = ['irrelevant']

        # Extract important information / details
        entities = {}
        if response_type in ['relevant', 'inform']:
            entities = self.parser.extract_details(query)

        return {
            "intents": intents,
            "entities": entities,
            "response_type": response_type
        }
        
def handle_nlu_output(nlu_result):
    print(f"\nNLU Output:") 
    pprint(nlu_result)
    
    response_type = nlu_result.get('response_type')
    intents = nlu_result.get('intents', [])
    entities = nlu_result.get('entities', {})

    if response_type == 'greeting':
        print("Orchestrator Action: Greet user back and ask for their query.")
    
    elif response_type == 'end_conversation':
        print("Orchestrator Action: Say goodbye and end the session. (e.g., 'Thanks for chatting, goodbye!')")
    
    elif response_type == 'irrelevant':
        print("Orchestrator Action: Inform user of capabilities and ask for a new query. (e.g., 'I can only help with airline questions. Do you have another query?')")
    
    elif response_type == 'inform':
        print(f"Orchestrator Action: Populate session state with entities {entities} and ask a clarifying question.")
    
    elif response_type == 'relevant':
        print(f"Orchestrator Action: Route to agent(s) for intent(s): {intents} with entities: {entities}.")
        # Check for multi-intent
        if len(intents) > 1:
            print("  -> Detected multiple intents. Triggering parallel processing.")
    
    elif response_type == 'error':
        print("Orchestrator Action: Handle error gracefully. (e.g., 'Sorry, I'm having trouble right now.')")
    
    else:
        print(f"Orchestrator Action: Received unknown response_type '{response_type}'. Defaulting to irrelevant.")