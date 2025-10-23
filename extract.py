import os
import json
import datetime
import google.generativeai as genai
from pprint import pprint

# --- Schema for controlled LLM output ---
ANSWER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "policy_compliance": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["reply", "policy_compliance"],
}

def safe_json_loads(possible_json_str):
    """Safely parses JSON with fallback for incomplete/malformed output."""
    try:
        return json.loads(possible_json_str)
    except json.JSONDecodeError:
        possible_json_str = possible_json_str.strip()
        if not possible_json_str.endswith('}'):
            possible_json_str += '}'
        try:
            return json.loads(possible_json_str)
        except json.JSONDecodeError:
            return {
                "reply": f"An error occurred. The model returned malformed JSON: {possible_json_str}",
                "policy_compliance": False,
                "explanation": "Returned output was not valid JSON.",
            }

class AirlineQueryParser:
    """
    Unified parser for airline queries:
    - Extracts structured entities from user input
    - Generates policy-compliant replies using context
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)

        # Extraction model (for entities)
        self.extraction_model = genai.GenerativeModel(
            model_name, 
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        # Generation model (for answers)
        self.generation_model = genai.GenerativeModel(model_name)
        print(f"✅ LLM Service (Gemini) initialized with model {model_name}.")

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """
        You are an intelligent airline assistant. Extract the following from user's query as a valid JSON object:
        - pnr, flight_number, source, destination, seat_number, departure_time, arrival_time, passenger_name.
        Rules:
        1. Output MUST be a single valid JSON object.
        2. Use 'Current Datetime' for normalizing relative times.
        3. Missing entities should be `null`.
        4. Do not add extra fields.
        """

    def extract_details(self, user_query: str) -> dict:
        """Extracts structured entities from a natural language query."""
        current_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prompt = (
            f"{self.system_prompt}\n\n"
            f"**Context:**\nCurrent Datetime (UTC): {current_time_utc}\n\n"
            f"**New Query:**\n{user_query}\nJSON:\n"
        )
        try:
            response = self.extraction_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"🚨 ERROR during entity extraction: {e}")
            return {"error": str(e)}

    def generate_answer_from_context(self, user_query: str, context: str) -> dict:
        """Generates a policy-compliant reply based on RAG context."""
        if not self.generation_model:
            return {
                "reply": "Model not initialized. Cannot generate reply.",
                "policy_compliance": False,
                "explanation": "Model initialization failed."
            }
        if not context:
            return {
                "reply": "No policy info found for that request.",
                "policy_compliance": False,
                "explanation": "No RAG context provided."
            }

        prompt = f"User query: {user_query}\nContext: {context}\nGenerate a polite, concise, policy-compliant reply."
        gen_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ANSWER_RESPONSE_SCHEMA,
            max_output_tokens=512,
            temperature=0.7
        )

        try:
            response = self.generation_model.generate_content(contents=[prompt], generation_config=gen_config)
            if not response.candidates:
                return {"reply": "No candidates returned.", "policy_compliance": False, "explanation": "Empty candidates."}
            
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return {"reply": "No content parts found.", "policy_compliance": False, "explanation": "Missing content parts."}
            
            candidate_text = candidate.content.parts[0].text
            return safe_json_loads(candidate_text)
        except Exception as e:
            print("[Orchestrator] LLM generation failed.", e)
            return {"reply": "Error during content generation.", "policy_compliance": False, "explanation": str(e)}

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    API_KEY = "YOUR_GEMINI_API_KEY"  # Or read from env var
    parser = AirlineQueryParser(api_key=API_KEY, model_name="gemini-2.5-flash-lite")
    
    # Test extraction
    test_query = "Please provide details for flight QF12 from Sydney to Los Angeles. My PNR is ZT34WX and I'm seated at 14F."
    print("\n--- Extract Details ---")
    details = parser.extract_details(test_query)
    pprint(details)

    # Test answer generation
    rag_context = "Company policy states that only 2 pets are allowed per flight."
    print("\n--- Generate Answer ---")
    answer = parser.generate_answer_from_context("How many pets can I bring on a flight?", rag_context)
    pprint(answer)

