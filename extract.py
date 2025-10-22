import os
import json
import datetime
from pprint import pprint
import google.generativeai as genai

class AirlineQueryParser:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        genai.configure(api_key=api_key)
        
        # Configure the model for JSON output
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json" 
        )
        
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash', 
            generation_config=self.generation_config
        )
        
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return """
        You are an intelligent airline assistant. Your sole responsibility is to extract specific pieces of information from a user's query and return them in a structured JSON format.

        **Entities to Extract:**
        - pnr: The 6-character alphanumeric Passenger Name Record (e.g., "AB12CD").
        - flight_number: The flight identifier (e.g., "BA249", "AI-101").
        - source: The origin city or airport (e.g., "London", "JFK").
        - destination: The destination city or airport (e.g., "Boston", "SFO").
        - seat_number: The passenger's seat (e.g., "22A", "14F").
        - departure_time: The normalized ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) departure time.
        - arrival_time: The normalized ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) arrival time.
        - passenger_name: The name of a passenger (e.g., "John Doe", "Ms. Smith").

        **Rules:**
        1.  Your output MUST be a single, valid JSON object.
        2.  Do NOT include any text, explanations, or markdown formatting (like ```json) before or after the JSON.
        3.  If an entity is not mentioned in the query, its value in the JSON MUST be `null`.
        4.  Do not add any fields that are not in the schema.
        5.  You will be provided with a 'Current Datetime (UTC)' in the 'Context' block.
        6.  You MUST use this 'Current Datetime' as a reference to normalize any relative times (like 'tomorrow', '5pm today', 'this Friday') into a full ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
        7.  If a time is partial (e.g., "tomorrow"), assume a logical time (e.g., 00:00:00Z for the start of the day).
        8.  Analyze the 'New Query' provided at the end and apply all rules.

        **JSON Schema to follow:**
        {
          "pnr": "string or null",
          "flight_number": "string or null",
          "source": "string or null",
          "destination": "string or null",
          "seat_number": "string or null",
          "departure_time": "string (ISO 8601) or null",
          "arrival_time": "string (ISO 8601) or null",
          "passenger_name": "string or null"
        }

        **Examples:**

        **Example 1:**
        Context:
        Current Datetime (UTC): 2025-10-23T15:00:00Z

        Query: "Hi, I need to check the status of my flight BA249 from London to Boston. My PNR is XJ45K. Is it arriving on time tomorrow?"
        JSON:
        {
          "pnr": "XJ45K",
          "flight_number": "BA249",
          "source": "London",
          "destination": "Boston",
          "seat_number": null,
          "departure_time": null,
          "arrival_time": "2025-10-24T00:00:00Z",
          "passenger_name": null
        }
        
        **Example 2:**
        Context:
        Current Datetime (UTC): 2025-10-23T15:00:00Z (Note: This is a Thursday)

        Query: "I'm flying from Dubai to NYC on EK201 this Friday at 8 PM."
        JSON:
        {
          "pnr": null,
          "flight_number": "EK201",
          "source": "Dubai",
          "destination": "NYC",
          "seat_number": null,
          "departure_time": "2025-10-24T20:00:00Z",
          "arrival_time": null,
          "passenger_name": null
        }

        **Example 3:**
        Context:
        Current Datetime (UTC): 2025-10-23T15:00:00Z

        Query: "Can I change my seat for John Doe on flight AI-101? I'm in 22A."
        JSON:
        {
          "pnr": null,
          "flight_number": "AI-101",
          "source": null,
          "destination": null,
          "seat_number": "22A",
          "departure_time": null,
          "arrival_time": null,
          "passenger_name": "John Doe"
        }
        """

    def extract_details(self, user_query):
        if not user_query:
            return {}
            
        current_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
        prompt = (
            f"{self.system_prompt}\n\n"
            f"**Context:**\n"
            f"Current Datetime (UTC): {current_time_utc}\n\n"
            f"**New Query:**\n{user_query}\nJSON:\n"
        )
        
        try:
            # Call the API
            response = self.model.generate_content(prompt)
            
            response_text = response.text
            
            # Parse the JSON string into a Python dictionary
            extracted_data = json.loads(response_text)
            
            return extracted_data
            
        except json.JSONDecodeError:
            print(f"ERROR: Model did not return valid JSON.\nResponse:\n{response_text}")
            return {"ERROR": "Failed to parse model output."}
        except Exception as e:
            print(f"An error occurred during API call: {e}")
            return {"ERROR": str(e)}

API_KEY = "AIzaSyC2uKCYvfg6YGU6WX1zWBotsyQuWqQnTyM"

if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set. Please set it and re-run.")

print("Initialising parser...")
parser = AirlineQueryParser(api_key=API_KEY)
print("Parser ready.\n")