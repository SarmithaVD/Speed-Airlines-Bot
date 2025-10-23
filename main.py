import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 
from contextlib import asynccontextmanager
from typing import List, Optional
from pprint import pprint

# --- RAG Imports (from Friend 1) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- NLU/LLM Imports (Your code) ---
from nlu_service import NLU_Service
from extract import AirlineQueryParser # Make sure this has your new logic

# --- Configuration ---
RAG_VECTOR_STORE_PATH = "faiss_index_jetblue" 
RAG_EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
RAG_RETRIEVAL_K = 3

NLU_MODEL_PATH = "./nlu_model"
GEMINI_API_KEY = "AIzaSyC2uKCYvfg6YGU6WX1zWBotsyQuWqQnTyM" # Your API Key

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set.")

app_state = {} 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all models on startup."""
    print("--- SERVER STARTUP ---")
    
    # 1. Load RAG components
    print("Loading RAG components...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=RAG_EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            RAG_VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        app_state["retriever"] = vector_store.as_retriever(search_kwargs={'k': RAG_RETRIEVAL_K})
        print("✅ RAG components loaded.")
    except Exception as e:
        print(f"🚨 FATAL RAG ERROR: {e}")
        app_state["retriever"] = None

    # 2. Load NLU (Intent) model
    # --- THIS IS THE FIX ---
    # NLU_Service needs the API key to initialize its internal parser
    app_state["nlu_service"] = NLU_Service(model_path=NLU_MODEL_PATH, gemini_api_key=GEMINI_API_KEY)
    # ----------------------
    if not app_state["nlu_service"].model:
        print("🚨 FATAL NLU ERROR: Intent model could not be loaded.")
        
    # 3. Load LLM (Gemini) service
    try:
        app_state["llm_parser"] = AirlineQueryParser(api_key=GEMINI_API_KEY)
        print("✅ LLM Service (Gemini) initialized successfully.")
    except Exception as e:
        print(f"🚨 FATAL LLM ERROR: {e}")
        app_state["llm_parser"] = None
        
    print("--- SERVER READY ---")
    yield
    print("--- SERVER SHUTDOWN ---")
    app_state.clear()

app = FastAPI(
    title="Unified Airline Bot API (Orchestrator)",
    description="This single API handles NLU, RAG, and LLM calls.",
    lifespan=lifespan
)

class UserQuery(BaseModel):
    query: str = Field(..., example="hi, what's your pet policy?")
    session_id: str = Field(..., example="session_12345")

class OrchestratorResponse(BaseModel):
    session_id: str
    query: str
    final_answer: str
    intent_analysis: dict
    entities: dict = None
    rag_context: list = None

@app.post("/chat", response_model=OrchestratorResponse)
async def chat_endpoint(request: UserQuery):
    
    nlu = app_state.get("nlu_service")
    llm = app_state.get("llm_parser")
    retriever = app_state.get("retriever")

    if not nlu or not llm or not retriever:
        raise HTTPException(status_code=503, detail="One or more services are not initialized.")

    print(f"\n[Orchestrator] Query: '{request.query}'")
    nlu_result = nlu.process_query(request.query)
    print(f"[Orchestrator] NLU Result: {nlu_result}")

    final_answer = ""
    entities = {}
    rag_chunks_for_response = []
    
    transactional = nlu_result.get('transactional_intents', [])
    informational = nlu_result.get('informational_intents', [])
    conversational = nlu_result.get('conversational_intents', [])

    if informational:
        print("[Orchestrator] Calling RAG retriever...")
        try:
            retrieved_docs: List[Document] = await retriever.ainvoke(request.query)
            context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            for doc in retrieved_docs:
                rag_chunks_for_response.append({"content": doc.page_content, "metadata": doc.metadata})

            print("[Orchestrator] Calling LLM to generate answer...")
            
            # --- THIS IS THE FIX ---
            # The function returns a dict, so we get the 'reply' field from it.
            answer_dict = llm.generate_answer_from_context(request.query, context_str)
            final_answer += answer_dict.get("reply", "I had trouble generating an answer.")
            # -----------------------
            
        except Exception as e:
            print(f"🚨 RAG/LLM Call Failed: {e}")
            final_answer += "I'm sorry, I couldn't connect to our policy database."

    if transactional:
        print(f"[Orchestrator] Calling LLM to extract entities...")
        try:
            entities = llm.extract_details(request.query)
            if not final_answer:
                final_answer += f"Understood. I've extracted this info: {entities}. (Ready for next step)"
        except Exception as e:
            print(f"🚨 LLM Entity Call Failed: {e}")
            final_answer += "I understood your request but had trouble processing the details."

    if not transactional and not informational:
        if 'greeting' in conversational:
            final_answer = "Hello! How can I help you?"
        elif 'end_conversation' in conversational:
            final_answer = "Goodbye!"
        elif 'inform' in conversational:
            entities = llm.extract_details(request.query)
            final_answer = f"Thanks for that information. How can I help? (Noted: {entities})"
        else:
            final_answer = "I'm sorry, I can only help with airline-related questions."

    print(final_answer)

    return OrchestratorResponse(
        session_id=request.session_id,
        query=request.query,
        final_answer=final_answer,
        intent_analysis=nlu_result,
        entities=entities,
        rag_context=rag_chunks_for_response
    )

if __name__ == "__main__":
    print("--- Starting Unified Server ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)