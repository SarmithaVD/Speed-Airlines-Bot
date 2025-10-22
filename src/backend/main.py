import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field 
from contextlib import asynccontextmanager
from typing import List, Optional

from langchain_core.documents import Document
VECTOR_STORE_PATH = "../../faiss_index_jetblue" 
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
RETRIEVAL_K = 3

app_state = {} # global state managed by lifespan

# -- RAG FUNCTIONS --
async def retrieve_relevant_chunks(query: str) -> List[Document]:
    if "retriever" not in app_state or app_state["retriever"] is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized.") # 503 Service Unavailable
    try:
        print(f"Retrieving documents for query: {query}")
        # Use async invoke for retriever
        retrieved_docs = await app_state["retriever"].ainvoke(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving documents.")
    
# Formats retrieved documents into a single string for the LLM context.
def format_context(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# -- LOADING EMBEDDING MODELS -- Lifespan management - (Load ONLY retrieval models on startup)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Loading RAG retrieval components...")
    try:
        # 1. Load Embedding Model
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        app_state["embeddings"] = embeddings # Store in app_state
        print("Embedding model loaded.")

        # 2. Load Vector Store
        print(f"Loading FAISS vector store from: {VECTOR_STORE_PATH}...")
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(f"Vector store path not found: {VECTOR_STORE_PATH}")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, app_state["embeddings"], allow_dangerous_deserialization=True
        )
        app_state["vector_store"] = vector_store # Store in app_state
        print("Vector store loaded successfully.")

        # 3. Create Retriever
        print(f"Creating retriever (k={RETRIEVAL_K})...")
        retriever = app_state["vector_store"].as_retriever(search_kwargs={'k': RETRIEVAL_K})
        app_state["retriever"] = retriever # Store in app_state
        print("Retriever created.")

        print("RAG retrieval components loaded successfully.")

    except Exception as e:
        print(f"FATAL ERROR during startup: Could not load RAG retrieval components: {e}")
        # Ensure keys exist even on failure, but set to None
        app_state["embeddings"] = None
        app_state["vector_store"] = None
        app_state["retriever"] = None

    yield # Application runs here
    # --- Cleanup on shutdown ---
    print("Application shutdown.")
    app_state.clear() # Clear the state dictionary

# -- PYDANTIC MODELS FOR REQ/RES -- 

# user req containing message from chat
class UserMessageRequest(BaseModel):
    user_id: str = Field(..., example="user123", description="Unique identifier for the user")
    session_id: str = Field(..., example="sessionABC", description="Unique identifier for the chat session")
    message: str = Field(..., min_length=1, example="How much does it cost to fly with a pet?", description="The user's message")

# single retrieved chunk
class RetrievedChunk(BaseModel):
    source_file: Optional[str] = None
    header_path: Optional[str] = None
    content: str

# used to represent the final response to send
class FinalResponse(BaseModel):
    session_id: str
    original_query: str
    answer: str

# <<< --- Use this response model TEMPORARILY TEST SWAGGER--- >>>
class RetrievalResponse(BaseModel):
    """Temporary response model to return retrieved chunks."""
    session_id: str
    original_query: str
    retrieved_chunks: List[RetrievedChunk]
    status: str = "success"

# -- API ENDPOINTS -- 
app = FastAPI(
    title="RAG Retrieval API",
    lifespan=lifespan        
)

@app.post("/process_message/", response_model=RetrievalResponse)
async def process_user_message(request: UserMessageRequest):
    """
    # Need to have relevance check
    # Need to have transactional agent
    # Informational agent : Receives a user message, retrieves relevant context using RAG and has the retrieved chunks.
    # Need to combine and generate user friendly response
    """

    # -- INFORMATIONAL AGENT (RAG)
    if "retriever" not in app_state or app_state["retriever"] is None:
        raise HTTPException(status_code=503, detail="Retriever not available due to startup error.")

    try:
        # --- RAG Retrieval Step ---
        print(f"Retrieving documents for query: {request.message}")
        retrieved_docs: List[Document] = await app_state["retriever"].ainvoke(request.message)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # --- Contains the relevant chunks for generating response ---
        response_chunks = []
        for doc in retrieved_docs:
            header2 = doc.metadata.get('Header 2', '')
            header3 = doc.metadata.get('Header 3', '')
            header_path = f"{header2} > {header3}" if header3 else header2
            response_chunks.append(RetrievedChunk(
                source_file=doc.metadata.get('source_file', 'Unknown'), # Assuming chunker adds this metadata
                header_path=header_path if header_path else "Introduction",
                content=doc.page_content
            ))
        
        # REMOVE THIS AFTER LLM OUTPUT IS READY, THIS IS ONLY FOR TESTING TOP CHUNKS, ALSO CHANGE IN RESPONSEMODEL IN THE FUNCTION CALL
        return RetrievalResponse(
            session_id=request.session_id,
            original_query=request.message,
            retrieved_chunks=response_chunks
        )

        # --- Return final response after generated by llm ---
        """
        return FinalResponse(
            session_id=request.session_id,
            original_query=request.message,
            answer=final_answer 
        )
        """

    except Exception as e:
        print(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during RAG retrieval: {e}")
