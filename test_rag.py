import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document # Import Document for type hinting

# --- Configuration (Should match your indexing script) ---
load_dotenv() # Load environment variables if needed (e.g., for API keys, though not strictly needed here)

VECTOR_STORE_PATH = "faiss_index_jetblue" # Make sure this path is correct
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- 1. Load Embeddings ---
print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    # Ensure sentence-transformers is installed: pip install sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Make sure 'sentence-transformers' and 'torch' (or 'tensorflow') are installed.")
    exit()

# --- 2. Load the FAISS Vector Store ---
print(f"Loading FAISS vector store from: {VECTOR_STORE_PATH}...")
if not os.path.exists(VECTOR_STORE_PATH):
    print(f"Error: Vector store path not found: {VECTOR_STORE_PATH}")
    print("Please run the indexing script first to create the vector store.")
    exit()

try:
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True # Be cautious in production
    )
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    print("Ensure the index files exist and were created with the same LangChain/FAISS versions.")
    exit()

# --- 3. Create the Retriever ---
print("Creating retriever (k=3)...")
# Retrieve top 3 most relevant chunks (adjust 'k' as needed for testing)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})
print("Retriever created.")

# --- 4. Define Test Queries ---
test_queries = [
    "How much does it cost to fly with a pet?",
    "What are the size limits for a pet carrier?",
    "Can I change a Blue Basic fare?",
    "What benefits do Mosaic members get?",
    "Are checked bags free for JetBlue cardmembers?",
    "What is the policy for emotional support animals?",
    "Tell me about same-day switches.",
]

# --- 5. Run Queries and Print Results ---
print("\n--- Testing Retriever ---")
for query in test_queries:
    print(f"\nQuery: \"{query}\"")
    print("Retrieving relevant documents...")
    try:
        # Use invoke() to get relevant documents
        retrieved_docs: List[Document] = retriever.invoke(query)

        if not retrieved_docs:
            print("  -> No relevant documents found.")
        else:
            print(f"  -> Found {len(retrieved_docs)} relevant documents:")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source_file', 'Unknown') # Get source file if available
                header2 = doc.metadata.get('Header 2', '')
                header3 = doc.metadata.get('Header 3', '')
                header_path = f"{header2} > {header3}" if header3 else header2 # Construct header path

                print(f"    --- Document {i+1} ---")
                print(f"    Source: {source}")
                print(f"    Header: {header_path}")
                # Print the first ~300 characters of the content
                print(f"    Content: {doc.page_content[:300]}...")
                print("-" * 20) # Separator

    except Exception as e:
        print(f"  -> An error occurred during retrieval: {e}")

print("\n--- Testing Complete ---")