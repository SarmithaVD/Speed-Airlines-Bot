import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader # Or use MarkdownLoader if preferred
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv() 
SOURCE_DOCUMENT = "deduplicated_output.md" 
VECTOR_STORE_PATH = "faiss_index_jetblue"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

print(f"Loading document: {SOURCE_DOCUMENT}...")
loader = TextLoader(SOURCE_DOCUMENT, encoding='utf-8')
documents = loader.load()
markdown_text = "\n\n".join([doc.page_content for doc in documents])
print(f"Document loaded. Total characters: {len(markdown_text)}")

print("Splitting document based on Markdown headers...")
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False # Keep headers in content
)
splits = markdown_splitter.split_text(markdown_text)
print(f"Document split into {len(splits)} chunks.")
if not splits:
    print("Error: No chunks were created. Check the document content and header structure.")
    exit()


print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
# use local embedding model (requires sentence-transformers)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

# store embedding in vector db (faiss) 
print("Creating or loading FAISS vector store...")
if os.path.exists(VECTOR_STORE_PATH):
    # load existing index
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True) # Be cautious with allow_dangerous_deserialization in production
    print(f"Loaded existing vector store from {VECTOR_STORE_PATH}")
else:
    # create and save new index
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Created and saved new vector store to {VECTOR_STORE_PATH}")

print("Creating retriever...")
# Retrieve top 3 most relevant chunks
retriever = vector_store.as_retriever(search_kwargs={'k': 3})
print("Retriever created.")
print(retriever)
