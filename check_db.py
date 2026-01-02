import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# --- Initialize ---
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Load your existing physics database
vector_store = Chroma(
    collection_name="physics_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

def inspect_database():
    print(f"--- Inspecting ChromaDB at {CHROMA_PATH} ---")
    
    # Retrieve all documents and metadata
    data = vector_store.get()
    metadatas = data.get('metadatas', [])
    documents = data.get('documents', [])
    
    if not metadatas:
        print("ALERT: No metadata found. Filtering will not work!")
        return

    # 1. Identify all unique topics
    unique_topics = set(meta.get('topic', 'MISSING') for meta in metadatas)
    print(f"Total chunks: {len(metadatas)}")
    print(f"Unique Topics Detected: {unique_topics}")
    print("-" * 40)

    # 2. Verify Sample Accuracy
    printed_topics = set()
    for i, meta in enumerate(metadatas):
        topic = meta.get('topic')
        if topic and topic not in printed_topics:
            print(f"TOPIC: [{topic}]")
            print(f"File: {meta.get('filename')}")
            print(f"Snippet: {documents[i][:150]}...")
            print("-" * 40)
            printed_topics.add(topic)

if __name__ == "__main__":
    inspect_database()