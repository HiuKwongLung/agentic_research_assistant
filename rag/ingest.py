import os
import re
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Paths ---
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
CSV_PATH = os.path.join(DOCS_DIR, "synthetic_knowledge_items.csv")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


# --- Text cleaning ---
def clean_text(text: str) -> str:
    """Remove markdown bold markers and extra whitespace"""
    text = re.sub(r"\*+", "", text)
    return text.strip()

# --- Document loading ---
 
def load_docs(csv_path: str) -> list[str]:
    """Load and format documents from the CSV file."""
    df = pd.read_csv(csv_path)
    docs = []
 
    for _, row in df.iterrows():
        topic = str(row.get("ki_topic", "")).strip()
        main_text = clean_text(str(row.get("ki_text", "")))
        alt_text = clean_text(str(row.get("alt_ki_text", "")))
 
        if main_text:
            docs.append(f"Topic: {topic}\n\nDetails:\n{main_text}")
 
        if alt_text:
            docs.append(f"Topic: {topic}\n\nDetails:\n{alt_text}")
 
    print(f"Loaded {len(docs)} documents from CSV.")
    return docs
 
 
# --- Chunking ---
 
def chunk_text(text: str, chunk_size: int = 128) -> list[str]:
    """Split text into word-based chunks."""
    words = text.split()
    chunks = []
 
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
 
    return chunks
 
 
def chunk_all_docs(docs: list[str], chunk_size: int = 128) -> list[str]:
    """Chunk all documents into smaller pieces."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc, chunk_size))
    print(f"Created {len(all_chunks)} chunks.")
    return all_chunks
 
 
# --- Embedding and indexing ---
 
def build_faiss_index(chunks: list[str]) -> None:
    """Embed chunks and save FAISS index and chunks to disk."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
 
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
 
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
 
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
 
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "faiss_index.index"))
    with open(os.path.join(VECTOR_STORE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
 
    print(f"FAISS index saved to: {VECTOR_STORE_DIR}")
 
 
# --- Main ---
 
if __name__ == "__main__":
    docs = load_docs(CSV_PATH)
    chunks = chunk_all_docs(docs)
    build_faiss_index(chunks)