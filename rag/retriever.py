import os 
import pickle
import faiss
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

# --- Paths ---

VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# --- Load vector store ---

def  _load_vector_store() -> tuple:
    """Load FAISS index, chunks, and embedding model from disk."""
    index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
    chunks_path = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "Vector store not found. Please run `python -m rag.ingest` first."
        )
    
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)

    return index, chunks, model

# --- Tool ---

@tool
def rag_search(query: str) -> str:
    """Search local documents using semantic similarity. 
    The local documents contain IT helpdesk knowledge items
    typical of a Fortune 500 company, covering topics like troubleshooting,
    software, hardware, networking, and common IT support procedures.
    Use this tool when the query is related to IT support or helpdesk topics.
    """
    try:
        results = _retrieve(query)

        if not results:
            return "No relevant documents found."
        
        return _format_results(results)
    
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        return f"RAG search failed: {str(e)}"
    
def _retrieve(query: str, k: int = 3) -> list[str]:
    """Embed the query and retrieve the top k matching chunks."""
    index, chunks, model = _load_vector_store()

    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k=k)

    return [chunks[i] for i in indices[0]]

def _format_results(results: list[str]) -> str:
    """Format retrieved chunks into a readable string for the agent."""
    formatted = []
    
    for i, chunk in enumerate(results, start=1):
        formatted.append(f"[{i}] {chunk}")
    
    return "\n\n".join(formatted)

# --- Quick test ---

if __name__ == "__main__":
    query = "How to reset a forgotten PIN?"
    print(f"Searching for: {query}\n")
    print(rag_search.invoke(query))