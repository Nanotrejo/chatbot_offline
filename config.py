# config.py

DB_PATH = "chroma_db/"
DATA_DIR = "documents/"

# Parámetros para fragmentación de documentos
CHUNK_SIZE = 1200  # Tamaño de cada fragmento
CHUNK_OVERLAP = 200  # Solapamiento entre fragmentos

# Retriever tuning
RETRIEVER_K = 5        # devolver final
RETRIEVER_FETCH_K = 20 # recuperar más para rerankear (usado por MMR/fetch)
TOP_K_RERANK = 5       # número final después de reranking

# Embedding/model selection
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3"
KWARGS = {"k": 12}
TEMPERATURE = 0.1
OLLAMA_BASE_URL = "http://localhost:11434"
