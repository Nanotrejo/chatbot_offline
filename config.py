# config.py

DB_PATH = "chroma_db/"
DATA_DIR = "documents/"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
LLM_MODEL = "phi3:mini"
KWARGS = {"k": 15}
TEMPERATURE = 0.2
OLLAMA_BASE_URL = "http://localhost:11434"
