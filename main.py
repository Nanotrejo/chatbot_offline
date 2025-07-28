# main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONSTANTES ---
DB_PATH = "./chroma_db"
DATA_DIR = "documents/"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3" # Modelo de Ollama que instalaste
KWARGS = {'k': 3} # Número de fragmentos relevantes a recuperar
TEMPERATURE = 0.1 # Temperatura para respuestas más precisas
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- INGESTA AUTOMÁTICA ---
def ensure_vector_db():
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        print("No existe la base de datos vectorial, creando...")
        # Eliminar la carpeta si existe (por si está corrupta o vacía)
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        # 1. Cargar el modelo de embeddings
        print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Modelo de embeddings cargado.")
        # 2. Buscar todos los archivos PDF y TXT en el directorio
        files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf') or f.lower().endswith('.txt')]
        if not files:
            print("❌ Error: No se encontraron archivos PDF o TXT en el directorio.")
            return
        print(f"Archivos encontrados: {files}")
        documents = []
        for file in files:
            print(f"Cargando documento desde: {file}...")
            file_path = os.path.join(DATA_DIR, file)
            if file.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs = [{"page_content": text, "metadata": {"source": file}}]
            else:
                continue
            if not docs:
                print(f"❌ Error: No se pudo cargar el documento o está vacío: {file}")
                continue
            documents.extend(docs)
            print(f"✅ Documento '{file}' cargado con {len(docs)} páginas o fragmentos.")
        print(f"Total de documentos/páginas cargados: {len(documents)}")
        # 3. Dividir los documentos en fragmentos (chunks)
        print("Dividiendo los documentos en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Documentos divididos en {len(chunks)} fragmentos.")
        # 4. Crear la base de datos vectorial y almacenar los fragmentos
        print("Creando y guardando la base de datos vectorial en ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print("✅ ¡Proceso completado! La base de datos vectorial está lista para ser usada.")
    else:
        print("La base de datos vectorial ya existe.")

# Ejecutar la ingesta automática antes de inicializar la API
ensure_vector_db()

# --- INICIALIZACIÓN DE COMPONENTES ---

# Inicializa la aplicación FastAPI
app = FastAPI(
    title="Chatbot API para Manual de Aplicación",
    description="API para interactuar con un chatbot que responde preguntas basadas en un manual.",
    version="1.0.0"
)

# Carga el modelo de embeddings (debe ser el mismo que en ingest.py)
print("Cargando modelo de embeddings...")
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)
print("✅ Modelo de embeddings cargado.")


# Carga la base de datos vectorial existente
print("Cargando base de datos vectorial...")
vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(
    search_type="similarity", # Tipo de búsqueda
    search_kwargs=KWARGS # Número de fragmentos relevantes a recuperar
)
print("✅ Base de datos vectorial cargada y lista.")

# Carga el modelo de lenguaje (LLM) desde Ollama
llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)
# llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

# Definición del Prompt (Plantilla de instrucciones para el LLM)
# ESTA ES LA PARTE CLAVE PARA CONTROLAR LA RESPUESTA
prompt_template = """
Eres un asistente experto que responde preguntas sobre un manual de aplicación.
Utiliza únicamente el siguiente CONTEXTO para responder la PREGUNTA del usuario.
Tu respuesta debe ser BREVE, DIRECTA y precisa.
Si la información no está en el contexto, responde amablemente: "Lo siento, no tengo información sobre eso en el manual."

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA CONCISA:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- CADENA DE PROCESAMIENTO (RAG CHAIN) ---

def format_docs(docs):
    """Función auxiliar para formatear los documentos recuperados en una sola cadena de texto."""
    return "\n\n".join(doc.page_content for doc in docs)

# Se define la cadena que orquesta todo el proceso
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- ENDPOINT DE LA API ---

class ChatRequest(BaseModel):
    question: str

@app.post("/chat", summary="Endpoint principal del chatbot")
def chat(request: ChatRequest):
    """
    Recibe una pregunta del usuario, busca en el manual y devuelve una respuesta generada por el LLM.
    """
    print(f"Pregunta recibida: {request.question}")
    response = rag_chain.invoke(request.question)
    print(f"Respuesta generada: {response}")
    return {"answer": response}


# --- EJECUCIÓN DEL SERVIDOR (para desarrollo) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)