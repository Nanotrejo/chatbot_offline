# main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama  # Importación actualizada
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
import uuid
import json
import re

# --- CONSTANTES ---
DB_PATH = "chroma_db/"
DATA_DIR = "documents/"
# EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
# LLM_MODEL = "llama3"  # Modelo de Ollama que instalaste
# LLM_MODEL = "gpt-oss:20b"
# LLM_MODEL = "gemma3:4b"
LLM_MODEL = "phi3:mini"
KWARGS = {"k": 15}  # Número de fragmentos relevantes a recuperar
TEMPERATURE = 0.2  # Temperatura para respuestas más precisas
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- INGESTA AUTOMÁTICA ---
def ensure_vector_db():
    print("No existe la base de datos vectorial, creando...")
    # Eliminar la carpeta si existe (por si está corrupta o vacía)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    # 1. Cargar el modelo de embeddings
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )
    print("✅ Modelo de embeddings cargado.")
    # 2. Buscar todos los archivos PDF y TXT en el directorio
    files = [
        f
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf") or f.lower().endswith(".txt")
    ]
    if not files:
        print("❌ Error: No se encontraron archivos PDF o TXT en el directorio.")
        return
    print(f"Archivos encontrados: {files}")
    documents = []
    for file in files:
        print(f"Cargando documento desde: {file}...")
        file_path = os.path.join(DATA_DIR, file)
        docs = []
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    print(f"❌ Error: El archivo '{file}' está vacío.")
                else:
                    docs = [Document(page_content=text, metadata={"source": file})]
            except Exception as e:
                print(f"❌ Error al cargar '{file}': {e}")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Documentos divididos en {len(chunks)} fragmentos.")
    # 4. Crear la base de datos vectorial y almacenar los fragmentos
    print("Creando y guardando la base de datos vectorial en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_PATH
    )
    print(
        "✅ ¡Proceso completado! La base de datos vectorial está lista para ser usada."
    )


# Ejecutar la ingesta automática antes de inicializar la API
# ensure_vector_db()

# --- INICIALIZACIÓN DE COMPONENTES ---

# Inicializa la aplicación FastAPI
app = FastAPI(
    title="Chatbot API para Manual de Aplicación",
    description="API para interactuar con un chatbot que responde preguntas basadas en un manual.",
    version="1.0.0",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto por los orígenes permitidos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo de embeddings (debe ser el mismo que en ingest.py)
print("Cargando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
)
print("✅ Modelo de embeddings cargado.")


# Carga la base de datos vectorial existente
print("Cargando base de datos vectorial...")
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_store.as_retriever(
    # search_type="similarity",  # Tipo de búsqueda
    search_type="mmr",  # Tipo de búsqueda
    # search_type="similarity_score_threshold",  # Tipo de búsqueda
    search_kwargs=KWARGS,  # Número de fragmentos relevantes a recuperar
)
print("✅ Base de datos vectorial cargada y lista.")

# Carga el modelo de lenguaje (LLM) desde Ollama
llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)

# Definición del Prompt (Plantilla de instrucciones para el LLM)
# ESTA ES LA PARTE CLAVE PARA CONTROLAR LA RESPUESTA
prompt_template = """
Eres un asistente experto en un manual de aplicación.

Responde la PREGUNTA del usuario usando **exclusivamente** el CONTEXTO proporcionado.
Tu respuesta debe ser **breve, directa y precisa**.
Incluye en la respuesta el **nombre del documento**, **número de página** y **subíndice** donde se encontró la información.
**No** digas frases como "según el manual" ni variantes similares.
Si la respuesta **no está en el CONTEXTO**, responde exactamente:
**"Lo siento, no tengo información sobre eso en el manual."**

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA CONCISA:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
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
    session_id: Optional[str] = None
    language: Optional[str] = "es"
    model: Optional[str] = None


# --- HISTORIAL DE SESIONES EN MEMORIA ---
session_histories = {}


@app.post("/chat", summary="Endpoint principal del chatbot")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    language = request.language or "es"
    model_name = request.model or LLM_MODEL  # Usa el modelo por defecto si no se especifica
    history = session_histories.get(session_id, [])
    history_context = "\n".join([
        f"Usuario: {q}\nBot: {a}" for q, a in history
    ])
    # history_context = ''
    pregunta_con_historial = f"{history_context}\nUsuario: {request.question}" if history_context else request.question
    print(f"Pregunta recibida: {request.question} (session_id={session_id}, language={language}, model={model_name})")

    # Mostrar los documentos recuperados por el retriever
    # docs_encontrados = retriever.get_relevant_documents(pregunta_con_historial)
    # print(f"Documentos recuperados por el retriever:\n{docs_encontrados}")

    prompt_template = f"""
    Eres un asistente experto en un manual técnico.

    Responde **únicamente** a la PREGUNTA usando el CONTEXTO dado.  
    No inventes información ni uses conocimientos externos.  
    Si el CONTEXTO no contiene la respuesta, indica claramente que no se encontró información.  
    Responde en el mismo idioma detectado de la pregunta o en el especificado: {language}.  

    Tu respuesta debe ser:
    - **Breve, directa y precisa**.
    - Incluir siempre: **nombre del documento**, **número de página** y **subíndice** exactos.
    - **No** uses frases como “según el manual” ni similares.

    ---
    CONTEXTO:
    {{context}}

    PREGUNTA:
    {{question}}

    RESPUESTA CONCISA EN {language}:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_local = ChatOllama(model=model_name, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)
    rag_chain_local = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_local
        | StrOutputParser()
    )
    response = rag_chain_local.invoke(pregunta_con_historial)
    print(f"Respuesta generada: {response}")
    history.append((request.question, response))
    session_histories[session_id] = history
    return {"answer": response, "session_id": session_id}


@app.post("/agent", summary="Modo agente: busca variable por contexto en model.txt y deja que el modelo razone y devuelva el nombre exacto")
def agent_search(request: ChatRequest):
    print(f"Petición de agente recibida: {request.question} (session_id={request.session_id}, language={request.language})")
    filename = "model.txt"
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return {"error": f"El archivo '{filename}' está vacío."}
        # Indexar el archivo como base de datos temporal
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        doc = Document(page_content=text, metadata={"source": filename})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents([doc])
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        # Buscar los fragmentos más relevantes
        docs_encontrados = retriever.invoke(request.question)
        if not docs_encontrados:
            return {"error": f"No se encontró información relevante en '{filename}'"}
        contexto = "\n".join(doc.page_content for doc in docs_encontrados)
        # Prompt para el agente
        prompt_agente = f"""
        Eres un agente experto en variables de control industrial.
        Tu tarea es identificar el nombre exacto de la variable y el valor que corresponde a la petición del usuario, usando únicamente el CONTEXTO proporcionado.
        Responde únicamente con un objeto JSON válido, sin explicaciones, sin formato markdown y sin ningún texto adicional. El JSON debe tener los campos:
        - variable: el nombre exacto de la variable (por ejemplo: ph.basic.cd.sp)
        - value: el valor que corresponde (true/false/número)
        - message: una explicación breve de la acción que se va a realizar, sin mencionar la variable.
        Si no existe la variable, responde exactamente: {{"variable": null, "value": null, "message": "No se encontró la variable solicitada."}}

        CONTEXTO:
        {contexto}

        PREGUNTA:
        {request.question}
        """
        llm_agente = ChatOllama(model=LLM_MODEL, temperature=0.1, base_url=OLLAMA_BASE_URL)
        respuesta = llm_agente.invoke(prompt_agente)
        # Extraer solo el JSON de la respuesta
        match = re.search(r'{[\s\S]*}', respuesta.content)
        if match:
            try:
                return json.loads(match.group())
            except Exception as e:
                return {"error": f"La IA no devolvió un JSON válido: {e}", "raw": match.group()}
        else:
            return {"error": "La IA no devolvió un JSON en la respuesta.", "raw": respuesta.content}
    except Exception as e:
        return {"error": f"No se pudo abrir el archivo '{filename}': {e}"}


# --- EJECUCIÓN DEL SERVIDOR (para desarrollo) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
