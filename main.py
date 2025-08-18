# main.py

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama  # Importación actualizada
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
import uuid
from db_utils import load_and_split_file, load_all_documents
from llm_utils import build_agent_prompt, extract_json_from_response, build_chat_prompt, build_manual_prompt, build_gpt_helper_prompt
from config import EMBEDDING_MODEL, LLM_MODEL, DATA_DIR, KWARGS, TEMPERATURE, OLLAMA_BASE_URL, DB_PATH, RETRIEVER_K, RETRIEVER_FETCH_K, TOP_K_RERANK

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
    # Usar función modular para cargar y dividir todos los documentos
    documents = load_all_documents(DATA_DIR)
    if not documents:
        print("❌ Error: No se encontraron archivos PDF o TXT en el directorio.")
        return
    print(f"✅ Documentos cargados y divididos en {len(documents)} fragmentos.")
    # 4. Crear la base de datos vectorial y almacenar los fragmentos
    print("Creando y guardando la base de datos vectorial en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=DB_PATH
    )
    print(
        "✅ ¡Proceso completado! La base de datos vectorial está lista para ser usada."
    )

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

# Variables globales para recursos compartidos
retriever = None
llm = None
prompt = None
rag_chain = None

def init_db():
    global retriever, llm, prompt, rag_chain
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
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )
    print("✅ Base de datos vectorial cargada y lista.")

    # Carga el modelo de lenguaje (LLM) desde Ollama
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)

    # Definición del Prompt (Plantilla de instrucciones para el LLM)
    prompt_template = build_manual_prompt(context="{context}", question="{question}")
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Se define la cadena que orquesta todo el proceso
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def format_docs(docs):
    """Función auxiliar para formatear los documentos recuperados en una sola cadena de texto."""
    return "\n\n".join(doc.page_content for doc in docs)

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
    model_name = request.model or LLM_MODEL
    use_history = getattr(request, "use_history", False)
    history = session_histories.get(session_id, [])
    if use_history and history:
        history_context = "\n".join([
            f"Usuario: {q}\nBot: {a}" for q, a in history
        ])
        pregunta_con_historial = f"{history_context}\nUsuario: {request.question}"
    else:
        pregunta_con_historial = f"Usuario: {request.question}"
    print(f"Pregunta recibida: {request.question} (session_id={session_id}, language={language}, model={model_name}, use_history={use_history})")
    prompt_text = build_chat_prompt(context="{context}", question="{question}", language=language)
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
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


@app.post("/chat_stream", summary="Endpoint de chatbot en streaming")
async def chat_stream(request: Request):
    body = await request.json()
    session_id = body.get("session_id") or str(uuid.uuid4())
    language = body.get("language") or "es"
    model_name = body.get("model") or LLM_MODEL
    use_history = body.get("use_history", True)
    question = body.get("question")
    history = session_histories.get(session_id, [])
    pregunta_con_historial = f"Usuario: {question}"
    if use_history and history:
        history_context = "\n".join([
            f"Usuario: {q}\nBot: {a}" for q, a in history
        ])
        pregunta_con_historial = f"{history_context}\nUsuario: {question}"
    print(f"Pregunta recibida: {question} (session_id={session_id}, language={language}, model={model_name}, use_history={use_history})")
    # prompt_text = build_chat_prompt("{context}", "{question}", language)
    prompt_text = build_gpt_helper_prompt(context="{context}", question="{question}", language=language)
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    llm_local = ChatOllama(model=model_name, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)
    rag_chain_local = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_local
        | StrOutputParser()
    )
    def stream_response():
        # Recupera los fragmentos relevantes de ChromaDB antes de generar la respuesta
        docs_relevantes = retriever.invoke(pregunta_con_historial)
        for doc in docs_relevantes:
            # print(f"- {getattr(doc, 'page_content', str(doc))[:120]}...")
            # crea un fichero md y añade el resultado
            with open("fragmentos.md", "a") as f:
                f.write(f"### Fragmento encontrado\n\n{getattr(doc, 'page_content', str(doc))}\n\n")
        # Genera la respuesta en streaming
        respuesta = ""
        for chunk in rag_chain_local.stream(pregunta_con_historial):
            text = getattr(chunk, "content", str(chunk))
            yield text
            respuesta += text
        history.append((question, respuesta))
        session_histories[session_id] = history
    return StreamingResponse(stream_response(), media_type="text/plain")


@app.post("/agent", summary="Modo agente: busca variable por contexto en model.txt y deja que el modelo razone y devuelva el nombre exacto y el valor")
def agent_search(request: ChatRequest):
    print(f"Petición de agente recibida: {request.question} (session_id={request.session_id}, language={request.language})")
    try:
        chunks = load_and_split_file("model.txt")
        if not chunks:
            return {"error": "El archivo 'model.txt' está vacío o no se pudo procesar."}
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever_agent = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        docs_encontrados = retriever_agent.invoke(request.question)
        if not docs_encontrados:
            return {"variable": None, "value": None, "message": "No se encontró la variable solicitada."}
        context = "\n".join(doc.page_content for doc in docs_encontrados)
        language = request.language or "es"
        prompt_agente = build_agent_prompt(context=context, question=request.question, language=language)
        llm_agente = ChatOllama(model=LLM_MODEL, temperature=0.1, base_url=OLLAMA_BASE_URL)
        respuesta = llm_agente.invoke(prompt_agente)
        return extract_json_from_response(respuesta.content)
    except Exception as e:
        return {"error": f"No se pudo procesar la petición: {e}"}


# --- EJECUCIÓN DEL SERVIDOR (para desarrollo) ---
if __name__ == "__main__":
    # Pregunta interactiva para ejecutar la ingesta automática
    respuesta = input("¿Deseas ejecutar la ingesta automática de documentos? (s/n): ").strip().lower()
    if respuesta == "s":
        ensure_vector_db()
    init_db()
    print("Inciado en http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
