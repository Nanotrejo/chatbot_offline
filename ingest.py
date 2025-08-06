# ingest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import shutil

# --- CONSTANTES ---
DATA_DIR = "documents/"  # Directory to search for PDF and TXT files
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2" 

def main():
    """
    Función principal para cargar, dividir y almacenar todos los documentos PDF y TXT en una base de datos vectorial.
    """
    print("🚀 Empezando el proceso de ingesta de datos...")

    # Eliminar la carpeta de la base de datos si existe
    if os.path.exists(DB_PATH):
        print(f"Eliminando carpeta de base de datos existente: {DB_PATH}")
        shutil.rmtree(DB_PATH)

    # 1. Cargar el modelo de embeddings
    # Este modelo convierte el texto en vectores numéricos para la búsqueda semántica.
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    # Usamos 'cpu' explícitamente para asegurar la compatibilidad si no hay GPU
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
            # Simulamos la estructura de documentos de langchain
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
    # Esto es crucial para que el modelo pueda encontrar el contexto más relevante.
    print("Dividiendo los documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamaño de cada fragmento
        chunk_overlap=200 # Superposición para no perder contexto entre fragmentos
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Documentos divididos en {len(chunks)} fragmentos.")

    # 4. Crear la base de datos vectorial y almacenar los fragmentos
    # ChromaDB se encargará de guardar los vectores y permitir búsquedas eficientes.
    print("Creando y guardando la base de datos vectorial en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH # Guardamos la DB en el disco
    )
    print("✅ ¡Proceso completado! La base de datos vectorial está lista para ser usada.")


if __name__ == "__main__":
    main()