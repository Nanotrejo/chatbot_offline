# db_utils.py


from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_and_split_file(filename, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Carga un archivo TXT y lo divide en fragmentos tipo Document.
    """
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return []
        doc = Document(page_content=text, metadata={"source": filename})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents([doc])
    except Exception as e:
        print(f"Error al cargar '{filename}': {e}")
        return []


def load_all_documents(data_dir=DATA_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Carga y divide todos los archivos PDF y TXT en el directorio indicado.
    Devuelve una lista de fragmentos (Document).
    """
    files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith('.pdf') or f.lower().endswith('.txt')
    ]
    documents = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        if file.lower().endswith('.pdf'):
            print(f"Leyendo PDF: {file}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = splitter.split_documents(docs)
        elif file.lower().endswith('.txt'):
            print(f"Leyendo TXT: {file}")
            docs = load_and_split_file(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            continue
        if docs:
            documents.extend(docs)
    return documents
