# 🤖 Chatbot Offline

Este proyecto es una API basada en FastAPI que permite consultar manuales técnicos usando modelos LLM locales vía Ollama y búsqueda semántica con ChromaDB.

Novedades incluidas en esta rama:
- Parámetros de chunking centralizados: `CHUNK_SIZE` y `CHUNK_OVERLAP` en `config.py`.
- Historial opcional por sesión (`use_history`) en los endpoints de chat.
- Reranking opcional con LLM controlado por `ENABLE_RERANK` (configurable en `config.py`).
- Endpoint `/chat_stream` que devuelve respuesta en streaming y registra los fragmentos recuperados en `fragmentos.md` para auditoría.
- Modo `agent` (`/agent`) para extraer nombre/valor de variables desde `model.txt` y devolver un JSON estructurado.
## 🛠️ Requisitos

- 🐍 Python 3.10+
- 🦙 Ollama instalado y corriendo en tu máquina
- 🧠 Modelos LLM compatibles descargados en Ollama (ejemplo: llama3, gemma3:1b, openai/gpt-oss-120b, etc.)
- 📦 Dependencias Python (ver `requirements.txt`)

## 🚀 Instalación

1. **Clona el repositorio:**
   ```bash
   git clone <URL_DEL_REPO>
   cd chatbot_offline
   ```

2. **Crea y activa un entorno virtual (recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Descarga los modelos en Ollama:**
   Por ejemplo:
   ```bash
   ollama pull llama3
   ollama pull gemma3:1b
   ollama pull openai/gpt-oss-120b
   ```

5. **Coloca los manuales en la carpeta `documents/`:**
   - 📄 Archivos PDF y TXT serán indexados automáticamente.

6. **Genera la base de datos vectorial (opcional al iniciar la API):**
   Puedes ejecutar la ingesta manualmente:
   ```bash
   python ingest.py
   ```
   O al iniciar la API, se te preguntará si deseas ejecutar la ingesta automática:
   ```bash
   python main.py
   ```
   > Se mostrará: `¿Deseas ejecutar la ingesta automática de documentos? (s/n):`
   > Si respondes "s", se reindexarán los documentos antes de iniciar la API.
   Por defecto la API inicia en `http://localhost:8000`

## 💡 Uso

### Endpoints principales

#### 1️⃣ Chat

`POST /chat`

- Permite conversar con el asistente sobre el contenido de los manuales.
- Mantiene el historial si usas el mismo `session_id`.
- Responde en el idioma solicitado y puedes elegir el modelo LLM.
- Mantiene el historial si usas el mismo `session_id` (usa el campo `use_history` para activarlo/desactivarlo por petición).
- Puedes activar/desactivar el reranker LLM desde `config.py` con `ENABLE_RERANK` (útil para ahorrar latencia si no quieres reordenar candidatos con el LLM).

**Body JSON:**
```json
{
  "question": "¿Cuál es la temperatura recomendada?",
  "session_id": "opcional",
  "language": "es",
  "model": "llama3"
   "use_history": true
}
```
**Respuesta:**
```json
{
  "answer": "...respuesta generada en el idioma solicitado...",
  "session_id": "..."
}
```

#### 2️⃣ Agent

`POST /agent`

- Permite buscar el nombre exacto y valor de una variable en el archivo `model.txt`.
- El modelo razona y responde con un JSON estructurado.

**Body JSON:**
```json
{
  "question": "¿Cuál es el valor actual de ph?",
  "language": "es"
}
```
**Respuesta:**
```json
{
  "variable": "ph.pv",
  "value": 7.2,
  "message": "Valor actual leído del sistema."
}
```
Si no se encuentra la variable:
```json
{
  "variable": null,
  "value": null,
  "message": "No se encontró la variable solicitada."
}
```

### 🧠 Memoria de sesión
- Si envías el mismo `session_id` en varias preguntas al chat, el chatbot recordará el historial y responderá con contexto.
- Si no envías `session_id`, el backend generará uno nuevo y lo devolverá en la respuesta.
 - Controla la memoria por petición con `use_history: true|false` en el body.

### 🌎 Idioma dinámico
- Usa el parámetro `language` para obtener la respuesta en el idioma que desees.
- El modelo intentará responder únicamente en ese idioma.

### 🔄 Cambiar el modelo LLM
Puedes especificar el modelo en el campo `model` de la petición al chat. Si no lo envías, se usará el modelo por defecto configurado en el código.

### ⚡ Modularidad y prompts
- Los prompts principales están centralizados en el archivo `llm_utils.py` para facilitar su personalización y mantenimiento.
- Puedes modificar el comportamiento del asistente editando las funciones de prompt en ese archivo.

## 📁 Estructura del proyecto y qué hace cada archivo
- `main.py` — API FastAPI: inicia la app, carga ChromaDB, configura el retriever y define los endpoints principales (`/chat`, `/chat_stream`, `/agent`). También contiene la función `rerank_with_llm` usada para reordenar fragmentos por relevancia.
- `config.py` — Variables de configuración globales: modelos, rutas, parámetros de retriever y ahora los parámetros para fragmentación (`CHUNK_SIZE`, `CHUNK_OVERLAP`). Modifica este archivo para ajustar comportamiento global.
- `db_utils.py` — Lógica de ingestión y chunking de documentos: carga PDFs/TXT y los divide en fragmentos (`Document`). Ahora toma `CHUNK_SIZE`, `CHUNK_OVERLAP` y `DATA_DIR` desde `config.py`.
- `llm_utils.py` — Prompts centralizados y utilidades de parsing. Contiene los prompts para modo agente (`build_agent_prompt`), chat (`build_chat_prompt`), manuales (`build_manual_prompt`) y helper/Biro (`build_gpt_helper_prompt`), más la función `extract_json_from_response`.
- `ingest.py` — (si existe) script para ingesta independiente de documentos (útil para pipelines o CI).
- `documents/` — Carpeta donde colocas PDFs y TXT que quieres indexar.
- `chroma_db/` — Directorio donde Chroma persiste vectores.
- `fragmentos.md` — Archivo de registro donde el endpoint `/chat_stream` añade los fragmentos recuperados en cada consulta (útil para depuración y auditoría).

Notas sobre `fragmentos.md`:
- El endpoint escribe los fragmentos en modo append (`a`) para conservar historial de consultas y facilitar auditoría.
- Si quieres que se sobrescriba (recrear el archivo con sólo los fragmentos de la consulta actual), el código se puede cambiar para abrir el archivo en modo `w` antes de escribir.
- Para evitar escrituras parciales o condiciones de carrera en entornos concurrentes, recomendamos escribir de forma atómica usando un fichero temporal y `os.replace()` o usar un lock (`filelock`).

Ejemplo de escritura atómica en el código:
```python
import tempfile, os

tmp_path = None
with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
   tmp_path = tmp.name
   for doc in docs_relevantes:
      tmp.write(f"### Fragmento encontrado\n\n{getattr(doc, 'page_content', str(doc))}\n\n")
os.replace(tmp_path, "fragmentos.md")
```

## ⚙️ Parámetros importantes en `config.py`
- `DATA_DIR`: carpeta de documentos a indexar (por defecto `documents/`).
- `CHUNK_SIZE`: tamaño en caracteres de cada fragmento (ej. 1200). Ajusta para controlar cuánto contexto aporta cada fragmento.
- `CHUNK_OVERLAP`: solapamiento entre fragmentos (ej. 200). Útil para evitar cortar información crítica entre fragments.
- `EMBEDDING_MODEL`: modelo usado para generar embeddings.
- `LLM_MODEL`: modelo usado por Ollama para respuestas.
- `RETRIEVER_K`, `RETRIEVER_FETCH_K`, `TOP_K_RERANK`: configuración del retriever y reranking.
- `ENABLE_RERANK`: booleano que activa/desactiva la etapa de reranking realizada por el LLM. Útil para controlar latencia.

## 🧭 Cómo inspeccionar qué guarda la base de datos y qué fragmentos se usan
- El endpoint `/chat_stream` añade los fragmentos que recupera desde ChromaDB a `fragmentos.md`. Abre ese archivo para ver los fragmentos exactos (texto y orden) que se usan para producir respuestas.
- Para una inspección más directa puedes reusar `db_utils.load_all_documents()` y crear un pequeño script que imprima `doc.metadata` y las primeras líneas de `doc.page_content`.

Ejemplo rápido para listar fragmentos en Python:
```python
from db_utils import load_all_documents

docs = load_all_documents()
for i, d in enumerate(docs[:50]):
   print(i, d.metadata.get('source'), '---', d.page_content[:200].replace('\n',' '))
```

## 🔧 Consejos para mejorar la precisión al responder usando solo documentos indexados
- Ajusta `CHUNK_SIZE` y `CHUNK_OVERLAP` en `config.py` y reindexa. Para manuales técnicos, valores como `1200/200` suelen funcionar bien.
- Usa un embeding multilingüe si trabajas con varios idiomas.
- Configura el retriever para recuperar más candidatos (`fetch_k`) y luego rerankear (`TOP_K_RERANK`).
- Refuerza los prompts en `llm_utils.py` para obligar al modelo a usar SOLO el contexto; ya hay prompts que devuelven respuestas concisas y piden citar documento/página.

## 🛠️ Reindexar / actualizar la base de datos
- Para reindexar manualmente: elimina `chroma_db/` o la ruta indicada por `DB_PATH` y ejecuta el script de ingesta o inicia `main.py` y responde "s" cuando pregunte por la ingesta automática.
- Si cambias el `EMBEDDING_MODEL` debes reindexar todo para regenerar vectores.

## 🧾 Feedback y mejora continua
- Actualmente no hay un endpoint oficial de feedback en el repo. Recomendación rápida: crea un endpoint `/feedback` que guarde entradas en `feedback.jsonl` (acción: accept/reject/correct). Las correcciones pueden convertirse en documentos que luego se reindexan en lote.
- Otra opción inmediata: añade manualmente archivos a `documents/` con correcciones y reindexa.

## ✅ Qué probar ahora
1. Añade tus documentos en `documents/` y ejecuta `python main.py`.
2. Responde "s" para crear la base vectorial si quieres reindexar.
3. Prueba `POST /chat` y `POST /chat_stream` desde `http://localhost:8000/docs`.
4. Abre `fragmentos.md` para ver los fragmentos que se recuperaron para cada consulta.

## 📬 Contacto
Para dudas o soporte, contacta a Nanotrejo.

### 📝 Notas
- ✨ Si cambias los documentos, ejecuta de nuevo la ingesta para reindexar.
- 🦙 Puedes agregar más modelos a Ollama según tus necesidades.
- 🌎 El sistema soporta preguntas y respuestas en español, inglés y otros idiomas.

### 🛡️ Ingesta interactiva
- Al iniciar la API, se te preguntará si deseas ejecutar la ingesta automática de documentos.
- Esto evita reindexaciones innecesarias y acelera el arranque si no hay cambios en los manuales.

## 📬 Contacto
Para dudas o soporte, contacta a Nanotrejo.
