# 🤖 Chatbot Offline - Manual Rosita

Este proyecto es una API basada en FastAPI que permite consultar manuales técnicos usando modelos LLM locales vía Ollama y búsqueda semántica con ChromaDB.

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

6. **Genera la base de datos vectorial:**
   ```bash
   python ingest.py
   ```

7. **Inicia la API:**
   ```bash
   python main.py
   ```
   Por defecto se inicia en `http://localhost:8000`

## 💡 Uso

### Endpoint principal

`POST /chat`

**Body JSON:**
```json
{
  "question": "¿Cuál es la temperatura recomendada?",
  "session_id": "opcional", // Para mantener el contexto de la conversación
  "language": "es", // Idioma de la respuesta (ej: "es", "en", "fr", etc.)
  "model": "llama3" // Puedes cambiar el modelo aquí (opcional)
}
```

**Respuesta:**
```json
{
  "answer": "...respuesta generada en el idioma solicitado...",
  "session_id": "..."
}
```

### 🧠 Memoria de sesión
- Si envías el mismo `session_id` en varias preguntas, el chatbot recordará el historial y responderá con contexto.
- Si no envías `session_id`, el backend generará uno nuevo y lo devolverá en la respuesta.

### 🌎 Idioma dinámico
- Usa el parámetro `language` para obtener la respuesta en el idioma que desees.
- El modelo intentará responder únicamente en ese idioma.

### 🔄 Cambiar el modelo LLM
Puedes especificar el modelo en el campo `model` de la petición. Si no lo envías, se usará el modelo por defecto configurado en el código.

## 📝 Notas
- ✨ Si cambias los documentos, ejecuta de nuevo `python ingest.py` para reindexar.
- 🦙 Puedes agregar más modelos a Ollama según tus necesidades.
- 🌎 El sistema soporta preguntas y respuestas en español, inglés y otros idiomas.

## 📬 Contacto
Para dudas o soporte, contacta a Nanotrejo.
