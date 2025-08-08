# 🤖 Chatbot Offline

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

**Body JSON:**
```json
{
  "question": "¿Cuál es la temperatura recomendada?",
  "session_id": "opcional",
  "language": "es",
  "model": "llama3"
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

### 🌎 Idioma dinámico
- Usa el parámetro `language` para obtener la respuesta en el idioma que desees.
- El modelo intentará responder únicamente en ese idioma.

### 🔄 Cambiar el modelo LLM
Puedes especificar el modelo en el campo `model` de la petición al chat. Si no lo envías, se usará el modelo por defecto configurado en el código.

### ⚡ Modularidad y prompts
- Los prompts principales están centralizados en el archivo `llm_utils.py` para facilitar su personalización y mantenimiento.
- Puedes modificar el comportamiento del asistente editando las funciones de prompt en ese archivo.

### 📝 Notas
- ✨ Si cambias los documentos, ejecuta de nuevo la ingesta para reindexar.
- 🦙 Puedes agregar más modelos a Ollama según tus necesidades.
- 🌎 El sistema soporta preguntas y respuestas en español, inglés y otros idiomas.

### 🛡️ Ingesta interactiva
- Al iniciar la API, se te preguntará si deseas ejecutar la ingesta automática de documentos.
- Esto evita reindexaciones innecesarias y acelera el arranque si no hay cambios en los manuales.

## 📬 Contacto
Para dudas o soporte, contacta a Nanotrejo.
