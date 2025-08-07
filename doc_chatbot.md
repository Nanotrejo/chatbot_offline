# 📚 Explicación de componentes y argumentos del chatbot

## 🔑 Argumentos principales
- **question**: Pregunta del usuario que se envía al chatbot.
- **session_id**: Identificador de la sesión para mantener el historial de la conversación.
- **language**: Idioma en el que se solicita la respuesta.
- **model**: Permite especificar el modelo LLM a utilizar (por ejemplo, "llama3", "phi3:mini").

## 🤖 LLM (Large Language Model)
- Es el modelo de lenguaje que genera las respuestas del chatbot.
- Se conecta a través de Ollama y puede ser cambiado según el parámetro `model`.
- Ejemplo de modelos: llama3, phi3:mini, gemma3:4b.

## 🗃️ Base de datos vectorial (ChromaDB)
- Almacena los fragmentos de los documentos en forma de vectores para realizar búsquedas semánticas.
- Permite encontrar el contexto relevante para cada pregunta.
- Se crea a partir de los documentos PDF/TXT usando embeddings.

## 🧩 Embeddings
- Modelo que convierte el texto en vectores numéricos para la búsqueda semántica.
- Ejemplo: "sentence-transformers/paraphrase-mpnet-base-v2".
- Debe ser el mismo tanto en la ingesta como en la consulta.

---

## ⚙️ Parámetros clave

### k (número de fragmentos relevantes a recuperar)
- Recomendado: **3-8** para precisión y contexto suficiente.
- Si el contexto es muy disperso, puedes subirlo a **10-15**.
- 🔼 Si recibes muchas respuestas "No tengo información...", sube k.
- 🔽 Si el modelo responde con información irrelevante, baja k.

### temperature (creatividad del modelo)
- Recomendado: **0.0 a 0.2** para respuestas técnicas y precisas.
- Si quieres respuestas más creativas, sube a **0.5 o más**.
- Para manuales y soporte técnico, lo óptimo es **0.1**.

---

## 🧱 Chunking (fragmentación de documentos)
- **chunk_size**: Tamaño de cada fragmento (recomendado: **1200**).
- **chunk_overlap**: Solapamiento entre fragmentos (recomendado: **200**).
- Si el chunk es muy pequeño, puedes perder contexto relevante.
- Si es muy grande, la búsqueda puede ser menos precisa y más lenta.

---

## 🏆 Modelos de embeddings recomendados

1. **paraphrase-multilingual-MiniLM-L12-v2**
   - 🌍 Multilingüe, rápido, buen rendimiento general.
   - ⚠️ Puede perder matices técnicos muy específicos.
2. **all-MiniLM-L6-v2**
   - ⚡ Muy eficiente, buena precisión en inglés y español.
   - ⚠️ Menos potente para textos largos o muy técnicos.
3. **sentence-transformers/paraphrase-mpnet-base-v2**
   - 🧠 Mejor comprensión semántica, más robusto para textos complejos.
   - ⚠️ Más pesado, requiere más recursos.
4. **distiluse-base-multilingual-cased-v2**
   - 🌍 Multilingüe, buen equilibrio entre velocidad y calidad.
   - ⚠️ Menos preciso que los modelos MPNet.

---

## 📝 Recomendaciones generales
- 🔼 Aumenta el valor de k en KWARGS si necesitas más contexto.
- 🔄 Prueba con otro modelo de embeddings si no encuentras resultados precisos.
- 🧱 Ajusta el tamaño de los fragmentos (chunk_size y chunk_overlap).
- 🔍 Cambia el tipo de búsqueda a "mmr" o "similarity_score_threshold" si la pregunta es ambigua.
- ✅ Verifica que los documentos estén bien indexados y el texto sea legible.

---

## 📊 Diferencias entre `/chat` y `/agent`
- **/chat**: Responde preguntas generales sobre los documentos, usando el modelo LLM y la base de datos vectorial completa.
- **/agent**: Busca y razona sobre variables técnicas en el archivo `model.txt`, devolviendo el nombre exacto, valor y mensaje en formato JSON.

---

## 🔬 EMBEDDING_MODEL
- Especifica el modelo de embeddings que convierte el texto en vectores numéricos para la búsqueda semántica.
- Ejemplo de valor: "sentence-transformers/paraphrase-mpnet-base-v2".
- Es fundamental que el mismo modelo se use tanto para la ingesta de documentos como para la consulta, para que los vectores sean compatibles y la búsqueda sea precisa.
- Cambiar el modelo puede afectar la calidad y el idioma de la búsqueda semántica.

---
