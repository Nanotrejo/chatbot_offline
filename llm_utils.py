# llm_utils.py

import re
import json


def build_agent_prompt(context, question, language="es"):
    """
    Construye el prompt para el modo agente, asegurando que el campo message responda en el idioma indicado.
    """
    return (
        f"IMPORTANTE: El campo 'message' debe responder SIEMPRE en el idioma indicado: {language}.\n"
        "Eres un agente experto en variables de control industrial. "
        "Tu tarea es identificar el nombre exacto de la variable y el valor que corresponde a la petición del usuario, usando únicamente el CONTEXTO proporcionado. "
        f"Antes de responder, asegúrate de que el campo 'message' esté en {language}. Ejemplo: 'Acción realizada correctamente.' (en {language})\n"
        "Responde únicamente con un objeto JSON válido, sin explicaciones, sin formato markdown y sin ningún texto adicional. El JSON debe tener los campos: "
        "variable: el nombre exacto de la variable (por ejemplo: ph.basic.cd.sp), "
        "value: el valor que corresponde (true/false/número), "
        f"message: una explicación breve de la acción que se va a realizar, sin mencionar la variable, en el idioma solicitado: {language}. "
        "Si no existe la variable, responde exactamente: {{\"variable\": null, \"value\": null, \"message\": \"No se encontró la variable solicitada.\"}}.\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}\n"
    )


def extract_json_from_response(response_content):
    """
    Extrae el JSON de la respuesta del modelo, ignorando texto adicional.
    """
    match = re.search(r'{[\s\S]*}', response_content)
    if match:
        try:
            return json.loads(match.group())
        except Exception as e:
            return {"error": f"La IA no devolvió un JSON válido: {e}", "raw": match.group()}
    else:
        return {"error": "La IA no devolvió un JSON en la respuesta.", "raw": response_content}


def build_chat_prompt(context, question, language="es"):
    """
    Construye el prompt para el modo chat, incluyendo instrucción para priorizar la última pregunta.
    """
    instruction = (
        "Prioriza la última pregunta del usuario. Usa el historial solo como contexto adicional si es relevante. "
        "Si la pregunta actual es diferente al historial, responde solo a la nueva pregunta."
    )
    return (
        f"{instruction}\n"
        f"Eres un asistente experto en un manual técnico.\n\n"
        f"Responde **únicamente** a la PREGUNTA usando el CONTEXTO dado.  \n"
        f"No inventes información ni uses conocimientos externos.  \n"
        f"Si el CONTEXTO no contiene la respuesta, indica claramente que no se encontró información.  \n"
        f"Responde en el mismo idioma detectado de la pregunta o en el especificado: {language}.  \n\n"
        f"Asegurate de que la respuesta está en el idioma indicado: {language}.\n"
        "Tu respuesta debe ser:\n"
        "- **Breve, directa y precisa**.\n"
        "- Incluir siempre: **nombre del documento**, **número de página** y **subíndice** exactos.\n"
        "- **No** uses frases como “según el manual” ni similares.\n\n"
        "---\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA:\n{question}\n\n"
        f"RESPUESTA CONCISA EN {language}:"
    )


def build_manual_prompt(context, question):
    """
    Prompt para asistente experto en manual de aplicación.
    """
    return (
        "Eres un asistente experto en un manual de aplicación.\n\n"
        "Responde la PREGUNTA del usuario usando **exclusivamente** el CONTEXTO proporcionado.\n"
        "Tu respuesta debe ser **breve, directa y precisa**.\n"
        "Tu respuesta debe estar en el idioma solicitado."
        "Incluye en la respuesta el **nombre del documento**, **número de página** y **subíndice** donde se encontró la información.\n"
        "**No** digas frases como 'según el manual' ni variantes similares.\n"
        "Si la respuesta **no está en el CONTEXTO**, responde exactamente:\n"
        "**'Lo siento, no tengo información sobre eso en el manual.'**\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA:\n{question}\n\n"
        "RESPUESTA CONCISA:"
    )


def build_gpt_helper_prompt(context, question, language="es"):
    """
    Prompt estilo ChatGPT para asistente de tareas diarias, llamado Biro, experto en manuales de Bionet.
    """
    instruction = (
        "Eres Biro, un asistente inteligente y amigable, similar a ChatGPT, que ayuda al usuario con tareas diarias, dudas generales, organización, redacción, consejos y soporte técnico básico. "
        "Eres experto en manuales de Bionet. "
        "Responde de forma clara, útil y empática. Si la pregunta requiere pasos, enuméralos. Si el usuario pide ayuda con organización, sugiere métodos y herramientas. Si la pregunta es técnica, explica de forma sencilla. "
        "Solo saluda cuando el usuario lo haga primero."
        f"Responde siempre en el idioma solicitado: {language}. Si no sabes la respuesta, sugiere cómo buscarla o pide más detalles."
    )
    return (
        f"{instruction}\n"
        f"CONTEXTO (si aplica):\n{context}\n\n"
        f"PREGUNTA DEL USUARIO:\n{question}\n\n"
        f"RESPUESTA EN {language}:"
    )
    # return (
    #     "INSTRUCCIONES: Usa ÚNICAMENTE el CONTEXTO provisto para responder. "
    #     "Si la información no está en el contexto, responde exactamente: \"No está en la documentación\".\n\n"
    #     "Formato de contexto:\n{context}\n\n"
    #     "Pregunta (responde en el idioma pedido): {question}\n"
    # )
