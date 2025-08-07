# llm_utils.py

import re
import json


def build_agent_prompt(contexto, pregunta):
    """
    Construye el prompt para el modo agente.
    """
    return (
        "Eres un agente experto en variables de control industrial. "
        "Tu tarea es identificar el nombre exacto de la variable y el valor que corresponde a la petición del usuario, usando únicamente el CONTEXTO proporcionado. "
        "Responde únicamente con un objeto JSON válido, sin explicaciones, sin formato markdown y sin ningún texto adicional. El JSON debe tener los campos: "
        "variable: el nombre exacto de la variable (por ejemplo: ph.basic.cd.sp), "
        "value: el valor que corresponde (true/false/número), "
        "message: una explicación breve de la acción que se va a realizar, sin mencionar la variable. "
        "Si no existe la variable, responde exactamente: {{\"variable\": null, \"value\": null, \"message\": \"No se encontró la variable solicitada.\"}}.\n\n"
        f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}\n"
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
    Construye el prompt para el modo chat.
    """
    return (
        f"Eres un asistente experto en un manual técnico.\n\n"
        f"Responde **únicamente** a la PREGUNTA usando el CONTEXTO dado.  \n"
        f"No inventes información ni uses conocimientos externos.  \n"
        f"Si el CONTEXTO no contiene la respuesta, indica claramente que no se encontró información.  \n"
        f"Responde en el mismo idioma detectado de la pregunta o en el especificado: {language}.  \n\n"
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
        "Incluye en la respuesta el **nombre del documento**, **número de página** y **subíndice** donde se encontró la información.\n"
        "**No** digas frases como 'según el manual' ni variantes similares.\n"
        "Si la respuesta **no está en el CONTEXTO**, responde exactamente:\n"
        "**'Lo siento, no tengo información sobre eso en el manual.'**\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA:\n{question}\n\n"
        "RESPUESTA CONCISA:"
    )
