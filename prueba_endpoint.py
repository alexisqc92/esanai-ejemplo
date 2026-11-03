"""Prueba local del endpoint de modelos en Azure AI Foundry.

Uso:
    python prueba_endpoint.py                  # demo completa (4 pasos)
    python prueba_endpoint.py --chat           # chat interactivo
    python prueba_endpoint.py --pregunta "..." # una sola pregunta

Autenticación:
    Solo Microsoft Entra ID (cero llaves). Requiere ``az login`` y el
    rol "Cognitive Services OpenAI User" sobre el recurso. No se usan
    API keys en ningún caso.
"""

import argparse
import os
import sys
import time

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
# gpt-5-mini es un modelo de razonamiento: el tope incluye los tokens que
# usa para "pensar". Con un tope bajo puede devolver una respuesta vacía.
MAX_TOKENS = 2000
# minimal | low | medium | high.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low") or None

SYSTEM_PROMPT = (
    "Eres el asistente institucional de la Universidad ESAN. "
    "Responde en español, de forma breve y clara. "
    "Si no sabes algo, dilo; no inventes datos."
)

CLASIFICADOR_PROMPT = (
    "Clasifica la consulta en una de estas áreas: "
    "comercial, dga, gthc, alumni, registros. "
    'Responde SOLO un JSON: {"area": "...", '
    '"tipo": "rag|transaccional", "motivo": "..."}'
)

AYUDA_ERRORES = {
    401: (
        "Credenciales inválidas o falta el rol 'Cognitive Services "
        "OpenAI User' (tarda hasta 5 min en propagarse). "
        "Revisa `az login`."
    ),
    403: "Sin permisos o red bloqueada (firewall / private endpoint).",
    404: (
        "Endpoint o deployment incorrecto. El endpoint NO debe "
        "incluir /openai/v1."
    ),
    429: "Cuota excedida (throttling). Espera unos segundos y reintenta.",
}


def crear_cliente() -> OpenAI:
    """Crea el cliente OpenAI apuntando al endpoint v1 de Azure.

    La autenticación es con Microsoft Entra ID: ``DefaultAzureCredential``
    toma la sesión de ``az login`` (en local) o la Managed Identity (en
    Azure) y el *token provider* obtiene y renueva el token solo.

    Returns:
        OpenAI: Cliente configurado para el recurso de Foundry.

    Raises:
        SystemExit: Si falta el endpoint en el archivo ``.env``.
    """
    if not ENDPOINT:
        sys.exit("Falta AZURE_OPENAI_ENDPOINT en el archivo .env")

    base_url = f"{ENDPOINT}/openai/v1/"

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), TOKEN_SCOPE
    )
    print(f"[auth] Entra ID | {base_url} | deployment: {DEPLOYMENT}")
    # "api_key" es el nombre del parámetro del SDK, pero recibe el token
    # provider de Entra ID: no hay ninguna llave en juego.
    return OpenAI(base_url=base_url, api_key=token_provider)


def opciones_razonamiento(esfuerzo: str | None = REASONING_EFFORT) -> dict:
    """Devuelve el parámetro de razonamiento si el modelo lo admite.

    Args:
        esfuerzo: Nivel de esfuerzo o ``None`` para no enviarlo.

    Returns:
        dict: ``{"reasoning_effort": ...}`` o un diccionario vacío.
    """
    return {"reasoning_effort": esfuerzo} if esfuerzo else {}


def construir_mensajes(pregunta: str, system: str = SYSTEM_PROMPT) -> list:
    """Arma la lista de mensajes para una consulta de un solo turno.

    Args:
        pregunta: Texto del usuario.
        system: Instrucciones de sistema para el modelo.

    Returns:
        list: Mensajes con roles ``system`` y ``user``.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": pregunta},
    ]


def pregunta_simple(client: OpenAI, pregunta: str) -> str:
    """Envía una pregunta y muestra respuesta, latencia y tokens.

    En Azure, el parámetro ``model`` es el NOMBRE DEL DEPLOYMENT, no el
    nombre del modelo base.

    Args:
        client: Cliente creado con :func:`crear_cliente`.
        pregunta: Texto del usuario.

    Returns:
        str: Texto de la respuesta del modelo.
    """
    inicio = time.perf_counter()
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=construir_mensajes(pregunta),
        max_completion_tokens=MAX_TOKENS,
        **opciones_razonamiento(),
    )
    ms = (time.perf_counter() - inicio) * 1000
    texto = resp.choices[0].message.content
    uso = resp.usage
    print(f"\n> {pregunta}\n{texto}")
    print(
        f"\n  [modelo: {resp.model} | {ms:.0f} ms | tokens in/out/total: "
        f"{uso.prompt_tokens}/{uso.completion_tokens}/{uso.total_tokens}"
        f" | fin: {resp.choices[0].finish_reason}]"
    )
    return texto


def pregunta_streaming(client: OpenAI, pregunta: str) -> None:
    """Muestra la respuesta a medida que el modelo la genera.

    Args:
        client: Cliente creado con :func:`crear_cliente`.
        pregunta: Texto del usuario.
    """
    print(f"\n> {pregunta}")
    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=construir_mensajes(pregunta),
        max_completion_tokens=MAX_TOKENS,
        stream=True,
        **opciones_razonamiento(),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def clasificar_consulta(client: OpenAI, consulta: str) -> str:
    """Clasifica una consulta por área institucional en formato JSON.

    Ilustra la base del enrutamiento: decidir si una consulta se
    resuelve con RAG documental o con lógica transaccional
    (Function Calling).

    Args:
        client: Cliente creado con :func:`crear_cliente`.
        consulta: Texto del usuario a clasificar.

    Returns:
        str: JSON con las claves ``area``, ``tipo`` y ``motivo``.
    """
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=construir_mensajes(consulta, system=CLASIFICADOR_PROMPT),
        response_format={"type": "json_object"},
        max_completion_tokens=1000,
        # Clasificar es simple: con razonamiento mínimo es más rápido.
        **opciones_razonamiento("minimal" if REASONING_EFFORT else None),
    )
    resultado = resp.choices[0].message.content
    print(f"\n> {consulta}\n{resultado}")
    return resultado


def chat_interactivo(client: OpenAI) -> None:
    """Abre un chat en consola que conserva el historial.

    El modelo no recuerda conversaciones: el historial lo guarda el
    cliente y se reenvía completo en cada llamada (en la plataforma se
    persistirá en Cosmos DB).

    Args:
        client: Cliente creado con :func:`crear_cliente`.
    """
    historial = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\nChat interactivo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "q"}:
            break
        if not pregunta:
            continue
        historial.append({"role": "user", "content": pregunta})
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=historial,
            max_completion_tokens=MAX_TOKENS,
            **opciones_razonamiento(),
        )
        texto = resp.choices[0].message.content
        historial.append({"role": "assistant", "content": texto})
        print(f"Asistente: {texto}")
        print(f"  [tokens de esta llamada: {resp.usage.total_tokens}]\n")


def ejecutar_demo(client: OpenAI) -> None:
    """Ejecuta la demo guiada de cuatro pasos para la clase.

    Args:
        client: Cliente creado con :func:`crear_cliente`.
    """
    print("\n=== Paso 1: pregunta simple ===")
    pregunta_simple(client, "¿Qué es la Universidad ESAN? En 2 líneas.")
    print("\n=== Paso 2: streaming ===")
    pregunta_streaming(client, "Dame 3 consejos para un primer día de clases.")
    print("\n=== Paso 3: salida JSON (clasificación de consulta) ===")
    clasificar_consulta(
        client, "¿Cuándo me devuelven el dinero de mi matrícula anulada?"
    )
    print("\n=== Paso 4: prueba el chat con: python prueba_endpoint.py --chat")


def parsear_argumentos() -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Argumentos ``chat`` y ``pregunta``.
    """
    parser = argparse.ArgumentParser(description="Prueba del endpoint Foundry")
    parser.add_argument("--chat", action="store_true", help="chat interactivo")
    parser.add_argument("--pregunta", help="hacer una sola pregunta")
    parser.add_argument(
        "--clasificar", help="clasificar una consulta por área y tipo"
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada: crea el cliente y ejecuta el modo elegido."""
    args = parsear_argumentos()
    client = crear_cliente()
    try:
        if args.chat:
            chat_interactivo(client)
        elif args.pregunta:
            pregunta_simple(client, args.pregunta)
        elif args.clasificar:
            clasificar_consulta(client, args.clasificar)
        else:
            ejecutar_demo(client)
    except APIStatusError as err:
        ayuda = AYUDA_ERRORES.get(err.status_code, "")
        sys.exit(f"\n[ERROR {err.status_code}] {err.message}\n{ayuda}")
    except ClientAuthenticationError as err:
        sys.exit(
            f"\n[ERROR de autenticación] {err.message}\n"
            "Ejecuta `az login` y verifica la suscripción activa."
        )
    except APIConnectionError as err:
        sys.exit(f"\n[ERROR de conexión] {err}. Revisa el endpoint y tu red.")


if __name__ == "__main__":
    main()
