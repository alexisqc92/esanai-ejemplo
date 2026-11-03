# Prueba local del endpoint de modelos (Azure AI Foundry)

Script para probar desde la laptop un modelo desplegado en la suscripción **ESAN-IA** y entender cómo se usa el endpoint antes de integrarlo en el backend Flask.

> **Regla del programa:** no se usan API keys. Toda llamada al endpoint se autentica con **Microsoft Entra ID** (en local con `az login`; en Azure con Managed Identity).

## 1. Preparar el entorno

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copia `.env.example` como `.env` y completa el endpoint y el nombre del deployment (Foundry → recurso → *Endpoint*; Foundry → *Models + endpoints* → nombre del deployment). El `.env` no lleva secretos.

## 2. Autenticarse con Entra ID

```bash
az login --tenant <tenant-esan>
az account set --subscription "ESAN-IA"
```

Tu usuario (o el grupo de tu equipo en Entra ID) necesita el rol **Cognitive Services OpenAI User** sobre el recurso. La asignación puede tardar hasta 5 minutos en propagarse.

`DefaultAzureCredential` usa esa sesión en local y, sin cambiar el código, la Managed Identity cuando el backend corra en Container Apps.

## 3. Ejecutar

```bash
python prueba_endpoint.py                          # demo de 4 pasos
python prueba_endpoint.py --pregunta "¿Qué es ESAN?"
python prueba_endpoint.py --chat                   # chat con memoria
```

La demo muestra: pregunta simple (con latencia y tokens), streaming, salida JSON para clasificar una consulta por área (base del enrutamiento con Function Calling) y chat con historial.

## 4. Prueba rápida con curl (para ver la petición HTTP "cruda")

```bash
TOKEN=$(az account get-access-token --resource https://cognitiveservices.azure.com --query accessToken -o tsv)

curl -X POST "$AZURE_OPENAI_ENDPOINT/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"model": "gpt-5-mini", "messages": [{"role": "user", "content": "Hola"}]}'
```

El token de Entra ID dura alrededor de una hora; el SDK lo renueva solo, con curl hay que volver a pedirlo.

## Conceptos clave para la clase

- En Azure, el parámetro `model` es el **nombre del deployment**, no el nombre del modelo base.
- La API v1 usa el cliente estándar `OpenAI()` con `base_url = <endpoint>/openai/v1/`; ya no se pasa `api-version`.
- El parámetro `api_key` del SDK recibe el *token provider* de Entra ID: no hay llaves en el código ni en el `.env`.
- El modelo **no recuerda** conversaciones: el historial se envía en cada llamada (en la plataforma se guardará en Cosmos DB).
- Cada llamada consume tokens de entrada + salida: eso es lo que se paga y lo que limita la cuota.
- Los modelos de razonamiento (ej. gpt-5-mini) no aceptan `temperature`; por eso el script no lo usa.

## Errores frecuentes

| Código | Causa probable |
|---|---|
| Error de autenticación | No se ejecutó `az login` o la suscripción activa no es ESAN-IA |
| 401 | Falta el rol *Cognitive Services OpenAI User* (o aún no se propaga) |
| 403 | Red bloqueada (firewall / private endpoint del recurso) |
| 404 | Endpoint con `/openai/v1` duplicado o nombre de deployment mal escrito |
| 429 | Cuota excedida (throttling) |

Referencia: [Azure OpenAI v1 API – Microsoft Learn](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle)
