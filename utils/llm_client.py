"""
utils/llm_client.py
===================
Cliente Ollama para el asistente IA de PLUTO.

    - System Prompt: rol Ingeniero Senior CTAG, tono asertivo y tecnico.
    - Semantica SHAP: (+) => Defecto, (-) => Calidad. Sin ambiguedad.
    - Tabla procesada: recibe los nombres industriales y valores netos de explainer.py.

Responsabilidades adicionales:
    - Verificar conectividad con `http://localhost:11434`.
    - Si el servidor responde pero el modelo no esta disponible,
      mostrar el comando exacto para descargarlo.
    - Hacer streaming de la respuesta mediante un generador Python.
"""

import json
import warnings
from typing import Generator

import requests

warnings.filterwarnings("ignore")

# ── Configuracion ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
LLM_MODEL        = "llama3"      # Cambiar si se usa otro modelo (ej. "mistral")
REQUEST_TIMEOUT  = 90            # segundos para streaming

# Umbral 
NOK_THRESHOLD = 0.4606  # esto estaría bien cambiarlo para que lo coja del pkl si se puede - Lucas


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt del Rol
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Eres un Ingeniero Senior de Calidad del proyecto PLUTO para la empresa CTAG. "
    "Tu mision es analizar el resultado del sistema automatico de clasificacion de piezas industriales "
    "y comunicar el diagnostico al operario de linea de forma directa, tecnica y sin ambiguedades. "
    "\n\n"
    "REGLAS DE INTERPRETACION OBLIGATORIAS:\n"
    f"1. El umbral de clasificacion es {NOK_THRESHOLD}. Si P(NOK) >= {NOK_THRESHOLD}, la pieza es NOK (defectuosa). "
    f"   Si P(NOK) < {NOK_THRESHOLD}, la pieza es OK (conforme). Este umbral es absoluto e inamovible.\n"
    "2. En el analisis SHAP:\n"
    "   - Un valor SHAP POSITIVO (+) significa que esa variable EMPUJA la prediccion hacia el DEFECTO (NOK).\n"
    "   - Un valor SHAP NEGATIVO (-) significa que esa variable APOYA la CALIDAD (OK).\n"
    "   - Los valores mostrados son impactos NETOS: ya incorporan la contribucion algebraica "
    "     de las variables derivadas a su sensor fisico original.\n"
    "3. Tono: asertivo, tecnico, sin rodeos. No usar expresiones vagas como 'podria ser' o 'quizas'.\n"
    "4. Estructura de respuesta obligatoria:\n"
    "   1. Diagnostico: resultado de clasificacion con probabilidad.\n"
    "   2. Variables criticas: las variables con mayor impacto SHAP y su interpretacion fisica.\n"
    "   3. Accion recomendada: que debe revisar el operario (si NOK) o confirmacion de conformidad (si OK).\n"
    "5. Si hay valores OK cercanos a sus limites operacionales, mencionarlos como alerta preventiva.\n"
    "6. Respuesta maxima: 15 segundos de lectura. Ser conciso."
)


# ══════════════════════════════════════════════════════════════════════════════
# Validacion de conectividad y modelos disponibles
# ══════════════════════════════════════════════════════════════════════════════

class OllamaStatus:
    """Resultado de la comprobacion de Ollama."""
    SERVER_UP    = "server_up"      # servidor OK, modelo OK
    SERVER_DOWN  = "server_down"    # sin conexion
    MODEL_MISSING= "model_missing"  # servidor OK, modelo NO disponible


def check_ollama(model: str = LLM_MODEL) -> tuple[str, str]:
    """
    Comprueba la disponibilidad de Ollama y del modelo.

    Returns:
        (status, message)
        status  -> OllamaStatus.*
        message -> Cadena legible para mostrar en la UI.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            OllamaStatus.SERVER_DOWN,
            f"Ollama no esta disponible en `{OLLAMA_BASE_URL}`.\n"
            "Para activarlo:\n"
            "```bash\n"
            f"ollama pull {model}\n"
            "ollama serve\n"
            "```\n"
            "Recarga la pagina tras completar estos pasos.",
        )
    except Exception as exc:
        return (
            OllamaStatus.SERVER_DOWN,
            f"Error al conectar con Ollama: {exc}",
        )

    # Servidor activo - verificar si el modelo esta disponible
    try:
        available = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        model_base = model.split(":")[0]
        if model_base not in available:
            return (
                OllamaStatus.MODEL_MISSING,
                f"El servidor Ollama esta activo, pero el modelo `{model}` "
                f"no esta instalado.\n\n"
                f"Descargalo con:\n"
                f"```bash\n"
                f"ollama pull {model}\n"
                f"```",
            )
    except Exception:
        pass  # Si no se puede parsear, asumimos que esta OK

    return (
        OllamaStatus.SERVER_UP,
        f"OK Ollama activo - modelo `{model}` disponible.",
    )


def build_prompt(
    label: str,
    proba: float,
    threshold: float,
    top_k: list[tuple[str, float, float]],
    user_query: str,
) -> str:
    """
    Construye el prompt estructurado para el LLM.

    El System Prompt (SYSTEM_PROMPT) define el rol y las reglas de interpretacion.
    Este user-prompt aporta los datos especificos de la inspeccion actual.

    Args:
        label      : "OK" o "NOK" (resultado del umbral)
        proba      : probabilidad de NOK (float 0-1)
        threshold  : umbral de decision (debe ser NOK_THRESHOLD = 0.6818)
        top_k      : lista procesada por explainer.py:
                     (nombre_industrial, valor_fisico, shap_neto)
                     Los nombres ya son industriales (sin sufijos _bin, sin latent_*).
                     Los valores SHAP son impactos netos algebraicos.
        user_query : pregunta del operario

    Returns:
        Prompt de usuario como string. Se envia junto con SYSTEM_PROMPT.
    """
    k = len(top_k)

    # Tabla SHAP procesada (ya viene con nombres industriales y valores netos)
    lines = []
    for name, val, sv in top_k:
        # Si es texto (string), lo formateamos como texto. Si es numero, con 4 decimales.
        val_str = f"{val:>10}" if isinstance(val, str) else f"{val:>10.4f}"
        
        lines.append(
            f"  {name:<45s}  valor={val_str}   SHAP_neto={sv:>+.5f}  "
            f"({'DEFECTO' if sv > 0 else 'CALIDAD'})"
        )
        
    feat_lines = "\n".join(lines)

    return (
        f"══ DATOS DE LA INSPECCION ══\n"
        f"Veredicto:             {label}\n"
        f"P(NOK):                {proba:.4f}  ({proba:.2%})\n"
        f"P(OK):                 {1 - proba:.4f}  ({1 - proba:.2%})\n"
        f"Umbral:        {threshold:.4f}\n"
        f"Decision:              {'P(NOK) >= umbral -> DEFECTO confirmado' if proba >= threshold else 'P(NOK) < umbral -> CONFORMIDAD confirmada'}\n"
        f"\n"
        f"══ ANALISIS SHAP - IMPACTO NETO POR SENSOR (Top {k}) ══\n"
        f"(Valores netos: suma algebraica sensor fisico + derivadas binarizadas)\n"
        f"{feat_lines}\n"
        f"\n"
        f"Recordatorio de interpretacion:\n"
        f"  SHAP_neto > 0  =>  contribucion al DEFECTO (NOK)\n"
        f"  SHAP_neto < 0  =>  contribucion a la CALIDAD (OK)\n"
        f"  Magnitud       =>  fuerza relativa del sensor en esta prediccion\n"
        f"\n"
        f"══ PREGUNTA DEL OPERARIO ══\n"
        f"{user_query}\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Streaming
# ══════════════════════════════════════════════════════════════════════════════

def stream_response(
    prompt: str,
    model: str = LLM_MODEL,
) -> Generator[str, None, None]:
    """
    Generador que envia el prompt a Ollama y hace streaming de la respuesta.
    Incluye el SYSTEM_PROMPT con el rol de Ingeniero Senior CTAG en el payload.

    Yields:
        Texto parcial acumulado (str) en cada chunk.
    """
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                buffer += chunk.get("response", "")
                yield buffer
                if chunk.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        yield (
            "**Ollama no esta disponible en `localhost:11434`.**\n\n"
            "Para activarlo:\n"
            "```bash\n"
            f"ollama pull {model}\n"
            "ollama serve\n"
            "```\n"
            "Recarga la pagina tras completar estos pasos."
        )
    except requests.exceptions.Timeout:
        yield "**Timeout:** El LLM tardo demasiado. Prueba con un modelo mas pequeno."
    except Exception as exc:
        yield f"Error al conectar con el LLM: {exc}"
