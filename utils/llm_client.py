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
NOK_THRESHOLD = 0.4606 


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt del Rol
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Eres un Ingeniero Senior de Calidad del proyecto PLUTO (CTAG). "
    "Recibes datos de un sistema automático de inspección de piezas industriales "
    "y respondes al operario de línea de forma directa, técnica y sin ambigüedades. "
    "\n\n"

    # ── Reglas invariantes ────────────────────────────────────────────────────
    "REGLAS ABSOLUTAS (nunca las ignores):\n"
    f"R1. Umbral fijo: si P(NOK) >= {NOK_THRESHOLD} → pieza NOK (defectuosa); "
    f"    si P(NOK) < {NOK_THRESHOLD} → pieza OK (conforme).\n"
    "R2. Semántica SHAP:\n"
    "    · Valor SHAP POSITIVO (+) → la variable empuja hacia DEFECTO (NOK).\n"
    "    · Valor SHAP NEGATIVO (-) → la variable empuja hacia CALIDAD (OK).\n"
    "    Los valores ya son impactos netos sobre el sensor físico original.\n"
    "R3. Responde SIEMPRE en español, tono técnico y asertivo. "
    "    Prohibido usar 'podría', 'quizás', 'tal vez'.\n"
    "R4. Usa SOLO las variables que aparecen en los datos. "
    "    No inventes ni menciones variables ausentes.\n"
    "R5. Si mencionas P(NOK), especifica siempre que es probabilidad de DEFECTO.\n"
    "\n"

    # ── Estructura adaptativa ────────────────────────────────────────────────
    "ESTRUCTURA DE RESPUESTA (adáptala a la pregunta del operario):\n"
    "\n"
    "· Si pregunta por el DIAGNÓSTICO GENERAL o el RESULTADO:\n"
    "  1. Veredicto con probabilidad.\n"
    "  2. Variables con mayor impacto SHAP y su dirección física.\n"
    "  3. Acción concreta para el operario.\n"
    "\n"
    "· Si pregunta por QUÉ la pieza es NOK o cuál es la CAUSA DEL DEFECTO:\n"
    "  1. Confirma el veredicto brevemente.\n"
    "  2. Lista ordenada de variables con SHAP positivo (causas del defecto), "
    "     de mayor a menor impacto. Explica en términos físicos qué implica cada una.\n"
    "  3. Acción de revisión específica.\n"
    "\n"
    "· Si pregunta por QUÉ la pieza es OK o qué APOYA LA CALIDAD:\n"
    "  1. Confirma el veredicto brevemente.\n"
    "  2. Lista ordenada de variables con SHAP negativo (factores de calidad), "
    "     de mayor magnitud a menor. Explica su aportación positiva.\n"
    "  3. Alerta preventiva si alguna variable OK está próxima al umbral.\n"
    "\n"
    "· Si pregunta por UNA VARIABLE CONCRETA:\n"
    "  1. Indica el valor medido y su impacto SHAP.\n"
    "  2. Explica en términos físicos qué significa ese valor y hacia dónde empuja.\n"
    "  3. Contexto: cómo afecta al veredicto final.\n"
    "\n"
    "· Si pregunta algo que los datos NO PERMITEN RESPONDER:\n"
    "  Indícalo con claridad y ofrece lo que sí puedes concluir con los datos disponibles.\n"
    "\n"
    "Longitud máxima: 15 segundos de lectura. Sin relleno."
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
    k = len(top_k)

    # Separamos variables por dirección SHAP para ayudar al modelo a razonar
    nok_drivers = [(n, v, sv) for n, v, sv in top_k if sv > 0]
    ok_drivers  = [(n, v, sv) for n, v, sv in top_k if sv <= 0]

    def fmt_var(name, val, sv):
        val_str = f"{val}" if isinstance(val, str) else f"{val:.4f}"
        direccion = "→ DEFECTO" if sv > 0 else "→ CALIDAD"
        return f"  · {name}: valor={val_str} | SHAP={sv:+.4f} {direccion}"

    nok_block = "\n".join(fmt_var(*v) for v in nok_drivers) or "  (ninguna)"
    ok_block  = "\n".join(fmt_var(*v) for v in ok_drivers)  or "  (ninguna)"

    decision_str = (
        f"P(NOK)={proba:.4f} >= umbral={threshold:.4f} → NOK confirmado"
        if proba >= threshold else
        f"P(NOK)={proba:.4f} < umbral={threshold:.4f} → OK confirmado"
    )

    return (
        f"══ DATOS DE INSPECCIÓN ══\n"
        f"Veredicto:              {label}\n"
        f"P(NOK) [prob. defecto]: {proba:.4f} ({proba:.2%})\n"
        f"Decisión:               {decision_str}\n"
        f"\n"
        f"── Variables que EMPUJAN a DEFECTO (SHAP +) ── top {len(nok_drivers)}/{k}\n"
        f"{nok_block}\n"
        f"\n"
        f"── Variables que APOYAN CALIDAD (SHAP -) ── top {len(ok_drivers)}/{k}\n"
        f"{ok_block}\n"
        f"\n"
        f"══ PREGUNTA DEL OPERARIO ══\n"
        f"{user_query}\n"
        f"\n"
        f"Responde adaptándote al tipo de pregunta según las reglas del sistema."
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