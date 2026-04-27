"""
utils/llm_client.py
===================
Cliente Ollama para el asistente IA de PLUTO.

Responsabilidades:
    - Al inicializar, verificar conectividad con `http://localhost:11434`.
    - Si el servidor responde pero el modelo configurado no está disponible,
      mostrar el comando exacto para descargarlo (`ollama pull <modelo>`).
    - Construir prompts dinámicos con contexto SHAP (top-K features).
    - Hacer streaming de la respuesta del LLM mediante un generador.

Rol del LLM: Ingeniero de Calidad de CTAG.
Tono: profesional, técnico pero pedagógico.
Respuestas: < 15 segundos de lectura.
"""

import json
import warnings
from typing import Generator

import requests

warnings.filterwarnings("ignore")

# ── Configuración ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3"      # Cambiar si usas otro modelo (ej. "mistral")
REQUEST_TIMEOUT = 90            # segundos para streaming


# ══════════════════════════════════════════════════════════════════════════════
# Validación de conectividad y modelos disponibles
# ══════════════════════════════════════════════════════════════════════════════

class OllamaStatus:
    """Resultado de la comprobación de Ollama."""
    SERVER_UP    = "server_up"      # servidor OK, modelo OK
    SERVER_DOWN  = "server_down"    # sin conexión
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

    # Servidor activo - verificar si el modelo está disponible
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
        pass  # Si no se puede parsear, asumimos que está OK

    return (
        OllamaStatus.SERVER_UP,
        f"OK Ollama activo - modelo `{model}` disponible.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Construcción del prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    label: str,
    proba: float,
    threshold: float,
    top_k: list[tuple[str, float, float]],
    user_query: str,
) -> str:
    """
    Construye el prompt estructurado para el LLM.

    Args:
        label      : "OK" o "NOK"
        proba      : probabilidad de NOK (float 0–1)
        threshold  : umbral de decisión
        top_k      : lista de (feature_name, feature_value, shap_value)
        user_query : pregunta del operario

    Returns:
        Prompt completo como string.
    """
    k = len(top_k)
    feat_lines = "\n".join(
        f"  • {name:40s}  valor={val:.4f}   SHAP={sv:+.5f}  "
        f"({'↑ NOK' if sv > 0 else '↓ OK'})"
        for name, val, sv in top_k
    )

    return (
        "Eres un Ingeniero de Calidad experto del proyecto PLUTO para la empresa CTAG. "
        "Tu misión es analizar el resultado del sistema de clasificación automática de "
        "piezas industriales y responder al operario de forma técnica pero comprensible.\n\n"
        "══ RESULTADO DE CLASIFICACIÓN ══\n"
        f"• Veredicto final:          {label}\n"
        f"• Probabilidad de NOK:      {proba:.2%}\n"
        f"• Probabilidad de OK:       {1 - proba:.2%}\n"
        f"• Umbral de decisión:       {threshold:.4f}\n\n"
        f"══ VARIABLES MÁS INFLUYENTES - análisis SHAP (Top {k}) ══\n"
        f"{feat_lines}\n\n"
        "Interpretación SHAP:\n"
        "  - Contribución positiva (+) -> la variable empuja la predicción hacia NOK\n"
        "  - Contribución negativa (−) -> la variable empuja la predicción hacia OK\n"
        "  - Las magnitudes indican la fuerza relativa de cada variable\n\n"
        "══ PREGUNTA DEL OPERARIO ══\n"
        f"{user_query}\n\n"
        "Responde de forma técnica pero pedagógica. Sé directo y conciso. "
        "Estructura tu respuesta así:\n"
        "1. Diagnóstico: explica brevemente el resultado.\n"
        "2. Variables críticas: describe qué variables han tenido más impacto.\n"
        "3. Acción recomendada: sugiere qué revisar (si hay NOK) o confirma conformidad.\n"
        "Si algún valor OK está cerca de sus límites, menciónalo como señal preventiva."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Streaming
# ══════════════════════════════════════════════════════════════════════════════

def stream_response(
    prompt: str,
    model: str = LLM_MODEL,
) -> Generator[str, None, None]:
    """
    Generador que envía el prompt a Ollama y hace streaming de la respuesta.

    Yields:
        Texto parcial acumulado (str) en cada chunk.
    """
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
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

