#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
# Fix Windows terminal encoding so Unicode prints work
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PLUTO — Sistema de Inspección de Calidad Industrial                        ║
║  Proyecto PIIA · Cliente CTAG · v1.0                                        ║
║                                                                              ║
║  Stack: Gradio · LightGBM · SHAP · Ollama (LLaMA 3 / Mistral)              ║
║  Ejecución: python main.py   →   http://localhost:7860                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura del módulo:
  ┌─ INICIALIZACIÓN ────────────────────────────────────────────────────────┐
  │  1. Carga del modelo .pkl (LightGBM Booster + threshold + feature list) │
  │  2. Lectura de metadatos del Excel (tipos, categorías, rangos)           │
  │  3. Inicialización del SHAP TreeExplainer                                │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ PIPELINE DE PREDICCIÓN ────────────────────────────────────────────────┐
  │  preprocess() → réplica exacta del Experimento 01                       │
  │  run_prediction() → predict + SHAP + banner HTML                        │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ LLM (OLLAMA) ──────────────────────────────────────────────────────────┐
  │  build_llm_prompt() → prompt engineering dinámico con contexto SHAP     │
  │  stream_ollama()    → streaming sobre API REST localhost:11434           │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ INTERFAZ GRADIO ───────────────────────────────────────────────────────┐
  │  Formulario dinámico con N variables agrupadas en acordeones            │
  │  Panel de resultados: banner OK/NOK + slider proba + tabla SHAP         │
  │  Chatbot integrado con streaming del LLM                                │
  └─────────────────────────────────────────────────────────────────────────┘
"""

import json
import warnings
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import requests
import shap

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_optimized_model.pkl"
DATA_PATH  = BASE_DIR / "data" / "raw" / "Dataset_v2_anonimizado.xlsx"
TARGET_COL = "Variable de Salida"

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3"          # Cambiar si usas otro modelo  (ej. "mistral")

TOP_K_SHAP  = 5                      # Número de features a mostrar en SHAP
GROUP_SIZE  = 4                      # Componentes por fila dentro de un acordeón
SECTION_SZ  = 20                     # Variables por sección (acordeón)

# Features añadidas por ingeniería en el Experimento 01 (no son inputs del usuario)
ENG_FEATS = frozenset(["num_sum", "num_mean", "num_std", "num_min", "num_max"])


# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN  (se ejecuta UNA VEZ al arrancar la aplicación)
# ══════════════════════════════════════════════════════════════════════════════

def _startup() -> tuple:
    """
    Carga el modelo, lee los metadatos del dataset y prepara el explainer SHAP.
    Retorna una tupla con todos los artefactos necesarios.
    """
    print("=" * 62)
    print("   PLUTO — Iniciando sistema de inspección de calidad")
    print("=" * 62)

    # ── 1. Modelo ─────────────────────────────────────────────────────────────
    print(f"\n[1/3] Cargando modelo: {MODEL_PATH.name} ...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH}\n"
            "Asegúrate de ejecutar main.py desde la raíz del proyecto PLUTO."
        )
    pkl       = joblib.load(MODEL_PATH)
    booster   = pkl["model"]          # lgb.Booster
    threshold = float(pkl["threshold"])
    all_feats = pkl["features"]       # Lista completa (incluye features de ingeniería)
    orig_feats = [f for f in all_feats if f not in ENG_FEATS]

    print(
        f"   [OK] Threshold={threshold:.4f} | "
        f"Features totales={len(all_feats)} | "
        f"Originales={len(orig_feats)}"
    )

    # ── 2. Metadatos del dataset ───────────────────────────────────────────────
    print(f"\n[2/3] Leyendo metadatos del dataset ({DATA_PATH.name}) ...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATA_PATH}")

    df_meta = pd.read_excel(DATA_PATH)
    df_meta = df_meta.dropna(subset=[TARGET_COL]).drop(columns=[TARGET_COL])

    # Sólo columnas que el modelo conoce (por si el Excel tiene columnas extra)
    known_cols = [c for c in orig_feats if c in df_meta.columns]
    df_meta    = df_meta[known_cols]

    cat_cols = df_meta.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df_meta.select_dtypes(include=[np.number]).columns.tolist()

    # Categorías ordenadas (el mismo orden que pandas usa con .astype('category'))
    categories: dict = {
        col: sorted(df_meta[col].dropna().unique().tolist())
        for col in cat_cols
    }

    # Estadísticos básicos para info de rangos y valores por defecto
    col_stats: dict = {
        col: {
            "min":  float(df_meta[col].min()),
            "max":  float(df_meta[col].max()),
            "mean": float(df_meta[col].mean()),
        }
        for col in num_cols
    }

    print(
        f"   [OK] Columnas categoricas: {len(cat_cols)} | "
        f"Columnas numericas: {len(num_cols)}"
    )

    # ── 3. SHAP Explainer ──────────────────────────────────────────────────────
    print("\n[3/3] Inicializando SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(booster)
    print("   [OK] Listo.\n")
    print("=" * 62)
    print(f"   Abre tu navegador en  http://localhost:7860")
    print("=" * 62 + "\n")

    return (
        booster, threshold, all_feats, orig_feats,
        cat_cols, num_cols, categories, col_stats,
        explainer,
    )


# Ejecutamos la inicialización al importar el módulo
(
    BOOSTER, THRESHOLD, ALL_FEATS, ORIG_FEATS,
    CAT_COLS, NUM_COLS, CATEGORIES, COL_STATS,
    EXPLAINER,
) = _startup()


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESADO  —  réplica exacta del Experimento 01
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(input_values: list) -> pd.DataFrame:
    """
    Convierte la lista de valores del formulario en un DataFrame listo
    para ser consumido por el modelo LightGBM.

    Pasos (idénticos al Experimento 01 - experiment_01_advanced.py):
      1. Mapeo input_values → columnas originales.
      2. Codificación de columnas categóricas como pd.Categorical
         (con el mismo orden de categorías que se usó en entrenamiento).
      3. Cálculo de las 5 features de ingeniería numéricas.
      4. Reordenación al orden estricto de ALL_FEATS.

    Args:
        input_values: lista de valores en el mismo orden que ORIG_FEATS.

    Returns:
        DataFrame de 1 fila con ALL_FEATS columnas.
    """
    row: dict = {}
    for feat, val in zip(ORIG_FEATS, input_values):
        if feat in CAT_COLS:
            row[feat] = str(val) if val is not None else CATEGORIES[feat][0]
        else:
            try:
                row[feat] = float(val)
            except (TypeError, ValueError):
                row[feat] = 0.0

    df = pd.DataFrame([row])

    # Codificar categóricas con pd.Categorical (replica .astype('category'))
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=CATEGORIES[col])

    # Features de ingeniería sobre las columnas numéricas
    num_values = df[NUM_COLS].values.astype(np.float64)
    df["num_sum"]  = np.nansum(num_values,  axis=1)
    df["num_mean"] = np.nanmean(num_values, axis=1)
    df["num_std"]  = np.nanstd(num_values,  axis=1)
    df["num_min"]  = np.nanmin(num_values,  axis=1)
    df["num_max"]  = np.nanmax(num_values,  axis=1)

    return df[ALL_FEATS]  # Orden estricto del entrenamiento


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DE PREDICCIÓN + SHAP
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(*args) -> tuple:
    """
    Pipeline completo: preprocesar → predecir → SHAP → formatear salidas.

    Args:
        *args: valores del formulario en orden ORIG_FEATS.

    Returns:
        Tupla (html_banner, prob_nok, shap_dataframe, state_dict)
    """
    PENDING_HTML = """
    <div class="pending-banner">
        <h1>⏳ Pendiente</h1>
        <p class="banner-sub">Introduce los valores y pulsa Comprobar</p>
    </div>"""

    try:
        df = preprocess(list(args))

        # ── Predicción (<2 s gracias a LightGBM) ─────────────────────────────
        proba = float(BOOSTER.predict(df)[0])
        label = "NOK" if proba >= THRESHOLD else "OK"

        # ── SHAP ──────────────────────────────────────────────────────────────
        sv = EXPLAINER.shap_values(df)
        # Para clasificación binaria LGBM nativo, shap_values puede ser:
        #   - 2D array (n_samples, n_features)  →  valores para clase positiva (NOK)
        #   - lista de dos arrays               →  [class_0_shap, class_1_shap]
        if isinstance(sv, list):
            vals = np.array(sv[1])[0]   # Tomamos clase 1 (NOK), primera muestra
        else:
            vals = np.array(sv)[0]

        feat_arr = np.array(ALL_FEATS)
        top_idx  = np.argsort(np.abs(vals))[::-1][:TOP_K_SHAP]
        top_k    = [
            (feat_arr[i], float(df.iloc[0, i]), float(vals[i]))
            for i in top_idx
        ]

        # ── Banner HTML ───────────────────────────────────────────────────────
        if label == "OK":
            html = f"""
            <div class="ok-banner">
                <h1>✅ PIEZA OK</h1>
                <p class="banner-sub">Pieza CONFORME · P(NOK) = {proba:.2%}</p>
            </div>"""
        else:
            html = f"""
            <div class="nok-banner">
                <h1>❌ PIEZA NOK</h1>
                <p class="banner-sub">Pieza DEFECTUOSA · P(NOK) = {proba:.2%}</p>
            </div>"""

        # ── Tabla SHAP ────────────────────────────────────────────────────────
        shap_df = pd.DataFrame(
            [
                {
                    "Variable":         name,
                    "Valor":            round(val, 4),
                    "Contribución SHAP": round(sv_val, 6),
                }
                for name, val, sv_val in top_k
            ]
        )

        state = {
            "label": label,
            "proba": proba,
            "top_k": top_k,
        }

        return html, proba, shap_df, state

    except Exception as exc:  # noqa: BLE001
        err_html = f"""
        <div class="pending-banner">
            <p style="color:#f87171;font-size:1.1em;">⚠️ Error durante la predicción</p>
            <p style="color:#94a3b8;font-size:0.9em;">{exc}</p>
        </div>"""
        return err_html, 0.0, pd.DataFrame(), {}


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO LLM  —  Ollama (LLaMA 3 / Mistral)
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama() -> bool:
    """Comprueba si Ollama está disponible en localhost:11434."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def build_llm_prompt(label: str, proba: float, top_k: list, user_query: str) -> str:
    """
    Construye el prompt estructurado que se envía al LLM.
    Inyecta: veredicto, probabilidad, top-k SHAP y pregunta del operario.
    """
    feat_lines = "\n".join(
        f"  • {name:40s}  valor={val:.4f}   SHAP={sv_val:+.5f}  "
        f"({'↑ NOK' if sv_val > 0 else '↓ OK'})"
        for name, val, sv_val in (top_k or [])
    )

    return f"""Eres un asistente experto en calidad industrial del proyecto PLUTO para la empresa CTAG.
Analiza el resultado del sistema de clasificación automática de piezas y responde la pregunta del operario.

══ RESULTADO DE CLASIFICACIÓN ══
• Veredicto final:          {label}
• Probabilidad de NOK:      {proba:.2%}
• Probabilidad de OK:       {1 - proba:.2%}
• Umbral de decisión:       {THRESHOLD:.4f}

══ VARIABLES MÁS INFLUYENTES — análisis SHAP (Top {TOP_K_SHAP}) ══
{feat_lines}

Interpretación SHAP:
  - Contribución positiva (+) → la variable empuja la predicción hacia NOK (pieza defectuosa)
  - Contribución negativa (−) → la variable empuja la predicción hacia OK (pieza conforme)
  - Las magnitudes indican la fuerza relativa de cada variable

══ PREGUNTA DEL OPERARIO ══
{user_query}

Responde de forma técnica pero comprensible para un operario de línea de producción.
Sé directo y conciso. Estructura tu respuesta así:
1. Diagnóstico: explica brevemente el resultado.
2. Variables críticas: describe qué variables han tenido más impacto y por qué.
3. Acción recomendada: sugiere qué revisar en el proceso (si hay NOK) o confirma la conformidad.
Si alguna variable OK está cerca de sus límites, menciónalo como señal de alerta preventiva."""


def stream_ollama(prompt: str):
    """
    Generador que envía el prompt a Ollama y hace streaming de la respuesta.
    Yields: texto parcial acumulado (str).
    """
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=90,
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            for raw_line in resp.iter_lines():
                if raw_line:
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
            "❌ **Ollama no está disponible en localhost:11434.**\n\n"
            "Para activarlo:\n"
            "```bash\n"
            f"ollama pull {LLM_MODEL}\n"
            "ollama serve\n"
            "```\n"
            "Después recarga la página."
        )
    except requests.exceptions.Timeout:
        yield "⏱️ **Timeout:** El LLM tardó demasiado en responder. Prueba con un modelo más pequeño."
    except Exception as exc:  # noqa: BLE001
        yield f"❌ Error al conectar con LLM: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# ESTILOS CSS — dark premium theme
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
/* ─── Fuente global ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ─── Banners de resultado ─────────────────────────────────────────────── */
.ok-banner {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 60%, #065f46 100%);
    border: 2px solid #34d399;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    animation: fadeIn 0.35s ease;
}
.ok-banner h1 {
    color: #34d399;
    font-size: 2.8em;
    font-weight: 900;
    margin: 0;
    letter-spacing: -1px;
    text-shadow: 0 0 24px rgba(52,211,153,0.5);
}
.nok-banner {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 60%, #991b1b 100%);
    border: 2px solid #f87171;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    animation: slamIn 0.25s ease;
}
.nok-banner h1 {
    color: #f87171;
    font-size: 2.8em;
    font-weight: 900;
    margin: 0;
    letter-spacing: -1px;
    text-shadow: 0 0 24px rgba(248,113,113,0.5);
}
.pending-banner {
    background: #1e293b;
    border: 2px dashed #3f4f63;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
}
.pending-banner h1 { color: #64748b; font-size: 2em; margin: 0; }
.banner-sub { color: #a3afc0; margin: 10px 0 0; font-size: 1.05em; }

@keyframes fadeIn  { from { opacity:0; transform:scale(0.96); } to { opacity:1; transform:scale(1); } }
@keyframes slamIn  { from { opacity:0; transform:scale(1.04); } to { opacity:1; transform:scale(1); } }

/* ─── Header ─────────────────────────────────────────────────────────────── */
#pluto-header {
    background: linear-gradient(135deg, #1a2844 0%, #0f172a 100%);
    border: 1px solid #2d4168;
    border-radius: 16px;
    padding: 22px 32px;
    margin-bottom: 22px;
}

/* ─── Secciones de información ──────────────────────────────────────────── */
.stat-pill {
    background: #162032;
    border: 1px solid #2d4168;
    border-radius: 10px;
    padding: 10px 18px;
    text-align: center;
    min-width: 130px;
}

/* ─── Separador ─────────────────────────────────────────────────────────── */
.section-divider {
    border: 0;
    border-top: 1px solid #1e2d42;
    margin: 28px 0;
}

/* ─── Tabla SHAP — color por signo ──────────────────────────────────────── */
.shap-table td:last-child { font-weight: 600; }

/* ─── Small label ────────────────────────────────────────────────────────── */
label.small { font-size: 0.78em !important; color: #64748b !important; }

/* ─── Gradio overrides ───────────────────────────────────────────────────── */
.gradio-container { max-width: 1500px !important; }
button.primary { font-weight: 700 !important; letter-spacing: 0.3px; }

/* ─── Dark theme overrides (CSS route, Gradio 6.x compatible) ───────────── */
body, .gradio-container {
    background: #0c1629 !important;
    color: #cbd5e1 !important;
}
.block, .form, .panel, .wrap {
    background: #111c30 !important;
    border-color: #1e2d45 !important;
}
label, .label-wrap span {
    color: #94a3b8 !important;
    font-size: 0.82em !important;
}
input, select, textarea, .wrap input {
    background: #0c1629 !important;
    border-color: #243552 !important;
    color: #e2e8f0 !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
}
button[class*="primary"] {
    background: #2563eb !important;
    border-color: #3b82f6 !important;
    color: white !important;
}
button[class*="primary"]:hover {
    background: #1d4ed8 !important;
}
button[class*="secondary"] {
    background: #1e293b !important;
    color: #cbd5e1 !important;
}
.accordion-header, details summary {
    background: #162032 !important;
    color: #7dd3fc !important;
}
.chatbot, .message-wrap { background: #111c30 !important; }
.svelte-1ed2p3z { background: #0c1629 !important; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE LA INTERFAZ GRADIO
# ══════════════════════════════════════════════════════════════════════════════

def _default_values() -> list:
    """Retorna la lista de valores por defecto para todos los inputs."""
    defaults = []
    for feat in ORIG_FEATS:
        if feat in CAT_COLS:
            cats = CATEGORIES[feat]
            defaults.append(cats[0] if cats else None)
        else:
            st = COL_STATS.get(feat, {})
            defaults.append(round(st.get("mean", 0.0), 4))
    return defaults


def build_app() -> gr.Blocks:
    """Construye y retorna la aplicación Gradio completa."""

    ollama_ok    = check_ollama()
    ollama_badge = (
        '<span style="color:#34d399;font-weight:700;">● Online</span>'
        if ollama_ok else
        '<span style="color:#f87171;font-weight:700;">● Offline</span>'
    )

    # Pre-computar secciones de acordeón
    sections: list[tuple[str, list[str]]] = []
    for i in range(0, len(ORIG_FEATS), SECTION_SZ):
        chunk      = ORIG_FEATS[i: i + SECTION_SZ]
        label_sec  = f"📂 {chunk[0]}  →  {chunk[-1]}"
        sections.append((label_sec, chunk))

    # ── Tema ──────────────────────────────────────────────────────────────────
    # Gradio 6.x: use gr.themes.Soft/Base with only constructor-level hues;
    # custom colours go via CSS variables injected through the css= parameter.
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )

    # ── Blocks ────────────────────────────────────────────────────────────────
    with gr.Blocks(
        theme=theme,
        css=CSS,
        title="PLUTO — Inspección de Calidad CTAG",
    ) as app:

        # ── ESTADO COMPARTIDO ──────────────────────────────────────────────────
        state_result = gr.State(value={})

        # ── CABECERA ───────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="pluto-header">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      flex-wrap:wrap;gap:16px;">
            <div>
              <div style="font-size:2.2em;font-weight:900;color:#7dd3fc;
                          letter-spacing:-1px;">
                🔬 PLUTO
              </div>
              <div style="color:#475f7b;font-size:0.875em;margin-top:3px;">
                Sistema de Inspección de Calidad Industrial · CTAG · v1.0
              </div>
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;">
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;
                            text-transform:uppercase;letter-spacing:1px;">Modelo</div>
                <div style="color:#e2e8f0;font-weight:700;font-size:0.95em;">
                  LightGBM Opt.
                </div>
                <div style="font-size:0.75em;color:#475f7b;">Experimento 01</div>
              </div>
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;
                            text-transform:uppercase;letter-spacing:1px;">Umbral NOK</div>
                <div style="color:#fbbf24;font-weight:700;font-size:1.1em;">
                  {THRESHOLD:.4f}
                </div>
                <div style="font-size:0.75em;color:#475f7b;">
                  P(NOK) &ge; umbral → NOK
                </div>
              </div>
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;
                            text-transform:uppercase;letter-spacing:1px;">
                  Variables
                </div>
                <div style="color:#e2e8f0;font-weight:700;font-size:1.1em;">
                  {len(ORIG_FEATS)}
                </div>
                <div style="font-size:0.75em;color:#475f7b;">
                  {len(NUM_COLS)} num · {len(CAT_COLS)} cat
                </div>
              </div>
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;
                            text-transform:uppercase;letter-spacing:1px;">
                  LLM Ollama
                </div>
                <div style="font-weight:700;">{ollama_badge}</div>
                <div style="font-size:0.75em;color:#475f7b;">{LLM_MODEL}</div>
              </div>
            </div>
          </div>
        </div>
        """)

        # ══════════════════════════════════════════════════════════════════════
        # FILA PRINCIPAL: FORMULARIO  |  RESULTADOS
        # ══════════════════════════════════════════════════════════════════════
        with gr.Row(equal_height=False):

            # ── COLUMNA IZQUIERDA: Formulario ──────────────────────────────────
            with gr.Column(scale=6, min_width=400):
                gr.Markdown("### 📋 Variables de Proceso")
                gr.Markdown(
                    f"Introduce las **{len(ORIG_FEATS)} variables** de proceso. "
                    "Los campos muestran las medias del conjunto de entrenamiento como "
                    "referencia. Los rangos válidos aparecen como tooltip."
                )

                input_comps: list = []   # Se llena en el loop de acordeones

                for sec_label, sec_feats in sections:
                    with gr.Accordion(label=sec_label, open=False):
                        for row_start in range(0, len(sec_feats), GROUP_SIZE):
                            row_feats = sec_feats[row_start: row_start + GROUP_SIZE]
                            with gr.Row():
                                for feat in row_feats:
                                    if feat in CAT_COLS:
                                        cats = CATEGORIES[feat]
                                        comp = gr.Dropdown(
                                            choices=cats,
                                            value=cats[0] if cats else None,
                                            label=feat,
                                            scale=1,
                                            min_width=120,
                                            elem_id=f"inp_{feat.replace(' ', '_')}",
                                        )
                                    else:
                                        st = COL_STATS.get(feat, {})
                                        rng = (
                                            f"Rango: [{st['min']:.2f}, {st['max']:.2f}]"
                                            if st else ""
                                        )
                                        comp = gr.Number(
                                            value=round(st.get("mean", 0.0), 4),
                                            label=feat if not rng else f"{feat}  ({rng})",
                                            scale=1,
                                            min_width=120,
                                            elem_id=f"inp_{feat.replace(' ', '_')}",
                                        )
                                    input_comps.append(comp)

                with gr.Row():
                    btn_predict = gr.Button(
                        "🔍  Comprobar Calidad",
                        variant="primary",
                        size="lg",
                        scale=3,
                    )
                    btn_clear = gr.Button(
                        "🔄  Limpiar",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

            # ── COLUMNA DERECHA: Resultados ────────────────────────────────────
            with gr.Column(scale=4, min_width=340):
                gr.Markdown("### 📊 Resultado de Inspección")

                result_html = gr.HTML(
                    value="""
                    <div class="pending-banner">
                        <h1>⏳ Pendiente</h1>
                        <p class="banner-sub">
                            Introduce las variables y pulsa <strong>Comprobar</strong>
                        </p>
                    </div>"""
                )

                result_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.001,
                    label=f"Probabilidad de NOK  (umbral = {THRESHOLD:.4f})",
                    interactive=False,
                )

                gr.Markdown(
                    "#### 🔎 Top Variables por Impacto SHAP\n"
                    "_SHAP (+) → empuja a NOK · SHAP (−) → empuja a OK_"
                )
                shap_table = gr.DataFrame(
                    value=None,
                    label="",
                    interactive=False,
                )

        # ══════════════════════════════════════════════════════════════════════
        # CHATBOT  —  Asistente IA con contexto SHAP
        # ══════════════════════════════════════════════════════════════════════
        gr.HTML('<hr class="section-divider">')
        gr.Markdown("### 💬 Asistente IA — Diagnóstico y Análisis")
        gr.Markdown(
            "Pregunta al asistente sobre la inspección actual. "
            "El contexto (veredicto, probabilidad y análisis SHAP) se incluye "
            "automáticamente en cada consulta.\n\n"
            + (
                f"> ℹ️ **Asistente LLM ({LLM_MODEL}):** activo y listo."
                if ollama_ok else
                f"> ⚠️ **Ollama no detectado.** Ejecuta `ollama pull {LLM_MODEL}` "
                "y `ollama serve`. Recarga la página tras activarlo."
            )
        )

        chatbot = gr.Chatbot(
            label="",
            height=420,
        )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder=(
                    "Ej: ¿Por qué esta pieza es NOK? "
                    "¿Qué variable tiene más peso en el fallo? "
                    "¿Hay algún valor cerca del límite?"
                ),
                label="",
                lines=2,
                scale=9,
                show_label=False,
                elem_id="chat_input",
            )
            btn_send = gr.Button(
                "Enviar →",
                variant="primary",
                scale=1,
                min_width=100,
                elem_id="btn_send",
            )

        # ══════════════════════════════════════════════════════════════════════
        # MANEJADORES DE EVENTOS
        # ══════════════════════════════════════════════════════════════════════

        # ── Predicción ─────────────────────────────────────────────────────────
        btn_predict.click(
            fn=run_prediction,
            inputs=input_comps,
            outputs=[result_html, result_slider, shap_table, state_result],
            show_progress="full",
        )

        # ── Limpiar formulario ─────────────────────────────────────────────────
        def on_clear():
            defs    = _default_values()
            pending = """
            <div class="pending-banner">
                <h1>⏳ Pendiente</h1>
                <p class="banner-sub">
                    Introduce las variables y pulsa <strong>Comprobar</strong>
                </p>
            </div>"""
            return (*defs, pending, 0.0, pd.DataFrame(), {})

        btn_clear.click(
            fn=on_clear,
            inputs=[],
            outputs=[*input_comps, result_html, result_slider, shap_table, state_result],
        )

        # ── Chat ───────────────────────────────────────────────────────────────
        def on_chat(user_msg: str, history: list, result: dict):
            """Envía la pregunta al LLM con contexto SHAP y hace streaming."""
            if not user_msg.strip():
                yield history, ""
                return

            if not result or not result.get("label"):
                reply = (
                    "⚠️ No hay una predicción activa todavía.\n\n"
                    "Por favor, introduce los valores de las variables de proceso "
                    "y pulsa **🔍 Comprobar Calidad** primero."
                )
                yield history + [(user_msg, reply)], ""
                return

            if not check_ollama():
                reply = (
                    f"❌ **Ollama no disponible** en `localhost:11434`.\n\n"
                    "Para activarlo:\n"
                    "```bash\n"
                    f"ollama pull {LLM_MODEL}\n"
                    "ollama serve\n"
                    "```\n"
                    "Recarga la página tras completar estos pasos."
                )
                yield history + [(user_msg, reply)], ""
                return

            prompt  = build_llm_prompt(
                result["label"],
                result["proba"],
                result.get("top_k", []),
                user_msg,
            )
            history = history + [(user_msg, "")]
            for partial in stream_ollama(prompt):
                history[-1] = (user_msg, partial)
                yield history, ""

        btn_send.click(
            fn=on_chat,
            inputs=[user_input, chatbot, state_result],
            outputs=[chatbot, user_input],
        )
        user_input.submit(
            fn=on_chat,
            inputs=[user_input, chatbot, state_result],
            outputs=[chatbot, user_input],
        )

    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    application = build_app()
    application.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )
