"""
app/ui.py
=========
Interfaz Gradio de PLUTO - Sistema de Inspección de Calidad Industrial (CTAG).

Responsabilidades:
    - Formulario manual agrupado en acordeones (variables numéricas y categóricas).
    - Botón de carga CSV para autocompletado automático de todos los campos.
    - Validación de rangos: warnings visuales (no bloqueantes) si un valor
      está fuera del rango detectado en el dataset de referencia.
    - Panel de resultados: banner OK/NOK + slider de probabilidad + tabla SHAP.
    - Chatbot con streaming del LLM Ollama, con contexto SHAP inyectado.

Motor: Experimento 05 (VAE + CatBoost) - cargado desde `utils/ml_engine.py`.
"""

import tempfile
import warnings
from html import escape
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from utils.ml_engine  import get_engine
from utils.explainer  import get_explainer
from utils.llm_client import (
    LLM_MODEL,
    OllamaStatus,
    build_prompt,
    check_ollama,
    stream_response,
)

warnings.filterwarnings("ignore")

# ── Configuración de la UI ────────────────────────────────────────────────────
GROUP_SIZE = 4    # Inputs por fila dentro de un acordeón
SECTION_SZ = 20   # Variables por sección (acordeón)

# ══════════════════════════════════════════════════════════════════════════════
# ESTILOS CSS - Dark premium theme
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* --- Banners ---------------------------------------------------------------- */
.ok-banner {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 60%, #065f46 100%);
    border: 2px solid #34d399; border-radius: 16px;
    padding: 32px 24px; text-align: center; animation: fadeIn 0.35s ease;
}
.ok-banner h1 {
    color: #34d399; font-size: 2.8em; font-weight: 900; margin: 0;
    letter-spacing: -1px; text-shadow: 0 0 24px rgba(52,211,153,0.5);
}
.nok-banner {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 60%, #991b1b 100%);
    border: 2px solid #f87171; border-radius: 16px;
    padding: 32px 24px; text-align: center; animation: slamIn 0.25s ease;
}
.nok-banner h1 {
    color: #f87171; font-size: 2.8em; font-weight: 900; margin: 0;
    letter-spacing: -1px; text-shadow: 0 0 24px rgba(248,113,113,0.5);
}
.pending-banner {
    background: #1e293b; border: 2px dashed #3f4f63;
    border-radius: 16px; padding: 32px 24px; text-align: center;
}
.pending-banner h1 { color: #64748b; font-size: 2em; margin: 0; }
.banner-sub { color: #a3afc0; margin: 10px 0 0; font-size: 1.05em; }
.warning-banner {
    background: #1c1a07; border: 1px solid #ca8a04;
    border-radius: 10px; padding: 12px 18px; margin-top: 8px;
    color: #fbbf24; font-size: 0.9em;
}

@keyframes fadeIn { from { opacity:0; transform:scale(0.96); } to { opacity:1; transform:scale(1); } }
@keyframes slamIn { from { opacity:0; transform:scale(1.04); } to { opacity:1; transform:scale(1); } }

/* --- Header ----------------------------------------------------------------- */
#pluto-header {
    background: linear-gradient(135deg, #1a2844 0%, #0f172a 100%);
    border: 1px solid #2d4168; border-radius: 16px;
    padding: 22px 32px; margin-bottom: 22px;
}

/* --- Stat pill -------------------------------------------------------------- */
.stat-pill {
    background: #162032; border: 1px solid #2d4168;
    border-radius: 10px; padding: 10px 18px;
    text-align: center; min-width: 130px;
}

/* --- Separador -------------------------------------------------------------- */
.section-divider { border: 0; border-top: 1px solid #1e2d42; margin: 28px 0; }

/* --- Gradio overrides ------------------------------------------------------- */
.gradio-container { max-width: 1500px !important; }
button.primary { font-weight: 700 !important; letter-spacing: 0.3px; }
body, .gradio-container { background: #0c1629 !important; color: #cbd5e1 !important; }
.block, .form, .panel, .wrap { background: #111c30 !important; border-color: #1e2d45 !important; }
label, .label-wrap span { color: #94a3b8 !important; font-size: 0.82em !important; }
input, select, textarea, .wrap input {
    background: #0c1629 !important; border-color: #243552 !important; color: #e2e8f0 !important;
}
input:focus, select:focus, textarea:focus { border-color: #3b82f6 !important; outline: none !important; }
button[class*="primary"] { background: #2563eb !important; border-color: #3b82f6 !important; color: white !important; }
button[class*="primary"]:hover { background: #1d4ed8 !important; }
button[class*="secondary"] { background: #1e293b !important; color: #cbd5e1 !important; }
.accordion-header, details summary { background: #162032 !important; color: #7dd3fc !important; }
.chatbot, .message-wrap { background: #111c30 !important; }
/* Componentes chatbot y codigo sin fondo blanco */
.chatbot .message, .chatbot .bot, .chatbot .user {
    background: #0f172a !important; border: 1px solid #1e2d45 !important;
}
/* Contraste texto Markdown: blanco industrial (Sec 7 GEMINI.md v2.2) */
.prose h1, .prose h2, .prose h3, .prose h4,
.prose p, .prose strong, .prose em, .prose span,
.prose li, .prose ul, .prose ol {
    color: #f8fafc !important;
}
/* Bloques de codigo integrados al tema oscuro */
.prose pre, .prose code {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
}
"""

PENDING_HTML = """
<div class="pending-banner">
    <h1>Pendiente</h1>
    <p class="banner-sub">Introduce los valores y pulsa <strong>Comprobar</strong></p>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _default_values(eng) -> list:
    """Valores por defecto para todos los inputs (media numérica / primera categoría)."""
    defaults = []
    for feat in eng.num_cols:
        st = eng.col_stats.get(feat, {})
        defaults.append(round(st.get("mean", 0.0), 4))
    for feat in eng.cat_cols:
        cats = eng.categories.get(feat, [])
        defaults.append(cats[0] if cats else None)
    return defaults


def _is_missing_value(value) -> bool:
    """True si el valor representa ausencia de dato en un input de Gradio."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validation_html(errors: list[str], warnings_list: list[str]) -> str:
    """Construye el bloque HTML de errores y avisos de validacion."""
    blocks = []
    if errors:
        blocks.append(
            '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
            '<b>Corrige estos valores antes de predecir:</b><br>'
            + "<br>".join(errors)
            + "</div>"
        )
    if warnings_list:
        blocks.append(
            '<div class="warning-banner">'
            '<b>Avisos de validacion:</b><br>'
            + "<br>".join(warnings_list)
            + "</div>"
        )
    return "".join(blocks)


def _build_manual_row(values: list, eng) -> tuple[dict, str, bool]:
    """
    Valida los inputs manuales y construye la fila interna para el modelo.

    - Numericos ausentes: se imputan con la media de entrenamiento y se avisa.
    - Numericos invalidos/no finitos: bloquean la prediccion.
    - Numericos fuera de rango: avisan, pero no bloquean.
    - Categoricos ausentes: se imputan con la primera categoria conocida.
    - Categoricos desconocidos: bloquean la prediccion.
    """
    row = {}
    errors: list[str] = []
    warnings_list: list[str] = []

    num_vals = values[: len(eng.num_cols)]
    cat_vals = values[len(eng.num_cols):]

    for feat, val in zip(eng.num_cols, num_vals):
        feat_html = escape(str(feat))
        st = eng.col_stats.get(feat, {})
        default = float(st.get("mean", 0.0))

        if _is_missing_value(val):
            row[feat] = default
            warnings_list.append(
                f"<b>{feat_html}</b>: valor ausente; se usa la media "
                f"<b>{default:.4f}</b>."
            )
            continue

        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(
                f"<b>{feat_html}</b>: valor numerico invalido "
                f"(<code>{escape(str(val))}</code>)."
            )
            continue

        if not np.isfinite(v):
            errors.append(
                f"<b>{feat_html}</b>: el valor debe ser finito "
                f"(<code>{escape(str(val))}</code>)."
            )
            continue

        row[feat] = v
        if st and (v < st["min"] or v > st["max"]):
            warnings_list.append(
                f"<b>{feat_html}</b>: valor <b>{v:.4f}</b> fuera del rango "
                f"[{st['min']:.4f}, {st['max']:.4f}]."
            )

    for feat, val in zip(eng.cat_cols, cat_vals):
        feat_html = escape(str(feat))
        cats = eng.categories.get(feat, [])

        if _is_missing_value(val):
            default = cats[0] if cats else ""
            row[feat] = default
            warnings_list.append(
                f"<b>{feat_html}</b>: categoria ausente; se usa "
                f"<b>{escape(str(default))}</b>."
            )
            continue

        text_val = str(val)
        if cats and text_val not in cats:
            errors.append(
                f"<b>{feat_html}</b>: categoria no reconocida "
                f"(<code>{escape(text_val)}</code>)."
            )
            continue

        row[feat] = text_val

    return row, _validation_html(errors, warnings_list), bool(errors)


# ══════════════════════════════════════════════════════════════════════════════
# Construcción de la app Gradio
# ══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    """Construye y devuelve la aplicación Gradio completa."""

    # ── Carga de artefactos ───────────────────────────────────────────────────
    eng      = get_engine()
    explainer= get_explainer(eng)

    ollama_status, ollama_msg = check_ollama(LLM_MODEL)
    ollama_ok = (ollama_status == OllamaStatus.SERVER_UP)
    ollama_badge = (
        '<span style="color:#34d399;font-weight:700;">* Online</span>'
        if ollama_ok else
        '<span style="color:#f87171;font-weight:700;">* Offline</span>'
    )

    # Pre-computar secciones de acordeon (chunks de SECTION_SZ variables)
    orig_feats = eng.orig_feats      # num_cols + cat_cols
    sections: list[list[str]] = [
        orig_feats[i: i + SECTION_SZ]
        for i in range(0, len(orig_feats), SECTION_SZ)
    ]

    # ── Tema ──────────────────────────────────────────────────────────────────
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Blocks
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Blocks(theme=theme, css=CSS, title="PLUTO - Inspeccion de Calidad CTAG") as app:

        state_result     = gr.State(value={})
        file_valid_state = gr.State(value=False)   # True solo cuando hay fichero valido cargado
        batch_state      = gr.State(value=None)    # DataFrame de resultados del lote activo

        # ── CABECERA ──────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="pluto-header">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
            <div>
              <div style="font-size:2.2em;font-weight:900;color:#7dd3fc;letter-spacing:-1px;">
                PLUTO
              </div>
              <div style="color:#475f7b;font-size:0.875em;margin-top:3px;">
                Sistema de Inspeccion de Calidad Industrial - CTAG - v2.2 - Motor: Exp_05 (VAE+CatBoost)
              </div>
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;">
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;text-transform:uppercase;letter-spacing:1px;">Modelo</div>
                <div style="color:#e2e8f0;font-weight:700;font-size:0.95em;">VAE + CatBoost</div>
                <div style="font-size:0.75em;color:#475f7b;">Experimento 05</div>
              </div>
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;text-transform:uppercase;letter-spacing:1px;">Umbral NOK</div>
                <div style="color:#fbbf24;font-weight:700;font-size:1.1em;">{eng.threshold:.4f}</div>
                <div style="font-size:0.75em;color:#475f7b;">P(NOK) >= umbral -> NOK</div>
              </div>
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;text-transform:uppercase;letter-spacing:1px;">Variables</div>
                <div style="color:#e2e8f0;font-weight:700;font-size:1.1em;">{len(eng.num_cols) + len(eng.cat_cols)}</div>
                <div style="font-size:0.75em;color:#475f7b;">{len(eng.num_cols)} num - {len(eng.cat_cols)} cat</div>
              </div>
              <div class="stat-pill">
                <div style="font-size:0.7em;color:#475f7b;text-transform:uppercase;letter-spacing:1px;">LLM Ollama</div>
                <div style="font-weight:700;">{ollama_badge}</div>
                <div style="font-size:0.75em;color:#475f7b;">{LLM_MODEL}</div>
              </div>
            </div>
          </div>
        </div>
        """)

        # ════════════════════════════════════════════════════════════════════
        # FILA PRINCIPAL: FORMULARIO | RESULTADOS
        # ════════════════════════════════════════════════════════════════════
        with gr.Row(equal_height=False):

            # ── Columna izquierda: Formulario ─────────────────────────────
            with gr.Column(scale=6, min_width=400):
                gr.Markdown("### Variables de Proceso")
                gr.Markdown(
                    f"Introduce las **{len(orig_feats)} variables** de proceso. "
                    "Los campos muestran las medias del conjunto de entrenamiento. "
                    "O bien carga un **CSV** para autocompletado automatico."
                )

                # Carga de fichero (CSV o Excel)
                with gr.Row():
                    csv_upload = gr.File(
                        label="Cargar fichero (CSV o Excel)",
                        file_types=[".csv", ".xlsx", ".xls"],
                        scale=1,
                        elem_id="csv_upload",
                    )

                range_warnings = gr.HTML(value="", label="")

                input_comps: list = []

                for sec_feats in sections:
                    sec_label = f"{sec_feats[0]}  ->  {sec_feats[-1]}"
                    with gr.Accordion(label=sec_label, open=False):
                        for row_start in range(0, len(sec_feats), GROUP_SIZE):
                            row_feats = sec_feats[row_start: row_start + GROUP_SIZE]
                            with gr.Row():
                                for feat in row_feats:
                                    if feat in eng.cat_cols:
                                        cats = eng.categories.get(feat, [])
                                        comp = gr.Dropdown(
                                            choices=cats,
                                            value=cats[0] if cats else None,
                                            label=feat,
                                            scale=1,
                                            min_width=120,
                                            elem_id=f"inp_{feat.replace(' ', '_')}",
                                        )
                                    else:
                                        st  = eng.col_stats.get(feat, {})
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
                        "Comprobar Calidad",
                        variant="primary", size="lg", scale=3,
                        elem_id="btn_predict",
                    )
                    btn_clear = gr.Button(
                        "Limpiar",
                        variant="secondary", size="lg", scale=1,
                        elem_id="btn_clear",
                    )

            # ── Columna derecha: Resultados ───────────────────────────────
            with gr.Column(scale=4, min_width=340):
                gr.Markdown("### Resultado de Inspeccion")

                result_html = gr.HTML(value=PENDING_HTML)

                result_slider = gr.Slider(
                    minimum=0, maximum=1, value=0, step=0.001,
                    label=f"Probabilidad de NOK  (umbral = {eng.threshold:.4f})",
                    interactive=False,
                    elem_id="result_slider",
                )

                gr.Markdown(
                    "#### Top Variables por Impacto SHAP\n"
                    "_SHAP (+) empuja a NOK - SHAP (-) empuja a OK_"
                )
                shap_table = gr.DataFrame(value=None, label="", interactive=False)

        # ════════════════════════════════════════════════════════════════════
        # PREDICCION POR LOTES
        # ════════════════════════════════════════════════════════════════════
        gr.HTML('<hr class="section-divider">')
        gr.Markdown("### Prediccion por Lotes")
        gr.Markdown(
            f"Sube un fichero con **multiples filas** para obtener el veredicto de cada pieza. "
            f"Las columnas ausentes se imputan con la media. "
            f"Columnas numericas con texto invalido bloquean el lote."
        )

        with gr.Row():
            batch_file = gr.File(
                label="Fichero de lote (CSV o Excel, multifila)",
                file_types=[".csv", ".xlsx", ".xls"],
                scale=4,
                elem_id="batch_file",
            )
            btn_batch = gr.Button(
                "Predecir Lote",
                variant="primary",
                size="lg",
                scale=1,
                min_width=160,
                elem_id="btn_batch",
            )

        batch_status   = gr.HTML(value="", label="")
        batch_table    = gr.DataFrame(value=None, label="Resultados del lote", interactive=False)
        batch_download = gr.File(label="Descargar resultados (CSV)", interactive=False, elem_id="batch_download")

        gr.Markdown(
            "Consulta al asistente sobre los resultados del lote. "
            "Puedes preguntar por piezas concretas, estadisticas o patrones."
        )
        batch_chatbot = gr.Chatbot(label="", height=320, elem_id="batch_chatbot", type="messages")
        with gr.Row():
            batch_user_input = gr.Textbox(
                placeholder=(
                    "Ej: ¿Cuantas piezas son NOK? "
                    "¿Que pieza tiene mayor probabilidad de defecto? "
                    "Describe la pieza 3."
                ),
                label="", lines=2, scale=9,
                show_label=False, elem_id="batch_chat_input",
            )
            btn_batch_send = gr.Button(
                "Enviar ->", variant="primary",
                scale=1, min_width=100, elem_id="btn_batch_send",
            )

        # ════════════════════════════════════════════════════════════════════
        # CHATBOT
        # ════════════════════════════════════════════════════════════════════
        gr.HTML('<hr class="section-divider">')
        gr.Markdown("### Asistente IA - Diagnostico y Analisis")
        gr.Markdown(
            "Pregunta al asistente sobre la inspeccion actual. "
            "El contexto (veredicto, probabilidad y analisis SHAP) se incluye automaticamente.\n\n"
            + (
                f"> **Asistente LLM ({LLM_MODEL}):** activo y listo."
                if ollama_ok else
                f"> **Ollama no detectado.** {ollama_msg}"
            )
        )

        chatbot = gr.Chatbot(label="", height=420, elem_id="chatbot", type="messages")

        with gr.Row():
            user_input = gr.Textbox(
                placeholder=(
                    "Ej: ¿Por qué esta pieza es NOK? "
                    "¿Qué variable tiene más peso? "
                    "¿Hay algún valor cerca del límite?"
                ),
                label="", lines=2, scale=9,
                show_label=False, elem_id="chat_input",
            )
            btn_send = gr.Button(
                "Enviar ->", variant="primary",
                scale=1, min_width=100, elem_id="btn_send",
            )

        # ════════════════════════════════════════════════════════════════════
        # MANEJADORES DE EVENTOS
        # ════════════════════════════════════════════════════════════════════

        # ── Predicción ────────────────────────────────────────────────────
        def on_predict(file_valid, *args):
            # Guardia: sin fichero valido no hay prediccion
            if not file_valid:
                blocked_html = (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Prediccion no disponible.</b> '
                    'Carga un fichero CSV o Excel valido antes de comprobar la calidad.'
                    '</div>'
                )
                return "", blocked_html, 0.0, pd.DataFrame(), {}

            values = list(args)
            row, validation_html, has_errors = _build_manual_row(values, eng)

            if has_errors:
                err_html = """
                <div class="pending-banner">
                    <p style="color:#f87171;font-size:1.1em;">Prediccion bloqueada por datos invalidos</p>
                    <p style="color:#94a3b8;font-size:0.9em;">Corrige los campos indicados antes de comprobar la calidad.</p>
                </div>"""
                return validation_html, err_html, 0.0, pd.DataFrame(), {}

            try:
                label, proba, df_fe = eng.predict(row)
                top_k = explainer.explain(df_fe, top_k_range=(3, 10))

                if label == "OK":
                    html = f"""
                    <div class="ok-banner">
                        <h1>PIEZA OK</h1>
                        <p class="banner-sub">Pieza CONFORME - P(NOK) = {proba:.2%}</p>
                    </div>"""
                else:
                    html = f"""
                    <div class="nok-banner">
                        <h1>PIEZA NOK</h1>
                        <p class="banner-sub">Pieza DEFECTUOSA - P(NOK) = {proba:.2%}</p>
                    </div>"""

                shap_df = pd.DataFrame(
                    [
                        {
                            "Variable":          name,
                            "Valor":             round(val, 4) if isinstance(val, (int, float)) else val,
                            "SHAP Neto":         round(sv, 6),
                            "Direccion":         "DEFECTO" if sv > 0 else "CALIDAD",
                        }
                        for name, val, sv in top_k
                    ],
                    columns=["Variable", "Valor", "SHAP Neto", "Direccion"],
                )

                state = {"label": label, "proba": proba, "top_k": top_k}
                return validation_html, html, proba, shap_df, state

            except Exception as exc:
                err_html = f"""
                <div class="pending-banner">
                    <p style="color:#f87171;font-size:1.1em;">Error durante la prediccion</p>
                    <p style="color:#94a3b8;font-size:0.9em;">{exc}</p>
                </div>"""
                return validation_html, err_html, 0.0, pd.DataFrame(), {}

        btn_predict.click(
            fn=on_predict,
            inputs=[file_valid_state, *input_comps],
            outputs=[range_warnings, result_html, result_slider, shap_table, state_result],
            show_progress="full",
        )

        # ── Limpiar ───────────────────────────────────────────────────────
        def on_clear():
            defs = _default_values(eng)
            # False: el operario debe volver a cargar fichero tras limpiar
            return ("", PENDING_HTML, 0.0, pd.DataFrame(), {}, False, *defs)

        btn_clear.click(
            fn=on_clear,
            inputs=[],
            outputs=[range_warnings, result_html, result_slider, shap_table, state_result,
                     file_valid_state, *input_comps],
        )

        # ── Autocompletado desde fichero (CSV o Excel) con validación estricta ──
        def on_csv_upload(file_obj):
            """
            Valida el fichero antes de autofill.
            Solo pone file_valid_state=True si el fichero se leyó correctamente
            Y todas las columnas numéricas presentes tienen valores parseables.
            Columnas ausentes del fichero se imputan silenciosamente (subset valido).
            """
            _no_warn   = ""     # sin aviso
            _defaults  = _default_values(eng)

            if file_obj is None:
                return _no_warn, False, *_defaults

            # 1. Leer el fichero (error de lectura = bloqueo)
            try:
                path = Path(file_obj.name)
                ext  = path.suffix.lower()
                if ext in (".xlsx", ".xls"):
                    df_csv = pd.read_excel(path, nrows=1)
                else:
                    df_csv = pd.read_csv(path, nrows=1)
            except Exception as e:
                err_html = (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Fichero no valido.</b> No se pudo leer el contenido.<br>'
                    f'<code style="font-size:0.85em;">{escape(str(e))}</code>'
                    '</div>'
                )
                return err_html, False, *_defaults

            # 2. Fichero vacío
            if df_csv.empty:
                err_html = (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Fichero vacio.</b> El archivo no contiene filas de datos.'
                    '</div>'
                )
                return err_html, False, *_defaults

            # 3. Normalización de cabeceras
            raw_cols       = {str(c).strip().lower(): str(c).strip() for c in df_csv.columns}
            row_data       = df_csv.iloc[0]
            vals: list     = []
            invalid_cols: list = []   # numéricas presentes con valor no parseable
            matched_count  = 0

            for feat in eng.orig_feats:  # num_cols + cat_cols
                feat_clean = feat.strip().lower()

                if feat_clean in raw_cols:
                    matched_count += 1
                    val = row_data[raw_cols[feat_clean]]

                    if feat in eng.num_cols:
                        try:
                            parsed = float(val)
                            if not np.isfinite(parsed):
                                raise ValueError("no finito")
                            vals.append(parsed)
                        except (ValueError, TypeError):
                            # Columna presente pero con valor inválido → registrar
                            invalid_cols.append(
                                f"<b>{escape(feat)}</b> "
                                f"(valor: <code>{escape(str(val))}</code>)"
                            )
                            # Placeholder; si hay invalidos el fichero se bloquea
                            vals.append(round(eng.col_stats.get(feat, {}).get("mean", 0.0), 4))
                    else:
                        vals.append(str(val))
                else:
                    # Columna ausente → imputa silenciosamente (subset valido)
                    if feat in eng.num_cols:
                        vals.append(round(eng.col_stats.get(feat, {}).get("mean", 0.0), 4))
                    else:
                        cats = eng.categories.get(feat, [])
                        vals.append(cats[0] if cats else None)

            # 4. Evaluación final del fichero
            if matched_count == 0:
                err_html = (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Fichero no reconocido.</b> Ninguna columna coincide con las '
                    'variables del modelo. Comprueba que el fichero corresponde a '
                    'datos de inspeccion CTAG.'
                    '</div>'
                )
                return err_html, False, *_defaults

            if invalid_cols:
                cols_list = "<br>".join(invalid_cols)
                n_inv = len(invalid_cols)
                err_html = (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    f'<b>Fichero invalido:</b> {n_inv} columna(s) numerica(s) '
                    f'contienen valores no numericos. Corrige el fichero y vuelve a subirlo.<br><br>'
                    f'{cols_list}'
                    '</div>'
                )
                return err_html, False, *_defaults

            # 5. Fichero limpio → autofill habilitado
            return _no_warn, True, *vals

        csv_upload.change(
            fn=on_csv_upload,
            inputs=[csv_upload],
            outputs=[range_warnings, file_valid_state, *input_comps],
        )

        # ── Prediccion por lotes ──────────────────────────────────────────
        def on_batch_predict(file_obj):
            """
            Valida y predice un fichero con N filas.
            NaN en columnas numericas: se imputa con la media (silencioso).
            Texto en columna numerica presente: bloquea el lote completo.
            """
            _empty = pd.DataFrame(columns=["Fila", "Veredicto", "P(NOK)", "P(OK)"])

            if file_obj is None:
                return (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Sin fichero.</b> Carga un CSV o Excel antes de predecir el lote.'
                    '</div>',
                    _empty, None,
                )

            # 1. Leer fichero completo
            try:
                path = Path(file_obj.name)
                ext  = path.suffix.lower()
                df_raw = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
            except Exception as e:
                return (
                    f'<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    f'<b>Fichero no valido.</b> No se pudo leer el contenido.<br>'
                    f'<code style="font-size:0.85em;">{escape(str(e))}</code></div>',
                    _empty, None,
                )

            if df_raw.empty:
                return (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Fichero vacio.</b> El archivo no contiene filas de datos.</div>',
                    _empty, None,
                )

            # 2. Normalizacion de cabeceras
            raw_cols  = {str(c).strip().lower(): str(c).strip() for c in df_raw.columns}
            orig_cols = eng.num_cols + eng.cat_cols

            if not any(f.strip().lower() in raw_cols for f in orig_cols):
                return (
                    '<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    '<b>Fichero no reconocido.</b> Ninguna columna coincide con las '
                    'variables del modelo. Comprueba que el fichero es de inspeccion CTAG.</div>',
                    _empty, None,
                )

            # 3. Validacion estricta: texto en numericas presentes (NaN se tolera)
            invalid_info: list[str] = []
            for feat in eng.num_cols:
                if len(invalid_info) >= 11:
                    break
                feat_clean = feat.strip().lower()
                if feat_clean not in raw_cols:
                    continue
                for row_idx, val in df_raw[raw_cols[feat_clean]].items():
                    if pd.isna(val):
                        continue
                    try:
                        fv = float(val)
                        if not np.isfinite(fv):
                            raise ValueError()
                    except (ValueError, TypeError):
                        invalid_info.append(
                            f"Fila {int(row_idx)+1}, columna <b>{escape(feat)}</b>: "
                            f"<code>{escape(str(val))}</code>"
                        )
                        if len(invalid_info) >= 10:
                            invalid_info.append("... (primeros 10 errores)")
                            break

            if invalid_info:
                return (
                    f'<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    f'<b>Lote invalido:</b> columnas numericas con valores no numericos.<br><br>'
                    f'{"<br>".join(invalid_info)}</div>',
                    _empty, None,
                )

            # 4. Alinear DataFrame con las 102 variables del motor
            X_aligned = pd.DataFrame(index=range(len(df_raw)))
            for feat in orig_cols:
                feat_clean = feat.strip().lower()
                if feat_clean in raw_cols:
                    X_aligned[feat] = df_raw[raw_cols[feat_clean]].values
                elif feat in eng.num_cols:
                    X_aligned[feat] = eng.col_stats.get(feat, {}).get("mean", 0.0)
                else:
                    cats = eng.categories.get(feat, [])
                    X_aligned[feat] = cats[0] if cats else ""

            # 5. Transformar y predecir el lote completo de una sola vez
            try:
                df_fe_batch = eng._transform(X_aligned)
                probas      = eng._model.predict_proba(df_fe_batch)[:, 1]
                labels      = ["NOK" if p >= eng.threshold else "OK" for p in probas]
            except Exception as e:
                return (
                    f'<div class="warning-banner" style="border-color:#ef4444;color:#fecaca;">'
                    f'<b>Error en prediccion:</b> {escape(str(e))}</div>',
                    _empty, None,
                )

            # 6. Tabla de resultados
            n_total = len(labels)
            n_nok   = sum(1 for lbl in labels if lbl == "NOK")
            n_ok    = n_total - n_nok
            results_df = pd.DataFrame({
                "Fila":      list(range(1, n_total + 1)),
                "Veredicto": labels,
                "P(NOK)":    [round(float(p), 4) for p in probas],
                "P(OK)":     [round(float(1 - p), 4) for p in probas],
            })

            status_html = (
                '<div class="warning-banner" '
                'style="border-color:#34d399;color:#34d399;background:#022c22;">'
                f'<b>Lote procesado:</b> {n_total} piezas &nbsp;|&nbsp; '
                f'OK: {n_ok} ({n_ok/n_total:.1%}) &nbsp;|&nbsp; '
                f'NOK: {n_nok} ({n_nok/n_total:.1%})'
                '</div>'
            )

            # 7. CSV temporal para descarga
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix="_pluto_lote.csv", delete=False, encoding="utf-8"
            )
            results_df.to_csv(tmp.name, index=False)
            tmp.close()

            return status_html, results_df, tmp.name, results_df, []

        btn_batch.click(
            fn=on_batch_predict,
            inputs=[batch_file],
            outputs=[batch_status, batch_table, batch_download, batch_state, batch_chatbot],
            show_progress="full",
        )

        # ── Chat del lote ──────────────────────────────────────────
        def _build_batch_prompt(df: pd.DataFrame, user_msg: str) -> str:
            """
            Construye el prompt de contexto para el LLM sobre el lote.
            - Lotes <=20 filas: tabla completa.
            - Lotes >20 filas: estadisticas + top-5 NOK + top-5 OK.
            """
            n_total = len(df)
            n_nok   = int((df["Veredicto"] == "NOK").sum())
            n_ok    = n_total - n_nok
            p_mean  = df["P(NOK)"].mean()
            p_max   = df["P(NOK)"].max()
            p_min   = df["P(NOK)"].min()
            fila_max = int(df.loc[df["P(NOK)"].idxmax(), "Fila"])
            fila_min = int(df.loc[df["P(NOK)"].idxmin(), "Fila"])

            resumen = (
                f"== RESUMEN DEL LOTE ==\n"
                f"Total piezas:   {n_total}\n"
                f"Conformes (OK): {n_ok} ({n_ok/n_total:.1%})\n"
                f"Defectuosas (NOK): {n_nok} ({n_nok/n_total:.1%})\n"
                f"P(NOK) media:   {p_mean:.4f}\n"
                f"P(NOK) maxima:  {p_max:.4f}  (Pieza {fila_max})\n"
                f"P(NOK) minima:  {p_min:.4f}  (Pieza {fila_min})\n"
                f"Umbral sagrado: {eng.threshold:.4f}\n"
            )

            if n_total <= 20:
                tabla_lines = df.to_string(index=False)
                tabla_ctx   = f"== TABLA COMPLETA ==\n{tabla_lines}\n"
            else:
                top_nok = df.nlargest(5, "P(NOK)")[["Fila", "Veredicto", "P(NOK)"]]
                top_ok  = df.nsmallest(5, "P(NOK)")[["Fila", "Veredicto", "P(NOK)"]]
                tabla_ctx = (
                    f"== TOP 5 PIEZAS CON MAYOR RIESGO (NOK) ==\n"
                    f"{top_nok.to_string(index=False)}\n\n"
                    f"== TOP 5 PIEZAS MAS CONFORMES (OK) ==\n"
                    f"{top_ok.to_string(index=False)}\n"
                    f"(Lote grande: se muestran solo los extremos. "
                    f"Para consultar una pieza concreta, menciona su numero de fila.)\n"
                )

            # Si el usuario menciona una pieza concreta, anadir su fila
            import re
            mention = re.search(r"pieza\s+(\d+)", user_msg, re.IGNORECASE)
            pieza_ctx = ""
            if mention:
                num = int(mention.group(1))
                filas = df[df["Fila"] == num]
                if not filas.empty:
                    row = filas.iloc[0]
                    pieza_ctx = (
                        f"\n== DETALLE PIEZA {num} ==\n"
                        f"Veredicto: {row['Veredicto']}\n"
                        f"P(NOK): {row['P(NOK)']:.4f}\n"
                        f"P(OK):  {row['P(OK)']:.4f}\n"
                    )
                else:
                    pieza_ctx = f"\nNota: La pieza {num} no existe en este lote (total: {n_total}).\n"

            return (
                f"{resumen}\n"
                f"{tabla_ctx}"
                f"{pieza_ctx}\n"
                f"== PREGUNTA DEL OPERARIO ==\n{user_msg}\n"
            )

        _BATCH_SYSTEM = (
            "Eres un Ingeniero Senior de Calidad del proyecto PLUTO para la empresa CTAG. "
            "Se te presenta el resultado de una inspeccion por lotes de piezas industriales. "
            "Responde de forma directa y tecnica. "
            f"El umbral de clasificacion es {eng.threshold:.4f}: "
            f"P(NOK) >= {eng.threshold:.4f} implica pieza DEFECTUOSA. "
            "Puedes referirte a piezas por su numero de fila. "
            "Sé conciso: maxima 15 segundos de lectura por respuesta."
        )

        # Parche: stream_response usa SYSTEM_PROMPT global; para el lote
        # creamos un wrapper que sobreescribe el system field del payload.


        def _stream_batch(prompt: str, model: str = LLM_MODEL):
            """Wrapper que envia el system prompt especifico de lote."""
            import json, requests as _req
            payload = {
                "model":  model,
                "system": _BATCH_SYSTEM,
                "prompt": prompt,
                "stream": True,
            }
            try:
                with _req.post(
                    "http://localhost:11434/api/generate",
                    json=payload, stream=True, timeout=90,
                ) as resp:
                    resp.raise_for_status()
                    buf = ""
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        buf += chunk.get("response", "")
                        yield buf
                        if chunk.get("done", False):
                            break
            except Exception as exc:
                yield f"Error al conectar con el LLM: {exc}"

        def on_batch_chat(user_msg: str, history: list, df_lote):  # noqa: F811
            if not user_msg.strip():
                yield history, ""
                return
            if df_lote is None or (hasattr(df_lote, "empty") and df_lote.empty):
                reply = "No hay resultados de lote. Pulsa **Predecir Lote** primero."
                yield history + [{"role": "user", "content": user_msg},
                                  {"role": "assistant", "content": reply}], ""
                return
            status, _ = check_ollama(LLM_MODEL)
            if status != OllamaStatus.SERVER_UP:
                reply = f"Ollama no disponible. Ejecuta: ollama serve"
                yield history + [{"role": "user", "content": user_msg},
                                  {"role": "assistant", "content": reply}], ""
                return
            prompt  = _build_batch_prompt(df_lote, user_msg)
            history = history + [{"role": "user", "content": user_msg},
                                  {"role": "assistant", "content": ""}]
            for partial in _stream_batch(prompt, model=LLM_MODEL):
                history[-1]["content"] = partial
                yield history, ""

        btn_batch_send.click(
            fn=on_batch_chat,
            inputs=[batch_user_input, batch_chatbot, batch_state],
            outputs=[batch_chatbot, batch_user_input],
        )
        batch_user_input.submit(
            fn=on_batch_chat,
            inputs=[batch_user_input, batch_chatbot, batch_state],
            outputs=[batch_chatbot, batch_user_input],
        )

        # ── Chat individual ─────────────────────────────────────────
        def on_chat(user_msg: str, history: list, result: dict):
            if not user_msg.strip():
                yield history, ""
                return

            if not result or not result.get("label"):
                reply = (
                    "No hay una predicción activa.\n\n"
                    "Introduce las variables y pulsa **Comprobar Calidad** primero."
                )
                yield history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}], ""
                return

            status, _ = check_ollama(LLM_MODEL)
            if status != OllamaStatus.SERVER_UP:
                reply = (
                    f"**Ollama no disponible** en `localhost:11434`.\n\n"
                    f"Ejecuta:\n```bash\nollama pull {LLM_MODEL}\nollama serve\n```"
                )
                yield history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}], ""
                return

            prompt  = build_prompt(
                result["label"],
                result["proba"],
                eng.threshold,
                result.get("top_k", []),
                user_msg,
            )
            
            # Añadimos el mensaje del usuario y un placeholder para el asistente
            history = history + [
                {"role": "user", "content": user_msg}, 
                {"role": "assistant", "content": ""}
            ]
            
            for partial in stream_response(prompt, model=LLM_MODEL):
                history[-1]["content"] = partial
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
