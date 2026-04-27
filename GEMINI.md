# 🧠 SYSTEM PROMPT: Proyecto PLUTO (CTAG) - Prototipo Funcional V1.2

## ⚠️ PROTOCOLO DE AUTOSINCRONIZACIÓN Y VERACIDAD
**Fuente de Verdad Única:** Este archivo (`GEMINI.md`) es la autoridad suprema.
**Verificación de Salida:** Re-lee antes de entregar:
1. Modelo Exp_05 activo.
2. Estructura modular `/app` y `/utils`.
3. **Local-First estricto** (0 APIs externas).
4. **Cero Emojis y Cero Tags** (`[PLUTO]`, `[Form]`, etc.) en la UI.

---

## 1. Contexto y Entorno Multiplataforma
* **Gestión de Rutas:** Uso obligatorio de `pathlib`. Compatible con Windows (OneDrive) y Linux.
* **Privacidad:** Todo el procesamiento (ML, SHAP, Ollama) es local.

---

## 2. El Modelo "Campeón" e Inferencia
* **Modelo:** Exp_05 (`src/models/experiment_05_ultimate_vae_fe.py`).
* **Justificación:** Recall OK de 0.33 (Prioridad CTAG).
* **Dataset:** `data/raw/Dataset_01_Anonimizado.xlsx`.

---

## 3. Arquitectura Modular
* `/app/ui.py`: Interfaz Gradio y lógica de vista.
* `/utils/ml_engine.py`: Motor de predicción (102 -> 123 vars).
* `/utils/explainer.py`: SHAP dinámico.
* `/utils/llm_client.py`: Orquestación Ollama.

---

## 4. Requisitos de Usabilidad y UX (PC Industrial)
Siguiendo las restricciones del AVP2 para entornos de planta:
* **Simplicidad:** Interfaz limpia para operarios con < 10 min de formación.
* **Feedback Inmediato:** Notificar estados (Analizando..., Error de formato, Listo) mediante cambios de color o texto claro.
* **Sin Ruidos Visuales:** * **ELIMINAR TODOS LOS EMOJIS** de la interfaz (nada de 🔬, 📋, 🔍, ✅, etc.).
    * **ELIMINAR TAGS DE TEXTO:** Borrar prefijos como `[PLUTO]`, `[Form]`, `[wait]`, `[Sec]`, `[Search]`, `[Reset]`, `[i]`, `[warn]`.
* **Entrada de Datos:** Formulario manual agrupado y botón de carga CSV con lógica fuzzy (mapeo inteligente de cabeceras).

---

## 5. Módulo de Explicabilidad y Ollama
* **Gestión de Ollama:** Si el servidor responde pero falta el modelo, mostrar comando: `ollama pull llama3`.
* **Integración SHAP:** Inyectar Top 3-5 variables en el prompt de forma transparente para el usuario.
* **Personalidad:** Ingeniero de Calidad CTAG. Tono profesional y técnico.

---

## 6. Estética Industrial y Legibilidad (Paleta CTAG)
Para garantizar legibilidad en planta con iluminación variable, aplica este esquema en el CSS de Gradio:
* **Fondo (Background):** `#0f172a` (Deep Navy Slate).
* **Contenedores:** `#1e293b` con bordes `#334155`.
* **Texto Principal:** `#f8fafc` (Blanco puro/gris muy claro para máximo contraste).
* **Texto Secundario/Labels:** `#94a3b8`.
* **Acentos (Botones):** Azul industrial `#3b82f6` (Primary) y `#475569` (Secondary).
* **Banners de Resultado:** * **OK:** Fondo `#064e3b`, texto `#4ade80`, borde `#22c55e`.
    * **NOK:** Fondo `#7f1d1d`, texto `#fca5a5`, borde `#ef4444`.
* **Componentes Críticos:** Los recuadros de código (Ollama) y el Chatbot **NO deben tener fondo blanco**. Deben integrarse en el tema oscuro usando fondos `#0f172a` y bordes sutiles.
* **Slider de Umbral:** Debe ser funcional y reflejar visualmente la probabilidad respecto al punto de corte **0.6818**.

---

## 7. Estado del Proyecto y Progreso
* [x] Refactorización Modular V2.0.
* [ ] **Misión Actual:** Rediseño UI V2.1 (Legibilidad Industrial y Limpieza de Tags).