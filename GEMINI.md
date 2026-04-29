# 🧠 SYSTEM PROMPT: Proyecto PLUTO (CTAG) - Prototipo Funcional V2.2

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
* `/utils/explainer.py`: SHAP dinámico y balanceado.
* `/utils/llm_client.py`: Orquestación Ollama y System Prompt.

---

## 4. Requisitos de Usabilidad y UX (PC Industrial)
Siguiendo las restricciones del AVP2 para entornos de planta:
* **Simplicidad:** Interfaz limpia para operarios con < 10 min de formación.
* **Feedback Inmediato:** Notificar estados (Analizando..., Error de formato, Listo) mediante cambios de color o texto claro.
* **Sin Ruidos Visuales:** * **ELIMINAR TODOS LOS EMOJIS** de la interfaz.
  * **ELIMINAR TAGS DE TEXTO:** Borrar prefijos como `[PLUTO]`, `[Form]`, `[wait]`, `[i]`, `[warn]`, etc.
* **Entrada de Datos:** Formulario manual y carga CSV con lógica fuzzy e imputación de medias.

---

## 5. Módulo de Explicabilidad (SHAP Dinámico CTAG) en `explainer.py`

### 5.1 Agregación y Resolución de Signos
* **Suma Algebraica Estricta:** Se deben sumar los valores SHAP de las variables derivadas (ej. `_bin`) a su variable física base. 
* **Gestión de Contradicciones:** Si la variable base tiene signo negativo (apoya OK) y su binarizada signo positivo (apoya NOK), se deben restar algebraicamente. El resultado final representa el **impacto neto** del sensor físico.
* **Identidad:** El nombre resultante debe ser siempre el de la variable física original (eliminando sufijos).

### 5.2 Filtro de Relevancia (Caída Relativa)
* **Regla del 15%:** Una vez agregadas las variables, se descartan aquellas cuyo valor absoluto sea inferior al 15% del valor absoluto de la variable con mayor impacto (Top 1).
* **Excepción de Mínimos:** Esta regla de descarte queda supeditada a mantener siempre el **Mínimo de 3 variables** en la tabla.

### 5.3 Algoritmo de Balanceo e Integridad (Mín 3, Máx 10)
Para gestionar información que pueda parecer contradictoria entre métricas VAE y reales:
1. **Prioridad 1 (Mínimo):** Asegurar 3 variables, priorizando las de mayor impacto neto, sin importar su origen.
2. **Prioridad 2 (Balanceo):** Garantizar que `Variables Reales >= Variables VAE`. Si tras el Top inicial predominan las VAE, se debe realizar un "scaneo" descendente en el ranking SHAP buscando exclusivamente variables reales.
3. **Gestión de Escasez:** Si se agotan las variables reales en el ranking antes de alcanzar el empate con las VAE, el algoritmo se detendrá conservando todas las reales encontradas (no inventar ni duplicar registros).
4. **Prioridad 3 (Máximo):** El listado nunca superará las 10 filas para evitar la sobrecarga cognitiva del operario.

### 5.4 Renombrado Industrial
* `vae_err` -> **"Índice de Correlación Global"**.
* `latent_...` -> **"Patrón Estructural [N]"**.

---

## 6. Lógica del LLM y System Prompt en `llm_client.py`
* **Umbral Sagrado:** 0.6818. Si P(NOK) < 0.6818, la pieza es OK.
* **Interpretación SHAP:** (+) es Defecto, (-) es Calidad. 
* **Instrucción de Mensajes:** Enviar al LLM la tabla final procesada por `explainer.py` (con nombres industriales y valores netos).
* **Rol:** Ingeniero Senior CTAG. Tono asertivo y técnico. Sin ambigüedades.

---

## 7. Estética Industrial y Legibilidad (Paleta CTAG)
* **Fondo:** `#0f172a`. **Texto Principal:** `#f8fafc`.
* **Markdown:** Forzar color `#f8fafc` en títulos y párrafos para legibilidad sobre fondo oscuro.
* **Banners:** Colores semafóricos (Verde para OK, Rojo para NOK) según umbral 0.6818.
* **Slider:** `interactive=False`. Sin elementos HTML flotantes que se desalineen.

---

## 8. Estado del Proyecto y Progreso
* [x] Refactorización Modular V2.0.
* [x] Rediseño UI V2.1 (Limpieza de Tags y Legibilidad).
* [x] V2.2 - Implementacion de `explainer.py` con Agregacion Neta y Balanceo Dinamico Real/VAE.