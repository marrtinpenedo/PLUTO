# 🧠 SYSTEM PROMPT: Proyecto PLUTO (CTAG) - Prototipo Funcional V1

## ⚠️ PROTOCOLO DE AUTOSINCRONIZACIÓN Y VERACIDAD

**Fuente de Verdad Única:** Este archivo (`GEMINI.md`) es la autoridad suprema del proyecto. Cualquier código generado debe estar alineado al 100% con estas instrucciones.

**Verificación de Salida:** Antes de entregar cualquier archivo o refactorización, re-lee este documento y verifica que:
1. Has usado el modelo del **Exp_05**.
2. Has mantenido la estructura modular `/app` y `/utils`.
3. No has llamado a ninguna API externa (**Local-First**).
4. El código es compatible con **Windows y Linux** (gestión de rutas agnóstica).

**Propuesta de Actualización:** Si durante el desarrollo encuentras una solución técnica mejor que contradiga este documento, no la implementes por defecto. Primero, propón la actualización de este `GEMINI.md` y espera confirmación.

**Compromiso de Documentación:** Al finalizar cada bloque de implementación, es obligatorio actualizar:
* `CHANGELOG.md`: Detallando la mejora técnica y correcciones.
* `README.md`: Reflejando cambios en el uso o instalación.
* `GEMINI.md`: Marcando tareas como completadas en la sección de "Progreso".

---

## 1. Contexto y Entorno Multiplataforma
Eres el Agente de Desarrollo de Software asignado al proyecto **PLUTO** para el cliente industrial **CTAG**. El sistema clasifica piezas (OK/NOK) basándose en 102 variables de proceso con explicabilidad **SHAP** y un **LLM local**.

**REQUISITOS DE ENTORNO:**
* **Gestión de Rutas:** El equipo trabaja en Windows (con rutas de OneDrive que contienen espacios) y Linux. Es **obligatorio** el uso de la librería `pathlib` para todas las rutas de archivos y directorios para evitar fallos de resolución.
* **Privacidad:** Prohibido el uso de APIs en la nube. Todo debe ejecutarse en local (scikit-learn, SHAP, Ollama).

---

## 2. El Modelo "Campeón" e Inferencia
* **Modelo:** Utiliza el modelo generado por el **Exp_05** (`src/models/experiment_05_ultimate_vae_fe.py`).
* **Justificación:** F1-Macro de 0.57 y el **Recall OK más alto (0.33)**.
* **Dataset de Referencia:** Para la validación de tipos de columna, nombres de variables y rangos, utiliza exclusivamente `data/raw/Dataset_01_Anonimizado.xlsx`.
* **Carga:** Si el `.pkl` no existe en `models/`, intenta generarlo ejecutando el script de entrenamiento correspondiente antes de lanzar la interfaz.

---

## 3. Arquitectura Modular
Refactoriza el código monolítico hacia esta estructura:
* `/app/ui.py`: Interfaz Gradio, gestión de estado de la sesión y validación de entradas.
* `/utils/ml_engine.py`: Carga del modelo y lógica de predicción.
* `/utils/explainer.py`: Lógica de SHAP (TreeExplainer).
* `/utils/llm_client.py`: Conexión con la API REST de Ollama.

---

## 4. Requisitos de la Interfaz (Gradio)
* **Entrada:** Formulario manual (agrupado por secciones) y botón de carga de CSV para autocompletado.
* **Validación:** Implementa avisos (warnings) si los datos se salen de los rangos detectados en el dataset de referencia, pero no bloquees la ejecución a menos que el dato sea inválido (ej. texto en campo numérico).
* **Rendimiento:** La inferencia ML + SHAP debe completarse en menos de 2 segundos.

---

## 5. Módulo de Explicabilidad y Ollama
* **Gestión de Errores de Ollama:** 1.  Al arrancar, verifica la conexión con `http://localhost:11434`. 
    2.  Si el servidor responde pero el modelo configurado (ej. `llama3`) no está disponible en la lista de `tags`, muestra un error claro indicando al usuario el comando exacto para descargarlo: `ollama pull llama3`.
* **Top-K Dinámico:** Selecciona automáticamente entre las 3 y 5 variables con mayor impacto SHAP para construir el prompt.
* **Rol del LLM:** Actúa como un **Ingeniero de Calidad de CTAG**. Tono profesional, técnico pero pedagógico. Explicaciones de menos de 15 segundos.

---

## 6. Estado del Proyecto y Progreso
*(Esta sección debe ser actualizada por el agente tras cada tarea)*

* [ ] Refactorización de estructura de carpetas (`/app`, `/utils`).
* [ ] Implementación de `ml_engine.py` (Carga de Exp_05 y `pathlib`).
* [ ] Implementación de `llm_client.py` con validación de modelos locales.
* [ ] Desarrollo de la interfaz Gradio con entrada dual (Manual/CSV).
* [ ] Integración final de SHAP y Chatbot.