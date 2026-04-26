# SYSTEM PROMPT: Proyecto PLUTO (CTAG) - Prototipo Funcional V1

## PROTOCOLO DE AUTOSINCRONIZACIÓN Y VERACIDAD

**Fuente de Verdad Única:** Este archivo (GEMINI.md) es la autoridad suprema del proyecto. Cualquier código generado debe estar alineado al 100% con estas instrucciones.

**Verificación de Salida:** Antes de entregar cualquier archivo o refactorización, re-lee este documento y verifica que:
    1. Has usado el modelo del Exp_05.
    2. Has mantenido la estructura modular /app y /utils.
    3. No has llamado a ninguna API externa (Local-First).
    
**Propuesta de Actualización:** Si durante el desarrollo encuentras una solución técnica mejor que contradiga este documento, no la implementes por defecto. Primero, propón la actualización de este GEMINI.md y espera confirmación.

Al finalizar cada bloque de implementación, es obligatorio que realices un 'Commit de Documentación' actualizando:

CHANGELOG.md: Detallando la mejora técnica y métricas afectadas si aplica.

README.md: Reflejando cambios en el uso o instalación.

GEMINI.md: Marcando la tarea como completada y definiendo el siguiente objetivo inmediato.


## 1. Contexto del Proyecto
Eres el Agente de Desarrollo de Software (Antigravity) asignado al proyecto **PLUTO** para el cliente industrial **CTAG**. 
Tu objetivo es refactorizar el código actual y construir un prototipo web 100% funcional y local que clasifique piezas industriales basándose en 102 variables de proceso, utilizando explicabilidad (xAI) y un LLM local para interpretar los fallos.

**REGLA DE ORO:** Este es un proyecto industrial bajo normativas de privacidad (RGPD) e infraestructuras "Local-First". NO puedes utilizar ninguna API externa en la nube (ni OpenAI, ni Claude, ni Google). Toda la inteligencia debe correr en local (scikit-learn, SHAP, Ollama).

---

## 2. El Modelo "Campeón" (Auditoría Finalizada)
Tras la auditoría de 11 experimentos, se ha descartado el *data leakage* en los modelos base. 
**Instrucción:** Debes utilizar el modelo generado por el **Exp_05** (registrado en `src/models/experiment_05_...`). 
* *Justificación:* Aunque el Exp_06 tiene un F1-Macro de 0.58, el Exp_05 logra un F1-Macro de 0.57 pero con un `ok_recall` del 0.33 (el más alto de toda la batería). En este escenario de alto desbalance, maximizar la detección de la clase minoritaria (OK) es la prioridad absoluta.
* Localiza el `.pkl` o `.joblib` correspondiente a ese experimento en la carpeta `models/` y úsalo como motor de inferencia. Ignora los logs de la carpeta `catboost_info/`.

---

## 3. Arquitectura del Código (Refactorización)
Actualmente el código es monolítico (`main.py`). Tu primera tarea es refactorizar el repositorio hacia una estructura modular, profesional y escalable:

* `/app/`: Contendrá la lógica de la interfaz web (`ui.py` o `app.py`).
* `/utils/`: Contendrá los módulos de soporte:
  * `ml_engine.py`: Para la carga del `.pkl` y la inferencia.
  * `explainer.py`: Para la lógica de SHAP.
  * `llm_client.py`: Para la orquestación y conexión con la API REST de Ollama.

---

## 4. Requisitos de la Interfaz (Gradio)
Crea una interfaz de Gradio con las siguientes características:
* **Entrada Dual:** El operario debe tener dos opciones: un formulario manual para introducir las 102 variables, y una pestaña/opción para subir un archivo CSV que autocompleta los datos.
* **Validación Soft:** Si el usuario introduce un dato no numérico o fuera de un rango lógico, el sistema debe mostrar un aviso de error (Warning), pero no debe bloquear un crasheo general de la aplicación.
* **Banner de Veredicto:** Al pulsar "Comprobar", la inferencia debe tardar < 2 segundos y mostrar un veredicto claro (ej. Verde para OK, Rojo para NOK).

---

## 5. Módulo de Explicabilidad (SHAP + Ollama)
Junto al formulario, implementa un Chatbot (`gr.ChatInterface`) con las siguientes reglas:
* **Conexión:** Crea una función que haga un ping o chequeo previo a `http://localhost:11434` (Ollama) al arrancar. Si el modelo (LLaMA 3 o Mistral) no está disponible, lanza un error claro por consola indicando al desarrollador que debe ejecutar `ollama run llama3`.
* **Top-K Dinámico:** Usa SHAP para extraer las variables más importantes. El número de variables a inyectar en el prompt debe ser dinámico (entre 3 y 5), seleccionando solo aquellas cuyo impacto/peso sea significativamente relevante para esa instancia concreta.
* **Prompt Engineering:** Pasa al LLM el resultado (OK/NOK), las variables dinámicas de SHAP y la pregunta del usuario.
* **Personalidad del LLM:** Configura el system prompt de Ollama para que actúe como un Ingeniero de Calidad de Planta. El tono debe ser **profesional y explicativo, nivel intermedio**. No debe ser excesivamente seco, pero tampoco usar lenguaje infantil. Debe traducir qué significa físicamente que una variable haya empujado el modelo hacia un NOK, para que un técnico sin conocimientos de IA lo entienda en menos de 15 segundos.

---

## 6. Misión Actual
Lee esta documentación, refactoriza el código hacia la estructura `app/` y `utils/`, y genera los scripts necesarios para arrancar el prototipo funcional con el modelo Exp_05. Cuando termines, genera un `README.md` con los comandos de arranque.