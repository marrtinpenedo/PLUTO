# 🔬 PLUTO — Sistema de Inspección de Calidad Industrial

**Proyecto PIIA · Cliente CTAG · v1.1 (Actualizado)**

Sistema de clasificación de piezas industriales (OK/NOK) a partir de variables de proceso. El sistema integra un modelo de Machine Learning para el veredicto, explicabilidad mediante **SHAP** y un diagnóstico en lenguaje natural utilizando un **LLM local (Ollama)**. 

Todo el ecosistema está diseñado bajo la filosofía **"Local-First"**, garantizando la privacidad de los datos industriales de CTAG sin depender de APIs en la nube.

---

## 🏗️ Arquitectura del Sistema <a name="arquitectura"></a>

El sistema se basa en una arquitectura modular para separar la interfaz de usuario de la lógica de inferencia y generación:

```
┌─────────────────────────────────────────────────────────┐
│                   INTERFAZ (Gradio)                     │
│      Formulario / CSV   │   Banner OK/NOK + Chatbot     │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │          BACKEND UTILS        │
        ├───────────────┬───────────────┤
        │   ML Engine   │   xAI (SHAP)  │
        │ (Exp_05 Model)│ (Dynamic Top-K)│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │       LLM CLIENT (Ollama)     │
        │          localhost:11434      │
        └───────────────────────────────┘
```

### Stack tecnológico
* **Frontend + Backend**: Gradio (Interfaz y servidor unificado).
* **Modelo ML**: Scikit-learn (Serializado en `.pkl`).
* **Explicabilidad**: SHAP (TreeExplainer para análisis de instancias).
* **LLM Local**: Ollama (Ejecutando LLaMA 3 o Mistral).
* **Comunicación**: API REST local para la orquestación del chatbot.

---

## 🏆 Modelo Seleccionado: Justificación <a name="modelo"></a>

### Modelo "Champion": **Experimento 05 (Ultimate VAE FE)**
* **Archivo**: `models/experiment_05_ultimate_vae_fe.pkl`
* **Origen**: `src/models/experiment_05_ultimate_vae_fe.py`

#### Razonamiento técnico
Tras auditar 11 experimentos, el **Exp_05** ha sido seleccionado como el modelo de producción por los siguientes motivos:
1.  **Máximo Recall OK (33%)**: Es el modelo con mayor capacidad para identificar correctamente las piezas buenas en un dataset altamente desbalanceado.
2.  **F1-Macro Robusto (0.57)**: Ofrece un equilibrio superior entre la clase mayoritaria (NOK) y la minoritaria (OK).
3.  **Auditoría de Leakage**: Se ha verificado que, a diferencia de los experimentos 10 y 11, este modelo no utiliza información de test durante el preprocesamiento o el filtrado de etiquetas.

---

## 🗂️ Estructura del Proyecto <a name="estructura"></a>

```
PLUTO/
├── app/                            ← Capa de Presentación
│   └── ui.py                       ← Interfaz Gradio y validación de rangos
│
├── utils/                          ← Lógica de Negocio
│   ├── ml_engine.py                ← Carga de modelos e inferencia
│   ├── explainer.py                ← Cálculo de SHAP values dinámicos
│   └── llm_client.py               ← Conexión con la API de Ollama
│
├── models/                         ← Almacén de modelos (.pkl)
│   └── experiment_05_ultimate.pkl  ← ✅ MODELO EN PRODUCCIÓN
│
├── src/models/                     ← Scripts de experimentación (01-11)
├── notebooks/                      ← Análisis exploratorio (EDA)
├── GEMINI.md                       ← Instrucciones para el agente de IA
└── CHANGELOG.md                    ← Registro de auditoría y versiones
```

---

## 💻 Requisitos e Instalación <a name="instalacion"></a>

### Requisitos previos
* **Python 3.9+**.
* **Ollama** instalado y en ejecución (`ollama serve`).
* **RAM**: 8GB mínimo (16GB recomendado para el LLM).

### Instalación
1.  **Clonar el repositorio** y crear un entorno virtual:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # .venv\Scripts\activate en Windows
    ```
2.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Descargar el modelo de lenguaje**:
    ```bash
    ollama pull llama3
    ```

---

## ▶️ Ejecución de la Aplicación <a name="ejecucion"></a>

Para iniciar el sistema PLUTO, ejecuta desde la raíz:

```bash
python app/ui.py
```

### Flujo de operación
1.  **Ingesta**: Introduce los datos manualmente o carga un CSV.
2.  **Clasificación**: El sistema devuelve un veredicto visual (Verde/Rojo) en menos de 2 segundos.
3.  **Explicación**: Consulta al chatbot por qué se ha tomado la decisión. El sistema inyectará los valores **SHAP** en el prompt para una respuesta técnica precisa.

---

## 📄 Licencia y Contacto
Proyecto desarrollado para **CTAG** por el equipo **CTG GT14** (Santiago Pereira, Martín Rodríguez, Lucas A. Martínez) en la **USC**. Todos los datos están anonimizados conforme al RGPD.