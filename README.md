# PLUTO - Sistema de Inspección de Calidad Industrial

**Proyecto PIIA - Cliente CTAG - v2.0**

Sistema de clasificación de piezas industriales (OK/NOK) a partir de 102 variables de proceso. Integra un modelo de Machine Learning para el veredicto, explicabilidad mediante **SHAP** y diagnóstico en lenguaje natural con un **LLM local (Ollama)**.

Todo el ecosistema está diseñado bajo la filosofía **"Local-First"**, garantizando la privacidad de los datos industriales de CTAG sin depender de APIs en la nube.

---

## Arquitectura del Sistema

El sistema tiene una arquitectura modular que separa la interfaz de la lógica de inferencia:

```
+----------------------------------------------------------+
|                   INTERFAZ (Gradio)                      |
|      Formulario / CSV   |   Banner OK/NOK + Chatbot      |
+------------------------+---------------------------------+
                         |
         +---------------+---------------+
         |          BACKEND UTILS        |
         +---------------+---------------+
         |   ML Engine   |   xAI (SHAP)  |
         | (Exp_05 Model)|  (Top-K 3-5)  |
         +---------------+---------------+
                         |
         +---------------v---------------+
         |       LLM CLIENT (Ollama)     |
         |          localhost:11434      |
         +-------------------------------+
```

### Stack tecnologico
* **Frontend + Backend**: Gradio (interfaz y servidor unificados).
* **Modelo ML**: VAE (PyTorch) + CatBoost serializado en `.pkl`.
* **Explicabilidad**: SHAP TreeExplainer con top-K dinamico (3-5 features).
* **LLM Local**: Ollama (LLaMA 3 o Mistral).
* **Rutas**: `pathlib` en todo el codigo (compatible Windows y Linux).

---

## Modelo Seleccionado

### Modelo Champion: **Experimento 05 (VAE + CatBoost)**

* **Artefacto**: `models/exp05_vae_catboost.pkl`
* **Script de origen**: `src/models/experiment_05_ultimate_vae_fe.py`
* **Script de exportacion**: `scripts/export_exp05_model.py`

#### Metricas (holdout 20%)

| Metrica | Valor |
|---|---|
| F1-Weighted | 0.69 |
| F1-Macro | 0.57 |
| Recall OK | **0.33** (mejor de los 11 experimentos) |
| Threshold | 0.6818 |
| Features totales | 123 (102 orig + bins + VAE latents) |

#### Razonamiento tecnico
Tras auditar 11 experimentos, el **Exp_05** es el modelo de produccion porque:
1. **Maximo Recall OK (33%)**: Mayor capacidad para identificar piezas buenas en dataset desbalanceado.
2. **F1-Macro Robusto (0.57)**: Equilibrio superior entre clase mayoritaria (NOK) y minoritaria (OK).
3. **Sin data leakage**: VAE, KBins y Scaler ajustados solo sobre el train pool de cada fold.

---

## Estructura del Proyecto

```
PLUTO/
+-- app/                            <- Capa de Presentacion
|   +-- ui.py                       <- Interfaz Gradio (formulario + CSV + chatbot)
|
+-- utils/                          <- Logica de Negocio
|   +-- ml_engine.py                <- Motor singleton: carga pkl + predict pipeline completo
|   +-- explainer.py                <- SHAP TreeExplainer (top-K dinamico 3-5)
|   +-- llm_client.py               <- Cliente Ollama con validacion de modelos
|
+-- scripts/
|   +-- export_exp05_model.py       <- Script one-shot para generar el pkl (~32 s)
|
+-- models/
|   +-- exp05_vae_catboost.pkl      <- MODELO EN PRODUCCION (threshold=0.6818, 123 feat)
|
+-- src/models/                     <- Scripts de experimentacion (Exp_01 - Exp_11)
+-- data/raw/                       <- Dataset de referencia (Dataset_01_Anonimizado.xlsx)
+-- notebooks/                      <- Analisis exploratorio (EDA)
+-- main.py                         <- Entry point (35 lineas)
+-- GEMINI.md                       <- Instrucciones maestras para el agente de IA
+-- CHANGELOG.md                    <- Registro de auditoria y versiones
```

---

## Requisitos e Instalacion

### Requisitos previos
* **Python 3.10+** (recomendado 3.11 o 3.12).
* **Ollama** instalado y en ejecucion (`ollama serve`).
* **RAM**: 8 GB minimo (16 GB recomendado para el LLM).

### Instalacion

1. **Clonar el repositorio** y crear un entorno virtual:
    ```bash
    python -m venv .venv
    source .venv/bin/activate      # Linux / macOS
    .venv\Scripts\activate         # Windows
    ```

2. **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Descargar el modelo de lenguaje** (solo la primera vez):
    ```bash
    ollama pull llama3
    ```

---

## Ejecucion de la Aplicacion

### Primera vez (generar el modelo)

Si `models/exp05_vae_catboost.pkl` no existe aun:

```bash
python scripts/export_exp05_model.py
```

Tiempo estimado: ~32 s en CPU.

### Arrancar la interfaz

```bash
python main.py
```

La interfaz queda disponible en: **http://localhost:7860**

### Flujo de operacion

1. **Ingesta**: Introduce los datos manualmente en el formulario agrupado por secciones, o carga un CSV para autocompletado automatico (tolerante a espacios y mayusculas en las cabeceras).
2. **Validacion**: Si algun valor numerico esta fuera del rango del dataset de referencia, aparece un aviso visual (no bloquea la prediccion).
3. **Clasificacion**: El sistema devuelve un banner OK/NOK y la probabilidad de fallo en menos de 2 segundos.
4. **Explicacion SHAP**: Se muestran las 3-5 variables con mayor impacto en la decision.
5. **Diagnostico LLM**: Consulta al chatbot para obtener un diagnostico tecnico con contexto SHAP inyectado automaticamente.

---

## Licencia y Contacto
Proyecto desarrollado para **CTAG** por el equipo **CTG GT14** (Santiago Pereira, Martin Rodriguez, Lucas A. Martinez) en la **USC**. Todos los datos estan anonimizados conforme al RGPD.