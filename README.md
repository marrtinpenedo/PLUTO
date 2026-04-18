# 🔬 PLUTO — Sistema de Inspección de Calidad Industrial

**Proyecto PIIA · Cliente CTAG · v1.0**

Sistema de clasificación de piezas industriales (OK/NOK) a partir de variables de proceso, con explicabilidad mediante SHAP y diagnóstico en lenguaje natural con un LLM local (Ollama). Todo funciona en local, sin APIs en la nube.

---

## 📋 Tabla de Contenidos

1. [Arquitectura del Sistema](#arquitectura)
2. [Requisitos del Sistema](#requisitos)
3. [Instalación de Dependencias](#instalacion)
4. [Configuración de Ollama (LLM Local)](#ollama)
5. [Ejecución de la Aplicación](#ejecucion)
6. [Guía de Uso](#uso)
7. [Modelo Seleccionado: Justificación](#modelo)
8. [Estructura del Proyecto](#estructura)

---

## 🏗️ Arquitectura del Sistema <a name="arquitectura"></a>

```
┌─────────────────────────────────────────────────────────┐
│                    INTERFAZ GRADIO                       │
│   Formulario N variables  │  Banner OK/NOK + SHAP       │
│   Chatbot con contexto    │  Streaming LLM              │
└────────────────┬──────────────────────┬─────────────────┘
                 │                      │
        ┌────────▼──────┐    ┌──────────▼──────────┐
        │  LightGBM     │    │  SHAP TreeExplainer  │
        │  (< 2 s)      │    │  top-5 features     │
        └───────────────┘    └─────────────────────┘
                 │                      │
        ┌────────▼──────────────────────▼─────────┐
        │        Prompt Engineering dinámico       │
        │   veredicto + proba + SHAP + pregunta   │
        └────────────────────┬────────────────────┘
                             │ REST API
                    ┌────────▼────────┐
                    │  Ollama Local   │
                    │  localhost:11434│
                    │  LLaMA 3 / Mistral│
                    └─────────────────┘
```

### Stack tecnológico

| Componente | Tecnología |
|---|---|
| Frontend + Backend | Gradio ≥ 4.7 |
| Modelo ML | LightGBM (nativo, `.pkl`) |
| Explicabilidad | SHAP `TreeExplainer` |
| LLM local | Ollama (LLaMA 3 / Mistral) |
| Comunicación LLM | API REST `localhost:11434` |
| Datos | pandas, openpyxl |

---

## 💻 Requisitos del Sistema <a name="requisitos"></a>

| Componente | Mínimo |
|---|---|
| Python | 3.9 o superior |
| RAM | 8 GB (recomendado 16 GB para el LLM) |
| Almacenamiento | ~6 GB libres (modelo LLM + entorno virtual) |
| SO | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| GPU | Opcional (acelera el LLM; no necesaria para ML) |

---

## 📦 Instalación de Dependencias <a name="instalacion"></a>

### 1. Clonar / abrir el proyecto

Asegúrate de estar en la raíz del proyecto `PLUTO/` (donde está `main.py`):

```bash
cd ruta/al/proyecto/PLUTO
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ La instalación puede tardar 2–5 minutos dependiendo de la conexión.

### Verificar instalación

```bash
python -c "import gradio, lightgbm, shap; print('✓ Instalación correcta')"
```

---

## 🤖 Configuración de Ollama (LLM Local) <a name="ollama"></a>

Ollama permite ejecutar modelos de lenguaje grandes (LLM) completamente en local.

### Paso 1: Instalar Ollama

**Windows:**
Descarga el instalador desde → **https://ollama.ai/download/windows**
Ejecuta el `.exe` y sigue el asistente de instalación.

**macOS:**
```bash
brew install ollama
# O descarga directamente desde https://ollama.ai
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Paso 2: Descargar el modelo LLM

Abre una terminal y ejecuta **uno** de los siguientes comandos:

```bash
# Opción A — LLaMA 3 8B (RECOMENDADO: mejor calidad, ~4.7 GB)
ollama pull llama3

# Opción B — Mistral 7B (alternativa, ~4.1 GB)
ollama pull mistral

# Opción C — LLaMA 3.2 3B (más ligero si tienes poca RAM, ~2 GB)
ollama pull llama3.2
```

> ⏱️ La descarga puede tardar varios minutos dependiendo de tu conexión.

Para verificar qué modelos tienes instalados:
```bash
ollama list
```

### Paso 3: Iniciar el servidor Ollama

```bash
ollama serve
```

Ollama quedará escuchando en `http://localhost:11434`. En Windows, Ollama se inicia automáticamente en el fondo tras la instalación.

Para verificar que está activo:
```bash
curl http://localhost:11434/api/tags
# Debe devolver un JSON con los modelos disponibles
```

### Cambiar el modelo en la aplicación

Si usas un modelo diferente a `llama3`, edita la línea en `main.py`:

```python
LLM_MODEL = "llama3"   # ← Cambia aquí por "mistral", "llama3.2", etc.
```

---

## ▶️ Ejecución de la Aplicación <a name="ejecucion"></a>

### Inicio rápido

Con el entorno virtual activo y Ollama en ejecución:

```bash
python main.py
```

La aplicación estará disponible en: **http://localhost:7860**

### Primera vez que arranque

Al iniciar, la aplicación realiza automáticamente:
1. Carga del modelo LightGBM desde `models/lightgbm_optimized_model.pkl`
2. Lectura de metadatos del dataset (tipos de columna, rangos, categorías)
3. Inicialización del SHAP `TreeExplainer`

Verás en la terminal:
```
══════════════════════════════════════════════════════════════
   PLUTO — Iniciando sistema de inspección de calidad
══════════════════════════════════════════════════════════════

[1/3] Cargando modelo: lightgbm_optimized_model.pkl ...
   ✓ Threshold=0.XXXX | Features totales=107 | Originales=102

[2/3] Leyendo metadatos del dataset (Dataset_01_Anonimizado.xlsx) ...
   ✓ Columnas categóricas: N | Columnas numéricas: M

[3/3] Inicializando SHAP TreeExplainer ...
   ✓ Listo.

══════════════════════════════════════════════════════════════
   Abre tu navegador en  http://localhost:7860
══════════════════════════════════════════════════════════════
```

### Latencias esperadas

| Operación | Tiempo estimado |
|---|---|
| Predicción (botón Comprobar) | **< 2 segundos** |
| Cálculo SHAP (incluido en Comprobar) | < 1 segundo |
| Respuesta LLM (según hardware) | 10–30 segundos |

---

## 📖 Guía de Uso <a name="uso"></a>

### 1. Introducir variables de proceso

El formulario muestra las variables agrupadas en acordeones. Cada acordeón contiene un bloque de ~20 variables. Los valores por defecto son las **medias del conjunto de entrenamiento**.

- **Variables numéricas**: introduce el valor medido (el tooltip muestra el rango `[min, max]` del dataset de entrenamiento).
- **Variables categóricas**: selecciona el valor del desplegable.

### 2. Obtener el veredicto

Pulsa **🔍 Comprobar Calidad**. En menos de 2 segundos:
- Aparece el **banner verde (OK)** o **rojo (NOK)** con la probabilidad.
- El **slider** muestra `P(NOK)` respecto al umbral de decisión.
- La **tabla SHAP** lista las 5 variables que más han influido en la predicción y su dirección:
  - **SHAP positivo (+)** → empuja hacia NOK
  - **SHAP negativo (−)** → empuja hacia OK

### 3. Consultar al Asistente IA

En la sección **💬 Asistente IA**, escribe una pregunta sobre el resultado:

- *"¿Por qué esta pieza es NOK?"*
- *"¿Qué variable tiene más impacto en el fallo?"*
- *"¿Hay alguna variable cerca de sus límites aunque la pieza sea OK?"*
- *"¿Qué proceso industrial debería revisarse?"*

El asistente recibe automáticamente el veredicto, la probabilidad y el análisis SHAP completo, y responde en lenguaje natural.

### 4. Limpiar y nueva inspección

Pulsa **🔄 Limpiar** para restaurar todos los campos a sus valores por defecto y comenzar una nueva inspección.

---

## 🏆 Modelo Seleccionado: Justificación <a name="modelo"></a>

### Modelo elegido: `lightgbm_optimized_model.pkl`

**Archivo:** `models/lightgbm_optimized_model.pkl`  
**Experimento origen:** `src/models/experiment_01_advanced.py`  
**Algoritmo:** LightGBM Booster nativo con `is_unbalance=True`

### Comparativa de modelos disponibles

| Archivo `.pkl` | Exp. origen | Algoritmo | Tamaño | Riesgo leakage |
|---|---|---|---|---|
| `lightgbm_optimized_model.pkl` | Exp 01 | LightGBM nativo | 23 KB | ✅ **Ninguno** |
| `xgboost_super_optimized.pkl` | Exp 02 | XGBoost + Optuna | 2.0 MB | ✅ Ninguno |
| `divide_and_conquer_experts.pkl` | Exp 04 | GMM + XGBoost por clúster | 1.8 MB | ⚠️ Bajo-Medio |

### Por qué se descartan los experimentos 10 y 11

Los **experimentos 10 (CleanLab)** y **11 (Ultimate Hybrid)** aplican Confident Learning (CleanLab) sobre el **dataset completo ANTES de hacer ningún split train/test**:

```python
# Experimento 10 — LEAKAGE: aplica CleanLab a todo el dataset sin holdout previo
df, target_col = load_and_preprocess_data()            # todas las filas
df_clean, drop_idx = extract_label_issues_with_cleanlab(df, target_col)  # usa todo
```

Esto constituye **data leakage claro**: el algoritmo de detección de ruido (CleanLab) usa probabilidades generadas sobre todo el conjunto (incluyendo lo que después sería test) para filtrar muestras. Las métricas reportadas sobre ese subset son optimistas e infladas. Adicionalmente, ninguno de estos experimentos guarda un modelo `.pkl` en `models/`.

### Por qué se elige el Experimento 01

El Experimento 01 implementa un pipeline honesto y limpio:

1. **Sin re-etiquetado:** Entrena con las etiquetas originales del dataset.
2. **Cross-validation estricta:** 5-Fold estratificado con predicciones OOF (Out-Of-Fold). El modelo de cada fold nunca ve sus datos de validación durante el entrenamiento.
3. **Threshold tuning sobre OOF:** El umbral de decisión óptimo se busca **únicamente** sobre las predicciones OOF, nunca sobre datos de test.
4. **Modelo final limpio:** Se re-entrena con todos los datos usando los hiperparámetros (no el threshold) derivados de la CV.
5. **Tamaño del modelo:** 23 KB vs. 2 MB del XGBoost. LightGBM es significativamente más ligero para producción.

### Estructura del `.pkl`

```python
{
    'model':     lgb.Booster,   # Modelo LightGBM nativo
    'threshold': float,          # Umbral óptimo (P(NOK) >= threshold → NOK)
    'features':  list[str],      # Nombres de todas las features (102 orig + 5 eng)
}
```

---

## 🗂️ Estructura del Proyecto <a name="estructura"></a>

```
PLUTO/
├── main.py                          ← Aplicación Gradio completa (este fichero)
├── requirements.txt                 ← Dependencias Python
├── README.md                        ← Este documento
│
├── models/
│   ├── lightgbm_optimized_model.pkl ← ✅ MODELO EN USO (Exp 01)
│   ├── xgboost_super_optimized.pkl  ← Exp 02 (no usado en producción)
│   └── divide_and_conquer_experts.pkl ← Exp 04 (no usado en producción)
│
├── data/
│   └── raw/
│       └── Dataset_01_Anonimizado.xlsx ← Dataset de entrenamiento/referencia
│
├── src/
│   └── models/
│       ├── experiment_01_advanced.py       ← Origen del modelo en producción
│       ├── experiment_02_super_ensemble.py
│       ├── experiment_03_pytorch_mlp.py
│       ├── experiment_04_cluster_and_conquer.py
│       ├── experiment_05_ultimate_vae_fe.py
│       ├── experiment_06_xgbod_sota.py
│       ├── experiment_07_insane_niche.py
│       ├── experiment_08_nature_sota.py
│       ├── experiment_09_supmin_tabm.py
│       ├── experiment_10_autogluon_cleanlab.py  ← ⚠️ leakage sospechoso
│       └── experiment_11_ultimate_hybrid.py     ← ⚠️ leakage sospechoso
│
└── notebooks/                       ← Análisis exploratorio (desarrollo)
```

---

## ❓ Solución de Problemas

| Problema | Solución |
|---|---|
| `FileNotFoundError: lightgbm_optimized_model.pkl` | Ejecuta `python main.py` desde la raíz del proyecto `PLUTO/` |
| `FileNotFoundError: Dataset_01_Anonimizado.xlsx` | Verifica que el Excel existe en `data/raw/` |
| Puerto 7860 ocupado | Edita en `main.py`: `application.launch(server_port=7861)` |
| Ollama offline (chatbot no responde) | Ejecuta `ollama serve` en una terminal separada |
| LLM muy lento | Usa `ollama pull llama3.2` (modelo de 3B, más ligero) y cambia `LLM_MODEL = "llama3.2"` en `main.py` |
| SHAP tarda mucho | Normal en la primera ejecución; se cacha el explainer para predicciones posteriores |

---

## 📄 Licencia y Contacto

Proyecto académico desarrollado para el cliente **CTAG** en el marco del **Grado en Inteligencia Artificial — USC**.  
Datos del dataset completamente anonimizados. No contiene información de producción real.
