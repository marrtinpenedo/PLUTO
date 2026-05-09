# PLUTO - Sistema de Inspeccion de Calidad Industrial

**Proyecto PIIA - Cliente CTAG - v2.2**

Sistema de clasificacion de piezas industriales (OK/NOK) a partir de **102 variables de proceso**.
Integra un pipeline VAE + CatBoost para el veredicto, explicabilidad via **SHAP nativo CatBoost**
con agregacion real/VAE y balanceo dinamico, y diagnostico en lenguaje natural con un
**LLM local (Ollama)** bajo el rol de Ingeniero Senior CTAG.

Todo el ecosistema sigue la filosofia **Local-First**: ningun dato sale de la maquina local,
garantizando la privacidad industrial de CTAG sin depender de APIs en la nube.

---

## Inicio Rapido

### Requisitos previos

- Python >= 3.10, < 3.13 (recomendado: 3.11.x)
- [Ollama](https://ollama.com/) instalado (para el chatbot LLM)
- Conda o venv

### Instalacion

```bash
# 1. Activar entorn
conda activate pluto
# o: python -m venv .venv && .venv\Scripts\activate

# 2. Instalar dependencias Python
pip install -r requirements.txt

# 3. Descargar el modelo de lenguaje (solo la primera vez)
ollama pull llama3

# 4. Arrancar Ollama en una terminal separada (necesario para el chatbot)
ollama serve
```

### Generar el artefacto del modelo (primera vez)

Si `models/exp05_vae_catboost_v2.pkl` no existe:

```bash
# Ejecutar desde la raiz del proyecto. Tarda ~40 s en CPU.
python scripts/export_exp05_model.py
```

### Arrancar la interfaz

```bash
python main.py
```

Interfaz disponible en: **http://localhost:7860**

---

## Arquitectura del Sistema

```
+-------------------------------------------------------------------+
|                  INTERFAZ (app/ui.py - Gradio 5.50)              |
|  Carga CSV/Excel -> Formulario 102 vars -> Banner OK/NOK          |
|  Slider P(NOK) | Tabla SHAP | Lotes | Ver Detalle | Chatbot LLM  |
+------------------------------+------------------------------------+
                               |
           +-------------------+------------------+
           |          BACKEND UTILS               |
           +--------+-----------+-----------------+
     ml_engine.py  explainer.py  llm_client.py
     (pkl v2)      (SHAP nativo) (Ollama streaming)
     102 -> 113    Real/VAE bal. System Prompt CTAG
     features      max 10 vars   NOK_THRESHOLD=0.4606
           |
+----------v-----------+   +----------------------------------+
|  LLM LOCAL (Ollama)  |   |  DATA (local, nunca sale)        |
|    localhost:11434   |   |  data/raw/Dataset_01_Anon.xlsx   |
|    llama3            |   |  models/exp05_vae_catboost_v2.pkl|
+----------------------+   +----------------------------------+
```

### Stack tecnologico

| Capa | Tecnologia | Detalle |
|---|---|---|
| UI | Gradio 5.50 (Base theme, Inter font) | Dark theme CTAG; requiere fichero valido para prediccion |
| ML | CatBoost + VAE (PyTorch) | Serializado en `.pkl` v2 con scaler + VAE state_dict + train_medians |
| xAI | CatBoost `get_feature_importance(ShapValues)` | Nativo CatBoost; balanceo Real/VAE; max 10 vars |
| LLM | Ollama (local) | LLaMA 3 o Mistral, sin llamadas externas |
| Rutas | pathlib | Compatible Windows (OneDrive) y Linux/macOS |

---

## Estructura del Proyecto

```
PLUTO/
+-- app/
|   +-- ui.py                        <- Interfaz Gradio (formulario + CSV + lotes + chatbot)
+-- utils/
|   +-- ml_engine.py                 <- Motor singleton: carga pkl v2 + predict pipeline
|   +-- explainer.py                 <- SHAP v2.2: SHAP nativo CatBoost + balanceo Real/VAE
|   +-- llm_client.py                <- Cliente Ollama: System Prompt + build_prompt + streaming
+-- scripts/
|   +-- export_exp05_model.py        <- Script one-shot para generar el pkl (~40 s)
+-- research/
|   +-- models/                      <- Experimentos Exp_01..11 + runner comparativo
|   +-- notebooks/                   <- Analisis exploratorio
|   +-- evaluation/                  <- Resultados de experimentos
+-- models/
|   +-- exp05_vae_catboost_v2.pkl    <- ARTEFACTO EN PRODUCCION (threshold=0.4606, 113 feats)
+-- data/
|   +-- raw/Dataset_01_Anonimizado.xlsx  <- Dataset de referencia (RGPD)
+-- doc/
|   +-- AVP1/                        <- Documentacion plan de proyecto (PDFs)
|   +-- AVP2/                        <- Documentacion requisitos funcionales
+-- main.py                          <- Entry point
+-- requirements.txt

```

---

## Artefacto PKL v2 - Contenido

`joblib.load()` devuelve un dict con estas claves:

| Clave | Tipo | Uso |
|---|---|---|
| `model` | CatBoostClassifier | Modelo de clasificacion |
| `vae_bytes` | bytes | state_dict VAE serializado con `torch.save` en BytesIO |
| `vae_input_dim` | int | Dimension de entrada para reconstruir la arquitectura VAE |
| `scaler` | StandardScaler | Ajustado solo sobre `num_cols` del train pool |
| `num_cols` | list[str] | Columnas numericas originales |
| `cat_cols` | list[str] | Columnas categoricas originales |
| `train_medians` | pd.Series | Medianas de entrenamiento para imputacion de nulos |
| `all_feats` | list[str] | Orden exacto de las 113 columnas que entran al modelo |
| `threshold` | float | Threshold calibrado OOF (produccion: **0.4606**) |
| `latent_dim` | int | Dimension latente VAE (valor: **12**) |
| `col_stats` | dict | `{col: {min, max, mean}}` para rangos de la UI |
| `categories` | dict | `{col: [values]}` para dropdowns de la UI |

> **v2 vs v1**: la v2 elimina `kbd` (KBinsDiscretizer), `oe` (OrdinalEncoder) y `high_var`.
> Las categoricas se pasan directamente como strings nativos a CatBoost.

---

## Modelo en Produccion: Experimento 05 (VAE + CatBoost)

### Metricas en holdout 20%

| Metrica | Valor |
|---|---|
| F1-Weighted | **0.67** |
| F1-Macro | **0.57** |
| F1 clase NOK | **0.77** |
| F1 clase OK | **0.36** |
| Recall NOK | **0.76** |
| **Recall OK** | **0.39** (mejor de 11 experimentos) |
| ROC-AUC | **0.63** |
| **Threshold de produccion** | **0.4606** (OOF F1-Macro) |
| Features de entrada | **113** |

### Pipeline de feature engineering (102 -> 113 variables)

```
Entrada: 102 variables originales (num_cols + cat_cols)
  1. Imputacion: mediana (numericas) / "missing" (categoricas)
  2. StandardScaler (ajustado solo en train, solo vars numericas)
  3. VAE Encoder (PyTorch)
       -> vae_err  (error de reconstruccion MSE)
       -> vae_l0 ... vae_l11  (12 dimensiones latentes)
  4. CatBoost con categoricas nativas (auto_class_weights=Balanced)
Output: 113 features -> CatBoostClassifier -> P(NOK) -> umbral 0.4606
```

### Arquitectura VAE 

```
Encoder: Linear(in,64)->ReLU->Linear(64,32)->ReLU -> fc_mu(32,12) + fc_logvar(32,12)
Decoder: Linear(12,32)->ReLU->Linear(32,64)->ReLU->Linear(64,in)
```

---

## Comparativa de Experimentos

| Exp | Modelo | F1-W | F1-M | OK-F1 | NOK-F1 | Recall-OK | Tiempo |
|-----|--------|------|------|-------|--------|-----------|--------|
| Exp_01 | LightGBM | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 4.6 s |
| Exp_02 | XGBoost + KMeans | 0.69 | 0.56 | 0.31 | 0.81 | 0.28 | 155 s |
| Exp_03 | PyTorch MLP | 0.68 | 0.56 | 0.31 | 0.80 | 0.28 | 225 s |
| Exp_04 | Cluster+Conquer | 0.69 | 0.57 | 0.32 | 0.81 | 0.29 | 113 s |
| **Exp_05** | **VAE+CatBoost v2** | **0.67** | **0.57** | **0.36** | **0.77** | **0.39** | **40 s** |
| Exp_06 | XGBOD | 0.71 | 0.58 | 0.34 | 0.83 | 0.28 | 246 s |
| Exp_07 | DART+UMAP | 0.69 | 0.57 | 0.32 | 0.82 | 0.27 | 757 s |
| Exp_08 | ASL+LightGBM | 0.69 | 0.56 | 0.31 | 0.81 | 0.27 | 15 s |
| Exp_09 | SupMin+TabM | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 221 s |
| Exp_10 | CleanLab+Stack | 0.69 | 0.51 | 0.16 | 0.86 | 0.09 | 43 s |
| Exp_11 | Ultimate Hybrid | 0.69 | 0.52 | 0.19 | 0.86 | 0.11 | 730 s |

**Criterio de seleccion:** Recall OK es la metrica prioritaria para CTAG.
Exp_05 v2 es el maximo absoluto en Recall OK con un tiempo de entrenamiento razonable.

---

## Explicabilidad SHAP v2.2

El modulo `utils/explainer.py` aplica cinco pasos en orden:

1. **SHAP crudos** via `get_feature_importance(pool, type='ShapValues')` (CatBoost nativo).
2. **Agregacion algebraica**: columnas `_bin` se suman al SHAP del sensor base (retrocompat. v1).
3. **Filtro 15%**: descarta features con `|SHAP| < 0.15 * |top1|` (minimo 3 vars).
4. **Balanceo Real/VAE**: garantiza `n_reales >= n_vae`, max 10, min 3.
5. **Renombrado industrial**: `vae_err` -> "Indice de Correlacion Global"; `vae_lN` -> "Patron Estructural N".

**Semantica de signo**: SHAP > 0 empuja hacia NOK (defecto); SHAP < 0 hacia OK (calidad).

---

## LLM Local - Configuracion

| Parametro | Valor |
|---|---|
| Endpoint | `http://localhost:11434/api/generate` |
| Modelo | `llama3` (configurable en `LLM_MODEL`) |
| Timeout | 90 s |
| Umbral | **0.4606** (`NOK_THRESHOLD` en `llm_client.py`) |

El prompt al LLM incluye veredicto, probabilidad, decision y tabla SHAP ya renombrada
(sin nombres tecnicos como `vae_l*`). La estructura de respuesta es adaptativa segun
el tipo de pregunta (diagnostico general, causa del defecto, variable concreta, etc.).

---

## Diseno UI Industrial

Paleta dark premium para operarios de planta:

| Elemento | Hex |
|---|---|
| Fondo global | `#0c1629` |
| Contenedores | `#111c30` |
| Texto principal | `#f8fafc` |
| Banner OK | fondo `#064e3b`, texto `#34d399` |
| Banner NOK | fondo `#7f1d1d`, texto `#f87171` |
| Boton primario | `#2563eb` |

**Reglas UI**: cero emojis, cero text-tags, `result_slider` no interactivo,
prediccion bloqueada hasta fichero valido cargado.

---

## Constantes Criticas

| Constante | Valor | Fichero(s) |
|---|---|---|
| NOK threshold | **0.4606** | `llm_client.py -> NOK_THRESHOLD`; `ml_engine.py -> self.threshold` |
| LATENT_DIM | **12** | `scripts/export_exp05_model.py` |
| VAE_EPOCHS | **15** | `scripts/export_exp05_model.py` |
| SHAP 15% filter | `0.15 * abs(top1)` | `explainer.py -> explain()` |
| SHAP table range | min=3, max=10 | `explainer.py -> explain(top_k_range=(3,10))` |
| SECTION_SZ | 20 vars/acordeon | `app/ui.py` |
| GROUP_SIZE | 4 inputs/fila | `app/ui.py` |

---

## Privacidad y Cumplimiento

- **Local-First**: ningun dato de proceso sale del equipo local. Sin APIs externas.
- **Dataset anonimizado**: `Dataset_01_Anonimizado.xlsx` cumple RGPD.
- **LLM local**: Ollama corre en `localhost`. Las consultas nunca abandonan la red local.
- **Sin telemetria**: el sistema no registra ni envia metricas de uso.

---

## Licencia y Contacto

Proyecto desarrollado para **CTAG** por el equipo **CTG GT14**:

- Santiago Pereira
- Martin Rodriguez
- Lucas A. Martinez

**Universidad de Santiago de Compostela (USC) - PIIA 2025/2026**

Todos los datos estan anonimizados conforme al Reglamento General de Proteccion de Datos (RGPD).