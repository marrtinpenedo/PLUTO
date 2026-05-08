# PLUTO - Sistema de Inspeccion de Calidad Industrial

**Proyecto PIIA - Cliente CTAG - v2.2**

Sistema de clasificacion de piezas industriales (OK/NOK) a partir de 102 variables de proceso.
Integra un modelo de Machine Learning para el veredicto, explicabilidad via **SHAP dinamico**
con agregacion neta Real/VAE, y diagnostico en lenguaje natural con un **LLM local (Ollama)**
bajo un rol de Ingeniero Senior CTAG.

Todo el ecosistema sigue la filosofia **Local-First**: ningun dato sale de la maquina local,
garantizando la privacidad industrial de CTAG sin depender de APIs en la nube.

---

## Inicio Rapido

### Instalacion

```bash
# 1. Clonar y crear entorno virtual
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 2. Instalar dependencias Python
pip install -r requirements.txt

# 3. Descargar el modelo de lenguaje (solo la primera vez)
ollama pull llama3
ollama run  llama3              # puede cerrarse mientras no se haga un stop
```

### Generar el modelo (primera vez)

Si `models/exp05_vae_catboost_v2.pkl` no existe aun:

```bash
python scripts/export_exp05_model_v2.py
```

El script entrena el pipeline completo (5-Fold OOF, VAE + CatBoost Nativo)
sobre el dataset de referencia y serializa el artefacto.

### Arrancar la interfaz

```bash
python main.py
```

Interfaz disponible en: **http://localhost:7860** 

---

## Arquitectura del Sistema

```
+--------------------------------------------------------------------+
|                      INTERFAZ (app/ui.py - Gradio)                 |
|  Carga CSV/Excel  ->  Autocompletado formulario (102 vars)          |
|  Banner OK/NOK + Slider P(NOK) | Tabla SHAP + Chatbot LLM          |
|  Prediccion por Lotes (N filas, descarga CSV resultados)           |
+------------------------------+-------------------------------------+
                               |
           +-------------------+-------------------+
           |             BACKEND UTILS              |
           +--------+------------+------------------+
           |        |            |                  |
     ml_engine.py  explainer.py  llm_client.py       |
     (Exp_05 pkl) (SHAP v2.2)   (Ollama Sistema)     |
     102 -> 113   Agregación   System Prompt CTAG   |
     features     Real/VAE      NOK_THRESHOLD=0.4606  |
           |        |            |                  |
           +--------+------------+                  |
                    |                               |
         +----------v-----------+                  |
         |  LLM LOCAL (Ollama)  |                  |
         |    localhost:11434   |                  |
         |    llama3 / mistral  |                  |
         +----------------------+                  |
                                                   |
+--------------------------------------------------+
|  DATA (local, nunca sale del equipo)             |
|  data/raw/Dataset_01_Anonimizado.xlsx            |
|  models/exp05_vae_catboost_v2.pkl                |
+--------------------------------------------------+
```

### Stack tecnologico

| Capa | Tecnologia | Detalle |
|---|---|---|
| UI | Gradio (Base theme, Inter font) | Dark theme CTAG; requiere CSV para activar prediccion |
| ML | CatBoost + VAE (PyTorch) | Serializado en `.pkl` con scaler + VAE state_dict |
| xAI | CatBoost `get_feature_importance(ShapValues)` | Nativo CatBoost; agrega derivadas `_bin` si existen; balanceo Real/VAE; max 10 vars |
| LLM | Ollama (local) | LLaMA 3 o Mistral, sin llamadas externas |
| Rutas | pathlib | Compatible Windows (OneDrive) y Linux |

---

## Estructura del Proyecto

```
PLUTO/
+-- app/
|   +-- ui.py                        <- Interfaz Gradio (formulario + CSV + chatbot)
|
+-- utils/
|   +-- ml_engine.py                 <- Motor singleton: carga pkl + predict pipeline
|   +-- explainer.py                 <- SHAP v2.2: agregacion neta + balanceo Real/VAE
|   +-- llm_client.py                <- Cliente Ollama: System Prompt + streaming
|
+-- scripts/
|   +-- export_exp05_model.py        <- Script one-shot para generar el pkl (~32 s)
|
+-- research/models/
|   +-- experiment_05_ultimate_vae_fe.py  <- Codigo fuente del modelo campeon
|   +-- experiment_01..11_*.py            <- Experimentos comparativos
|   +-- run_all_experiments.py            <- Runner de comparacion de experimentos
|
+-- models/
|   +-- exp05_vae_catboost.pkl       <- ARTEFACTO DEPRECATED (threshold=0.6818)
    +-- exp05_vae_catboost_v2.pkl       <- ARTEFACTO EN PRODUCCIÓN (threshold=0.4606, 113 feats)
|
+-- data/
|   +-- raw/Dataset_01_Anonimizado.xlsx   <- Dataset de referencia (RGPD-safe)
|
+-- doc/AVP2/                        <- Documentacion de requisitos del cliente
+-- main.py                          <- Entry point (35 lineas)
+-- CHANGELOG.md                     <- Registro de auditoria y versiones
```

---

## Modelo en producción: Experimento 05 (VAE + CatBoost)

### Metricas en holdout 20% (datos nunca vistos durante entrenamiento)

| Metrica | Valor |
|---|---|
| F1-Weighted | **0.67** |
| F1-Macro | **0.57** |
| F1 clase NOK | **0.77** |
| F1 clase OK | **0.36** |
| Recall NOK | **0.76** |
| **Recall OK** | **0.39** (mejor de 11 experimentos) |
| ROC-AUC | **0.63** |
| **Threshold de produccion** | **0.4606** (optimizado OOF sobre F1-Macro) |
| Features de entrada al modelo | 113 (102 orig escaladas + 1 vae_err + 12 latents - 2 excluidas) |

### Por que Exp_05 es el modelo de produccion

1. **Maximo Recall OK (39%)**: Prioridad CTAG. Un false negative (pieza OK clasificada
   como NOK) es tolerable; un false positive (pieza NOK marcada como OK) no lo es.
   El Recall OK indica la capacidad de evitar desperdiciar piezas buenas.
2. **F1-Macro robusto (0.57)**: Equilibrio entre la clase mayoritaria (NOK) y la
   minoritaria (OK) en un dataset fuertemente desbalanceado.
3. **Sin data leakage**: VAE y StandardScaler se ajustan
   exclusivamente sobre el train pool de cada fold. El threshold se optimiza
   sobre predicciones OOF (Out-Of-Fold), nunca sobre el holdout.

### Pipeline de feature engineering (102 -> 113 variables)

```
Entrada (102 vars originales)
   |
   +-- Imputacion: mediana (numericas) / "missing" (categoricas)
   |
   +-- StandardScaler (ajustado solo en train, solo vars numericas)
   |
   +-- VAE Encoder (PyTorch, ajustado solo en train)
   |       -> vae_err  (error de reconstruccion)
   |       -> vae_l0 ... vae_l11  (12 dimensiones latentes)
   |
   +-- CatBoost con categorias nativas (auto_class_weights=Balanced)
   |
   -> 113 features totales -> CatBoostClassifier
                                   -> P(NOK) -> umbral 0.4606
```

---

## Flujo de Operacion

### 1. Ingesta de datos

**Carga de fichero**
- El operario carga un fichero **CSV o Excel** con una fila de datos de inspeccion.
- Hasta que se carga un fichero valido, el boton "Comprobar Calidad" permanece bloqueado.
- Matching tolerante: insensible a mayusculas/minusculas y espacios en las cabeceras.
- Las columnas ausentes se imputan con la media (numericas) o la primera categoria.
- Una columna numerica presente con valor no parseable bloquea la carga (error rojo).

**Formulario manual (visualizacion y edicion)**
- Tras la carga, el formulario muestra los valores del fichero en 102 campos agrupados en acordeones de 20 variables cada uno.
- Cada campo numerico muestra el rango `[min, max]` del dataset de referencia.
- El operario puede editar los valores antes de predecir.

**Prediccion por lotes**
- Seccion separada para subir un fichero con **N filas** y obtener el veredicto de cada pieza.
- Resultado descargable como CSV con columnas `Fila`, `Veredicto`, `P(NOK)`, `P(OK)`.

### 2. Validacion de rangos

Si un valor numerico esta fuera del rango `[min, max]` del dataset de entrenamiento,
aparece un bloque de aviso visual (`warning-banner` amarillo). La prediccion no se bloquea,
pero el operario es informado del valor anomalo.

### 3. Clasificacion

- `ml_engine.predict(row)` ejecuta el pipeline completo: scaler, VAE, CatBoost.
- Devuelve `(label, proba, df_fe)` donde `df_fe` contiene las 113 features transformadas.
- El banner cambia de color segun el umbral **0.4606**:
  - `P(NOK) >= umbral` -> **Banner rojo NOK** (pieza DEFECTUOSA).
  - `P(NOK) < umbral`  -> **Banner verde OK** (pieza CONFORME).

### 4. Tabla SHAP Neta (v2.2)

Ver seccion detallada mas abajo.

### 5. Diagnostico LLM

Ver seccion detallada mas abajo.

---

## Explicabilidad SHAP v2.2 - Logica Detallada

El modulo `utils/explainer.py` implementa cuatro transformaciones secuenciales
sobre los valores SHAP crudos de CatBoost para obtener una tabla industrialmente
significativa y sin ruido cognitivo.

### Paso 1 - Calculo de SHAP crudos

Se usa el metodo **nativo de CatBoost** `get_feature_importance(pool, type='ShapValues')`.
Se construye un `cb.Pool` con `cat_features` para que CatBoost reconozca las categoricas.
La matriz devuelta tiene forma `(1, n_features + 1)`; la ultima columna es el `base_value`
(valor esperado del modelo) y se descarta.

```
Entrada:  cb.Pool(df_fe_113_features, cat_features=cat_cols)
Salida:   array de 113 valores SHAP (uno por feature)
          SHAP > 0  =>  la feature empuja la prediccion hacia NOK (defecto)
          SHAP < 0  =>  la feature empuja la prediccion hacia OK (calidad)
```

### Paso 2 - Agregacion por variable

El nuevo pipeline (v2) elimina el `KBinsDiscretizer`, por lo que ya no existen
variables derivadas con sufijo `_bin`. Cada sensor fisico tiene exactamente un
valor SHAP directo. Las variables VAE (`vae_err`, `vae_l0..vae_l11`) tampoco
se cruzan con sensores fisicos.

```
SHAP_neto(sensor_X) = SHAP(sensor_X)   # sin derivadas binarias
```

Las variables VAE (`vae_err`, `vae_l*`) no tienen base fisica y no se agregan con nadie.

**Resultado:** un diccionario con un unico registro por sensor fisico, con su SHAP directo.

### Paso 3 - Filtro de caida relativa del 15% (Seccion 5.2)

Tras ordenar por `|SHAP_neto|` descendente:

```
threshold_15 = 0.15 * |SHAP_neto_top1|

Se descartan todas las variables donde:
    |SHAP_neto| < threshold_15

Excepcion: nunca se baja de 3 variables en la tabla.
```

Este filtro elimina el ruido de variables con impacto marginal que confundarian
al operario sin aportar informacion diagnostica relevante.

### Paso 4 - Balanceo Real/VAE (Seccion 5.3)

El modelo incluye variables sinteticas generadas por el VAE (`vae_err` y `latent_*`).
Aunque informativamente validas, tienen nombres abstractos que resultan menos
accionables para el operario que los sensores fisicos reales. El algoritmo garantiza:

```
Prioridad 1 (Minimo):  siempre >= 3 variables en la tabla.
Prioridad 2 (Balanceo): n_variables_reales >= n_variables_VAE.
    -> Si las VAE dominan el top inicial, se escanea el ranking
       descendente buscando variables reales adicionales.
    -> Si se agotan las variables reales disponibles antes del
       empate, el algoritmo se detiene (no duplica ni inventa).
Prioridad 3 (Maximo):  nunca > 10 variables en la tabla.
```

**Ejemplo ilustrativo:**

| Ranking inicial | Tipo | SHAP neto |
|---|---|---|
| latent_3 | VAE | +0.182 |
| latent_7 | VAE | +0.141 |
| presion_entrada | Real | -0.098 |
| temperatura | Real | +0.071 |

El ranking tiene 2 VAE y 2 Real -> equilibrio cumplido. La tabla se devuelve tal cual.

Si hubiera 3 VAE y 1 Real, el algoritmo bajaria en el ranking buscando
2 variables reales adicionales para igualar.

### Paso 5 - Renombrado industrial (Seccion 5.4)

| Nombre tecnico | Nombre industrial en UI |
|---|---|
| `vae_err` | Indice de Correlacion Global |
| `vae_l0` | Patron Estructural 0 |
| `vae_l5` | Patron Estructural 5 |
| `vae_l11` | Patron Estructural 11 |
| Resto | Nombre del sensor fisico original |

### Columnas de la tabla SHAP en la UI

| Columna | Descripcion |
|---|---|
| Variable | Nombre industrial del sensor o metrica VAE |
| Valor | Valor real medido del sensor en esta pieza |
| SHAP Neto | Impacto neto algebraico de este sensor en la prediccion |
| Direccion | `DEFECTO` (SHAP > 0) o `CALIDAD` (SHAP < 0) |

---

## LLM Local - Logica y System Prompt

### Configuracion

| Parametro | Valor |
|---|---|
| Endpoint | `http://localhost:11434/api/generate` |
| Modelo por defecto | `llama3` (configurable en `llm_client.py`) |
| Timeout streaming | 90 s |
| Umbral sagrado | **0.4606** (constante `NOK_THRESHOLD` en el modulo) |

### Verificacion de disponibilidad

Al arrancar la app, `check_ollama()` realiza dos comprobaciones:
1. **Ping al servidor**: `GET /api/tags` con timeout de 3 s.
2. **Verificacion del modelo**: comprueba si el modelo configurado aparece en la lista.

El badge en la cabecera de la UI muestra el estado en tiempo real (Online / Offline).

### System Prompt - Rol del LLM

El campo `"system"` del payload de Ollama contiene las instrucciones de rol permanentes:

```
Eres un Ingeniero Senior de Calidad del proyecto PLUTO para la empresa CTAG.
Tu mision es analizar el resultado del sistema automatico de clasificacion de piezas
industriales y comunicar el diagnostico al operario de linea de forma directa,
tecnica y sin ambiguedades.

REGLAS DE INTERPRETACION OBLIGATORIAS:
1. El umbral de clasificacion es 0.4606.
   Si P(NOK) >= 0.4606 => NOK (defectuosa).
   Si P(NOK) < 0.4606 => OK (conforme). Absoluto e inamovible.
2. En el analisis SHAP:
   - SHAP POSITIVO (+) => la variable EMPUJA la prediccion hacia el DEFECTO (NOK).
   - SHAP NEGATIVO (-) => la variable APOYA la CALIDAD (OK).
   - Los valores son impactos NETOS (ya incorporan la contribucion algebraica).
3. Tono: asertivo, tecnico, sin rodeos.
   No usar expresiones vagas como "podria ser" o "quizas".
4. Estructura de respuesta obligatoria:
   1. Diagnostico: resultado con probabilidad.
   2. Variables criticas: mayor impacto SHAP.
   3. Accion recomendada: que revisar (si NOK) o confirmacion de conformidad (si OK).
5. Si hay valores OK cercanos a sus limites operacionales, mencionarlos como alerta
   preventiva.
6. Respuesta maxima: 15 segundos de lectura. Ser conciso.
```

### User Prompt - Datos de la inspeccion

Cada consulta al LLM inyecta automaticamente el contexto en formato estructurado:

```
══ INSTRUCCIONES ESTRICTAS PARA ESTA RESPUESTA ══
1. RESPONDE ÚNICA Y EXCLUSIVAMENTE EN ESPAÑOL.
2. Basa tu diagnostico SOLO en las N variables listadas abajo. No menciones variables fantasma.
3. REGLA DE PORCENTAJES: El valor P(NOK) es la probabilidad de DEFECTO.
   Si la mencionas, aclara siempre que es probabilidad de DEFECTO.

══ DATOS DE LA INSPECCION ══
Veredicto:             NOK
P(NOK) [Prob. Defecto]: 0.7431  (74.31%)
Umbral:                0.4606
Decision:              P(NOK) >= umbral -> DEFECTO confirmado

ANALISIS SHAP - IMPACTO NETO POR SENSOR (Top N)
(Valores netos: suma algebraica sensor fisico + derivadas binarizadas)
  presion_entrada           valor=    2.3100   SHAP_neto=+0.18200  (DEFECTO)
  temperatura_camara        valor=  187.5000   SHAP_neto=+0.11400  (DEFECTO)
  Indice de Correlacion Global     valor=    0.0043   SHAP_neto=+0.09100  (DEFECTO)
  velocidad_extrusora       valor=   34.2000   SHAP_neto=-0.06700  (CALIDAD)

══ PREGUNTA DEL OPERARIO ══
[pregunta introducida por el operario]
```

El LLM nunca recibe datos crudos ni nombres tecnicos de variables VAE:
la tabla ya viene procesada por `explainer.py` con nombres industriales
y valores SHAP netos.

---

## Diseno UI Industrial (v2.2)

Interfaz disenada para operarios de planta con menos de 10 minutos de formacion
(requisito AVP2 del cliente CTAG).

### Paleta de colores

| Elemento | Color | Hex |
|---|---|---|
| Fondo global | Deep Navy Slate | `#0c1629` |
| Contenedores | Navy Dark | `#111c30` |
| Bordes | Slate Border | `#1e2d45` |
| Texto principal | White Industrial | `#f8fafc` |
| Texto secundario | Slate Gray | `#94a3b8` |
| Boton primario | Electric Blue | `#2563eb` |
| Banner OK | Verde Calidad | fondo `#064e3b`, texto `#34d399` |
| Banner NOK | Rojo Defecto | fondo `#7f1d1d`, texto `#f87171` |
| Umbral NOK | Ambar Alerta | `#fbbf24` |
| Chatbot/codigo | Navy Profundo | `#0f172a` |



## Comparativa de Experimentos

| Exp | Modelo | F1-W | F1-M | OK-F1 | NOK-F1 | Recall-OK | Tiempo |
|-----|--------|------|------|-------|--------|-----------|--------|
| Exp_01 | LightGBM | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 4.6s |
| Exp_02 | XGBoost + KMeans | 0.69 | 0.56 | 0.31 | 0.81 | 0.28 | 155s |
| Exp_03 | PyTorch MLP | 0.68 | 0.56 | 0.31 | 0.80 | 0.28 | 225s |
| Exp_04 | Cluster+Conquer | 0.69 | 0.57 | 0.32 | 0.81 | 0.29 | 113s |
| **Exp_05** | **VAE+CatBoost v2** | **0.67** | **0.57** | **0.36** | **0.77** | **0.39** | **40s** |
| Exp_06 | XGBOD | 0.71 | 0.58 | 0.34 | 0.83 | 0.28 | 246s |
| Exp_07 | DART+UMAP | 0.69 | 0.57 | 0.32 | 0.82 | 0.27 | 757s* |
| Exp_08 | ASL+LightGBM | 0.69 | 0.56 | 0.31 | 0.81 | 0.27 | 15s |
| Exp_09 | SupMin+TabM | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 221s |
| Exp_10 | CleanLab+Stack | 0.69 | 0.51 | 0.16 | 0.86 | 0.09 | 43s |
| Exp_11 | Ultimate Hybrid | 0.69 | 0.52 | 0.19 | 0.86 | 0.11 | 730s |

**Criterio de seleccion:** Recall OK es la metrica prioritaria para CTAG.
Exp_05 v2 es el maximo absoluto en Recall OK con un tiempo de entrenamiento razonable (40s).

---

## Privacidad y Cumplimiento

- **Local-First**: ningun dato de proceso sale del equipo local. Sin APIs externas.
- **Dataset anonimizado**: `Dataset_01_Anonimizado.xlsx` cumple RGPD.
- **LLM local**: Ollama corre en `localhost`. La consulta del operario nunca
  abandona la red local de planta.
- **Sin telemetria**: el sistema no registra ni envia metricas de uso.

---

## Licencia y Contacto

Proyecto desarrollado para **CTAG** por el equipo **CTG GT14**:

- Santiago Pereira
- Martin Rodriguez
- Lucas A. Martinez

**Universidad de Santiago de Compostela (USC) - PIIA 2025/2026**

Todos los datos estan anonimizados conforme al Reglamento General de Proteccion de Datos (RGPD).