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

### Requisitos previos

| Requisito | Version minima | Notas |
|---|---|---|
| Python | 3.10+ | Recomendado 3.11 o 3.12 |
| Ollama | cualquiera | Debe estar ejecutando `ollama serve` |
| RAM | 8 GB min | 16 GB recomendado para el LLM |
| OS | Windows / Linux | Rutas gestionadas con `pathlib` |

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
ollama serve                    # dejar en ejecucion en otra terminal
```

### Generar el modelo (primera vez)

Si `models/exp05_vae_catboost.pkl` no existe aun:

```bash
python scripts/export_exp05_model.py
```

Tiempo estimado: ~32 s en CPU. El script entrena el pipeline completo
(VAE + CatBoost) sobre el dataset de referencia y serializa el artefacto.

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
|   Formulario manual (102 vars)  |  Carga CSV (fuzzy match)         |
|   Banner OK/NOK + Slider P(NOK) |  Tabla SHAP Neta + Chatbot LLM   |
+------------------------------+------------------------------------- +
                               |
           +-------------------+-------------------+
           |             BACKEND UTILS              |
           +--------+------------+------------------+
           |        |            |                  |
    ml_engine.py  explainer.py  llm_client.py       |
    (Exp_05 pkl) (SHAP v2.2)   (Ollama Sistema)     |
    102 -> 123   Agregación   System Prompt CTAG   |
    features     Real/VAE      NOK_THRESHOLD=0.6818  |
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
|  models/exp05_vae_catboost.pkl                   |
+--------------------------------------------------+
```

### Stack tecnologico

| Capa | Tecnologia | Detalle |
|---|---|---|
| UI | Gradio | Interfaz + servidor unificados, dark theme CTAG |
| ML | CatBoost + VAE (PyTorch) | Serializado en `.pkl` con scaler + VAE state_dict |
| xAI | SHAP TreeExplainer | Agregacion neta, balanceo Real/VAE, max 10 vars |
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
+-- src/models/
|   +-- experiment_05_ultimate_vae_fe.py  <- Codigo fuente del modelo campeon
|   +-- experiment_01..11_*.py            <- Experimentos comparativos
|   +-- run_all_experiments.py            <- Runner de comparacion de experimentos
|
+-- models/
|   +-- exp05_vae_catboost.pkl       <- ARTEFACTO EN PRODUCCION (threshold=0.6818)
|
+-- data/
|   +-- raw/Dataset_01_Anonimizado.xlsx   <- Dataset de referencia (RGPD-safe)
|
+-- notebooks/                       <- Analisis exploratorio (EDA)
+-- doc/AVP2/                        <- Documentacion de requisitos del cliente
+-- main.py                          <- Entry point (35 lineas)
+-- CHANGELOG.md                     <- Registro de auditoria y versiones
```

---

## Modelo Campeon: Experimento 05 (VAE + CatBoost)

### Metricas en holdout 20% (datos nunca vistos durante entrenamiento)

| Metrica | Valor |
|---|---|
| F1-Weighted | 0.69 |
| F1-Macro | 0.57 |
| F1 clase NOK | 0.80 |
| F1 clase OK | 0.35 |
| **Recall OK** | **0.33** (mejor de 11 experimentos) |
| **Threshold de produccion** | **0.6818** |
| Features de entrada al modelo | 123 (102 orig + 10 bins + 1 vae_err + 12 latents - 2 excluidas) |

### Por que Exp_05 es el modelo de produccion

1. **Maximo Recall OK (33%)**: Prioridad CTAG. Un false negative (pieza OK clasificada
   como NOK) es tolerable; un false positive (pieza NOK marcada como OK) no lo es.
   El Recall OK indica la capacidad de evitar desperdiciar piezas buenas.
2. **F1-Macro robusto (0.57)**: Equilibrio entre la clase mayoritaria (NOK) y la
   minoritaria (OK) en un dataset fuertemente desbalanceado.
3. **Sin data leakage**: VAE, KBinsDiscretizer y StandardScaler se ajustan
   exclusivamente sobre el train pool de cada fold. El threshold se optimiza
   sobre predicciones OOF (Out-Of-Fold), nunca sobre el holdout.

### Pipeline de feature engineering (102 -> 123 variables)

```
Entrada (102 vars originales)
   |
   +-- StandardScaler (ajustado solo en train)
   |
   +-- KBinsDiscretizer (10 variables numericas -> _bin)
   |
   +-- VAE Encoder (PyTorch, ajustado solo en train)
   |       -> vae_err  (error de reconstruccion)
   |       -> latent_0 ... latent_11  (12 dimensiones latentes)
   |
   -> 123 features totales -> CatBoostClassifier
                                  -> P(NOK) -> umbral 0.6818 -> OK / NOK
```

---

## Flujo de Operacion

### 1. Ingesta de datos

**Formulario manual**
- 102 variables agrupadas en acordeones de 20 variables cada uno.
- Cada campo muestra la media del dataset de entrenamiento como valor por defecto.
- Los campos numericos muestran el rango `[min, max]` del dataset de referencia.

**Carga CSV**
- El operario puede subir un fichero CSV con cualquier subconjunto de las 102 variables.
- Matching tolerante: insensible a mayusculas/minusculas y espacios en las cabeceras.
- Las columnas ausentes se imputan con la media (numericas) o la primera categoria.

### 2. Validacion de rangos

Si un valor numerico esta fuera del rango `[min, max]` del dataset de entrenamiento,
aparece un bloque de aviso visual (`warning-banner` amarillo). La prediccion no se bloquea,
pero el operario es informado del valor anomalo.

### 3. Clasificacion

- `ml_engine.predict(row)` ejecuta el pipeline completo: scaler, bins, VAE, CatBoost.
- Devuelve `(label, proba, df_fe)` donde `df_fe` contiene las 123 features transformadas.
- El banner cambia de color segun el umbral sagrado **0.6818**:
  - `P(NOK) >= 0.6818` -> **Banner rojo NOK** (pieza DEFECTUOSA).
  - `P(NOK) < 0.6818`  -> **Banner verde OK** (pieza CONFORME).

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

Se usa `shap.TreeExplainer` sobre el modelo CatBoost.
Para clasificacion binaria, se extraen los valores SHAP de la **clase 1 (NOK)**.

```
Entrada:  df_fe con 123 features
Salida:   array de 123 valores SHAP (uno por feature)
          SHAP > 0  =>  la feature empuja la prediccion hacia NOK (defecto)
          SHAP < 0  =>  la feature empuja la prediccion hacia OK (calidad)
```

### Paso 2 - Agregacion algebraica 

El pipeline genera variables derivadas con sufijo `_bin` para cada variable numerica
binariazada con `KBinsDiscretizer`. Estas variables derivadas no son independientes de
su sensor fisico: miden lo mismo desde otra perspectiva. Por ello se **suman algebraicamente**:

```
SHAP_neto(sensor_X) = SHAP(sensor_X) + SHAP(sensor_X_bin)
```

**Gestion de contradicciones de signo:**
Si `SHAP(sensor_X) = -0.05` (apoya OK) y `SHAP(sensor_X_bin) = +0.03` (apoya NOK),
el impacto neto es `-0.02`. El signo final refleja el **efecto neto real** del sensor.

Las variables VAE (`vae_err`, `latent_*`) no tienen base fisica y no se agregan con nadie.

**Resultado:** un diccionario con un unico registro por sensor fisico, con su SHAP neto.

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
| `latent_0` | Patron Estructural 0 |
| `latent_5` | Patron Estructural 5 |
| `latent_11` | Patron Estructural 11 |
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
| Umbral sagrado | **0.6818** (constante `NOK_THRESHOLD` en el modulo) |

### Verificacion de disponibilidad

Al arrancar la app, `check_ollama()` realiza dos comprobaciones:
1. **Ping al servidor**: `GET /api/tags` con timeout de 3 s.
2. **Verificacion del modelo**: comprueba si el modelo configurado aparece en la lista.

El badge en la cabecera de la UI muestra el estado en tiempo real (Online / Offline).

### System Prompt - Rol del LLM

El campo `"system"` del payload de Ollama contiene las instrucciones de rol permanentes:

```
Eres un Ingeniero Senior de Calidad del proyecto PLUTO para la empresa CTAG.
Tu mision es analizar el resultado del sistema automatico de clasificacion
de piezas industriales y comunicar el diagnostico al operario de linea de
forma directa, tecnica y sin ambiguedades.

REGLAS DE INTERPRETACION OBLIGATORIAS:
1. Umbral sagrado: 0.6818.
   - P(NOK) >= 0.6818  =>  pieza NOK (defectuosa). Absoluto e inamovible.
   - P(NOK) < 0.6818   =>  pieza OK (conforme).
2. Semantica SHAP:
   - SHAP POSITIVO (+) => la variable EMPUJA hacia el DEFECTO (NOK).
   - SHAP NEGATIVO (-) => la variable APOYA la CALIDAD (OK).
   - Los valores son NETOS: ya incorporan la contribucion de las derivadas.
3. Tono: asertivo, tecnico, sin rodeos. Prohibido usar expresiones vagas.
4. Estructura de respuesta obligatoria:
   1. Diagnostico
   2. Variables criticas
   3. Accion recomendada
5. Respuesta maxima: 15 segundos de lectura.
```

### User Prompt - Datos de la inspeccion

Cada consulta al LLM inyecta automaticamente el contexto de la prediccion activa:

```
DATOS DE LA INSPECCION
Veredicto:             NOK
P(NOK):                0.7431  (74.31%)
P(OK):                 0.2569  (25.69%)
Umbral sagrado:        0.6818
Decision:              P(NOK) >= umbral -> DEFECTO confirmado

ANALISIS SHAP - IMPACTO NETO POR SENSOR (Top N)
(Valores netos: suma algebraica sensor fisico + derivadas binarizadas)
  presion_entrada           valor=    2.3100   SHAP_neto=+0.18200  (DEFECTO)
  temperatura_camara        valor=  187.5000   SHAP_neto=+0.11400  (DEFECTO)
  Indice de Correlacion Global     valor=    0.0043   SHAP_neto=+0.09100  (DEFECTO)
  velocidad_extrusora       valor=   34.2000   SHAP_neto=-0.06700  (CALIDAD)

PREGUNTA DEL OPERARIO
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

### Reglas UX (requisito AVP2)

- **Cero emojis** en toda la interfaz.
- **Cero tags de texto** (`[PLUTO]`, `[Form]`, `[ERR]`, `[wait]`, etc.).
- Slider `P(NOK)` en modo `interactive=False` (es un indicador de salida, no un input).
- Sin elementos HTML flotantes que se superpongan a otros componentes.
- Feedback de estado inmediato: banners de color semaforo (verde/rojo) segun el umbral.

### Elemento slider

```
0%                    68.18%                  100%
|---------------------|------------------------|
                       ^
                  Umbral NOK (0.6818)
                  comunicado via label del slider
```

El label del slider informa directamente: `"Probabilidad de NOK (umbral = 0.6818)"`.
No se usan elementos HTML flotantes (eliminados en v3.4).

---

## Comparativa de Experimentos

| Exp | Modelo | F1-W | F1-M | OK-F1 | NOK-F1 | Recall-OK | Tiempo |
|-----|--------|------|------|-------|--------|-----------|--------|
| Exp_01 | LightGBM | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 4.6s |
| Exp_02 | XGBoost + KMeans | 0.69 | 0.56 | 0.31 | 0.81 | 0.28 | 155s |
| Exp_03 | PyTorch MLP | 0.68 | 0.56 | 0.31 | 0.80 | 0.28 | 225s |
| Exp_04 | Cluster+Conquer | 0.69 | 0.57 | 0.32 | 0.81 | 0.29 | 113s |
| **Exp_05** | **VAE+CatBoost** | **0.69** | **0.57** | **0.35** | **0.80** | **0.33** | **40s** |
| Exp_06 | XGBOD | 0.71 | 0.58 | 0.34 | 0.83 | 0.28 | 246s |
| Exp_07 | DART+UMAP | 0.69 | 0.57 | 0.32 | 0.82 | 0.27 | 757s* |
| Exp_08 | ASL+LightGBM | 0.69 | 0.56 | 0.31 | 0.81 | 0.27 | 15s |
| Exp_09 | SupMin+TabM | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 221s |
| Exp_10 | CleanLab+Stack | 0.69 | 0.51 | 0.16 | 0.86 | 0.09 | 43s |
| Exp_11 | Ultimate Hybrid | 0.69 | 0.52 | 0.19 | 0.86 | 0.11 | 730s |

*Exp_07 supera el timeout de 20 min. Ejecutar individualmente.

**Criterio de seleccion:** Recall OK es la metrica prioritaria para CTAG.
Exp_05 es el maximo absoluto en Recall OK con un tiempo de entrenamiento razonable (40s).

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