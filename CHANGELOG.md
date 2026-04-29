# CHANGELOG - Proyecto PLUTO

## [v2.2.1] - 2026-04-29 - Hotfix Sec 5.4: Renombrado industrial VAE corregido

**Scope:** `utils/explainer.py`, `README.md`
**Autor:** Antigravity / santipereeira

### Motivo
El GEMINI.md fue actualizado con la terminologia definitiva para las variables VAE:
- Anterior: `vae_err` -> "Anomalia de Correlacion General", `latent_*` -> "Desviacion de Patron Estructural".
- Nuevo (Sec 5.4 GEMINI.md): `vae_err` -> **"Indice de Correlacion Global"**, `latent_N` -> **"Patron Estructural N"**.

### Cambios en `utils/explainer.py`

- `_INDUSTRIAL_NAMES["vae_err"]`: `"Anomalia de Correlacion General"` -> `"Indice de Correlacion Global"`.
- `_industrial_name()`: ahora extrae el numero de dimension `N` del nombre tecnico `latent_N`
  y devuelve `"Patron Estructural N"` (en lugar de una cadena generica sin numero).
  - `latent_0`  -> `"Patron Estructural 0"`
  - `latent_5`  -> `"Patron Estructural 5"`
  - `latent_11` -> `"Patron Estructural 11"`
- Docstring del modulo actualizado con los nuevos nombres.

### Cambios en `README.md`

- Tabla de renombrado industrial (Paso 5 de la seccion SHAP) actualizada con ejemplos por dimension.
- Ejemplo de user prompt del LLM: `"Anomalia de Correlacion General"` -> `"Indice de Correlacion Global"`.

---

## [v2.2] - 2026-04-29 - Explainer Dinamico CTAG + LLM Senior + UI Ajustes

**Scope:** `utils/explainer.py`, `utils/llm_client.py`, `app/ui.py`
**Autor:** Antigravity / santipereeira
**GEMINI.md:** Secciones 5, 6 y 7 implementadas segun directiva v2.2.

### Cambios en `utils/explainer.py` (reescritura completa - Sec 5)

#### 5.1 - Agregacion algebraica de variables derivadas
- **Implementado:** Suma algebraica estricta de los valores SHAP de variables `_bin`
  al SHAP de su variable fisica base.
- **Gestion de signos:** Si la base apoya OK (-) y la binarizada apoya NOK (+),
  se restan algebraicamente. El resultado es el **impacto neto** del sensor fisico.
- **Identidad:** El nombre resultante es siempre el de la variable fisica original
  (se elimina el sufijo `_bin`).

#### 5.2 - Filtro de caida relativa del 15%
- **Regla del 15%:** Tras la agregacion, se descartan variables cuyo |SHAP_neto|
  sea inferior al 15% del |SHAP_neto| de la variable Top-1.
- **Excepcion de minimos:** El descarte nunca reduce el listado por debajo de 3 variables.

#### 5.3 - Algoritmo de balanceo Real/VAE (min 3, max 10)
- **Prioridad 1 (Minimo):** Se garantizan al menos 3 variables, priorizando mayor impacto neto.
- **Prioridad 2 (Balanceo):** `n_reales >= n_vae`. Si domina VAE, se escanea el ranking
  descendente buscando exclusivamente variables reales hasta alcanzar el empate.
- **Gestion de Escasez:** Si se agotan las reales antes del empate, el algoritmo para
  conservando todas las encontradas (no se inventan ni duplican registros).
- **Prioridad 3 (Maximo):** El listado nunca supera 10 filas.

#### 5.4 - Renombrado industrial
- `vae_err`    ->  "Anomalia de Correlacion General"
- `latent_*`   ->  "Desviacion de Patron Estructural"

---

### Cambios en `utils/llm_client.py` (Sec 6)

#### SYSTEM_PROMPT con rol Ingeniero Senior CTAG
- **Nuevo:** Constante `SYSTEM_PROMPT` con el rol y reglas de interpretacion.
  Se inyecta en el campo `"system"` del payload de Ollama (API nativa).
- **Rol:** Ingeniero Senior de Calidad del proyecto PLUTO / CTAG.
- **Tono:** Asertivo, tecnico, sin ambiguedades ni expresiones vagas.

#### Umbral Sagrado explicitado
- **Constante:** `NOK_THRESHOLD = 0.6818` definida a nivel de modulo.
- **Regla en SYSTEM_PROMPT:** P(NOK) < 0.6818 => OK (absoluto e inamovible).

#### Semantica SHAP inambigua
- **SHAP (+) => DEFECTO (NOK).** **SHAP (-) => CALIDAD (OK).**
- Explicitado tanto en el SYSTEM_PROMPT como en la seccion de datos del user-prompt.

#### Tabla ya procesada
- `build_prompt()` recibe la lista final de `explainer.py` con nombres industriales
  y valores SHAP netos. La columna `"SHAP_neto"` sustituye a `"SHAP"` crudo.
- Formato de tabla actualizado con columna `"Direccion"` (DEFECTO / CALIDAD).

---

### Cambios en `app/ui.py` (Sec 7)

#### CSS .prose ampliado
- Añadidos `.prose li`, `.prose ul`, `.prose ol` a la regla `color: #f8fafc !important;`
  para garantizar consistencia en listas dentro de bloques Markdown.

#### Tabla SHAP actualizada
- `gr.DataFrame` recibe ahora columnas explicitas:
  `["Variable", "Valor", "SHAP Neto", "Direccion"]`.
- Columna "Direccion" muestra "DEFECTO" / "CALIDAD" sin simbolos ni emojis.

#### Top-K actualizado
- `explainer.explain(..., top_k_range=(3, 10))` en consonancia con el max=10 de Sec 5.3.

#### Version en cabecera
- Actualizada de `v2.1` a `v2.2`.

#### Slider
- Mantenido `interactive=False` (ya correcto desde v3.4).
- Sin elementos HTML flotantes (ya correcto desde v3.4).

---

## [v3.4] - 2026-04-27 - Bugfix: Eliminacion marcador flotante de umbral + normalizacion CSS

**Scope:** `app/ui.py`
**Autor:** Antigravity / santipereeira

### Cambios en `app/ui.py`

#### 1. Eliminacion del marcador HTML flotante del slider (Seccion 4 GEMINI.md)
- **Bug:** El bloque `gr.HTML` con `position:relative` y los divs `.threshold-label`/`.threshold-mark`
  se descuadraban visualmente, generando superposicion sobre banners y el slider.
- **Fix:** Eliminado el `gr.HTML(f"<div id='threshold-slider'...")` completo.
- **Eliminadas clases CSS:** `#threshold-slider`, `.threshold-mark`, `.threshold-label`.
- **Label del slider suficiente:** `"Probabilidad de NOK  (umbral = 0.6818)"` ya comunica
  la informacion relevante al operario sin elementos superpuestos.
- **Slider mantiene** `interactive=False` (medidor de salida, no input).

#### 2. Limpieza de constantes huerfanas
- **Eliminadas:** `_THRESHOLD = 0.6818` y `_THRESHOLD_PCT = f"..."` que ya no se referenciaban.

#### 3. Normalizacion del string CSS
- **Convertido:** `CSS = f"""..."""` -> `CSS = """..."""` (ya no habia interpolaciones).
- **Normalizadas:** todas las llaves dobles `{{`/`}}` a llaves simples `{`/`}` en el CSS.
- **Resultado:** CSS valido y legible, sin riesgo de errores de format-string.

#### 4. Registro de cambio manual del usuario
- `gr.Chatbot(..., type="tuples")` añadido por el usuario para compatibilidad Gradio.

---

## [v3.3] - 2026-04-27 - Hotfix UI: Contraste Markdown + Codigo Oscuro + Tags Residuales


**Scope:** `app/ui.py`, `utils/llm_client.py`
**Autor:** Antigravity / santipereeira

### Cambios en `app/ui.py`

#### 1. Contraste de textos Markdown (Seccion 6 GEMINI.md)
- **Problema:** Titulos y parrafos de `gr.Markdown` se renderizaban oscuros e ilegibles en el tema dark.
- **Fix:** Nuevas reglas CSS sobre clases `.prose` de Gradio:
  ```css
  .prose h1, .prose h2, .prose h3, .prose h4,
  .prose p, .prose strong, .prose em, .prose span {
      color: #f8fafc !important;
  }
  ```
- **Resultado:** Texto blanco industrial (#f8fafc) en todos los elementos Markdown.

#### 2. Bloques de codigo integrados al tema oscuro (Seccion 6 GEMINI.md)
- **Problema:** Los bloques `\`\`\`bash` del chatbot (comandos Ollama) mostraban fondo blanco, rompiendo la estetica industrial.
- **Fix:** Nuevas reglas CSS:
  ```css
  .prose pre, .prose code {
      background-color: #0f172a !important;
      color: #e2e8f0 !important;
      border: 1px solid #334155 !important;
  }
  ```

### Cambios en `utils/llm_client.py`

#### 3. Eliminacion de tags residuales (Seccion 4 GEMINI.md)
- **Tags eliminados:** `[ERR]`, `[warn]`, `[timeout]` de todos los mensajes de error.
- **Afecta:** `check_ollama()` (3 mensajes) y `stream_response()` (3 mensajes).
- **Detalle de cambios:**
  - `"[ERR] Ollama no esta disponible..."` -> `"Ollama no esta disponible..."`
  - `"[ERR] Error al conectar con Ollama: {exc}"` -> `"Error al conectar con Ollama: {exc}"`
  - `"[warn] El servidor Ollama esta activo..."` -> `"El servidor Ollama esta activo..."`
  - `"[ERR] **Ollama no esta disponible**..."` -> `"**Ollama no esta disponible**..."`
  - `"[timeout] **Timeout:**..."` -> `"**Timeout:**..."`
  - `"[ERR] Error al conectar con el LLM: {exc}"` -> `"Error al conectar con el LLM: {exc}"`

---

## [v3.2] - 2026-04-27 - Rediseno UI V2.1: Limpieza de Tags + Slider con Marca de Umbral


**Scope:** `app/ui.py`, `README.md`
**Autor:** Antigravity (asistente IA) / santipereeira

### Cambios en `app/ui.py`

#### 1. Eliminacion de todos los tags de texto (Seccion 4 GEMINI.md)
- **Cambio:** Eliminados todos los prefijos de tag de la interfaz:
  `[PLUTO]`, `[Form]`, `[Sec]`, `[Search]`, `[Reset]`, `[Results]`, `[Inspect]`, `[Chat]`, `[i]`, `[warn]`, `[ERR]`, `[wait]`.
- **Afecta:** Cabecera, titulos de secciones, botones, banners OK/NOK/Error/Pending,
  mensajes del chatbot, warnings de rango, etiquetas de acordeon.
- **Motivo:** Cumplimiento AVP2 - entorno industrial: interfaz limpia sin ruido visual.

#### 2. Slider vinculado al motor + marca visual del umbral (Seccion 6 GEMINI.md)
- **El slider ya estaba vinculado:** `result_slider` es output directo de `on_predict`,
  recibiendo el valor `proba` retornado por `eng.predict()`. No habia desconexion.
- **Nuevo: marca visual del umbral:**
  - Constante `_THRESHOLD = 0.6818` y `_THRESHOLD_PCT` para calculo de posicion CSS.
  - CSS convertido a f-string para inyectar `_THRESHOLD_PCT` en tiempo de arranque.
  - Clases `#threshold-slider`, `.threshold-mark` (linea vertical amarilla `#fbbf24`)
    y `.threshold-label` (tooltip con borde amarillo) posicionadas con CSS absoluto
    en el `{_THRESHOLD_PCT}` del eje horizontal del slider.
  - HTML auxiliar insertado sobre el `gr.Slider` con las clases de marca.
- **Label del slider actualizada:** `"P(NOK) -- umbral de corte: 0.6818"`.
- **elem_id:** `"result_slider"` para facilitar tests e inspeccion.

#### 3. Mejoras CSS de componentes criticos (Seccion 6 GEMINI.md)
- `.chatbot .message`, `.chatbot .bot`, `.chatbot .user`: fondo `#0f172a` con borde
  `#1e2d45` (sin fondo blanco). Cumple requisito de integracion en tema oscuro.
- `#threshold-slider input[type=range]`: `accent-color: #3b82f6`, `cursor: not-allowed`
  (indicador visual de solo lectura).
- Version en cabecera actualizada: `v2.0 -> v2.1`.

#### 4. Banners limpios
- OK-banner: `"PIEZA OK"` (eliminado prefijo redundante `"OK "`).
- NOK-banner: `"PIEZA NOK"` (eliminado `"[ERR]"`).
- Pending-banner: `"Pendiente"` (eliminado `"[wait]"`).

---


## [v3.1] - 2026-04-26 - Hotfix UI: ASCII compatibility + CSV header matching

**Scope:** `app/ui.py`, `main.py`, `README.md`
**Autor:** santipereeira (edicion manual post-lanzamiento)

### Cambios en `app/ui.py`

#### 1. Normalizacion ASCII (compatibilidad cp1252 Windows)
- **Cambio:** Sustitucion de todos los caracteres Unicode/emoji por equivalentes ASCII.
- **Afecta:** labels de botones, banners HTML, mensajes de warning, titulos de secciones.
- **Motivo:** Evitar `UnicodeEncodeError` en terminales Windows con codec cp1252.

#### 2. Mejora del autocompletado CSV (`on_csv_upload`)
- **Antes:** Dos loops separados (primero `num_cols`, luego `cat_cols`), matching exacto por nombre de columna.
- **Ahora:** Un unico loop sobre `eng.orig_feats` (num + cat en orden correcto), con matching case-insensitive y tolerante a espacios en los headers del CSV.
- **Detalle:** `raw_cols = {str(c).strip().lower(): str(c).strip() for c in df_csv.columns}` permite que "Temperatura ", "TEMPERATURA" o "temperatura" se mapeen correctamente a la variable del modelo.
- **Fallback:** Si la columna no existe en el CSV, se usa la media (numerica) o la primera categoria (categorica).
- **Log de errores:** `print(f"[ERR] Error en autocompletado CSV: {e}")` para trazabilidad.

### Cambios en `main.py`

- Sustitucion de caracteres Unicode en docstring (compatibilidad ASCII).
- Linea en blanco al final del fichero (PEP 8).

### Cambios en `README.md`

- Reescritura completa para reflejar la arquitectura v2.0.
- Correccion de la ruta del modelo (`experiment_05_ultimate.pkl` -> `exp05_vae_catboost.pkl`).
- Instrucciones de ejecucion actualizadas: paso previo `export_exp05_model.py`.
- Descripcion del flujo CSV mejorada (tolerancia a cabeceras con espacios/mayusculas).
- Eliminacion de caracteres Unicode/emoji para compatibilidad multiplataforma.

---

## [v3.0] — 2026-04-26 · Refactorización Modular + Motor Exp_05


**Scope:** Arquitectura completa — `main.py` (monolito) -> `/app` + `/utils`
**Motor:** Migración de Exp_01 (LightGBM) a **Exp_05 (VAE + CatBoost)**
**Threshold:** 0.6818 | **Features:** 123 (102 orig + 10 bins + 1 vae_err + 12 vae_latents - 2 excluidas)

### Archivos nuevos

| Archivo | Descripción |
|---|---|
| `scripts/export_exp05_model.py` | Script one-shot: entrena Exp_05 y serializa pkl (32 s en CPU) |
| `models/exp05_vae_catboost.pkl` | Artefacto serializado: CatBoost + VAE state_dict + scalers |
| `utils/__init__.py` | Paquete utils raíz |
| `utils/ml_engine.py` | Motor de predicción singleton, pathlib-safe, con pipeline completo |
| `utils/explainer.py` | SHAP TreeExplainer con top-K dinámico (3-5 features) |
| `utils/llm_client.py` | Cliente Ollama: check server + check model + streaming |
| `app/__init__.py` | Paquete app |
| `app/ui.py` | Interfaz Gradio completa (manual + CSV + warnings + chatbot) |

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `main.py` | Reducido de 876 líneas a 35 (solo entry point) |
| `GEMINI.md` | Progreso marcado como completado |

### Decisiones técnicas

- **pathlib** en todas las rutas (compatibilidad Windows/Linux con espacios en OneDrive).
- VAE serializado como `bytes` dentro del pkl (evita dependencia de ficheros .pt separados).
- Singleton pattern en `get_engine()` y `get_explainer()` para cargar una sola vez al arrancar.
- Top-K dinámico SHAP: si las 3 primeras features acumulan >=80% del impacto, se muestran 3; si no, 5.
- Ollama: comprobación en dos pasos (server ping + model in /api/tags) con mensaje de error exacto.
- Warnings de rangos: no bloqueantes, HTML visual, calculados contra estadísticos del train pool.

---

## [v2.0] — 2026-04-25 · Auditoría de Integridad y Blindaje de Datos


**Merge:** `New-tests` → `prototipo-1`  
**Commit:** `6e99e12` · Author: santipereeira  
**Scope:** Experimentos 01–11 (`src/models/experiment_*.py`)  
**Objetivo:** Eliminar data leakage en todos los pipelines de ML y corregir inconsistencias arquitectónicas detectadas tras la integración de nuevos tests.

---

### Commits incluidos en este release

| Hash | Fecha | Mensaje | Autor |
|------|-------|---------|-------|
| `6e99e12` | 2026-04-25 | Merge branch 'New-tests' into prototipo-1 | santipereeira |
| `26c7b20` | 2026-04-21 | Fixed: data leakage??? | Lucas |
| `7688656` | 2026-04-18 | Primer pipeline prototipo + requirements + README exhaustivo | santipereeira |
| `520c60a` | 2026-04-16 | Added: New tests | Martin |

---

### Archivos modificados

```
M  README.md
M  main.py
M  requirements.txt
M  models/lightgbm_optimized_model.pkl
M  notebooks/pruebas_clasif_rawdata.ipynb
M  src/models/experiment_01_advanced.py
M  src/models/experiment_02_super_ensemble.py
M  src/models/experiment_03_pytorch_mlp.py
M  src/models/experiment_04_cluster_and_conquer.py
M  src/models/experiment_05_ultimate_vae_fe.py
M  src/models/experiment_06_xgbod_sota.py
M  src/models/experiment_07_insane_niche.py
M  src/models/experiment_08_nature_sota.py
M  src/models/experiment_09_supmin_tabm.py
M  src/models/experiment_10_autogluon_cleanlab.py
M  src/models/experiment_11_ultimate_hybrid.py
A  src/models/run_all_experiments.py          ← NUEVO
A  data/raw/Dataset_v2_anonimizado.xlsx       ← NUEVO
A  data/scaled/data_basic_10k.csv             ← NUEVO
A  doc/AVP2/AVP2.pdf                          ← NUEVO
A  doc/AVP2/UML_diagram.drawio                ← NUEVO
```

---

### Cambios Globales (todos los scripts)

#### 1. `load_raw_data()` — Solo carga, sin transformaciones
- **Antes:** aplicaba `.cat.codes`, `fillna` y `StandardScaler` globalmente sobre el dataset completo.
- **Ahora:** función estrictamente de carga. Solo lee el Excel, limpia el target (`OK→0`, `NOK→1`) y devuelve DataFrames crudos.
- **Motivo:** garantizar que el holdout sea invisible para cualquier cálculo estadístico previo al split.

#### 2. `OrdinalEncoder` en lugar de `.cat.codes`
- **Cambio:** sustitución de `pandas.Categorical.codes` por `sklearn.preprocessing.OrdinalEncoder`.
- **Config:** `handle_unknown='use_encoded_value', unknown_value=-1`
- **Motivo:** `.cat.codes` asigna códigos inconsistentes entre conjuntos; `OrdinalEncoder` aprende el mapeo en train y lo aplica de forma determinista en validación/test.

#### 3. Imputación post-split (Statistical Isolation)
- **Cambio:** `fillna` y cualquier imputación estadística se aplican **después** del `train_test_split`.
- **Motivo:** evitar que la distribución del test influya en los estadísticos de imputación del train (data leakage por imputación).

#### 4. Filtro `real_num_cols`
- **Cambio:** creación de una lista separada de columnas numéricas reales, excluyendo columnas ordinalmente encodizadas.
- **Impacto:** polinomios, cálculo de varianza y binning operan solo sobre variables continuas reales. Evita cálculos matemáticamente inválidos (varianza de identificadores nominales codificados como enteros).

#### 5. Feature Engineering per-fold (dentro del CV loop)
- **Cambio:** todo FE con estado (KMeans, UMAP, VAE, PolynomialFeatures, KBinsDiscretizer, StandardScaler) se ajusta **solo** con los datos de train de cada fold y se aplica (transform) al fold de validación.
- **Motivo:** evitar leakage de validación en el proceso de selección de hiperparámetros.

#### 6. Threshold tuning aislado
- **Cambio:** la búsqueda del threshold óptimo se realiza **solo** sobre las predicciones OOF (Out-Of-Fold) del train pool, nunca con datos del holdout.
- **Métricas reportadas:** únicamente sobre el holdout test (datos nunca vistos durante el entrenamiento).

---

### Cambios por Experimento

#### Exp 01, 02, 04, 07 — Consistencia de firmas (`apply_fe_per_fold`)
- **Bug:** `apply_fe_per_fold(X_tr, X_vl, num_cols)` llamada con 3 args, función definida con 4 (`num_cols, cat_cols`).
- **Fix:** sincronización de argumentos en todas las llamadas (CV loop, Optuna objective, evaluación final holdout).
- **Tipo de error:** `TypeError: missing 1 required positional argument: 'cat_cols'`

#### Exp 02 — XGBoost + KMeans Ensemble
- KMeans y StandardScaler ahora se ajustan per-fold (solo en train).
- Optuna inner CV opera exclusivamente sobre el train pool (3-fold).

#### Exp 03, 09 — Deep Learning (PyTorch)
- **Arquitectura:** conexión real entre el encoder (SupMinEncoder/VAE) y el clasificador final.
- **Flujo:** latentes concatenados a la entrada del clasificador.
- **Fix técnico:** uso de `.detach()` para separar gradientes entre el encoder y el clasificador, evitando backprop no deseada.

#### Exp 05 — VAE + CatBoost
- **Bug 1:** `apply_fe_per_fold` retorna 3 valores (`X_tr, X_vl, cat_features`) pero se desempaquetaba como 2 → `ValueError: too many values to unpack`.
- **Fix 1:** `X_tr, X_vl, _ = apply_fe_per_fold(...)` en todos los call sites.
- **Bug 2:** `KBinsDiscretizer` con `encode='ordinal'` produce floats; CatBoost rechaza floats en `cat_features`.
- **Fix 2:** `.astype(int)` en columnas `_bin`; eliminación de `cat_features` en `CatBoostClassifier` (todas las columnas ya están ordinalmente encodizadas a numérico).
- **VAE:** entrenamiento exclusivamente sobre el train pool; reconstrucción de error calculada post-split.

#### Exp 06 — XGBOD / Outlier Detection
- **Fix:** remapeo de etiquetas PyOD (`outlier=1`) a convención del proyecto (`NOK=1, OK=0`).
- **Fix:** inversión de probabilidades en el reporte final para reflejar la probabilidad de fallo real (no de outlier genérico).

#### Exp 07 — DART + UMAP + Polynomial + RUSBoost
- **Bug:** `RUSBoostClassifier` (internamente AdaBoost) fallaba con `ValueError: BaseClassifier worse than random` en el espacio de features post-UMAP/Poly.
- **Fix:** sustitución de `RUSBoostClassifier` por `BalancedBaggingClassifier` (mismo paquete `imblearn`), que usa bagging en lugar de boosting y es robusto a este fallo.
- **Nota operacional:** este experimento excede el timeout de 20 min del runner (757s). Debe ejecutarse de forma independiente.

#### Exp 08 — ASL + ORD (LightGBM)
- Detección de overlaps (ORD) realizada exclusivamente sobre el train pool.

#### Exp 10 — CleanLab + Meta-Stack
- CleanLab audita únicamente el train pool (nunca el holdout).
- Meta-Stack entrenado sobre datos limpios; evaluación final sobre holdout original completo.

#### Exp 11 — Ultimate Hybrid (VAE + Polynomials + CleanLab + Optuna + Stack)
- VAE ajustado solo sobre train pool.
- CleanLab aplicado post-split.
- Optuna HPO ejecutado sobre train limpio (nunca sobre holdout).

---

### Vulnerabilidades auditadas

| Vulnerabilidad | Estado | Mitigación |
|---|---|---|
| Target Leakage | ✅ Corregido | Holdout split al 20% antes de cualquier procesamiento |
| Categorical Shift | ✅ Corregido | `OrdinalEncoder` con `handle_unknown='use_encoded_value'` |
| Variance / Math Leakage | ✅ Corregido | Filtro `real_num_cols` excluye columnas encodizadas |
| Signature Mismatch (TypeError) | ✅ Corregido | Refactorización de firmas y lambdas en Optuna |
| Imputation Leakage | ✅ Corregido | `fillna` post-split en todos los scripts |
| FE State Leakage | ✅ Corregido | Scalers, KMeans, UMAP, VAE ajustados per-fold solo en train |
| Threshold Leakage | ✅ Corregido | Threshold tuning solo sobre OOF del train pool |
| CatBoost dtype error (float cat_features) | ✅ Corregido | `astype(int)` en bins + eliminación de `cat_features` |
| AdaBoost instability (RUSBoost) | ✅ Corregido | Reemplazado por `BalancedBaggingClassifier` |

---

### Resultados post-refactorización (`experiment_results.json`)

| Exp | F1-Weighted | F1-Macro | OK-F1 | NOK-F1 | OK-Recall | Tiempo |
|-----|------------|----------|-------|--------|-----------|--------|
| Exp_01 | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 4.6s |
| Exp_02 | 0.69 | 0.56 | 0.31 | 0.81 | 0.28 | 155s |
| Exp_03 | 0.68 | 0.56 | 0.31 | 0.80 | 0.28 | 225s |
| Exp_04 | 0.69 | 0.57 | 0.32 | 0.81 | 0.29 | 113s |
| Exp_05 | 0.69 | 0.57 | 0.35 | 0.80 | 0.33 | 40s |
| Exp_06 | 0.71 | 0.58 | 0.34 | 0.83 | 0.28 | 246s |
| Exp_07 | 0.69 | 0.57 | 0.32 | 0.82 | 0.27 | 757s* |
| Exp_08 | 0.69 | 0.56 | 0.31 | 0.81 | 0.27 | 15s |
| Exp_09 | 0.68 | 0.51 | 0.17 | 0.85 | 0.11 | 221s |
| Exp_10 | 0.69 | 0.51 | 0.16 | 0.86 | 0.09 | 43s |
| Exp_11 | 0.69 | 0.52 | 0.19 | 0.86 | 0.11 | 730s |

*Exp_07 supera el timeout de 20 min del runner. Ejecutar individualmente.

---

### Archivos de referencia

- `experiment_results.json` — métricas completas de todos los experimentos
- `src/models/run_all_experiments.py` — runner secuencial con parsing automático de métricas
- `requirements.txt` — dependencias actualizadas (imblearn, catboost, optuna, pyod, cleanlab, umap-learn, torch)
