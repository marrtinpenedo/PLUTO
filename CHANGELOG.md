# CHANGELOG — Proyecto PLUTO

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
