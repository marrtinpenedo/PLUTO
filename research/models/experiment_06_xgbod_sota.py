#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 06 - XGBOD (State-of-the-Art Outlier Detection)
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - OrdinalEncoding e Imputación realizados POST-SPLIT de forma segura.
  - StandardScaler DENTRO del CV loop (fit fold-train, transform fold-val).
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales sobre holdout test.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pyod.models.xgbod import XGBOD
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'


def maximize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_target = -1.0
    for t in np.linspace(0.01, 0.99, 150):
        preds = (preds_proba >= t).astype(int)
        r_nok = recall_score(y_val, preds, pos_label=1)
        if r_nok < 0.85:
            continue
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        score = f1_w + (r_ok * 0.3)
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target


def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    # XGBOD: OK=1 (outlier/minority), NOK=0 (inlier/majority) for PyOD internals
    df['target'] = df[target_col].map({'NOK': 0, 'OK': 1})
    df = df.drop(columns=[target_col])
    df = df.drop(columns=['ID','Variable 02'], errors='ignore')
    
    y = df['target'].values
    # CORRECCIÓN: Devolvemos el DataFrame crudo, sin transformar
    X = df.drop(columns=['target'])

    return X, y


def main():
    print("=" * 60)
    print(" EXPERIMENTO 06 (XGBOD SOTA) — SIN LEAKAGE")
    print("=" * 60)
    start_time = time.time()

    print("[1] Cargando datos...")
    X_df, y_pyod = load_raw_data()
    
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_df.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # HOLDOUT SPLIT (Sobre los datos crudos)
    print("[2] Separando holdout test (20%) ANTES de preprocesar...")
    X_pool_df, X_holdout_df, y_pool, y_holdout = train_test_split(
        X_df, y_pyod, test_size=0.20, stratify=y_pyod, random_state=RANDOM_STATE
    )
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    # CORRECCIÓN: Preprocesamiento de categorías y nulos POST-SPLIT
    print("\n[2.5] Aplicando Imputación y Codificación Ordinal Segura...")
    X_pool_df[num_cols] = X_pool_df[num_cols].fillna(0)
    X_holdout_df[num_cols] = X_holdout_df[num_cols].fillna(0)

    if len(cat_cols) > 0:
        X_pool_df[cat_cols] = X_pool_df[cat_cols].fillna('missing')
        X_holdout_df[cat_cols] = X_holdout_df[cat_cols].fillna('missing')
        
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_pool_df[cat_cols] = oe.fit_transform(X_pool_df[cat_cols])
        X_holdout_df[cat_cols] = oe.transform(X_holdout_df[cat_cols])
        
    # Convertimos a arrays de NumPy para el resto del código
    X_pool = X_pool_df.values
    X_holdout = X_holdout_df.values

    # 5-Fold CV on train pool with per-fold scaling
    print("\n[3] 5-Fold CV con XGBOD (scaler per-fold)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_train_raw, y_train = X_pool[train_idx], y_pool[train_idx]
        X_val_raw, y_val = X_pool[val_idx], y_pool[val_idx]

        # Scaler per-fold (Correcto)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        model = XGBOD(random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        preds_array = model.predict_proba(X_val)
        if preds_array.ndim == 1:
            oof_proba[val_idx] = preds_array
        else:
            oof_proba[val_idx] = preds_array[:, 1]
        print(f"    - Fold {fold + 1} completado.")

    # Remap to business convention: OK=0, NOK=1
    business_nok_proba = 1.0 - oof_proba
    y_business_pool = np.where(y_pool == 1, 0, 1)  

    # Threshold tuning on OOF (train pool only)
    print("\n[4] Threshold tuning sobre OOF del train pool...")
    best_threshold, _ = maximize_threshold(y_business_pool, business_nok_proba)
    print(f"    - Threshold (NOK): {best_threshold:.4f}")

    # HOLDOUT evaluation
    print("\n[5] Evaluación Final sobre HOLDOUT TEST...")
    final_scaler = StandardScaler()
    X_pool_scaled = final_scaler.fit_transform(X_pool)
    X_holdout_scaled = final_scaler.transform(X_holdout)

    final_model = XGBOD(random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X_pool_scaled, y_pool)

    holdout_proba_raw = final_model.predict_proba(X_holdout_scaled)
    if holdout_proba_raw.ndim == 1:
        holdout_ok_proba = holdout_proba_raw
    else:
        holdout_ok_proba = holdout_proba_raw[:, 1]

    holdout_nok_proba = 1.0 - holdout_ok_proba
    y_business_holdout = np.where(y_holdout == 1, 0, 1)

    holdout_preds = (holdout_nok_proba >= best_threshold).astype(int)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_business_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_business_holdout, holdout_preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"Tiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()