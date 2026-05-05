#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 10 - CleanLab Confident Learning + Meta-Stack
---------------------------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de CleanLab o cualquier procesamiento.
  - Codificación Categórica e Imputación realizadas DESPUÉS del holdout split.
  - CleanLab aplicado SÓLO sobre train pool.
  - is_unbalance=False para CleanLab's LGB (mejor calibración de probas).
  - Stacker entrenado en train pool limpio, evaluado en holdout test ORIGINAL.
  - Métricas finales sobre holdout test.
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder # Añadido
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

try:
    from cleanlab.filter import find_label_issues
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import StackingClassifier
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'

RANDOM_STATE = 42


def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)
    df = df.drop(columns=['ID','Variable 02'], errors='ignore')
    
    # LEAKAGE CORREGIDO: Devolvemos los datos crudos, sin transformar.
    df[target_col] = df[target_col].map({'NOK': 1, 'OK': 0})
    return df, target_col


def cleanlab_on_train_only(X_train, y_train):
    """
    Confident Learning applied ONLY to training data.
    Uses default LGB (no is_unbalance) for better calibrated probabilities.
    """
    print("\n[!] CleanLab: Auditando sólo train pool...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_pred_probs = np.zeros((len(y_train), 2))

    for train_idx, val_idx in skf.split(X_train, y_train):
        Xt, yt = X_train[train_idx], y_train[train_idx]
        Xv, yv = X_train[val_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
        model.fit(Xt, yt)
        cv_pred_probs[val_idx] = model.predict_proba(Xv)

    ranked_label_issues = find_label_issues(
        labels=y_train,
        pred_probs=cv_pred_probs,
        return_indices_ranked_by='self_confidence',
    )

    print(f"    - CleanLab detectó {len(ranked_label_issues)} etiquetas problemáticas.")

    # Remove problematic indices from training data
    clean_mask = np.ones(len(y_train), dtype=bool)
    clean_mask[ranked_label_issues] = False

    X_clean = X_train[clean_mask]
    y_clean = y_train[clean_mask]
    print(f"    - Train original: {len(y_train)} -> Train limpio: {len(y_clean)}")

    return X_clean, y_clean


def main():
    start_time = time.time()
    print("=" * 60)
    print(" EXPERIMENTO 10 (CLEANLAB + META-STACK) — SIN LEAKAGE")
    print("=" * 60)

    print("[1] Cargando datos...")
    df, target_col = load_raw_data()
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # Identificamos columnas para preprocesamiento posterior
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # HOLDOUT SPLIT — BEFORE CleanLab AND Preprocessing
    print("[2] Separando holdout test (20%) ANTES de preprocesar...")
    X_pool_df, X_holdout_df, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    print("[3] Aplicando preprocesamiento (OrdinalEncoding + Imputación)...")
    # LEAKAGE CORREGIDO: Imputación y codificación POST-split
    X_pool_df[num_cols] = X_pool_df[num_cols].fillna(0)
    X_holdout_df[num_cols] = X_holdout_df[num_cols].fillna(0)

    if len(cat_cols) > 0:
        X_pool_df[cat_cols] = X_pool_df[cat_cols].fillna('missing')
        X_holdout_df[cat_cols] = X_holdout_df[cat_cols].fillna('missing')
        
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_pool_df[cat_cols] = oe.fit_transform(X_pool_df[cat_cols])
        X_holdout_df[cat_cols] = oe.transform(X_holdout_df[cat_cols])
        
    X_pool = X_pool_df.values
    X_holdout = X_holdout_df.values

    # CleanLab on train pool ONLY
    X_clean, y_clean = cleanlab_on_train_only(X_pool, y_pool)

    # Stack ensemble trained on CLEANED train, evaluated on ORIGINAL holdout
    print("\n[4] Entrenando Meta-Stack sobre datos limpios...")
    cb = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3,
        loss_function='Logloss', verbose=0, random_state=RANDOM_STATE
    )
    xgb_model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
    )
    meta = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.01,
        random_state=RANDOM_STATE, verbosity=-1
    )

    stacker = StackingClassifier(
        estimators=[('cb', cb), ('xgb', xgb_model)],
        final_estimator=meta,
        cv=5,
        n_jobs=-1
    )

    stacker.fit(X_clean, y_clean)

    # HOLDOUT evaluation (original, un-cleaned holdout)
    print("\n[5] Evaluación Final sobre HOLDOUT TEST (datos originales)...")
    preds = stacker.predict(X_holdout)

    print("\n" + "*" * 50)
    print(" RESULTADOS FINALES: CLEANLAB + ADVANCED STACK")
    print("*" * 50)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()