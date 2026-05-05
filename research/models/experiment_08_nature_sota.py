#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 08 - NATURE 2025 & ASYMMETRIC LOSS SOTA (ASL + ORD)
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - Imputación, Codificación y Feature Engineering aplicados POST-SPLIT.
  - ORD computado SÓLO sobre train pool (inner CV).
  - Sample weights derivados de ORD sólo sobre train pool.
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales sobre holdout test.
"""

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import OrdinalEncoder
from scipy.special import expit
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'


def custom_asymmetric_loss(y_true, y_pred):
    p = expit(y_pred)
    alpha = 0.8
    grad = np.where(y_true == 0, alpha * (p - y_true), (1 - alpha) * (p - y_true))
    hess = np.where(y_true == 0, alpha * p * (1 - p), (1 - alpha) * p * (1 - p))
    return grad, hess


def maximize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_target = -1.0
    for t in np.linspace(0.1, 0.9, 150):
        preds = (preds_proba >= t).astype(int)
        r_nok = recall_score(y_val, preds, pos_label=1)
        if r_nok < 0.85:
            continue
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        score = f1_w + (r_ok * 0.4)
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target


def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])
    df = df.drop(columns=['ID','Variable 02'], errors='ignore')
    
    y = df['target'].values
    
    # CORRECCIÓN: Devolvemos datos crudos
    X = df.drop(columns=['target'])
    
    return X, y


def preprocess_data(X_pool_df, X_holdout_df):
    """
    Applies imputation, encoding, and feature engineering post-split.
    """
    X_pool = X_pool_df.copy()
    X_holdout = X_holdout_df.copy()
    
    cat_cols = X_pool.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_pool.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Imputación de nulos
    X_pool[num_cols] = X_pool[num_cols].fillna(0)
    X_holdout[num_cols] = X_holdout[num_cols].fillna(0)

    if len(cat_cols) > 0:
        X_pool[cat_cols] = X_pool[cat_cols].fillna('missing')
        X_holdout[cat_cols] = X_holdout[cat_cols].fillna('missing')
        
        # Codificación Ordinal Segura
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_pool[cat_cols] = oe.fit_transform(X_pool[cat_cols])
        X_holdout[cat_cols] = oe.transform(X_holdout[cat_cols])

    # Feature Engineering de la suma (ahora num_cols incluye las nuevas generadas)
    current_num_cols = X_pool.select_dtypes(include=[np.number]).columns.tolist()
    if current_num_cols:
        X_pool['num_sum'] = X_pool[current_num_cols].sum(axis=1)
        X_holdout['num_sum'] = X_holdout[current_num_cols].sum(axis=1)

    return X_pool, X_holdout


def apply_ord_on_train(X_pool, y_pool):
    """
    Overlap Region Detection — computed ONLY on train pool (inner CV).
    Returns sample_weights array for train pool.
    """
    from sklearn.ensemble import RandomForestClassifier

    print("\n[!] ORD: Detectando overlap sólo en train pool...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    overlap_flags = np.zeros(len(y_pool), dtype=bool)

    for train_idx, val_idx in skf.split(X_pool, y_pool):
        X_train, y_train = X_pool.iloc[train_idx], y_pool[train_idx]
        X_val, y_val = X_pool.iloc[val_idx], y_pool[val_idx]

        rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, class_weight='balanced')
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)

        # NOK (1) misclassified as OK (0) = overlap region
        false_oks = (y_val == 1) & (preds == 0)
        for i, is_overlap in zip(val_idx, false_oks):
            if is_overlap:
                overlap_flags[i] = True

    overlap_count = overlap_flags.sum()
    print(f"    - Overlap detectados: {overlap_count}")

    sample_weights = np.ones(len(y_pool))
    sample_weights[overlap_flags] = 0.05
    return sample_weights


def main():
    print("=" * 60)
    print(" EXPERIMENTO 08 (ASL + ORD) — SIN LEAKAGE")
    print("=" * 60)
    start_time = time.time()

    print("[1] Cargando datos...")
    X_raw, y = load_raw_data()

    # HOLDOUT SPLIT (Sobre los datos crudos)
    print("[2] Separando holdout test (20%) ANTES de preprocesar...")
    X_pool_raw, X_holdout_raw, y_pool, y_holdout = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool_raw = X_pool_raw.reset_index(drop=True)
    X_holdout_raw = X_holdout_raw.reset_index(drop=True)
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    # Preprocesamiento POST-SPLIT
    print("\n[2.5] Aplicando Imputación, Codificación y Feature Engineering...")
    X_pool, X_holdout = preprocess_data(X_pool_raw, X_holdout_raw)

    # ORD on train pool ONLY
    sample_weights = apply_ord_on_train(X_pool, y_pool)

    # 5-Fold CV with ASL on train pool
    print("\n[3] 5-Fold CV con LightGBM + ASL...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_train, y_train = X_pool.iloc[train_idx], y_pool[train_idx]
        X_val, y_val = X_pool.iloc[val_idx], y_pool[val_idx]
        w_train = sample_weights[train_idx]

        model_asl = lgb.LGBMClassifier(
            learning_rate=0.05, num_leaves=31, max_depth=5, n_estimators=300,
            verbosity=-1, random_state=RANDOM_STATE, objective=custom_asymmetric_loss
        )
        model_asl.fit(X_train, y_train, sample_weight=w_train)

        raw_preds = model_asl.predict(X_val, raw_score=True)
        oof_proba[val_idx] = expit(raw_preds)

    # Threshold tuning on OOF (train pool only)
    print("\n[4] Threshold tuning sobre OOF del train pool...")
    best_t, _ = maximize_threshold(y_pool, oof_proba)
    if best_t == 0.5 and maximize_threshold(y_pool, oof_proba)[1] == -1.0:
        best_t = 0.55
    print(f"    - Threshold: {best_t:.4f}")

    # HOLDOUT evaluation
    print("\n[5] Evaluación Final sobre HOLDOUT TEST...")
    final_model = lgb.LGBMClassifier(
        learning_rate=0.05, num_leaves=31, max_depth=5, n_estimators=300,
        verbosity=-1, random_state=RANDOM_STATE, objective=custom_asymmetric_loss
    )
    final_model.fit(X_pool, y_pool, sample_weight=sample_weights)

    holdout_raw = final_model.predict(X_holdout, raw_score=True)
    holdout_proba = expit(holdout_raw)
    holdout_preds = (holdout_proba >= best_t).astype(int)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()