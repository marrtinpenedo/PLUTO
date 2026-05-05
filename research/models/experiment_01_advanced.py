#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 01 - Optmización Continua y Threshold Tuning en LightGBM
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios respecto al original:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - Feature engineering (estadísticas por fila) aplicado PER-FOLD.
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales reportadas ÚNICAMENTE sobre holdout test.
  - num_boost_round del modelo final = mediana de todos los folds.
"""

import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuración base
RANDOM_STATE = 42
K_FOLDS = 5
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
MODEL_SAVE_PATH = '../../models/lightgbm_optimized_model.pkl'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'
    MODEL_SAVE_PATH = 'models/lightgbm_optimized_model.pkl'


def load_raw_data():
    """Carga el dataset y devuelve X (sin procesar) e y."""
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])
    df = df.drop(columns=['ID','Variable 02'])
    y = df['target'].values
    X = df.drop(columns=['target'])
    return X, y


def apply_feature_engineering(X_train, X_val, cat_cols, num_cols):
    """
    Aplica FE per-fold sin leakage y SIN desincronizar las categorías.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    # 1. Sincronización estricta de variables categóricas
    for col in cat_cols:
        # Train crea el diccionario base
        X_train[col] = X_train[col].astype('category')
        
        # Val/Holdout adopta EXACTAMENTE las mismas categorías que Train
        X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories)

    # 2. Feature Engineering por fila (Totalmente seguro contra leakage)
    if num_cols:
        X_train['num_sum'] = X_train[num_cols].sum(axis=1)
        X_train['num_mean'] = X_train[num_cols].mean(axis=1)
        X_train['num_std'] = X_train[num_cols].std(axis=1)
        X_train['num_min'] = X_train[num_cols].min(axis=1)
        X_train['num_max'] = X_train[num_cols].max(axis=1)

        X_val['num_sum'] = X_val[num_cols].sum(axis=1)
        X_val['num_mean'] = X_val[num_cols].mean(axis=1)
        X_val['num_std'] = X_val[num_cols].std(axis=1)
        X_val['num_min'] = X_val[num_cols].min(axis=1)
        X_val['num_max'] = X_val[num_cols].max(axis=1)

    return X_train, X_val


def main():
    print("=" * 60)
    print(" EXPERIMENTO 01 (LIGHTGBM AVANZADO) — SIN LEAKAGE")
    print("=" * 60)

    start_time = time.time()

    # 1. CARGA DE DATOS
    print("[1] Cargando dataset crudo...")
    X, y = load_raw_data()

    # 2. HOLDOUT TEST SPLIT (20%) — ANTES de cualquier procesamiento
    print("[2] Separando holdout test (20%) ANTES de cualquier FE...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool = X_pool.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)

    print(f"    - Train pool: {X_pool.shape[0]} | Holdout test: {X_holdout.shape[0]}")
    print(f"    - Distribución pool: {np.bincount(y_pool)} | holdout: {np.bincount(y_holdout)}")

    # Identificar tipos de columnas (antes de FE)
    cat_cols = X_pool.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_pool.select_dtypes(include=[np.number]).columns.tolist()

    # 3. CROSS-VALIDATION Y OOF (sólo sobre train pool)
    print(f"\n[3] Ejecutando {K_FOLDS}-Fold Stratified CV sobre train pool...")
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds_proba = np.zeros(len(y_pool))
    models = []
    best_iterations = []

    ratio = np.sum(y_pool == 0) / np.sum(y_pool == 1)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'is_unbalance': True,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr_raw, y_tr = X_pool.iloc[train_idx], y_pool[train_idx]
        X_vl_raw, y_vl = X_pool.iloc[val_idx], y_pool[val_idx]

        # FE per-fold (no leakage — row-level stats are independent per sample)
        X_tr, X_vl = apply_feature_engineering(X_tr_raw, X_vl_raw, cat_cols, num_cols)

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_vl, label=y_vl, categorical_feature=cat_cols, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],  # FIX: sólo val_data para early stopping
            callbacks=callbacks
        )

        models.append(model)
        best_iterations.append(model.best_iteration)
        oof_preds_proba[val_idx] = model.predict(X_vl, num_iteration=model.best_iteration)
        print(f"    - Fold {fold + 1}/{K_FOLDS} (Mejor iteración: {model.best_iteration})")

    # 4. THRESHOLD TUNING (sólo sobre OOF del train pool)
    print("\n[4] Optimización de Threshold sobre OOF del train pool...")
    best_threshold = 0.5
    best_f1_w = -1

    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        preds_t = (oof_preds_proba >= t).astype(int)
        f1_w = f1_score(y_pool, preds_t, average='weighted')
        if f1_w > best_f1_w:
            best_f1_w = f1_w
            best_threshold = t

    print(f"    - Threshold óptimo (train OOF): {best_threshold:.4f}  F1-W(OOF): {best_f1_w:.4f}")

    # 5. EVALUACIÓN FINAL SOBRE HOLDOUT TEST
    print("\n[5] Evaluación Final sobre HOLDOUT TEST (datos nunca vistos)...")

    # Preparar holdout con FE
    X_pool_fe, X_holdout_fe = apply_feature_engineering(X_pool, X_holdout, cat_cols, num_cols)

    # Re-entrenar modelo final sobre todo el train pool
    median_iters = int(np.median(best_iterations))
    print(f"    - Re-entrenando con {median_iters} iteraciones (mediana de folds)...")

    final_train_data = lgb.Dataset(X_pool_fe, label=y_pool, categorical_feature=cat_cols)
    final_model = lgb.train(lgb_params, final_train_data, num_boost_round=median_iters)

    # Predecir holdout test
    holdout_proba = final_model.predict(X_holdout_fe, num_iteration=final_model.best_iteration)
    holdout_preds = (holdout_proba >= best_threshold).astype(int)

    target_names = ['OK (Minoritario)', 'NOK (Mayoritario)']
    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=target_names))

    f1_macro = f1_score(y_holdout, holdout_preds, average='macro')
    f1_weighted = f1_score(y_holdout, holdout_preds, average='weighted')
    print(f"    => F1 Macro (HOLDOUT):    {f1_macro:.4f}")
    print(f"    => F1 Weighted (HOLDOUT): {f1_weighted:.4f}")

    # Guardar modelo
    print("\n[6] Guardando modelo final...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model_data = {
        'model': final_model,
        'threshold': best_threshold,
        'features': X_pool_fe.columns.tolist()
    }
    joblib.dump(model_data, MODEL_SAVE_PATH)
    print(f"    [!] Modelo guardado en: {MODEL_SAVE_PATH}")
    print(f"Tiempo total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
