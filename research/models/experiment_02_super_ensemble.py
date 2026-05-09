#!/usr/bin/env python


# -*- coding: utf-8 -*-

"""
EXPERIMENTO 02 - Híper-Optimización de Modelo Industrial (XGBoost)
--------------------------------------------------------------------
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - KMeans + StandardScaler DENTRO del CV loop (fit sólo en fold-train).
  - Optuna ejecutado SÓLO sobre train pool (inner 3-fold CV).
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales reportadas ÚNICAMENTE sobre holdout test.
"""


import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OrdinalEncoder # Añadido OrdinalEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
MODEL_SAVE_PATH = '../../models/xgboost_super_optimized.pkl'

def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target', 'ID', 'Variable 02'], errors='ignore')

    # CORRECCIÓN: No codificamos aquí. Devolvemos crudo para el OrdinalEncoder post-split.
    return X, y

def apply_fe_per_fold(X_train_raw, X_val_raw, num_cols, cat_cols):
    """
    Feature engineering per-fold:
    - Encoding, Scaling y KMeans aprenden SOLO de X_train.
    """
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()

    # 1. Encoding Categórico Post-Split
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].fillna('missing'))
        X_val[cat_cols] = encoder.transform(X_val[cat_cols].fillna('missing'))

    # 2. Row-level aggregations (Sin leakage por definición)
    if num_cols:
        for df in [X_train, X_val]:
            df['num_sum'] = df[num_cols].sum(axis=1)
            df['num_mean'] = df[num_cols].mean(axis=1)
            df['num_std'] = df[num_cols].std(axis=1)

    # 3. KMeans: Fit en train, predict en val
    if num_cols:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train[num_cols].fillna(0))
        X_vl_scaled = scaler.transform(X_val[num_cols].fillna(0))

        kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=5)
        X_train['cluster_id'] = kmeans.fit_predict(X_tr_scaled)
        X_val['cluster_id'] = kmeans.predict(X_vl_scaled)

    return X_train, X_val


def optimize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (preds_proba >= t).astype(int)
        recall_nok = recall_score(y_val, preds, pos_label=1)
        if recall_nok < 0.85:
            continue
        recall_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        custom_target = f1_w + (recall_ok * 0.15)
        if custom_target > best_score:
            best_score = custom_target
            best_t = t
    return best_t, best_score


def optuna_objective(trial, X_pool, y_pool, num_cols, cat_cols):
    """Optuna inner CV — runs ONLY on the train pool."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 1.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    scores = []
    for train_idx, val_idx in skf.split(X_pool, y_pool):
        X_tr_raw, y_tr = X_pool.iloc[train_idx], y_pool[train_idx]
        X_vl_raw, y_vl = X_pool.iloc[val_idx], y_pool[val_idx]

        # FE per-fold inside Optuna
        X_tr, X_vl = apply_fe_per_fold(X_tr_raw, X_vl_raw, num_cols, cat_cols)

        model = xgb.XGBClassifier(**param, n_estimators=600, early_stopping_rounds=40)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        preds_proba = model.predict_proba(X_vl)[:, 1]
        _, best_val_score = optimize_threshold(y_vl, preds_proba)
        scores.append(best_val_score)

    return np.mean(scores)


def main():
    print("=" * 60)
    print(" EXPERIMENTO 02 (XGBOOST + KMEANS) — SIN LEAKAGE")
    print("=" * 60)

    start_time = time.time()

    # 1. Load raw data
    print("[1] Cargando datos...")
    X, y = load_raw_data()

    # 2. HOLDOUT SPLIT — before any FE
    print("[2] Separando holdout test (20%)...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool = X_pool.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    num_cols = X_pool.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_pool.select_dtypes(include=['object']).columns.tolist()

    # 3. Optuna HPO (inner CV on train pool only)
    print("\n[3] Optuna HPO (3-Fold CV sobre train pool)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optuna_objective(trial, X_pool, y_pool, num_cols, cat_cols),
                   n_trials=25, n_jobs=1)
    print(f"    [OK] Mejor Trial Score: {study.best_value:.4f}")
    best_params = study.best_params

    # 4. Final 5-Fold CV evaluation on train pool (with tuned params)
    print("\n[4] 5-Fold CV (train pool) con parámetros óptimos...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_proba = np.zeros(len(y_pool))

    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'n_jobs': -1
    })

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr_raw, y_tr = X_pool.iloc[train_idx], y_pool[train_idx]
        X_vl_raw, y_vl = X_pool.iloc[val_idx], y_pool[val_idx]

        X_tr, X_vl = apply_fe_per_fold(X_tr_raw, X_vl_raw, num_cols, cat_cols)

        model = xgb.XGBClassifier(**final_params, early_stopping_rounds=60)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        oof_preds_proba[val_idx] = model.predict_proba(X_vl)[:, 1]

    # 5. Threshold tuning on OOF (train pool only)
    print("\n[5] Threshold tuning sobre OOF del train pool...")
    best_threshold = 0.5
    best_score = -1
    for t in np.linspace(0.1, 0.9, 150):
        preds_t = (oof_preds_proba >= t).astype(int)
        recall_nok = recall_score(y_pool, preds_t, pos_label=1)
        if recall_nok < 0.85:
            continue
        f1_ok = f1_score(y_pool, preds_t, pos_label=0)
        f1_w = f1_score(y_pool, preds_t, average='weighted')
        target_score = f1_w + (f1_ok * 0.5)
        if target_score > best_score:
            best_score = target_score
            best_threshold = t

    print(f"    - Threshold: {best_threshold:.4f}")

    # 6. HOLDOUT evaluation
    print("\n[6] Evaluación Final sobre HOLDOUT TEST...")
    X_pool_fe, X_holdout_fe = apply_fe_per_fold(X_pool, X_holdout, num_cols, cat_cols)

    master_model = xgb.XGBClassifier(**final_params)
    master_model.fit(X_pool_fe, y_pool, verbose=False)

    holdout_proba = master_model.predict_proba(X_holdout_fe)[:, 1]
    holdout_preds = (holdout_proba >= best_threshold).astype(int)

    target_names = ['OK (Minoritario)', 'NOK (Mayoritario)']
    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=target_names))

    print(f"Tiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
