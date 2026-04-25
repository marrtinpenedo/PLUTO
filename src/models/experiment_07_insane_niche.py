#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 07 - DART + RUSBoost + UMAP + Polynomial Interactions
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - PolynomialFeatures fitted PER-FOLD (stateless but feature selection on train).
  - UMAP fitted PER-FOLD (fit on fold-train, transform fold-val).
  - StandardScaler fitted PER-FOLD.
  - Supervised feature selection (LightGBM) PER-FOLD.
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales sobre holdout test.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from imblearn.ensemble import RUSBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import umap
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
        score = f1_w + (r_ok * 0.35)
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
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target'])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes

    X = X.fillna(0)
    return X, y


def apply_fe_per_fold(X_train, X_val, y_train):
    """
    Per-fold feature engineering:
    1. Polynomial interactions (fit on train variance selection)
    2. UMAP (fit on train, transform val)
    3. Supervised feature selection (fit on train only)
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Polynomial: select top-10 variance cols FROM TRAIN
    variances = X_train[num_cols].var().sort_values(ascending=False)
    top_10_vars = variances.head(10).index.tolist()

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly_tr = poly.fit_transform(X_train[top_10_vars])
    X_poly_vl = poly.transform(X_val[top_10_vars])
    poly_cols = [f"poly_{i}" for i in range(X_poly_tr.shape[1])]

    df_poly_tr = pd.DataFrame(X_poly_tr, columns=poly_cols, index=X_train.index)
    df_poly_vl = pd.DataFrame(X_poly_vl, columns=poly_cols, index=X_val.index)
    X_train = pd.concat([X_train, df_poly_tr], axis=1)
    X_val = pd.concat([X_val, df_poly_vl], axis=1)

    # 2. UMAP: fit on train, transform val
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_vl_scaled = scaler.transform(X_val)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    X_umap_tr = reducer.fit_transform(X_tr_scaled)
    X_umap_vl = reducer.transform(X_vl_scaled)

    X_train['umap_x'] = X_umap_tr[:, 0]
    X_train['umap_y'] = X_umap_tr[:, 1]
    X_val['umap_x'] = X_umap_vl[:, 0]
    X_val['umap_y'] = X_umap_vl[:, 1]

    # 3. Feature selection: fit LightGBM on TRAIN only
    model_fs = lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, is_unbalance=True, verbose=-1)
    model_fs.fit(X_train, y_train)
    importances = model_fs.feature_importances_
    valid_indices = np.where(importances >= 5)[0]
    cols_to_keep = X_train.columns[valid_indices]

    return X_train[cols_to_keep], X_val[cols_to_keep]


def get_niche_ensemble():
    dart = xgb.XGBClassifier(
        booster='dart', rate_drop=0.1, skip_drop=0.5, max_depth=5,
        scale_pos_weight=0.8, random_state=RANDOM_STATE, n_estimators=300, n_jobs=-1
    )
    rus = RUSBoostClassifier(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE)
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=30, min_samples_leaf=4,
        class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[('dart', dart), ('rus', rus), ('et', et)],
        voting='soft', weights=[2.0, 1.5, 1.0]
    )
    return ensemble


def main():
    print("=" * 60)
    print(" EXPERIMENTO 07 (DART+UMAP+POLY+RUS) — SIN LEAKAGE")
    print("=" * 60)
    start_time = time.time()

    print("[1] Cargando datos...")
    X, y = load_raw_data()

    # HOLDOUT SPLIT
    print("[2] Separando holdout test (20%)...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool = X_pool.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    # 5-Fold CV with per-fold FE
    print("\n[3] 5-Fold CV con FE per-fold (Poly+UMAP+FS)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr_raw, y_tr = X_pool.iloc[train_idx], y_pool[train_idx]
        X_vl_raw, y_vl = X_pool.iloc[val_idx], y_pool[val_idx]

        # All FE per-fold (poly, UMAP, feature selection — all fitted on train)
        X_tr, X_vl = apply_fe_per_fold(X_tr_raw, X_vl_raw, y_tr)

        ensemble = get_niche_ensemble()
        ensemble.fit(X_tr, y_tr)
        oof_proba[val_idx] = ensemble.predict_proba(X_vl)[:, 1]
        print(f"    - Fold {fold + 1}/5 listo.")

    # Threshold tuning on OOF (train pool only)
    print("\n[4] Threshold tuning sobre OOF del train pool...")
    best_threshold, _ = maximize_threshold(y_pool, oof_proba)
    print(f"    - Threshold: {best_threshold:.4f}")

    # HOLDOUT evaluation
    print("\n[5] Evaluación Final sobre HOLDOUT TEST...")
    X_pool_fe, X_holdout_fe = apply_fe_per_fold(X_pool, X_holdout, y_pool)

    final_ensemble = get_niche_ensemble()
    final_ensemble.fit(X_pool_fe, y_pool)

    holdout_proba = final_ensemble.predict_proba(X_holdout_fe)[:, 1]
    holdout_preds = (holdout_proba >= best_threshold).astype(int)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
