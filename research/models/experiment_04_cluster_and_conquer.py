#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 04 - "Divide & Conquer" 
--------------------------------------------------------------------
Protocolo aplicado:
  - Holdout test (20%) separado en crudo.
  - OrdinalEncoder con memoria para evitar desincronización de categorías.
  - GMM + StandardScaler ajustados únicamente con el Train Pool.
  - Entrenamiento de expertos locales con validación cruzada interna.
  - Umbral óptimo calculado sobre predicciones fuera de la muestra (OOF).
"""

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'


def load_raw_data():
    """Carga datos crudos sin procesar."""
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    # Excluimos IDs y target, devolvemos crudo
    X = df.drop(columns=[target_col, 'target', 'ID', 'Variable 02'], errors='ignore')
    return X, y


def preprocess_pipeline(X_pool_df, X_holdout_df):
    """Encapsula la codificación e imputación post-split."""
    X_p = X_pool_df.copy()
    X_h = X_holdout_df.copy()
    
    cat_cols = X_p.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_p.select_dtypes(exclude=['object']).columns.tolist()
    
    # 1. Imputación de nulos
    X_p[num_cols] = X_p[num_cols].fillna(0)
    X_h[num_cols] = X_h[num_cols].fillna(0)
    
    # 2. Ordinal Encoding Seguro (con memoria de categorías)
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_p[cat_cols] = oe.fit_transform(X_p[cat_cols].astype(str).fillna('missing'))
        X_h[cat_cols] = oe.transform(X_h[cat_cols].astype(str).fillna('missing'))
        
    # 3. Feature Engineering por fila (independiente por muestra)
    for df in [X_p, X_h]:
        df['num_sum'] = df[num_cols].sum(axis=1)
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std'] = df[num_cols].std(axis=1)
        
    return X_p, X_h, num_cols


def optimize_threshold(y_val, preds_proba):
    """Encuentra el umbral que maximiza el F1 manteniendo Recall NOK > 0.85."""
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (preds_proba >= t).astype(int)
        recall_nok = recall_score(y_val, preds, pos_label=1)
        if recall_nok < 0.85:
            continue
        recall_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        # Target industrial: F1 con premio a la precisión en OK
        custom_target = f1_w + (recall_ok * 0.20)
        if custom_target > best_score:
            best_score = custom_target
            best_t = t
    return best_t, best_score


def cluster_optuna_objective(trial, X_c, y_c):
    """Optimización local para cada experto del clúster."""
    folds = 3 if len(y_c) > 300 else 2
    try:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        param = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 1.5),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        scores = []
        for train_idx, val_idx in skf.split(X_c, y_c):
            X_tr, y_tr = X_c.iloc[train_idx], y_c[train_idx]
            X_vl, y_vl = X_c.iloc[val_idx], y_c[val_idx]
            
            model = xgb.XGBClassifier(**param, n_estimators=200, early_stopping_rounds=20)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            
            p_val = model.predict_proba(X_vl)[:, 1]
            _, b_score = optimize_threshold(y_vl, p_val)
            scores.append(b_score if b_score != -1.0 else 0.0)
            
        return np.mean(scores) if scores else 0.0
    except Exception:
        return 0.0


def main():
    print("=" * 60)
    print(" EXPERIMENTO 04 (DIVIDE & CONQUER GMM) — BLINDADO")
    print("=" * 60)
    start_time = time.time()

    print("[1] Cargando datos crudos...")
    X_raw, y = load_raw_data()

    # 2. HOLDOUT SPLIT (Separación total del futuro)
    X_pool_raw, X_holdout_raw, y_pool, y_holdout = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    
    # Preprocesamiento post-split
    X_pool, X_holdout, original_num_cols = preprocess_pipeline(X_pool_raw, X_holdout_raw)
    X_pool = X_pool.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)

    # 
    # 3. Clustering GMM (solo con Train Pool)
    print("\n[2] Enrutamiento de datos (GMM Clustering)...")
    N_CLUSTERS = 3
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool[original_num_cols])
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='tied', random_state=RANDOM_STATE)
    pool_cluster_labels = gmm.fit_predict(X_pool_scaled)

    # 4. Entrenamiento de Expertos Locales
    print("[3] Entrenando Expertos Locales (Optuna)...")
    expert_models = {}
    oof_preds_global = np.zeros(len(y_pool))

    for c in range(N_CLUSTERS):
        idx_c = np.where(pool_cluster_labels == c)[0]
        X_c, y_c = X_pool.iloc[idx_c].reset_index(drop=True), y_pool[idx_c]

        if len(np.unique(y_c)) < 2:
            print(f"    - Clúster {c} es puro. Asignando clase constante.")
            oof_preds_global[idx_c] = float(y_c[0])
            expert_models[c] = None
            continue

        print(f"    - Optimizando Experto para Clúster {c}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: cluster_optuna_objective(t, X_c, y_c), n_trials=10)

        best_p = study.best_params
        best_p.update({'objective': 'binary:logistic', 'random_state': RANDOM_STATE, 'n_estimators': 400})

        # Generar OOF para este experto
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for t_idx, v_idx in skf.split(X_c, y_c):
            m = xgb.XGBClassifier(**best_p, early_stopping_rounds=20)
            m.fit(X_c.iloc[t_idx], y_c[t_idx], eval_set=[(X_c.iloc[v_idx], y_c[v_idx])], verbose=False)
            oof_preds_global[idx_c[v_idx]] = m.predict_proba(X_c.iloc[v_idx])[:, 1]
        
        # Modelo final del experto para producción
        master_m = xgb.XGBClassifier(**best_p)
        master_m.fit(X_c, y_c, verbose=False)
        expert_models[c] = master_m

    # 5. Threshold Tuning Global
    print("\n[4] Threshold Tuning Final (OOF Train Pool)...")
    best_t_global, _ = optimize_threshold(y_pool, oof_preds_global)
    print(f"    - Threshold Óptimo: {best_t_global:.4f}")

    # 6. Evaluación en Holdout (Simulación de Producción)
    print("\n[5] Evaluación sobre HOLDOUT TEST...")
    X_holdout_scaled = scaler.transform(X_holdout[original_num_cols])
    holdout_cluster_labels = gmm.predict(X_holdout_scaled)
    holdout_probs = np.zeros(len(y_holdout))

    for c in range(N_CLUSTERS):
        idx_h = np.where(holdout_cluster_labels == c)[0]
        if len(idx_h) == 0: continue
        
        if expert_models[c] is None:
            # Caso de clúster puro en train
            idx_p_c = np.where(pool_cluster_labels == c)[0]
            holdout_probs[idx_h] = float(y_pool[idx_p_c[0]])
        else:
            holdout_probs[idx_h] = expert_models[c].predict_proba(X_holdout.iloc[idx_h])[:, 1]

    holdout_preds = (holdout_probs >= best_t_global).astype(int)
    
    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación:\n", classification_report(y_holdout, holdout_preds))
    print(f"Tiempo total: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()