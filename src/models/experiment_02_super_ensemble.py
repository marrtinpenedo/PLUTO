#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 02 - Híper-Optimización de Modelo Industrial
--------------------------------------------------------------------
- Tuning agresivo de XGBoost (Class Weights / Scale Pos Weight invertido y focalizado) 
- Feature Engineering profundo: Agrupaciones y KMeans locales.
- Función objetivo penalizada hacia F1-Weighted pero forzando separación explícita
  de la clase OK sin destruir Recall de NOK.
"""

import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
MODEL_SAVE_PATH = '../../models/xgboost_super_optimized.pkl'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'
    MODEL_SAVE_PATH = 'models/xgboost_super_optimized.pkl'

def load_and_preprocess_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # Mapping: NOK=1, OK=0
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])
    
    y = df['target'].values
    X = df.drop(columns=['target'])
    
    # Identify dtypes
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Label encode explicitly for XGBoost since it handles ints well
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    # Agregaciones numéricas elementales
    if num_cols:
        X['num_sum'] = X[num_cols].sum(axis=1)
        X['num_mean'] = X[num_cols].mean(axis=1)
        X['num_std'] = X[num_cols].std(axis=1)
    
    # Clustering Feature (para identificar mini-anomalias latentes)
    if num_cols:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[num_cols].fillna(0))
        # KMeans rápido
        kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=5)
        X['cluster_id'] = kmeans.fit_predict(X_scaled)
        # One-hot de cluster (para ayudar a xgboost a aislar comportamientos)
        clusters_dummies = pd.get_dummies(X['cluster_id'], prefix='cluster')
        X = pd.concat([X, clusters_dummies], axis=1)
        X.drop(columns=['cluster_id'], inplace=True)
    
    return X, y

def optimize_threshold_for_focal_business(y_val, preds_proba):
    best_t = 0.5
    best_score = -1.0
    
    # Explorar todos los percentiles 
    for t in np.linspace(0.1, 0.9, 100):
        preds = (preds_proba >= t).astype(int)
        
        recall_nok = recall_score(y_val, preds, pos_label=1)
        recall_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        
        # Penalizamos duramente cualquier combinación que rompa el recall de NOK (< 0.85)
        if recall_nok < 0.85:
            continue
            
        # Queremos maximizar el F1 ponderado, pero bonificar si el recall de OK sube.
        # Esto empujará la optimización a que diferencia realmente la clase OK.
        custom_target = f1_w + (recall_ok * 0.15)
        
        if custom_target > best_score:
            best_score = custom_target
            best_t = t
            
    return best_t, best_score

def optuna_objective(trial, X, y):
    # k-fold robusto dentro de Optuna
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # Hiperparámetros de XGBoost
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
        # NOK(1) es mayoritario (~3 veces OK(0)).  
        # scale_pos_weight > 1 da más peso a C1 (NOK). 
        # scale_pos_weight < 1 da más peso a C0 (OK).
        # Exploremos este parámetro crucial (0.1 a 0.9 para dar peso masivo a OK)
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 1.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**param, n_estimators=600, early_stopping_rounds=40)
        
        # Fit with evaluation
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds_proba = model.predict_proba(X_val)[:, 1]
        _, best_val_score = optimize_threshold_for_focal_business(y_val, preds_proba)
        
        scores.append(best_val_score)
        
    return np.mean(scores)


def main():
    print("="*60)
    print(" PASO 3 [EXTRA] - HIPER TUNE Y SEPARACIÓN PROFUNDA (XGBOOST)")
    print("="*60)
    
    start_time = time.time()
    
    print("[1] Cargando y preprocesando datos con clustering integrado...")
    X, y = load_and_preprocess_data()
    print(f"    - Dimensiones Finales: {X.shape}")
    
    print("\n[2] Lanzando Optuna sobre 3-Fold CV (Esto puede tomar unos minutos)...")
    study = optuna.create_study(direction='maximize')
    # Limitando a 25 trials para que se resuelva en tiempo útil.
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=25, n_jobs=1)
    
    print(f"    [OK] Mejor Trial Score: {study.best_value:.4f}")
    best_params = study.best_params
    
    print("\n[3] Re-Entrenando y Validando con 5-Fold Estratificado (Evaluación Final)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds_proba = np.zeros(len(X))
    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'n_estimators': 1500, # mas largo al final
        'n_jobs': -1
    })
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**final_params, early_stopping_rounds=60)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        oof_preds_proba[val_idx] = model.predict_proba(X_val)[:, 1]
    
    print("\n[4] Análisis y Threshold Tuning Definitivo...")
    best_threshold = 0.5
    best_f1_ok = -1
    best_f1_w = -1
    
    # Exploración detallada asegurando Recall NOK >= 85% y maximizando OK
    for t in np.linspace(0.1, 0.9, 150):
        preds_t = (oof_preds_proba >= t).astype(int)
        
        recall_nok = recall_score(y, preds_t, pos_label=1)
        if recall_nok < 0.85:
            continue
            
        f1_ok = f1_score(y, preds_t, pos_label=0)
        f1_w = f1_score(y, preds_t, average='weighted')
        
        target_score = f1_w + (f1_ok*0.5)
        
        if target_score > best_f1_w:
            best_f1_w = target_score
            best_f1_ok = f1_ok
            best_threshold = t
            
    final_preds = (oof_preds_proba >= best_threshold).astype(int)
    
    print("\n" + "*"*50)
    print(" RESULTADOS CRÍTICOS LOGRADOS (OUT-OF-FOLD COMPLETO)")
    print("*"*50)
    target_names = ['OK (Minotitario)', 'NOK (Mayoritario)']
    print("\nMatriz de Confusión:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=target_names))
    
    print(f"    - Umbral Optimizado: {best_threshold:.4f}")
    print(f"    - Parámetros Optimos Encontrados: {best_params}")
    
    print("\n[5] Retraining del modelo maestro completo y guardado...")
    master_model = xgb.XGBClassifier(**final_params)
    master_model.fit(X, y, verbose=False)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump({"model": master_model, "threshold": best_threshold}, MODEL_SAVE_PATH)
    
    print(f"    [!] Exportado en: {MODEL_SAVE_PATH}")
    print(f"Tiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == '__main__':
    main()
