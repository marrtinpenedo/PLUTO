#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 04 - "Divide & Conquer" (Mixture of Local Experts)
--------------------------------------------------------------------
Esta técnica avanzada no utiliza un modelo universal. En lugar de ello:
1. Aplica un modelo probabilístico (Gaussian Mixture Model) para fragmentar la base
   de datos en 'sub-fábricas' o dominios operacionales.
2. Entrena un modelo Maestro Independiente (Local Expert) exclusivo para los datos 
   de cada clúster.
3. Al predecir, el sistema evalúa a qué clúster pertenece el registro y delega
   su evaluación a su experto específico.
"""

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'

def load_and_preprocess_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # Mapping: NOK=1, OK=0
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])
    
    y = df['target'].values
    X = df.drop(columns=['target'])
    
    # Label encode explicitly 
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X['num_sum'] = X[num_cols].sum(axis=1)
        X['num_mean'] = X[num_cols].mean(axis=1)
        X['num_std'] = X[num_cols].std(axis=1)
        
    return X, y, num_cols

def optimize_threshold_for_focal_business(y_val, preds_proba):
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (preds_proba >= t).astype(int)
        recall_nok = recall_score(y_val, preds, pos_label=1)
        recall_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        
        if recall_nok < 0.85:
            continue
            
        custom_target = f1_w + (recall_ok * 0.20)
        if custom_target > best_score:
            best_score = custom_target
            best_t = t
    return best_t, best_score

def cluster_optuna_objective(trial, X_c, y_c):
    # Si hay muy pocos datos, simplificamos la validación
    folds = 3 if len(y_c) > 300 else 2
    try:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 1.5),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        scores = []
        for train_idx, val_idx in skf.split(X_c, y_c):
            X_train, y_train = X_c.iloc[train_idx], y_c[train_idx]
            X_val, y_val = X_c.iloc[val_idx], y_c[val_idx]
            
            # Si un fold se queda con 1 sola clase, saltamos.
            if len(np.unique(y_train)) < 2:
                continue
                
            model = xgb.XGBClassifier(**param, n_estimators=200, early_stopping_rounds=20)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds_proba = model.predict_proba(X_val)[:, 1]
            
            # Si por penalizar fallamos, el threshold fallback es 0.5
            _, best_val_score = optimize_threshold_for_focal_business(y_val, preds_proba)
            if best_val_score == -1.0:
                best_val_score = f1_score(y_val, (preds_proba>0.5).astype(int), average='weighted')
            scores.append(best_val_score)
            
        return np.mean(scores) if scores else 0.0
    except Exception as e:
        return 0.0

def main():
    print("="*60)
    print(" PASO 3 [VANGUARDIA] - MIXTURE OF EXPERTS (DIVIDE & CONQUER)")
    print("="*60)
    
    start_time = time.time()
    
    print("[1] Cargando datos y Entrenando Segmentación GMM...")
    X, y, num_cols = load_and_preprocess_data()
    
    # 1. Aplicamos GMM (Gaussian Mixture) a las features escaladas
    # Usamos 3 dominios para no diluir demasiado la clase minoritaria
    N_CLUSTERS = 3
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[num_cols].fillna(0))
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='tied', random_state=RANDOM_STATE)
    
    cluster_labels = gmm.fit_predict(X_scaled)
    X['Cluster'] = cluster_labels
    
    for c in range(N_CLUSTERS):
        sub_y = y[cluster_labels == c]
        print(f"    - Sub-fábrica (Clúster) {c}: {len(sub_y)} registros (NOK: {np.sum(sub_y==1)}, OK: {np.sum(sub_y==0)})")
        
    print("\n[2] Entrenando Expertos Locales vía Optuna...")
    
    expert_models = {}
    best_thresholds = {}
    
    # Matriz global out-of-fold para reportar la métrica exacta
    oof_preds_proba_global = np.zeros(len(y))
    oof_y_global = np.zeros(len(y))
    
    for c in range(N_CLUSTERS):
        print(f"    -> Iniciando Experto del Clúster {c}...")
        
        idx_c = np.where(cluster_labels == c)[0]
        X_c = X.iloc[idx_c].drop(columns=['Cluster']).reset_index(drop=True)
        y_c = y[idx_c]
        
        # Si un clúster captura el 100% de NOK o OK y no hay de la otra, el experto predice cte.
        if len(np.unique(y_c)) < 2:
            print(f"       [!] Clúster puro. Experto devolverá clase: {y_c[0]}")
            oof_preds_proba_global[idx_c] = y_c[0]
            continue
            
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: cluster_optuna_objective(trial, X_c, y_c), n_trials=10, n_jobs=1)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'n_estimators': 400
        })
        
        # K-Fold a nivel de Experto para no falsear validación
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_expert_proba = np.zeros(len(y_c))
        
        for train_idx, val_idx in skf.split(X_c, y_c):
            X_train, y_train = X_c.iloc[train_idx], y_c[train_idx]
            X_val, y_val = X_c.iloc[val_idx], y_c[val_idx]
            
            # Chequeo seguridad
            if len(np.unique(y_train)) < 2:
                model = None
                oof_expert_proba[val_idx] = np.mean(y_train)
            else:
                model = xgb.XGBClassifier(**best_params, early_stopping_rounds=20)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                oof_expert_proba[val_idx] = model.predict_proba(X_val)[:, 1]
                
        oof_preds_proba_global[idx_c] = oof_expert_proba
        
        # Train final master expert for this cluster
        master_expert = xgb.XGBClassifier(**best_params)
        master_expert.fit(X_c, y_c, verbose=False)
        expert_models[c] = master_expert

    print("\n[3] Fusión de Multi-Expertos y Meta-Resolución...")
    
    best_t_global = 0.5
    best_score_global = -1
    
    for t in np.linspace(0.1, 0.9, 150):
        preds_t = (oof_preds_proba_global >= t).astype(int)
        
        recall_nok = recall_score(y, preds_t, pos_label=1)
        f1_ok = f1_score(y, preds_t, pos_label=0)
        f1_w = f1_score(y, preds_t, average='weighted')
        
        if recall_nok < 0.85:
            continue
            
        target_score = f1_w + (f1_ok * 0.5)
        if target_score > best_score_global:
            best_score_global = target_score
            best_t_global = t
            
    final_preds = (oof_preds_proba_global >= best_t_global).astype(int)

    print("\n" + "*"*50)
    print(" RESULTADOS LOGRADOS POR ENSAMBLADOR DIVIDE & CONQUER")
    print("*"*50)
    target_names = ['OK (Minotitario)', 'NOK (Mayoritario)']
    print("\nMatriz de Confusión Global:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=target_names))
    
    print("\n[4] Guardando Meta-Modelo GMM + Híbridos...")
    model_export = {
        'gmm': gmm,
        'scaler': scaler,
        'experts': expert_models,
        'threshold': best_t_global,
        'num_cols': num_cols
    }
    joblib.dump(model_export, '../../models/divide_and_conquer_experts.pkl' if os.path.exists('../../models/') else 'models/divide_and_conquer_experts.pkl')
    
    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == '__main__':
    main()
