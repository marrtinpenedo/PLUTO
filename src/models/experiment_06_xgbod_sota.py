#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 06 - State-of-the-Art (XGBOD)
--------------------------------------------------------------------
Extreme Gradient Boosting Outlier Detection (Zhao et al.)
Técnica académica superior para detección de anomalías supervisada / semi-supervisada.
Compila de forma automática decenas de puntuaciones de anomalía de algoritmos 
clásicos (kNN, LOF, LSCP) y los fusiona de manera end-to-end dentro de 
un espacio XGBoost.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from pyod.models.xgbod import XGBOD
from sklearn.preprocessing import StandardScaler
import joblib
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
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        
        # Constraint absoluto de fábrica
        if r_nok < 0.85:
            continue
            
        score = f1_w + (r_ok * 0.3)
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target

def load_and_scale():
    print("[1] Cargando Base de Datos Cruda...")
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # XGBOD busca "Outliers" explícitamente etiquetados como 1 y normalidad como 0.
    # Dado que "OK" (25%) es la anomalía matemática (minoritario):
    # Asignamos OK = 1 (Outlier), NOK = 0 (Inlier) temporalmente para la estructura teórica interna de PyOD.
    # ¡Al medir métricas de negocio invertiremos la matriz para mantener tu estándar visual OK=0, NOK=1!
    df['target'] = df[target_col].map({'NOK': 0, 'OK': 1})
    
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target'])
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    
    return X_scaled, y

def main():
    print("="*60)
    print(" EXPERIMENTO 06 - XGBOD (SOTA OUTLIER DETECTION)")
    print("="*60)
    start_time = time.time()
    
    X, y_pyod = load_and_scale()
    
    print("\n[2] Inicializando Meta-Framework XGBOD...")
    # XGBOD generará scores desde kNN (varios K), LOF, etc., como meta-features.
    clf = XGBOD(random_state=RANDOM_STATE, n_jobs=-1, estimator_params={'scale_pos_weight': 0.8})
    
    print("\n[3] Entrenamiento y 5-Fold Evaluation de XGBOD...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pyod))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_pyod)):
        X_train, y_train = X[train_idx], y_pyod[train_idx]
        X_val, y_val = X[val_idx], y_pyod[val_idx]
        
        # Train XGBOD on this fold
        model = XGBOD(random_state=RANDOM_STATE, n_jobs=-1, estimator_params={'scale_pos_weight': 0.8})
        model.fit(X_train, y_train)
        
        preds_array = model.predict_proba(X_val)
        if preds_array.ndim == 1:
            oof_proba[val_idx] = preds_array
        else:
            oof_proba[val_idx] = preds_array[:, 1]
        print(f"    - Fold {fold+1} completado.")
        
    print("\n[4] Re-mapeando predicciones de vuelta al Estándar Industrial...")
    # En el código previo: OK=0, NOK=1.
    # En XGBOD pusimos: OK=1, NOK=0.
    # Así que la probabilidad de OK(1) es `oof_proba`. 
    # Por tanto, la probabilidad de NOK (modelo de negocio) es: `1 - oof_proba`
    business_nok_proba = 1.0 - oof_proba
    y_business = np.where(y_pyod == 1, 0, 1) # Revierte a OK=0, NOK=1
    
    best_threshold, _ = maximize_threshold(y_business, business_nok_proba)
    if best_threshold == 0.5 and maximize_threshold(y_business, business_nok_proba)[1] == -1.0:
        best_threshold = 0.5 
        
    final_preds = (business_nok_proba >= best_threshold).astype(int)
    
    print("\n" + "*"*50)
    print(" RESULTADOS INVESTIGACIÓN SOTA (XGBOD)")
    print("*"*50)
    
    print(f"    - Umbral Optimizado (Para NOK): {best_threshold:.4f}")
    print("\nMatriz de Confusión:\n", confusion_matrix(y_business, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y_business, final_preds, target_names=['OK (Min)','NOK (May)']))
    
    print(f"Tiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
