#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 07 - THE NICHE FRONTIER (DART + RUSBoost + UMAP)
--------------------------------------------------------------------
Pipeline ultra-masivo y destructivo a nivel de variables:
1. UMAP (Uniform Manifold Approximation): Inyecta topología espacial.
2. Polynomial Interactions: Cruza variables Top por varianza.
3. Drop de Variables: Purga masiva de variables ruido (Feature Selection).
4. DART (Dropout Additive Regression Trees): Algoritmo de mutación en árboles (nicho para sobreajuste).
5. RUSBoost (Random Under-Sampling Boosting): Algoritmo específico de academia para anomalías.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from imblearn.ensemble import RUSBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import umap
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
        
        # Constraint absoluto de fábrica (podemos aflojar a 0.84 si hay mucha ganancia, pero por defecto 0.85)
        if r_nok < 0.85:
            continue
            
        score = f1_w + (r_ok * 0.35)
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target

def load_and_niche_fe():
    print("[1] Cargando Datos...")
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # 0 = OK, 1 = NOK
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target'])
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Rellenar NaNs para operaciones matemáticas
    X = X.fillna(0)
    
    # --- 1. Polynomial Splitting (Creación de Nuevas Variables) ---
    print("    - Creando interacciones polinómicas locas de Grado 2 sobre Top Variables...")
    variances = X[num_cols].var().sort_values(ascending=False)
    top_10_vars = variances.head(10).index.tolist()
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X[top_10_vars])
    poly_cols = [f"poly_{i}" for i in range(X_poly.shape[1])]
    df_poly = pd.DataFrame(X_poly, columns=poly_cols, index=X.index)
    X = pd.concat([X, df_poly], axis=1)
    
    # --- 2. UMAP Topological Map (Proyección a Espacio Latente) ---
    print("    - Proyectando UMAP (Uniform Manifold Approximation) para rescatar la topología global...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    X_umap = reducer.fit_transform(X_scaled)
    X['umap_x'] = X_umap[:, 0]
    X['umap_y'] = X_umap[:, 1]
    
    print(f"    [!] Base de datos expandida a {X.shape[1]} columnas.")
    return X, y

def ruthless_feature_selection(X, y):
    print("\n[2] Feature Selection Despiadado (Eliminando variables ruido)...")
    # Entrenamos un estimador rápido pero profundo (LGBM)
    model = lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, is_unbalance=True, verbose=-1)
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    # Cortamos todo lo que tenga < 10 de importancia (puro ruido en el árbol)
    valid_indices = np.where(importances >= 5)[0]
    cols_to_keep = X.columns[valid_indices]
    
    print(f"    - Variables originales evaluadas: {X.shape[1]}")
    print(f"    - Cortadas (Basura): {X.shape[1] - len(cols_to_keep)}")
    print(f"    - Conservadas: {len(cols_to_keep)}")
    
    return X[cols_to_keep]

def get_niche_ensemble():
    # 1. DART (XGBoost con Dropouts para prevenir overfitting a la minoría)
    dart = xgb.XGBClassifier(
        booster='dart',
        rate_drop=0.1,
        skip_drop=0.5,
        max_depth=5,
        scale_pos_weight=0.8, # Focalizado inverso
        random_state=RANDOM_STATE,
        n_estimators=300,
        n_jobs=-1
    )
    
    # 2. RUSBoost (Técnica académica híbrida de Random UnderSampling Boosting)
    # Por defecto usa DecisionTrees pequeños.
    rus = RUSBoostClassifier(
        n_estimators=400,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    
    # 3. ExtraTrees Extremadamente restrictivo
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ('dart', dart),
            ('rus', rus),
            ('et', et)
        ],
        voting='soft',
        weights=[2.0, 1.5, 1.0] # DART y RUS tienen mayor peso deductivo
    )
    return ensemble

def main():
    print("="*60)
    print(" EXPERIMENTO 07 - THE ARCHITECTURE OF MADNESS")
    print(" (DART + UMAP + POLYNOMIALS + RUSBOOST + RUTHLESS RFE)")
    print("="*60)
    start_time = time.time()
    
    # 1. Feature Engineering
    X_expanded, y = load_and_niche_fe()
    
    # 2. Selección de Atrbutos
    X_clean = ruthless_feature_selection(X_expanded, y)
    
    # 3. 5-Fold Evaluation de la Locura
    print("\n[3] Evaluando Ensamble de Vanguardia (DART+RUS+ET) en 5-Folds...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
        X_train, y_train = X_clean.iloc[train_idx], y[train_idx]
        X_val, y_val = X_clean.iloc[val_idx], y[val_idx]
        
        ensemble = get_niche_ensemble()
        ensemble.fit(X_train, y_train)
        
        oof_proba[val_idx] = ensemble.predict_proba(X_val)[:, 1]
        print(f"    - Fold {fold+1}/5 listo.")
        
    best_threshold, _ = maximize_threshold(y, oof_proba)
    if best_threshold == 0.5 and maximize_threshold(y, oof_proba)[1] == -1.0:
        best_threshold = 0.55
        
    final_preds = (oof_proba >= best_threshold).astype(int)
    
    print("\n" + "*"*50)
    print(" RESULTADOS FINALES PIPELINE EXTREMO")
    print("*"*50)
    
    print(f"    - Umbral Optimizado: {best_threshold:.4f}")
    print("\nMatriz de Confusión:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=['OK (Min)','NOK (May)']))
    
    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
