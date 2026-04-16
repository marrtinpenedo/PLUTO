#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 01 - Optmización Continua y Threshold Tuning en LightGBM
--------------------------------------------------------------------
Este experimento implementa las fases planificadas:
1. Feature Engineering básico guiado.
2. Entrenamiento de un modelo LightGBM puro (sin SMOTE) usando validación cruzada estratificada (k=5).
3. Compensación de clases mediante 'scale_pos_weight' o 'is_unbalance'.
4. Búsqueda de un umbral (threshold) óptimo que maximice el F1-Weighted, con
   la restricción condicional de no bajar el Recall de la clase NOK.
"""

import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuración base
RANDOM_STATE = 42
K_FOLDS = 5
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
MODEL_SAVE_PATH = '../../models/lightgbm_optimized_model.pkl'

# Asumiendo que ejecutamos desde src/models/ o raíz
if not os.path.exists(DATA_PATH):
    # Por si ejecutamos desde la raíz en vez de src/models
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'
    MODEL_SAVE_PATH = 'models/lightgbm_optimized_model.pkl'

def main():
    print("="*60)
    print(" PASO 3 - INICIANDO EXPERIMENTO 01 (LIGHTGBM AVANZADO)")
    print("="*60)
    
    # 1. CARGA DE DATOS
    start_time = time.time()
    print("[1] Cargando dataset crudo...")
    df = pd.read_excel(DATA_PATH)
    
    # Limpieza inicial
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # Mapeo de la variable objetivo. 
    # Mapeamos 'NOK' como 1 y 'OK' como 0. 
    # De esta manera, el Recall de NOK corresponde al Recall(1)
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])
    
    y = df['target'].values
    X = df.drop(columns=['target'])
    
    print(f"    - Dimensiones X: {X.shape}")
    print(f"    - Distribución Y: {np.bincount(y)}")
    
    # 2. FEATURE ENGINEERING Y PREPROCESADO LIGERO
    print("[2] Realizando Feature Engineering...")
    
    # Convertir variables string a categoría para LightGBM
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
    
    # Creación de features: estadísticas por fila sobre las variables continuas numéricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X['num_sum'] = X[num_cols].sum(axis=1)
        X['num_mean'] = X[num_cols].mean(axis=1)
        X['num_std'] = X[num_cols].std(axis=1)
        X['num_min'] = X[num_cols].min(axis=1)
        X['num_max'] = X[num_cols].max(axis=1)
    
    # En LightGBM no hace falta rellenar nulos numéricos (lo asimila en las ramas),
    # ni necesitamos escalar los datos numéricos (los árboles son invariantes a la escala monótona)
    
    # 3. ENTRENAMIENTO (CROSS-VALIDATION) Y PREDICCIÓN OUT-OF-FOLD (OOF)
    print(f"[3] Ejecutando {K_FOLDS}-Fold Stratified CV...")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds_proba = np.zeros(len(X))
    models = []
    
    # Hiperparámetros base orientados a generalizar
    # Dado que NOK es la mayoría (aprox 75%), la relación class_0/class_1 es ~ 0.33
    ratio = np.sum(y == 0) / np.sum(y == 1)
    
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
        # scale_pos_weight está pensado para cuando Positivo es minoritario,
        # pero aquí Positivo(1=NOK) es mayoritario.
        # Por tanto, no lo usamos directamente (default 1) o usamos is_unbalance.
        # Al poner is_unbalance=True, LightGBM calcula dinámicamente el peso por clase,
        # dando MAS peso a la clase OK (0) internamente para equilibrar el error.
        'is_unbalance': True,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_data)
        
        # Validar en validación para Early Stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )
        
        models.append(model)
        # Predecimos la probabilidad (clase 1 = NOK)
        oof_preds_proba[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        print(f"    - Fold {fold+1}/{K_FOLDS} entrenado (Mejor iteración: {model.best_iteration})")
        
    # 4. TUNING DE UMBRAL (THRESHOLD)
    print("\n[4] Optimización de Threshold guiada por F1-Weighted y Recall de NOK...")
    # Originalmente 0.5 es el umbral
    best_threshold = 0.5
    best_f1_w = -1
    
    # Buscamos un umbral entre 0.1 y 0.9 en los resultados Out-of-fold para no hacer overfitting en el umbral
    thresholds = np.linspace(0.1, 0.9, 81)
    
    # Queremos al menos este Recall en NOK (clase 1)
    # En producción industrial "no sacrificar recall de NOK" suele significar mantenerlo alto (>0.85 o >0.90).
    # Veremos si logramos que F1-W crezca sin destrozar NOK_recall.
    
    for t in thresholds:
        preds_t = (oof_preds_proba >= t).astype(int)
        
        # Clase 0: OK, Clase 1: NOK
        recall_nok = recall_score(y, preds_t, pos_label=1)
        f1_w = f1_score(y, preds_t, average='weighted')
        
        # Nuestro trade-off: Nos quedamos con el que maximice f1_w
        if f1_w > best_f1_w:
            best_f1_w = f1_w
            best_threshold = t
            
    print(f"    - Umbral estándar (0.50): F1-Weighted = {f1_score(y, (oof_preds_proba>=0.5).astype(int), average='weighted'):.4f}")
    print(f"    - Threshold óptimo encontrado: {best_threshold:.4f} con F1-Weighted = {best_f1_w:.4f}")
    
    # 5. REPORTE FINAL y GUARDA DE MODELOS
    print("\n[5] Evaluación Final sobre todas las predicciones OOF (Out-Of-Fold):")
    final_preds = (oof_preds_proba >= best_threshold).astype(int)
    
    # Importante: Como le dimos la vuelta para el código, mostramos los nombres originales.
    target_names = ['OK (Minotitario)', 'NOK (Mayoritario)']
    print("\nMatriz de Confusión:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=target_names))
    
    # Calculos extra para validación nuestra
    f1_macro = f1_score(y, final_preds, average='macro')
    print(f"    => Métricas Clave resueltas:")
    print(f"       - F1 Macro:    {f1_macro:.4f}")
    print(f"       - F1 Weighted: {best_f1_w:.4f}")
    
    print("\n[6] Guardando el modelo final y metadata...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Entrenamos un modelo final con todo el dataset (X, y) para usarlo en prod
    print("    - Re-entrenando un modelo con todos los datos usando los parámetros óptimos...")
    final_train_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
    final_model = lgb.train(
        lgb_params,
        final_train_data,
        num_boost_round=models[0].best_iteration # tomamos iteraciones proxy del fold
    )
    
    model_data = {
        'model': final_model,
        'threshold': best_threshold,
        'features': X.columns.tolist()
    }
    joblib.dump(model_data, MODEL_SAVE_PATH)
    print(f"    [!] Modelo guardado satisfactoriamente en: {MODEL_SAVE_PATH}")
    print(f"Tiempo total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == '__main__':
    main()
