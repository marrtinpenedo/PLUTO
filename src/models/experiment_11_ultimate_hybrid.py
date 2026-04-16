#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 11 - THE ULTIMATE HYBRID FRAMEWORK
----------------------------------------------------------------------------------
Fusión Absoluta del Conocimiento Antiguo y Nuevo:
1. VAE Anomaly Scoring: Red Neuronal Pytorch mapeando las dimensiones y sacando score.
2. Polinomiales y Binning sobre Features Maestras.
3. CleanLab MIT: Purgado de datos humanos basura sobre la data masivamente enriquecida.
4. Optuna Genetic Search sobre CatBoost y XGBoost.
5. Meta-Stacking Level 2 con LightGBM.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import warnings

from cleanlab.filter import find_label_issues
import optuna
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'

RANDOM_STATE = 42

# ==============================================================================
# [A] VAE Pytorch Architecture 
# ==============================================================================
class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(TabularVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim) # NO sigmoid we assume normalized standard data
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def get_vae_reconstruction_error(X_scaled):
    print("    [!] Entrenando Variational Autoencoder (VAE) para detectar Anomalías Estructurales...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = TabularVAE(input_dim=X_scaled.shape[1], latent_dim=12).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=256, shuffle=True)
    
    vae.train()
    for __ in range(25): # 25 Epochs quick train
        for batch in loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
    vae.eval()
    with torch.no_grad():
        recon_all, _, _ = vae(tensor_x.to(device))
        mse_errors = torch.mean((tensor_x.to(device) - recon_all)**2, dim=1).cpu().numpy()
        
    return mse_errors

# ==============================================================================
# PIPELINES MAESTROS
# ==============================================================================

def orchestrate_hybrid_fe():
    print("\n[1] FASE I: INGENIERÍA HÍBRIDA MULTIDIMENSIONAL (Carga + VAE + Polinomiales)...")
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
        
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target']).fillna(0)
    
    # Variables de alta varianza -> Polinomiales
    vars_var = X.var().sort_values(ascending=False).head(5).index
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X[vars_var])
    poly_df = pd.DataFrame(X_poly, columns=[f"poly_{i}" for i in range(X_poly.shape[1])], index=X.index)
    
    # Binning robusto a variables mediocres
    vars_med = X.var().sort_values(ascending=False)[6:15].index
    est = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
    X_bin = est.fit_transform(X[vars_med])
    bin_df = pd.DataFrame(X_bin, columns=[f"bin_{i}" for i in range(X_bin.shape[1])], index=X.index)
    
    X_concat = pd.concat([X, poly_df, bin_df], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_concat)
    
    # Extraer el Error VAE y adjuntarlo
    vae_errors = get_vae_reconstruction_error(X_scaled)
    X_concat['VAE_Reconstruction_Error'] = vae_errors
    print(f"    - Dataset Enriquecido con éxito. Dimensiones Finales: {X_concat.shape[1]}")
    
    return X_concat, y

def orchestrate_cleanlab(X, y):
    print("\n[2] FASE II: CONFIDENT LEARNING MIT (Identificando Basura Humana apoyado por VAE)...")
    X_arr = X.values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_probs = np.zeros((len(y), 2))
    
    # Usamos LightGBM + Robustez por VAE para purgar
    for train_idx, val_idx in skf.split(X_arr, y):
        model = lgb.LGBMClassifier(random_state=RANDOM_STATE, is_unbalance=True, verbosity=-1)
        model.fit(X_arr[train_idx], y[train_idx])
        cv_probs[val_idx] = model.predict_proba(X_arr[val_idx])
        
    ranked_issues = find_label_issues(
        labels=y,
        pred_probs=cv_probs,
        return_indices_ranked_by='self_confidence',
    )
    
    print(f"    - CleanLab Saneó a muerte: Detectados {len(ranked_issues)} errores catastróficos operativos.")
    X_clean = np.delete(X_arr, ranked_issues, axis=0)
    y_clean = np.delete(y, ranked_issues, axis=0)
    
    print(f"    - DATASET INMACULADO (Filas: {len(y_clean)}). Listo para la fase genética.")
    return X_clean, y_clean

def optimize_optuna_cv(X_train, y_train):
    print("\n[3] FASE III: BUSQUEDA GENÉTICA OPTUNA MÁXIMA...\n")
    # Buscamos hipers ultra refinados en Catboost
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'loss_function': 'Logloss',
            'verbose': 0,
            'random_state': RANDOM_STATE,
            'task_type': 'CPU'
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        f1_scores = []
        for tr_i, va_i in skf.split(X_train, y_train):
            cb = CatBoostClassifier(**params)
            cb.fit(X_train[tr_i], y_train[tr_i])
            f1_scores.append(f1_score(y_train[va_i], cb.predict(X_train[va_i]), average='weighted'))
            
        return np.mean(f1_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    print("    [!] Corriendo docenas de simulaciones paramétricas paralelas en CatBoost... (Espere)")
    study.optimize(objective, n_trials=15) # Reducido a 15 for resource safety but provides great value
    print(f"    - Mejor F1-W hallado por Optuna durante cruces genéticos: {study.best_value:.4f}")
    
    return study.best_params

def orchestrate_ultimate_stack(X, y_clean, cb_params):
    print("\n[4] FASE IV: CONSTRUYENDO META-STACK NIVEL 2...")
    X_train, X_test, y_tr, y_te = train_test_split(X, y_clean, test_size=0.15, stratify=y_clean, random_state=RANDOM_STATE)
    
    cb_params['loss_function'] = 'Logloss'
    cb_params['verbose'] = 0
    cb_params['random_state'] = RANDOM_STATE
    
    cb = CatBoostClassifier(**cb_params)
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, colsample_bytree=0.8, random_state=RANDOM_STATE)
    # Lvl 2 Logistic or LightGBM
    meta = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.02, random_state=RANDOM_STATE, verbosity=-1)
    
    stacker = StackingClassifier(
        estimators=[('cb', cb), ('xgb', xgb)],
        final_estimator=meta,
        cv=5,
        n_jobs=-1
    )
    
    stacker.fit(X_train, y_tr)
    
    print("\n" + "="*60)
    print(" EVALUACIÓN FINAL: THE ULTIMATE HYBRID")
    print("="*60)
    
    preds = stacker.predict(X_test)
    print("\nMatriz de Confusión:\n", confusion_matrix(y_te, preds))
    print("\nReporte de Clasificación:\n", classification_report(y_te, preds, target_names=['OK (Min)','NOK (May)']))
    print("="*60)

def main():
    start_time = time.time()
    print("="*60)
    print(" EXPERIMENTO 11 - THE ULTIMATE HYBRID FRAMEWORK")
    print(" (VAE + Polinomios + CleanLab MIT + Optuna Genetic + Stack)")
    print("="*60)
    
    X_hybrid, y = orchestrate_hybrid_fe()
    X_clean, y_clean = orchestrate_cleanlab(X_hybrid, y)
    best_cb = optimize_optuna_cv(X_clean, y_clean)
    orchestrate_ultimate_stack(X_clean, y_clean, best_cb)
    
    print(f"\nTiempo Total Monolito 11: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
