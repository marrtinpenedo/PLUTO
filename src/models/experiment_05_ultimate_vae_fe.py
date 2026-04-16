#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 05 - The Absolute Frontier
--------------------------------------------------------------------
1. Extreme Feature Engineering: Discretization (Binning) & Math Interactions.
2. Variational AutoEncoder (VAE): Extracción de Latent Space y Error de 
   Reconstrucción como meta-features.
3. Clasificación Maestra: CatBoost (resistente a overfitting con FE agresivo).
"""

import os
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from xgboost import XGBClassifier
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'

# -------------------------------------------------------------
# 1. Variational AutoEncoder en PyTorch (Unsupervised / EVT-like proxy)
# -------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Latent space metrics
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.relu(self.dec1(z))
        h = self.relu(self.dec2(h))
        return self.dec3(h) # Raw logits/values

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def loss_function(x_recon, x, mu, logvar):
    # Mean Squared Error for reconstruction 
    MSE = nn.functional.mse_loss(x_recon, x, reduction='sum')
    # KL Divergence to force normal gaussian distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 0.1 * KLD # beta-VAE weight

# -------------------------------------------------------------
# 2. Pipeline Principal
# -------------------------------------------------------------
def load_and_fe():
    print("[1] Cargando y Ejecutando Feature Engineering Extremo...")
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # 0 = OK (minoritario, detectar!), 1 = NOK (mayoritario)
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=['Variable de Salida', 'target'])
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 2.A Bining & Discretización (Transformación no lineal nativa)
    # Seleccionamos las TOP columnas por varianza para no explotar la ram
    variances = X[num_cols].var().sort_values(ascending=False)
    high_var_cols = variances.head(20).index.tolist()
    
    # Discretizamos en quartiles
    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    X_binned = est.fit_transform(X[high_var_cols].fillna(0))
    for i, col in enumerate(high_var_cols):
        X[f'{col}_binned'] = X_binned[:, i]
        cat_cols.append(f'{col}_binned') # Treated as categorical!
        
    # 2.B Codificación estándar para categorias
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    # Scale numerical part for VAE
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[num_cols].fillna(0))
    X_num_scaled = torch.tensor(X_num_scaled, dtype=torch.float32)
    
    return X, y, X_num_scaled, cat_cols

def run_vae(X_num_scaled):
    print("[2] Entrenando VAE (Variational AutoEncoder) para EVT/Anomalías...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae = VAE(input_dim=X_num_scaled.shape[1], hidden_dim=64, latent_dim=12).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    dataset = TensorDataset(X_num_scaled)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    vae.train()
    for epoch in range(15): # Entrenamiento rápido para atrapar manifold
        for batch in loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch_data)
            loss = loss_function(recon_batch, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()
            
    # Extracción de Latent Features y Error de Anomalía
    vae.eval()
    with torch.no_grad():
        X_tensor = X_num_scaled.to(device)
        recon, mu, logvar = vae(X_tensor)
        
        # Mean Squared Error per row (Anomaly Score)
        recon_error = torch.mean((X_tensor - recon)**2, dim=1).cpu().numpy()
        latent_mu = mu.cpu().numpy()
        
    return recon_error, latent_mu

def maximize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_target = -1.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (preds_proba >= t).astype(int)
        r_nok = recall_score(y_val, preds, pos_label=1)
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        
        # Constraint absoluto de fábrica
        if r_nok < 0.85:
            continue
            
        score = f1_w + (r_ok * 0.25) # Recompensamos agresivamente detectar OK
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target

def optuna_catboost(trial, X, y, cat_features):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    cb_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'iterations': 500,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 1.5),
        'cat_features': cat_features,
        'verbose': False,
        'thread_count': -1,
        'random_seed': RANDOM_STATE
    }
    
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # Handle cat features that might have unseen values in val
        # CatBoost handles this fine if they are passed correctly.
        model = cb.CatBoostClassifier(**cb_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
        
        preds_proba = model.predict_proba(X_val)[:, 1]
        _, best_score = maximize_threshold(y_val, preds_proba)
        
        # Fallback if constraint impossible
        if best_score == -1.0:
            best_score = f1_score(y_val, (preds_proba>0.5).astype(int), average='weighted')
            
        scores.append(best_score)
        
    return np.mean(scores)

def main():
    print("="*60)
    print(" EXPERIMENTO 05 - VAE, DISCRETIZATION Y CATBOOST")
    print("="*60)
    start_time = time.time()
    
    # 1. Feature Engineering
    X, y, X_num_scaled, cat_cols = load_and_fe()
    
    # 2. VAE
    recon_error, latent_mu = run_vae(X_num_scaled)
    X['vae_recon_error'] = recon_error
    for i in range(latent_mu.shape[1]):
        X[f'vae_latent_{i}'] = latent_mu[:, i]
        
    print(f"    - Dimensiones Expandidas: {X.shape}")
    
    # 3. Optuna on CatBoost
    print("\n[3] Buscando arquitectura de CatBoost + VAE features...")
    # Convert cat_cols datatypes properly for catboost
    for c in cat_cols:
        X[c] = X[c].astype(int)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optuna_catboost(trial, X, y, cat_cols), n_trials=15, n_jobs=1)
    
    print(f"    [OK] Mejor Trail Score: {study.best_value:.4f}")
    
    # 4. Evaluacion Definitiva (5-fold)
    print("\n[4] 5-Fold Evaluation de Vanguardia...")
    best_params = study.best_params
    best_params.update({
        'loss_function': 'Logloss',
        'iterations': 1000,
        'cat_features': cat_cols,
        'verbose': False,
        'random_seed': RANDOM_STATE,
        'thread_count': -1
    })
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = cb.CatBoostClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40, verbose=False)
        oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
        
    best_threshold, _ = maximize_threshold(y, oof_proba)
    if best_threshold == 0.5 and maximize_threshold(y, oof_proba)[1] == -1.0:
        # Prevent crash if boundaries fail
        best_threshold = 0.6 
        
    final_preds = (oof_proba >= best_threshold).astype(int)
    
    print("\n" + "*"*50)
    print(" RESULTADOS EXTREMOS (VAE + CATBOOST + BINNING)")
    print("*"*50)
    
    print(f"    - Umbral Optimizado: {best_threshold:.4f}")
    print("\nMatriz de Confusión:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=['OK (Min)','NOK (May)']))
    
    # Save Model
    master_model = cb.CatBoostClassifier(**best_params)
    master_model.fit(X, y, verbose=False)
    
    save_path = '../../models/ultimate_catboost_vae.pkl' if os.path.exists('../../models/') else 'models/ultimate_catboost_vae.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"model": master_model, "threshold": best_threshold}, save_path)
    
    print(f"Tiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
