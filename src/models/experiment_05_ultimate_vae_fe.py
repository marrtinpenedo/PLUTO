#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 05 - VAE + Discretization + CatBoost
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test separado antes de cualquier transformación.
  - OrdinalEncoder aplicado post-split para evitar fugas.
  - VAE y Binning ajustados únicamente con datos de entrenamiento de cada fold.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OrdinalEncoder
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'

# -------------------------------------------------------------
# VAE Architecture (Mantenida igual, es robusta)
# -------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
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
        return self.dec3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(x_recon, x, mu, logvar):
    # Pérdida: MSE (Reconstrucción) + KLD (Divergencia KL)
    MSE = nn.functional.mse_loss(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 0.1 * KLD

def train_vae_and_encode(X_train_scaled, X_val_scaled, latent_dim=12, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=X_train_scaled.shape[1], latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

    vae.train()
    for _ in range(epochs):
        for batch in loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch_data)
            loss = vae_loss_function(recon, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()

    vae.eval()
    def get_latents(data_np):
        tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            recon, mu, _ = vae(tensor)
            err = torch.mean((tensor - recon)**2, dim=1).cpu().numpy()
            lat = mu.cpu().numpy()
        return err, lat

    tr_err, tr_lat = get_latents(X_train_scaled)
    vl_err, vl_lat = get_latents(X_val_scaled)
    return (tr_err, tr_lat), (vl_err, vl_lat)

# -------------------------------------------------------------
# Pipeline Helpers
# -------------------------------------------------------------
def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target', 'ID', 'Variable 02'], errors='ignore')
    return X, y

def apply_fe_per_fold(X_train_raw, X_val_raw, y_train):
    X_tr = X_train_raw.copy()
    X_vl = X_val_raw.copy()
    
    num_cols = X_tr.select_dtypes(exclude=['object']).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=['object']).columns.tolist()

    # 1. Imputación y Encoding seguro
    X_tr[num_cols] = X_tr[num_cols].fillna(0)
    X_vl[num_cols] = X_vl[num_cols].fillna(0)
    
    current_cat_features = []
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = oe.fit_transform(X_tr[cat_cols].fillna('missing').astype(str))
        X_vl[cat_cols] = oe.transform(X_vl[cat_cols].fillna('missing').astype(str))
        current_cat_features.extend(cat_cols)

    # 2. Binning de variables de alta varianza (Solo sobre Train)
    variances = X_tr[num_cols].var().sort_values(ascending=False)
    high_var = variances.head(10).index.tolist()
    
    kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    X_tr_bin = kbd.fit_transform(X_tr[high_var])
    X_vl_bin = kbd.transform(X_vl[high_var])
    
    for i, col in enumerate(high_var):
        name = f"{col}_bin"
        X_tr[name] = X_tr_bin[:, i].astype(int)
        X_vl[name] = X_vl_bin[:, i].astype(int)
        current_cat_features.append(name)

    # 3. VAE Features
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr[num_cols])
    X_vl_sc = scaler.transform(X_vl[num_cols])
    
    (tr_err, tr_lat), (vl_err, vl_lat) = train_vae_and_encode(X_tr_sc, X_vl_sc)
    
    X_tr['vae_err'] = tr_err
    X_vl['vae_err'] = vl_err
    for i in range(tr_lat.shape[1]):
        X_tr[f'vae_l{i}'] = tr_lat[:, i]
        X_vl[f'vae_l{i}'] = vl_lat[:, i]

    return X_tr, X_vl, current_cat_features

def main():
    print("=" * 60)
    print(" EXPERIMENTO 05 (VAE + CATBOOST) — SIN LEAKAGE")
    print("=" * 60)
    start_time = time.time()

    X_df, y = load_raw_data()

    # HOLDOUT SPLIT
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))

    cb_params = {
        'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 
        'verbose': False, 'random_seed': RANDOM_STATE
    }

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr_fold, X_vl_fold, _ = apply_fe_per_fold(X_pool.iloc[tr_idx], X_pool.iloc[vl_idx], y_pool[tr_idx])
        
        model = cb.CatBoostClassifier(**cb_params)
        model.fit(X_tr_fold, y_pool[tr_idx], eval_set=(X_vl_fold, y_pool[vl_idx]), early_stopping_rounds=30)
        oof_proba[vl_idx] = model.predict_proba(X_vl_fold)[:, 1]
        print(f"    - Fold {fold + 1}/5 listo.")

    # Tuning e Holdout
    best_t = 0.5
    best_f1 = -1
    for t in np.linspace(0.1, 0.9, 100):
        p = (oof_proba >= t).astype(int)
        if recall_score(y_pool, p) < 0.85: continue
        score = f1_score(y_pool, p, average='weighted')
        if score > best_f1: best_f1, best_t = score, t

    print(f"\n[4] Threshold: {best_t:.4f}")

    # Evaluación Final
    X_p_fe, X_h_fe, _ = apply_fe_per_fold(X_pool, X_holdout, y_pool)
    
    final_model = cb.CatBoostClassifier(**cb_params)
    final_model.fit(X_p_fe, y_pool)
    
    h_proba = final_model.predict_proba(X_h_fe)[:, 1]
    h_preds = (h_proba >= best_t).astype(int)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, h_preds))
    print("\nReporte:\n", classification_report(y_holdout, h_preds))
    print(f"Tiempo: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()