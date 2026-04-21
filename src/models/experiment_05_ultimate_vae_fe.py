#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 05 - VAE + Discretization + CatBoost
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - KBinsDiscretizer fitted PER-FOLD (fit en fold-train, transform fold-val).
  - Variance-based feature selection computed PER-FOLD.
  - VAE trained PER-FOLD (fit en fold-train, encode fold-train + fold-val).
  - Optuna on train pool only.
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales sobre holdout test.
"""

import os
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'


# -------------------------------------------------------------
# VAE Architecture
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
    MSE = nn.functional.mse_loss(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 0.1 * KLD


def train_vae_and_encode(X_train_scaled, X_val_scaled, latent_dim=12, epochs=15):
    """Train VAE on X_train_scaled ONLY, then encode both train and val."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_scaled.shape[1]

    vae = VAE(input_dim=input_dim, hidden_dim=64, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

    vae.train()
    for epoch in range(epochs):
        for batch in loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch_data)
            loss = vae_loss_function(recon_batch, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()

    # Encode both splits using fold-specific VAE
    vae.eval()
    results = {}
    for name, data in [('train', X_train_scaled), ('val', X_val_scaled)]:
        tensor = torch.tensor(data, dtype=torch.float32).to(device)
        with torch.no_grad():
            recon, mu, logvar = vae(tensor)
            recon_error = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
            latent_mu = mu.cpu().numpy()
        results[name] = (recon_error, latent_mu)

    return results['train'], results['val']


# -------------------------------------------------------------
# Pipeline helpers
# -------------------------------------------------------------
def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target'])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes

    return X, y, cat_cols, num_cols


def apply_fe_per_fold(X_train, X_val, num_cols, cat_cols_base):
    """
    Per-fold FE: binning, VAE features — all fitted on train only.
    Returns augmented DataFrames and updated cat_cols list.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Variance-based selection on TRAIN only
    variances = X_train[num_cols].var().sort_values(ascending=False)
    high_var_cols = variances.head(20).index.tolist()

    # KBinsDiscretizer fitted on TRAIN only
    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    X_train_binned = est.fit_transform(X_train[high_var_cols].fillna(0))
    X_val_binned = est.transform(X_val[high_var_cols].fillna(0))

    new_cat_cols = list(cat_cols_base)
    for i, col in enumerate(high_var_cols):
        bname = f'{col}_binned'
        X_train[bname] = X_train_binned[:, i].astype(int)
        X_val[bname] = X_val_binned[:, i].astype(int)
        new_cat_cols.append(bname)

    # StandardScaler on num_cols fitted on TRAIN only
    scaler = StandardScaler()
    X_tr_num_scaled = scaler.fit_transform(X_train[num_cols].fillna(0))
    X_vl_num_scaled = scaler.transform(X_val[num_cols].fillna(0))

    # VAE trained on TRAIN only
    (tr_recon_err, tr_latent), (vl_recon_err, vl_latent) = train_vae_and_encode(
        X_tr_num_scaled, X_vl_num_scaled, latent_dim=12, epochs=15
    )

    X_train['vae_recon_error'] = tr_recon_err
    X_val['vae_recon_error'] = vl_recon_err
    for i in range(tr_latent.shape[1]):
        X_train[f'vae_latent_{i}'] = tr_latent[:, i]
        X_val[f'vae_latent_{i}'] = vl_latent[:, i]

    # Convert cat cols to int for CatBoost
    for c in new_cat_cols:
        X_train[c] = X_train[c].astype(int)
        X_val[c] = X_val[c].astype(int)

    return X_train, X_val, new_cat_cols


def maximize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_target = -1.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (preds_proba >= t).astype(int)
        r_nok = recall_score(y_val, preds, pos_label=1)
        if r_nok < 0.85:
            continue
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        score = f1_w + (r_ok * 0.25)
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target


def main():
    print("=" * 60)
    print(" EXPERIMENTO 05 (VAE + CATBOOST) — SIN LEAKAGE")
    print("=" * 60)
    start_time = time.time()

    print("[1] Cargando datos...")
    X, y, cat_cols, num_cols = load_raw_data()

    # HOLDOUT SPLIT
    print("[2] Separando holdout test (20%)...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool = X_pool.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    # 5-Fold CV on train pool (VAE + FE per fold)
    print("\n[3] 5-Fold CV con VAE + FE per-fold...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))

    # CatBoost params (fixed reasonable defaults — Optuna removed to keep manageable runtime
    # since VAE training per fold is already expensive)
    cb_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'iterations': 600,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_strength': 1.0,
        'verbose': False,
        'thread_count': -1,
        'random_seed': RANDOM_STATE
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr_raw, y_tr = X_pool.iloc[train_idx], y_pool[train_idx]
        X_vl_raw, y_vl = X_pool.iloc[val_idx], y_pool[val_idx]

        # All FE per-fold (bins, VAE, scaler — all fitted on train only)
        X_tr, X_vl, fold_cat_cols = apply_fe_per_fold(X_tr_raw, X_vl_raw, num_cols, cat_cols)

        model = cb.CatBoostClassifier(**cb_params, cat_features=fold_cat_cols)
        model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=40, verbose=False)
        oof_proba[val_idx] = model.predict_proba(X_vl)[:, 1]
        print(f"    - Fold {fold + 1}/5 listo.")

    # Threshold tuning on OOF (train pool only)
    print("\n[4] Threshold tuning sobre OOF del train pool...")
    best_threshold, _ = maximize_threshold(y_pool, oof_proba)
    print(f"    - Threshold: {best_threshold:.4f}")

    # HOLDOUT evaluation
    print("\n[5] Evaluación Final sobre HOLDOUT TEST...")
    X_pool_fe, X_holdout_fe, final_cat_cols = apply_fe_per_fold(X_pool, X_holdout, num_cols, cat_cols)

    final_model = cb.CatBoostClassifier(**cb_params, cat_features=final_cat_cols)
    final_model.fit(X_pool_fe, y_pool, verbose=False)

    holdout_proba = final_model.predict_proba(X_holdout_fe)[:, 1]
    holdout_preds = (holdout_proba >= best_threshold).astype(int)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"Tiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
