#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 11 - THE ULTIMATE HYBRID FRAMEWORK (VAE + CleanLab + Optuna + Stack)
----------------------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de CUALQUIER procesamiento.
  - VAE trained SÓLO sobre train pool.
  - PolynomialFeatures + KBinsDiscretizer fitted SÓLO sobre train pool.
  - CleanLab applied SÓLO sobre train pool post-FE.
  - Optuna CV SÓLO sobre train pool limpio.
  - Stacker entrenado en train pool limpio.
  - Métricas finales sobre holdout test (transformado con pipeline fit en train).
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'

RANDOM_STATE = 42


# ==============================================================================
# [A] VAE Architecture
# ==============================================================================
class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(TabularVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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


def train_vae_and_get_errors(X_train_scaled, X_val_scaled):
    """Train VAE on X_train_scaled ONLY, compute recon error for both splits."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = TabularVAE(input_dim=X_train_scaled.shape[1], latent_dim=12).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)

    tensor_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_train), batch_size=256, shuffle=True)

    vae.train()
    for _ in range(25):
        for batch in loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

    vae.eval()
    errors = {}
    for name, data_np in [('train', X_train_scaled), ('val', X_val_scaled)]:
        tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            recon_all, _, _ = vae(tensor)
            mse_errors = torch.mean((tensor - recon_all) ** 2, dim=1).cpu().numpy()
        errors[name] = mse_errors

    return errors['train'], errors['val']


# ==============================================================================
# Feature Engineering Pipeline (fit on train, transform val)
# ==============================================================================
class FEPipeline:
    """Encapsulates all feature engineering to ensure train-only fitting."""

    def __init__(self):
        self.poly = None
        self.binner = None
        self.scaler = None
        self.top_var_cols = None
        self.med_var_cols = None

    def fit_transform(self, X_train, num_cols_hint=None):
        """Fit all transformers on X_train and return transformed X_train."""
        X = X_train.copy()

        # Polynomial on top-5 variance columns
        all_num = X.select_dtypes(include=[np.number]).columns.tolist()
        vars_sorted = X[all_num].var().sort_values(ascending=False)
        self.top_var_cols = vars_sorted.head(5).index.tolist()

        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = self.poly.fit_transform(X[self.top_var_cols])
        poly_df = pd.DataFrame(X_poly, columns=[f"poly_{i}" for i in range(X_poly.shape[1])], index=X.index)

        # Binning on next 9 columns
        self.med_var_cols = vars_sorted[5:14].index.tolist()
        if len(self.med_var_cols) > 0:
            self.binner = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
            X_bin = self.binner.fit_transform(X[self.med_var_cols])
            bin_df = pd.DataFrame(X_bin, columns=[f"bin_{i}" for i in range(X_bin.shape[1])], index=X.index)
        else:
            bin_df = pd.DataFrame(index=X.index)

        X_concat = pd.concat([X, poly_df, bin_df], axis=1)

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_concat)

        return X_concat, X_scaled

    def transform(self, X_val):
        """Transform X_val using fitted transformers."""
        X = X_val.copy()

        X_poly = self.poly.transform(X[self.top_var_cols])
        poly_df = pd.DataFrame(X_poly, columns=[f"poly_{i}" for i in range(X_poly.shape[1])], index=X.index)

        if self.binner is not None:
            X_bin = self.binner.transform(X[self.med_var_cols])
            bin_df = pd.DataFrame(X_bin, columns=[f"bin_{i}" for i in range(X_bin.shape[1])], index=X.index)
        else:
            bin_df = pd.DataFrame(index=X.index)

        X_concat = pd.concat([X, poly_df, bin_df], axis=1)
        X_scaled = self.scaler.transform(X_concat)

        return X_concat, X_scaled


# ==============================================================================
# Main Pipeline
# ==============================================================================
def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target']).fillna(0)

    return X, y


def cleanlab_on_train(X_arr, y_arr):
    """CleanLab applied ONLY to training data."""
    print("    [!] CleanLab sobre train pool...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_probs = np.zeros((len(y_arr), 2))

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        cv_probs[val_idx] = model.predict_proba(X_arr[val_idx])

    ranked_issues = find_label_issues(
        labels=y_arr, pred_probs=cv_probs, return_indices_ranked_by='self_confidence',
    )
    print(f"    - Detectados {len(ranked_issues)} errores.")

    clean_mask = np.ones(len(y_arr), dtype=bool)
    clean_mask[ranked_issues] = False
    return X_arr[clean_mask], y_arr[clean_mask]


def main():
    start_time = time.time()
    print("=" * 60)
    print(" EXPERIMENTO 11 (ULTIMATE HYBRID) — SIN LEAKAGE")
    print(" (VAE + Polinomios + CleanLab MIT + Optuna + Stack)")
    print("=" * 60)

    print("\n[1] Cargando datos...")
    X, y = load_raw_data()

    # HOLDOUT SPLIT — BEFORE EVERYTHING
    print("[2] Separando holdout test (20%)...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool = X_pool.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    # Feature Engineering: fit on train pool ONLY
    print("\n[3] Feature Engineering (fit en train pool)...")
    fe = FEPipeline()
    X_pool_concat, X_pool_scaled = fe.fit_transform(X_pool)
    X_holdout_concat, X_holdout_scaled = fe.transform(X_holdout)

    # VAE on train pool ONLY
    print("[4] VAE sobre train pool...")
    pool_vae_err, holdout_vae_err = train_vae_and_get_errors(X_pool_scaled, X_holdout_scaled)
    X_pool_concat['VAE_Reconstruction_Error'] = pool_vae_err
    X_holdout_concat['VAE_Reconstruction_Error'] = holdout_vae_err
    print(f"    - Dimensiones finales: {X_pool_concat.shape[1]}")

    # CleanLab on train pool ONLY
    print("\n[5] CleanLab sobre train pool...")
    X_pool_arr = X_pool_concat.values
    X_clean, y_clean = cleanlab_on_train(X_pool_arr, y_pool)
    print(f"    - Train limpio: {len(y_clean)}")

    # Optuna on cleaned train
    print("\n[6] Optuna HPO sobre train limpio...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'loss_function': 'Logloss', 'verbose': 0,
            'random_state': RANDOM_STATE, 'task_type': 'CPU'
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        f1_scores = []
        for tr_i, va_i in skf.split(X_clean, y_clean):
            cb_m = CatBoostClassifier(**params)
            cb_m.fit(X_clean[tr_i], y_clean[tr_i])
            f1_scores.append(f1_score(y_clean[va_i], cb_m.predict(X_clean[va_i]), average='weighted'))
        return np.mean(f1_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    print(f"    - Mejor F1-W Optuna: {study.best_value:.4f}")
    cb_params = study.best_params

    # Meta-Stack trained on cleaned train, evaluated on holdout
    print("\n[7] Meta-Stack sobre train limpio → evaluación en holdout...")
    cb_params['loss_function'] = 'Logloss'
    cb_params['verbose'] = 0
    cb_params['random_state'] = RANDOM_STATE

    cb_model = CatBoostClassifier(**cb_params)
    xgb_model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                              colsample_bytree=0.8, random_state=RANDOM_STATE)
    meta = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.02,
                              random_state=RANDOM_STATE, verbosity=-1)

    stacker = StackingClassifier(
        estimators=[('cb', cb_model), ('xgb', xgb_model)],
        final_estimator=meta, cv=5, n_jobs=-1
    )

    stacker.fit(X_clean, y_clean)

    # HOLDOUT evaluation
    print("\n" + "=" * 60)
    print(" EVALUACIÓN FINAL: THE ULTIMATE HYBRID (HOLDOUT)")
    print("=" * 60)

    X_holdout_arr = X_holdout_concat.values
    preds = stacker.predict(X_holdout_arr)
    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
