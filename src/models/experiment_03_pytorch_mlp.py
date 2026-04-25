#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 03 - Red Neuronal Residual con Focal Loss (PyTorch)
--------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage.
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - StandardScaler DENTRO del CV loop (fit en fold-train, transform fold-val).
  - Threshold tuning sólo sobre OOF del train pool.
  - Métricas finales reportadas ÚNICAMENTE sobre holdout test.
  - set_seed incluye cudnn.deterministic.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.4):
        super(TabularResNet, self).__init__()
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        self.res1_lin = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn = nn.BatchNorm1d(hidden_dim)
        self.res1_act = nn.Mish()
        self.res1_drop = nn.Dropout(dropout_rate)

        self.res2_lin = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn = nn.BatchNorm1d(hidden_dim)
        self.res2_act = nn.Mish()
        self.res2_drop = nn.Dropout(dropout_rate)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.entry(x)
        x_res = self.res1_drop(self.res1_act(self.res1_bn(self.res1_lin(x))))
        x = x + x_res
        x_res = self.res2_drop(self.res2_act(self.res2_bn(self.res2_lin(x))))
        x = x + x_res
        return self.head(x)


def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])

    y = df['target'].values
    X = df.drop(columns=['target'])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X['num_sum'] = X[num_cols].sum(axis=1)
        X['num_mean'] = X[num_cols].mean(axis=1)
        X['num_std'] = X[num_cols].std(axis=1)

    X = X.fillna(0)
    return X.values, y


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("=" * 60)
    print(" EXPERIMENTO 03 (PYTORCH RESNET + FOCAL) — SIN LEAKAGE")
    print("=" * 60)

    start_time = time.time()
    set_seed(RANDOM_STATE)

    print("[1] Cargando datos...")
    X, y = load_raw_data()

    # HOLDOUT SPLIT before scaling
    print("[2] Separando holdout test (20%)...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    - Computando en: {device}")

    EPOCHS = 60
    BATCH_SIZE = 128

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_proba = np.zeros(len(y_pool))

    print("[3] Iniciando K-Fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_train_raw, y_train = X_pool[train_idx], y_pool[train_idx]
        X_val_raw, y_val = X_pool[val_idx], y_pool[val_idx]

        # FIX: scaler fitted on train fold ONLY
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        input_dim = X_train.shape[1]

        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = TabularResNet(input_dim, hidden_dim=256, dropout_rate=0.4).to(device)
        criterion = FocalLoss(alpha=0.6, gamma=2.5)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        best_model_weights = None

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict().copy()

        model.load_state_dict(best_model_weights)
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.extend(probs)

        oof_preds_proba[val_idx] = np.array(fold_preds).flatten()
        print(f"    - Fold {fold + 1}/5 listo. Mejor Val Loss: {best_val_loss:.4f}")

    # Threshold tuning on OOF (train pool only)
    print("\n[4] Threshold tuning sobre OOF del train pool...")
    best_t = 0.5
    best_target_score = -1.0
    for t in np.linspace(0.1, 0.9, 150):
        preds_t = (oof_preds_proba >= t).astype(int)
        recall_nok = recall_score(y_pool, preds_t, pos_label=1)
        if recall_nok < 0.85:
            continue
        f1_ok = f1_score(y_pool, preds_t, pos_label=0)
        f1_w = f1_score(y_pool, preds_t, average='weighted')
        target_score = f1_w + (f1_ok * 0.5)
        if target_score > best_target_score:
            best_target_score = target_score
            best_t = t

    print(f"    - Threshold: {best_t:.4f}")

    # HOLDOUT evaluation
    print("\n[5] Evaluación Final sobre HOLDOUT TEST...")
    # Retrain scaler on full pool, apply to holdout
    final_scaler = StandardScaler()
    X_pool_scaled = final_scaler.fit_transform(X_pool)
    X_holdout_scaled = final_scaler.transform(X_holdout)

    input_dim = X_pool_scaled.shape[1]
    # Retrain model on full pool
    final_model = TabularResNet(input_dim, hidden_dim=256, dropout_rate=0.4).to(device)
    criterion = FocalLoss(alpha=0.6, gamma=2.5)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-4)

    train_ds = TabularDataset(X_pool_scaled, y_pool)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in train_ld:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    final_model.eval()
    holdout_ds = TabularDataset(X_holdout_scaled, y_holdout)
    holdout_ld = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)
    holdout_probs = []
    with torch.no_grad():
        for batch_X, _ in holdout_ld:
            batch_X = batch_X.to(device)
            probs = torch.sigmoid(final_model(batch_X)).cpu().numpy()
            holdout_probs.extend(probs)

    holdout_proba = np.array(holdout_probs).flatten()
    holdout_preds = (holdout_proba >= best_t).astype(int)

    target_names = ['OK (Minoritario)', 'NOK (Mayoritario)']
    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=target_names))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
