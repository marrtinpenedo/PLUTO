#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 03 - Red Neuronal Residual con Focal Loss (PyTorch)
--------------------------------------------------------------------
Cambios:
  - Holdout test (20%) separado ANTES de cualquier procesamiento.
  - OrdinalEncoder e Imputación post-split para evitar fugas.
  - StandardScaler DENTRO del CV loop (fit en fold-train, transform fold-val).
  - Feature Engineering (num_sum, etc.) aplicado post-encoding.
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
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target', 'ID', 'Variable 02'], errors='ignore')
    return X, y


def preprocess_step(X_train_df, X_val_df, cat_cols, num_cols):
    X_tr = X_train_df.copy()
    X_vl = X_val_df.copy()
    
    # Imputación y Codificación Categórica
    X_tr[num_cols] = X_tr[num_cols].fillna(0)
    X_vl[num_cols] = X_vl[num_cols].fillna(0)
    
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = oe.fit_transform(X_tr[cat_cols].astype(str).fillna('missing'))
        X_vl[cat_cols] = oe.transform(X_vl[cat_cols].astype(str).fillna('missing'))
        
    # Feature Engineering (basado en numéricas reales)
    for df in [X_tr, X_vl]:
        df['num_sum'] = df[num_cols].sum(axis=1)
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std'] = df[num_cols].std(axis=1)
        
    return X_tr, X_vl


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
    X_df, y = load_raw_data()
    
    cat_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_df.select_dtypes(exclude=['object']).columns.tolist()

    # 2. HOLDOUT SPLIT
    print("[2] Separando holdout test (20%)...")
    X_pool_raw, X_holdout_raw, y_pool, y_holdout = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_pool_raw = X_pool_raw.reset_index(drop=True)
    X_holdout_raw = X_holdout_raw.reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 60
    BATCH_SIZE = 128

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_proba = np.zeros(len(y_pool))

    print("[3] Iniciando K-Fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool_raw, y_pool)):
        X_tr_raw, y_tr = X_pool_raw.iloc[train_idx], y_pool[train_idx]
        X_vl_raw, y_vl = X_pool_raw.iloc[val_idx], y_pool[val_idx]

        # Preprocesamiento y FE por fold
        X_tr_fe, X_vl_fe = preprocess_step(X_tr_raw, X_vl_raw, cat_cols, num_cols)

        # Escalamiento por fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_tr_fe)
        X_val = scaler.transform(X_vl_fe)

        input_dim = X_train.shape[1]
        train_loader = DataLoader(TabularDataset(X_train, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TabularDataset(X_val, y_vl), batch_size=BATCH_SIZE, shuffle=False)

        model = TabularResNet(input_dim).to(device)
        criterion = FocalLoss(alpha=0.6, gamma=2.5)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        best_model_weights = None

        for epoch in range(EPOCHS):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    v_loss += criterion(model(batch_X), batch_y).item() * batch_X.size(0)
            v_loss /= len(val_idx)
            scheduler.step(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_weights = model.state_dict().copy()

        model.load_state_dict(best_model_weights)
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                fold_probs.extend(torch.sigmoid(model(batch_X.to(device))).cpu().numpy())

        oof_preds_proba[val_idx] = np.array(fold_probs).flatten()
        print(f"    - Fold {fold + 1}/5 completado.")

    # 4. Threshold tuning on OOF
    print("\n[4] Threshold tuning...")
    best_t = 0.5
    best_target_score = -1.0
    for t in np.linspace(0.1, 0.9, 150):
        preds_t = (oof_preds_proba >= t).astype(int)
        if recall_score(y_pool, preds_t, pos_label=1) < 0.85: continue
        score = f1_score(y_pool, preds_t, average='weighted') + (f1_score(y_pool, preds_t, pos_label=0) * 0.5)
        if score > best_target_score:
            best_target_score, best_t = score, t

    # 5. Evaluación Final
    print(f"\n[5] Evaluación Final sobre HOLDOUT (Threshold: {best_t:.4f})")
    X_pool_fe, X_holdout_fe = preprocess_step(X_pool_raw, X_holdout_raw, cat_cols, num_cols)
    
    final_scaler = StandardScaler()
    X_pool_scaled = final_scaler.fit_transform(X_pool_fe)
    X_holdout_scaled = final_scaler.transform(X_holdout_fe)

    final_model = TabularResNet(X_pool_scaled.shape[1]).to(device)
    train_loader = DataLoader(TabularDataset(X_pool_scaled, y_pool), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.6, gamma=2.5)

    final_model.train()
    for _ in range(EPOCHS):
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            criterion(final_model(b_X), b_y).backward()
            optimizer.step()

    final_model.eval()
    holdout_loader = DataLoader(TabularDataset(X_holdout_scaled, y_holdout), batch_size=BATCH_SIZE, shuffle=False)
    holdout_probs = []
    with torch.no_grad():
        for b_X, _ in holdout_loader:
            holdout_probs.extend(torch.sigmoid(final_model(b_X.to(device))).cpu().numpy())

    holdout_preds = (np.array(holdout_probs).flatten() >= best_t).astype(int)
    print("\nMatriz de Confusión:\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte:\n", classification_report(y_holdout, holdout_preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)

if __name__ == '__main__':
    main()