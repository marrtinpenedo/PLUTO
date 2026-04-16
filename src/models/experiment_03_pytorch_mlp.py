#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
import joblib

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
MODEL_SAVE_PATH = '../../models/pytorch_dl_model.pth'

if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'
    MODEL_SAVE_PATH = 'models/pytorch_dl_model.pth'

# -------------------------------------------------------------
# 1. Pytorch Dataset & Focal Loss
# -------------------------------------------------------------
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
        # Usaremos BCEWithLogitsLoss internamente para mayor estabilidad numérica
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# -------------------------------------------------------------
# 2. Arquitectura Residual Dinámica (Deep Tabular ResNet)
# -------------------------------------------------------------
class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.4):
        super(TabularResNet, self).__init__()
        
        # Bloque de Entrada
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        
        # Bloque Residual 1
        self.res1_lin = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn = nn.BatchNorm1d(hidden_dim)
        self.res1_act = nn.Mish()
        self.res1_drop = nn.Dropout(dropout_rate)
        
        # Bloque Residual 2
        self.res2_lin = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn = nn.BatchNorm1d(hidden_dim)
        self.res2_act = nn.Mish()
        self.res2_drop = nn.Dropout(dropout_rate)
        
        # Salida Colapsada
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, 1) # Salida cruda (Logit) para BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.entry(x)
        
        # Skip connection 1
        x_res = self.res1_lin(x)
        x_res = self.res1_bn(x_res)
        x_res = self.res1_act(x_res)
        x_res = self.res1_drop(x_res)
        x = x + x_res # Add
        
        # Skip connection 2
        x_res = self.res2_lin(x)
        x_res = self.res2_bn(x_res)
        x_res = self.res2_act(x_res)
        x_res = self.res2_drop(x_res)
        x = x + x_res # Add
        
        return self.head(x)

# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------
def load_and_preprocess():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    # Mapping: NOK=1, OK=0
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    df = df.drop(columns=[target_col])
    
    y = df['target'].values
    X = df.drop(columns=['target'])
    
    # Label encode explicitly 
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
         X['num_sum'] = X[num_cols].sum(axis=1)
         X['num_mean'] = X[num_cols].mean(axis=1)
         X['num_std'] = X[num_cols].std(axis=1)

    X = X.fillna(0)
    return X.values, y, X.columns.tolist()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    print("="*60)
    print(" PASO 3 [DL] - RED NEURONAL RESIDUAL CON FOCAL LOSS")
    print("="*60)
    
    start_time = time.time()
    set_seed(RANDOM_STATE)
    
    print("[1] Cargando y Escalando Datos...")
    X, y, feature_names = load_and_preprocess()
    
    # En Deep Learning, el escalado fino de TODOS los features es obligatorio y estricto.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    
    # Definir factor Alpha de FocalLoss basado en la asimetría
    # NOK(1) es 75%, OK(0) es 25%. Alpha pondera la clase 1 respecto a la 0.
    # Dado que nos enfocamos en que OK acierte mas (clase 0), invertimos el alpha drásticamente
    # Un alpha bajo (ej: 0.25) da peso a la clase 0 nativamente si 1 es el objetivo en FocalLoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    - Computando en: {device}")
    
    EPOCHS = 60
    BATCH_SIZE = 128
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_proba = np.zeros(len(y))
    
    print("[2] Iniciando Entrenamiento K-Fold...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_val, y_val = X_scaled[val_idx], y[val_idx]
        
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = TabularResNet(input_dim, hidden_dim=256, dropout_rate=0.4).to(device)
        
        # Focal Loss
        criterion = FocalLoss(alpha=0.6, gamma=2.5) 
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # AdamW previene overfitting via L2
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
            
            # Validación
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
                best_model_weights = model.state_dict()
                
        # Cargamos los mejores pesos para inferir Out-of-fold
        model.load_state_dict(best_model_weights)
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                 batch_X = batch_X.to(device)
                 outputs = model(batch_X)
                 # Usamos Sigmoid para sacar la probabilidad
                 probs = torch.sigmoid(outputs).cpu().numpy()
                 fold_preds.extend(probs)
                 
        oof_preds_proba[val_idx] = np.array(fold_preds).flatten()
        print(f"    - Fold {fold+1}/5 listo. Mejor Val Loss: {best_val_loss:.4f}")

    print("\n[3] Optimización Numérica Especial del Umbral para Deep Learning...")
    best_t = 0.5
    best_target_score = -1.0
    
    for t in np.linspace(0.1, 0.9, 150):
        preds_t = (oof_preds_proba >= t).astype(int)
        
        recall_nok = recall_score(y, preds_t, pos_label=1)
        f1_ok = f1_score(y, preds_t, pos_label=0)
        f1_w = f1_score(y, preds_t, average='weighted')
        
        # En DL a veces empujar el umbral destroza una de las métricas. Restringimos
        if recall_nok < 0.85:
            continue
            
        target_score = f1_w + (f1_ok * 0.5)
        
        if target_score > best_target_score:
            best_target_score = target_score
            best_t = t
            
    final_preds = (oof_preds_proba >= best_t).astype(int)
    
    print("\n" + "*"*50)
    print(" RESULTADOS CRÍTICOS LOGRADOS POR LA RED RESIDUAL")
    print("*"*50)
    target_names = ['OK (Minotitario)', 'NOK (Mayoritario)']
    
    print(f"    - Umbral Optimizado: {best_t:.4f}")
    
    print("\nMatriz de Confusión:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=target_names))
    
    # Exportar el pipeline completo (Scaler + Pytorch Model Structure)
    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)
    
    # Aquí podríamos reentrenar sobre todo el dataset, pero dejamos el proof de CV primero.

if __name__ == '__main__':
    main()
