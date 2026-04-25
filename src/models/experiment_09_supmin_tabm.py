#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 09 - SupMin + TabM (Vanguard Architecture)
----------------------------------------------------------------------------------
VERSIÓN CORREGIDA — Sin data leakage y con arquitecturas conectadas.
Cambios:
  - Holdout test separado antes del preprocesamiento.
  - Imputación y OrdinalEncoding aplicados DENTRO del CV loop.
  - StandardScaler aplicado DENTRO del CV loop.
  - TabMClassifier ahora procesa: [Variables Originales + Representaciones Latentes].
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'


# =====================================================================
# 1. DWB LOSS
# =====================================================================
class DWBLoss(nn.Module):
    def __init__(self, base_alpha=0.5, gamma=2.0, momentum=0.9):
        super(DWBLoss, self).__init__()
        self.gamma = gamma
        self.momentum = momentum
        self.current_alpha = base_alpha

    def update_alpha(self, empirical_minority_error):
        target_alpha = min(0.99, max(0.1, empirical_minority_error))
        self.current_alpha = self.momentum * self.current_alpha + (1 - self.momentum) * target_alpha

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = torch.where(targets == 0, self.current_alpha, 1 - self.current_alpha)
        dwb_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return dwb_loss.mean()


# =====================================================================
# 2. SUPMIN ENCODER
# =====================================================================
class SupMinEncoder(nn.Module):
    def __init__(self, input_dim=102, projection_dim=32):
        super(SupMinEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.projector = nn.Sequential(nn.Linear(128, projection_dim))

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        return z


def asymmetric_nt_xent_loss(z, labels, temperature=0.1):
    device = z.device
    batch_size = z.shape[0]
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    sim_matrix.masked_fill_(mask, -9e15)
    minority_mask = (labels == 0).float()
    pos_mask = torch.outer(minority_mask, minority_mask)
    pos_mask.masked_fill_(mask, 0.0)
    exp_sim = torch.exp(sim_matrix)
    denominators = exp_sim.sum(dim=1, keepdim=True)
    log_probs = sim_matrix - torch.log(denominators)
    pos_log_probs = log_probs * pos_mask
    sum_pos_log_probs = pos_log_probs.sum(dim=1)
    num_pos = pos_mask.sum(dim=1)
    num_pos = torch.where(num_pos > 0, num_pos, torch.ones_like(num_pos))
    loss = -(sum_pos_log_probs / num_pos)
    loss = loss.sum() / (minority_mask.sum() + 1e-8)
    return loss


# =====================================================================
# 3. TABM CLASSIFIER (BatchEnsemble)
# =====================================================================
class TabMBatchEnsembleLayer(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=32):
        super(TabMBatchEnsembleLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)
        self.R_in = nn.Parameter(torch.Tensor(ensemble_size, in_features))
        self.S_out = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        nn.init.normal_(self.R_in, mean=1.0, std=0.05)
        nn.init.normal_(self.S_out, mean=1.0, std=0.05)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.dim() == 2:
            x_b = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        else:
            x_b = x
        r_in = self.R_in.unsqueeze(1)
        z = x_b * r_in
        batch_size = z.shape[1]
        z_flat = z.reshape(-1, self.in_features)
        h_flat = F.linear(z_flat, self.W)
        h = h_flat.view(self.ensemble_size, batch_size, self.out_features)
        s_out = self.S_out.unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        out = (h * s_out) + bias
        return out


class TabMClassifier(nn.Module):
    # CAMBIO: Agregado projection_dim para aceptar la representación latente
    def __init__(self, input_dim=102, projection_dim=32, ensemble_size=32):
        super(TabMClassifier, self).__init__()
        self.ensemble_size = ensemble_size
        combined_dim = input_dim + projection_dim
        self.layer1 = TabMBatchEnsembleLayer(combined_dim, 256, ensemble_size)
        self.norm1 = nn.LayerNorm(256)
        self.layer2 = TabMBatchEnsembleLayer(256, 128, ensemble_size)
        self.norm2 = nn.LayerNorm(128)
        self.head = TabMBatchEnsembleLayer(128, 1, ensemble_size)

    def forward(self, x):
        h = self.layer1(x)
        h = F.gelu(self.norm1(h))
        h = self.layer2(h)
        h = F.gelu(self.norm2(h))
        logits = self.head(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        mean_probs = probs.mean(dim=0)
        mean_logits = torch.log(mean_probs / (1 - mean_probs + 1e-9))
        return mean_logits


# =====================================================================
# 4. TRAINING PIPELINE
# =====================================================================
def vanguard_training_pipeline(train_loader, val_loader, input_dim, projection_dim=32, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SupMinEncoder(input_dim=input_dim, projection_dim=projection_dim).to(device)
    classifier = TabMClassifier(input_dim=input_dim, projection_dim=projection_dim, ensemble_size=32).to(device)
    
    dwb_loss = DWBLoss(base_alpha=0.5).to(device)
    opt_contrastive = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    opt_tabm = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        # 1. Entrenar Encoder
        encoder.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt_contrastive.zero_grad()
            z_latents = encoder(x_b)
            loss_c = asymmetric_nt_xent_loss(z_latents, y_b)
            loss_c.backward()
            opt_contrastive.step()

        # 2. Entrenar Clasificador
        classifier.train()
        encoder.eval() # Modo evaluación para extraer latents limpios
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device).float()
            opt_tabm.zero_grad()
            
            # CAMBIO: Conectar arquitecturas (con detach para no alterar el encoder aquí)
            with torch.no_grad():
                z_latents = encoder(x_b)
            combined_x = torch.cat([x_b, z_latents], dim=1)
            
            logits = classifier(combined_x)
            loss_t = dwb_loss(logits, y_b)
            loss_t.backward()
            opt_tabm.step()

        # 3. Validación y actualización de DWB alpha
        classifier.eval()
        minority_errors = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                
                # CAMBIO: Conectar en validación también
                z_latents_v = encoder(x_v)
                combined_v = torch.cat([x_v, z_latents_v], dim=1)
                
                logits_v = classifier(combined_v)
                preds = (torch.sigmoid(logits_v) >= 0.5).int()
                minority_mask = (y_v == 0)
                if minority_mask.sum() > 0:
                    errors = (preds[minority_mask] != y_v[minority_mask]).float().mean()
                    minority_errors.append(errors.item())

        emp_min_err = sum(minority_errors) / len(minority_errors) if minority_errors else 0.0
        dwb_loss.update_alpha(emp_min_err)

    return encoder, classifier


def maximize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_target = -1.0
    for t in np.linspace(0.1, 0.9, 150):
        preds = (preds_proba >= t).astype(int)
        r_nok = recall_score(y_val, preds, pos_label=1)
        if r_nok < 0.85:
            continue
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        score = f1_w + (r_ok * 0.35)
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target


def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)
    df = df.drop(columns=['ID','Variable 02'], errors='ignore')
    
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=['target', target_col], errors='ignore')
    
    # CAMBIO: Devuelve el DataFrame para poder procesarlo en cada fold
    return X, y


# =====================================================================
# FUNCIÓN DE PREPROCESAMIENTO PER-FOLD
# =====================================================================
def preprocess_fold(X_train_df, X_val_df, cat_cols, num_cols, is_holdout=False):
    # Copias para no alterar los DataFrames originales
    X_train = X_train_df.copy()
    X_val = X_val_df.copy()
    
    # Imputación de nulos
    X_train[num_cols] = X_train[num_cols].fillna(0)
    X_val[num_cols] = X_val[num_cols].fillna(0)
    
    if len(cat_cols) > 0:
        X_train[cat_cols] = X_train[cat_cols].fillna('missing')
        X_val[cat_cols] = X_val[cat_cols].fillna('missing')
        
        # Ordinal Encoder (Seguro contra leakage y categorías desconocidas)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
        X_val[cat_cols] = oe.transform(X_val[cat_cols])
        
    # Escalado StandardScaler
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    
    return X_train.values, X_val.values


def main():
    print("=" * 60)
    print(" EXPERIMENTO 09 (SUPMIN + TABM) — ARQUITECTURAS CONECTADAS")
    print("=" * 60)
    start_time = time.time()

    print("[1] Cargando datos...")
    X, y = load_raw_data()
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # HOLDOUT SPLIT (Sobre los datos crudos)
    print("[2] Separando holdout test (20%)...")
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    - Pool: {len(y_pool)} | Holdout: {len(y_holdout)}")

    # 5-Fold CV with per-fold scaling & encoding
    print("\n[3] 5-Fold CV con scaler y encoding per-fold...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_train_raw = X_pool.iloc[train_idx]
        y_train = y_pool[train_idx]
        X_val_raw = X_pool.iloc[val_idx]
        y_val = y_pool[val_idx]

        # CAMBIO: Preprocesamiento sin leakage aplicado aquí
        X_train_arr, X_val_arr = preprocess_fold(X_train_raw, X_val_raw, cat_cols, num_cols)

        input_dim = X_train_arr.shape[1]

        X_train_t = torch.tensor(X_train_arr, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_arr, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

        encoder, classifier = vanguard_training_pipeline(train_loader, val_loader, input_dim=input_dim, epochs=20)

        classifier.eval()
        encoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_probs = []
        with torch.no_grad():
            for x_v, _ in val_loader:
                x_v = x_v.to(device)
                
                # CAMBIO: Conectar arquitecturas para las predicciones OOF
                z_latents_v = encoder(x_v)
                combined_v = torch.cat([x_v, z_latents_v], dim=1)
                
                logits_v = classifier(combined_v)
                probs = torch.sigmoid(logits_v).cpu().numpy()
                fold_probs.extend(probs)

        oof_proba[val_idx] = np.array(fold_probs)
        print(f"    - Fold {fold + 1}/5 completado.")

    # Threshold tuning on OOF (train pool)
    print("\n[4] Threshold tuning sobre OOF del train pool...")
    best_t, _ = maximize_threshold(y_pool, oof_proba)
    if best_t == 0.5 and maximize_threshold(y_pool, oof_proba)[1] == -1.0:
        best_t = 0.55
    print(f"    - Threshold: {best_t:.4f}")

    # HOLDOUT evaluation
    print("\n[5] Evaluación Final sobre HOLDOUT TEST...")
    
    # CAMBIO: Preprocesamiento final usando todo el Train Pool para evaluar el Holdout
    X_pool_arr, X_holdout_arr = preprocess_fold(X_pool, X_holdout, cat_cols, num_cols)

    input_dim = X_pool_arr.shape[1]
    X_pool_t = torch.tensor(X_pool_arr, dtype=torch.float32)
    y_pool_t = torch.tensor(y_pool, dtype=torch.long)
    X_holdout_t = torch.tensor(X_holdout_arr, dtype=torch.float32)
    y_holdout_t = torch.tensor(y_holdout, dtype=torch.long)

    final_train_ld = DataLoader(TensorDataset(X_pool_t, y_pool_t), batch_size=256, shuffle=True)
    # Subset of train as proxy for validation during final fit
    final_val_ld = DataLoader(TensorDataset(X_pool_t[:256], y_pool_t[:256]), batch_size=256, shuffle=False)

    final_encoder, final_classifier = vanguard_training_pipeline(final_train_ld, final_val_ld, input_dim=input_dim, epochs=20)

    final_classifier.eval()
    final_encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_probs = []
    holdout_ld = DataLoader(TensorDataset(X_holdout_t, y_holdout_t), batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for x_h, _ in holdout_ld:
            x_h = x_h.to(device)
            
            # CAMBIO: Conectar arquitecturas para las predicciones del Holdout
            z_latents_h = final_encoder(x_h)
            combined_h = torch.cat([x_h, z_latents_h], dim=1)
            
            logits = final_classifier(combined_h)
            probs = torch.sigmoid(logits).cpu().numpy()
            holdout_probs.extend(probs)

    holdout_proba = np.array(holdout_probs)
    holdout_preds = (holdout_proba >= best_t).astype(int)

    print("\nMatriz de Confusión (HOLDOUT):\n", confusion_matrix(y_holdout, holdout_preds))
    print("\nReporte de Clasificación (HOLDOUT):\n",
          classification_report(y_holdout, holdout_preds, target_names=['OK (Min)', 'NOK (May)']))

    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main