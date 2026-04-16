#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXPERIMENTO 09 - The Vanguard Architecture: SupMin + TabDDPM + TabM + DWB Loss
----------------------------------------------------------------------------------
Esqueleto Funcional en PyTorch.
1. SupMin Contrastive Encoder: Convergencia de minoría usando Asymmetric NT-Xent.
2. OVERLAP Ternary Transmutation & TabDDPM Generator.
3. TabM Classifier: BatchEnsemble de multiplicadores de rango uno (K=32).
4. DWB Loss: Dinámica ponderada.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================
# 1. DYNAMICALLY WEIGHTED BALANCED (DWB) LOSS
# =====================================================================
class DWBLoss(nn.Module):
    def __init__(self, base_alpha=0.5, gamma=2.0, momentum=0.9):
        """
        Dynamically Weighted Balanced (DWB) Loss.
        Ajusta la ponderación de la clase minoritaria (y_true=0) dinámicamente
        época tras época en función del error empírico / "hardness".
        No utiliza SMOTE ni scale_pos_weight estático.
        """
        super(DWBLoss, self).__init__()
        self.gamma = gamma
        self.momentum = momentum
        self.current_alpha = base_alpha 

    def update_alpha(self, empirical_minority_error):
        """
        Actualiza el factor de penalización basándose en el error de validación
        de los hard minority samples.
        """
        target_alpha = min(0.99, max(0.1, empirical_minority_error))
        self.current_alpha = self.momentum * self.current_alpha + (1 - self.momentum) * target_alpha

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Asymmetric weighting dynamically aligned
        # Asumiendo targets = 1 (NOK/Mayoritaria) y targets = 0 (OK/Minoritaria)
        alpha_t = torch.where(targets == 0, self.current_alpha, 1 - self.current_alpha)
        
        dwb_loss = alpha_t * (1 - pt)**self.gamma * bce_loss
        return dwb_loss.mean()

# =====================================================================
# 2. SUPMIN CONTRASTIVE ENCODER (Asymmetric NT-Xent)
# =====================================================================
class SupMinEncoder(nn.Module):
    def __init__(self, input_dim=102, projection_dim=32):
        super(SupMinEncoder, self).__init__()
        # Red de extracción de features (Preservando Integridad Geométrica)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.projector = nn.Sequential(
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1) # Proyectado al hiperesfera unitaria
        return z

def asymmetric_nt_xent_loss(z, labels, temperature=0.1):
    """
    Colapso Métrico InfoNCE Asimétrico:
    Forzamos un colapso esférico exclusivamente para la clase minoritaria (OK=0),
    mientras la clase mayoritaria (NOK=1) no sufre penalización de atracción,
    permitiendo dispersión uniforme natural conservando su alta varianza empírica.
    """
    device = z.device
    batch_size = z.shape[0]
    
    # Matriz de similaridad coseno (Latent Dot Product)
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Masking self-similarity
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    sim_matrix.masked_fill_(mask, -9e15)

    # Buscar "Positive Pairs" pero SOLO dictados por la Minoría (0)
    # Si un sample es 1 (NOK), no ejerce fuerza atractiva explícita (Spread libre)
    minority_mask = (labels == 0).float()
    
    # Pairs matrix: 1 if both are minority, else 0
    pos_mask = torch.outer(minority_mask, minority_mask)
    pos_mask.masked_fill_(mask, 0.0) # avoid self

    # Exponentiated similarities
    exp_sim = torch.exp(sim_matrix)
    
    # Sum over all negative samples for denominator
    denominators = exp_sim.sum(dim=1, keepdim=True)
    
    # InfoNCE solo calculada para los anchor minoritarios contra los positivos minoritarios
    log_probs = sim_matrix - torch.log(denominators)
    
    # Computar pérdida solo donde existen positive pairs forzando colapso de Minoría
    pos_log_probs = log_probs * pos_mask
    sum_pos_log_probs = pos_log_probs.sum(dim=1)
    
    num_pos = pos_mask.sum(dim=1)
    # Evitar division by zero
    num_pos = torch.where(num_pos > 0, num_pos, torch.ones_like(num_pos))
    
    loss = - (sum_pos_log_probs / num_pos)
    # Only average over minority samples in batch
    loss = loss.sum() / (minority_mask.sum() + 1e-8)
    
    return loss

# =====================================================================
# 3. TERNARY OVERLAP & TABDDPM GENERATOR (SKELETON)
# =====================================================================
class OverlapTransmutator:
    def __init__(self, density_threshold=0.85):
        self.density_threshold = density_threshold
        
    def fit_transmute(self, z_latents, labels):
        """
        Algoritmo de densidades fronterizas. Tras la proyección SupMin,
        evalúa vecindades en el espacio latente. 
        Si un NOK (1) está inmerso en la hiper-esfera defensiva OK (0), es transmutado a OVERLAP (2).
        """
        # Placeholder: Distancia L2 KNN al centroide minoritario
        minority_centroid = z_latents[labels == 0].mean(dim=0)
        distances = torch.norm(z_latents - minority_centroid, dim=1)
        
        ternary_labels = labels.clone()
        
        # Definición geométrica del overlap (heurística)
        overlap_mask = (labels == 1) & (distances < self.density_threshold)
        ternary_labels[overlap_mask] = 2 # OVERLAP
        
        return ternary_labels

class TabDDPM_Generator(nn.Module):
    def __init__(self, latent_dim=102, num_classes=3):
        super(TabDDPM_Generator, self).__init__()
        # Esqueleto de Denoising Diffusion. 
        # Modela la síntesis condicional bajo 3 clases.
        self.time_embed = nn.Embedding(1000, 32)
        self.class_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x, t_steps, y_ternary):
        t_emb = self.time_embed(t_steps)
        y_emb = self.class_embed(y_ternary)
        cond_emb = torch.cat([t_emb, y_emb], dim=1)
        h = torch.cat([x, cond_emb], dim=1)
        return self.net(h)
        
    def sample_ok_minority(self, num_samples, n_steps=1000):
        """
        Sintetiza la clase OK (0) respetando la frontera tri-clásica evitando
        caer en el pozo de OVERLAP (2) ni NOK (1).
        """
        device = next(self.parameters()).device
        x_synth = torch.randn((num_samples, 102), device=device)
        y_cond = torch.zeros(num_samples, dtype=torch.long, device=device) # Clase 0
        
        # Diffusion Reverse Process (Skeleton) Loop
        for t in reversed(range(n_steps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
            predicted_noise = self.forward(x_synth, t_tensor, y_cond)
            # Aplicar ecuación generativa Reverse (Euler / DDPM estandar)
            x_synth = x_synth - 0.05 * predicted_noise # Simplified Step
            
        return x_synth

# =====================================================================
# 4. TABM CLASSIFIER (BatchEnsemble)
# =====================================================================
class TabMBatchEnsembleLayer(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=32):
        super(TabMBatchEnsembleLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        
        # Shared Weight Matrix
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)
        
        # Rank-One Multipliers (Fast Ensembles Parameters)
        self.R_in = nn.Parameter(torch.Tensor(ensemble_size, in_features))
        self.S_out = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        
        nn.init.normal_(self.R_in, mean=1.0, std=0.05)
        nn.init.normal_(self.S_out, mean=1.0, std=0.05)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        x: [batch_size, in_features] o [batch_size, ensemble_size, in_features]
        Se vectoriza el producto Hadamard usando los multiplicadores de rango uno.
        """
        # [ensemble_size, batch_size, in_features]
        if x.dim() == 2:
            x_b = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        else:
            # Asumiendo input [ensemble_size, batch_size, in]
            x_b = x

        # Hadamard product con el In-Multiplier
        # r_in = [ensemble_size, 1, in_features]
        r_in = self.R_in.unsqueeze(1)
        z = x_b * r_in
        
        # Aplicación Matriz de Pesos Compartida
        batch_size = z.shape[1]
        z_flat = z.reshape(-1, self.in_features)
        h_flat = F.linear(z_flat, self.W)
        h = h_flat.view(self.ensemble_size, batch_size, self.out_features)
        
        # Hadamard product con el Out-Multiplier y sumar el bias
        s_out = self.S_out.unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        
        out = (h * s_out) + bias
        return out

class TabMClassifier(nn.Module):
    def __init__(self, input_dim=102, ensemble_size=32):
        super(TabMClassifier, self).__init__()
        self.ensemble_size = ensemble_size
        
        self.layer1 = TabMBatchEnsembleLayer(input_dim, 256, ensemble_size)
        self.norm1 = nn.LayerNorm(256)
        
        self.layer2 = TabMBatchEnsembleLayer(256, 128, ensemble_size)
        self.norm2 = nn.LayerNorm(128)
        
        self.head = TabMBatchEnsembleLayer(128, 1, ensemble_size)
        
    def forward(self, x):
        h = self.layer1(x)
        h = F.gelu(self.norm1(h))
        
        h = self.layer2(h)
        h = F.gelu(self.norm2(h))
        
        # [ensemble_size, batch_size, 1]
        logits = self.head(h).squeeze(-1)
        
        # Marginalización: Calculamos la probabilidad media sobre la asamblea (K=32)
        # Permitiendo interacciones masivas de rango 1 pero reduciendo la var de predicción
        probs = torch.sigmoid(logits)
        mean_probs = probs.mean(dim=0)
        mean_logits = torch.log(mean_probs / (1 - mean_probs + 1e-9))
        return mean_logits

# =====================================================================
# 5. TRAINING ORCHESTRATION SKELETON
# =====================================================================
def vanguard_training_pipeline(train_loader, val_loader, input_dim=102, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Init Components
    encoder = SupMinEncoder(input_dim=input_dim).to(device)
    classifier = TabMClassifier(input_dim=input_dim, ensemble_size=32).to(device)
    ddpm_gen = TabDDPM_Generator(latent_dim=input_dim).to(device)
    
    dwb_loss = DWBLoss(base_alpha=0.5).to(device)
    transmutator = OverlapTransmutator(density_threshold=0.85)
    
    # Phase Optimizers
    opt_contrastive = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    opt_tabm = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    # opt_ddpm = ... 
    
    print("[INIT] Orchestrating The Vanguard Architecture...")
    
    for epoch in range(epochs):
        # ----------------------------------------------------
        # FASE 1: SupMin Contrastive Representation Learning
        # ----------------------------------------------------
        encoder.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            
            opt_contrastive.zero_grad()
            z_latents = encoder(x_b)
            loss_c = asymmetric_nt_xent_loss(z_latents, y_b)
            loss_c.backward()
            opt_contrastive.step()
            
            # Dinámica Ternaria a nivel Latente
            with torch.no_grad():
                y_ternary = transmutator.fit_transmute(z_latents, y_b)
                # Aquí se invocaría el entrenamiento de self.ddpm_gen(x_b, t, y_ternary)
                
        # ----------------------------------------------------
        # FASE 2: Clasificación BatchEnsemble TabM
        # ----------------------------------------------------
        classifier.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device).float()
            
            # (Opcional): Concatenar sintéticos desde TabDDPM_Generator
            # x_synth = ddpm_gen.sample_ok_minority(100)
            
            opt_tabm.zero_grad()
            logits = classifier(x_b) # TabM Ensemble Forward Propagation
            loss_t = dwb_loss(logits, y_b)
            loss_t.backward()
            opt_tabm.step()
            
        # ----------------------------------------------------
        # FASE 3: Validación OOF & Dynamic Optimization (DWB)
        # ----------------------------------------------------
        classifier.eval()
        minority_errors = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                logits_v = classifier(x_v)
                preds = (torch.sigmoid(logits_v) >= 0.5).int()
                
                # Identificar errores solo en clase minoritaria (0)
                minority_mask = (y_v == 0)
                if minority_mask.sum() > 0:
                    errors = (preds[minority_mask] != y_v[minority_mask]).float().mean()
                    minority_errors.append(errors.item())
                    
        emp_min_err = sum(minority_errors)/len(minority_errors) if minority_errors else 0.0
        
        # Ponderación Dinámica! "forzando convergencia sobre hard minority samples"
        dwb_loss.update_alpha(emp_min_err)
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] - Contrastive Loss: {loss_c.item():.4f} - DWB Alpha: {dwb_loss.current_alpha:.4f}")
            
    print("[SUCCESS] Vanguard Tabular Topology Generated & Trained.")
    return encoder, classifier

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import time

DATA_PATH = '../../data/raw/Dataset_01_Anonimizado.xlsx'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/Dataset_01_Anonimizado.xlsx'

def maximize_threshold(y_val, preds_proba):
    best_t = 0.5
    best_target = -1.0
    for t in np.linspace(0.1, 0.9, 150):
        preds = (preds_proba >= t).astype(int)
        r_nok = recall_score(y_val, preds, pos_label=1)
        r_ok = recall_score(y_val, preds, pos_label=0)
        f1_w = f1_score(y_val, preds, average='weighted')
        
        if r_nok < 0.85:
            continue
            
        score = f1_w + (r_ok * 0.35) 
        if score > best_target:
            best_target = score
            best_t = t
    return best_t, best_target

def load_data():
    df = pd.read_excel(DATA_PATH)
    target_col = 'Variable de Salida'
    df = df.dropna(subset=[target_col])
    
    df['target'] = df[target_col].map({'NOK': 1, 'OK': 0})
    y = df['target'].values
    X = df.drop(columns=[target_col, 'target'])
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    return X_scaled, y

def main():
    print("="*60)
    print(" EXPERIMENTO 09 - THE VANGUARD ARCHITECTURE EXECUTION")
    print("="*60)
    start_time = time.time()
    
    X, y = load_data()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y))
    input_dim = X.shape[1]
    
    print(f"[!] Evaluando sobre Dimensiones Paramétricas: {input_dim}\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        from torch.utils.data import TensorDataset, DataLoader
        
        X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
        X_val_t = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)
        
        encoder, classifier = vanguard_training_pipeline(train_loader, val_loader, input_dim=input_dim, epochs=20)
        
        # Validacion Fold
        classifier.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_probs = []
        with torch.no_grad():
            for x_v, _ in val_loader:
                x_v = x_v.to(device)
                logits_v = classifier(x_v)
                probs = torch.sigmoid(logits_v).cpu().numpy()
                fold_probs.extend(probs)
                
        oof_proba[val_idx] = np.array(fold_probs)
        print(f"--- Fold {fold+1} completado ---\n")
        
    best_t, _ = maximize_threshold(y, oof_proba)
    if best_t == 0.5 and maximize_threshold(y, oof_proba)[1] == -1.0:
        best_t = 0.55
        
    final_preds = (oof_proba >= best_t).astype(int)
    
    print("\n" + "*"*50)
    print(" RESULTADOS INVESTIGACIÓN PYTORCH VANGUARD")
    print("*"*50)
    
    print(f"    - Umbral Optimizado: {best_t:.4f}")
    print("\nMatriz de Confusión:\n", confusion_matrix(y, final_preds))
    print("\nReporte de Clasificación:\n", classification_report(y, final_preds, target_names=['OK (Min)','NOK (May)']))
    
    print(f"\nTiempo Total: {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
