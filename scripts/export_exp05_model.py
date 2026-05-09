#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_exp05_model_v2.py
========================
Pipeline optimizado: VAE + CatBoost Nativo.
- Aprovecha el manejo nativo de categóricas de CatBoost.
- Gestiona el desbalanceo extremo (90% NOK) con auto_class_weights.
- Evalúa y calibra considerando AMBAS clases (OK y NOK).
"""

import io
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import sklearn
import catboost as cb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -- Rutas --------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "Dataset_01_Anonimizado.xlsx"
MODEL_OUT  = BASE_DIR / "models" / "exp05_vae_catboost_v2.pkl"

RANDOM_STATE = 42
LATENT_DIM   = 12
VAE_EPOCHS   = 15

# -- Seeds globales -----------------------------------------------------------
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ==============================================================================
# VAE
# ==============================================================================
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.enc1      = nn.Linear(input_dim, hidden_dim)
        self.enc2      = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec1      = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec2      = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec3      = nn.Linear(hidden_dim, input_dim)
        self.relu      = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.relu(self.dec1(z))
        h = self.relu(self.dec2(h))
        return self.dec3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def _vae_loss(x_recon, x, mu, logvar):
    mse = nn.functional.mse_loss(x_recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + 0.1 * kld


def train_vae(X_scaled: np.ndarray, latent_dim: int, epochs: int, device) -> VAE:
    vae = VAE(input_dim=X_scaled.shape[1], latent_dim=latent_dim).to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)),
        batch_size=64, shuffle=True,
    )
    vae.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = vae(batch)
            _vae_loss(recon, batch, mu, logvar).backward()
            opt.step()
    return vae


def get_latents(vae: VAE, X_scaled: np.ndarray, device) -> tuple:
    vae.eval()
    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, mu, _ = vae(tensor)
        err = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
        lat = mu.cpu().numpy()
    return err, lat


# ==============================================================================
# Helpers de datos
# ==============================================================================
def load_raw_data(path: Path) -> tuple:
    print(f"  Leyendo {path.name} ...")
    df = pd.read_excel(path)
    target_col = "Variable de Salida"
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df["target"] = df[target_col].map({"NOK": 1, "OK": 0})
    y = df["target"].values
    X = df.drop(columns=[target_col, "target", "ID", "Variable 02"], errors="ignore")
    return X, y


def apply_fe(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    y_train: np.ndarray,
    *,
    scaler: Optional[StandardScaler]   = None,
    vae: Optional[VAE]                 = None,
    train_medians: Optional[pd.Series] = None,
    device=None,
    fit: bool = True,
) -> tuple:
    """
    Feature Engineering simplificado y potente:
    1. Imputa nulos numéricos (mediana) y categóricos ("missing").
    2. Escala solo las numéricas para pasarlas al VAE.
    3. Extrae features latentes del VAE.
    4. Mantiene las categóricas como strings nativos para CatBoost.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = X_train_raw.copy(deep=True)
    X_vl = X_val_raw.copy(deep=True)

    num_cols = X_tr.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=["object"]).columns.tolist()

    # 1. Imputación
    if fit:
        train_medians = X_tr[num_cols].median()
    X_tr[num_cols] = X_tr[num_cols].fillna(train_medians)
    X_vl[num_cols] = X_vl[num_cols].fillna(train_medians)
    
    X_tr[cat_cols] = X_tr[cat_cols].fillna("missing").astype(str)
    X_vl[cat_cols] = X_vl[cat_cols].fillna("missing").astype(str)

    # 2. Escalado (Solo para el VAE, mantenemos las numéricas escaladas en X final)
    if fit:
        if scaler is None:
            scaler = StandardScaler()
            X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        else:
            X_tr[num_cols] = scaler.transform(X_tr[num_cols])
    else:
        X_tr[num_cols] = scaler.transform(X_tr[num_cols])

    X_vl[num_cols] = scaler.transform(X_vl[num_cols])

    # 3. VAE features
    tr_err, tr_lat = get_latents(vae, X_tr[num_cols].values, device)
    vl_err, vl_lat = get_latents(vae, X_vl[num_cols].values, device)

    X_tr["vae_err"] = tr_err
    X_vl["vae_err"] = vl_err
    for i in range(tr_lat.shape[1]):
        X_tr[f"vae_l{i}"] = tr_lat[:, i]
        X_vl[f"vae_l{i}"] = vl_lat[:, i]

    artifacts = {
        "scaler": scaler, "vae": vae,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "train_medians": train_medians,
    }
    return X_tr, X_vl, artifacts


# ==============================================================================
# Entrenamiento principal
# ==============================================================================
def main():
    t0 = time.time()
    print("=" * 65)
    print("  EXPORT EXP_05 - VAE + CatBoost NATIVO (Balanceado) -> pkl")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATA_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")

    X_df, y = load_raw_data(DATA_PATH)

    # Distribución de clases real
    nok_pct = (y == 1).mean() * 100
    print(f"  Distribución target: OK(0): {100-nok_pct:.1f}% | NOK(1): {nok_pct:.1f}%")

    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train pool : {len(X_pool):,}  |  Holdout : {len(X_holdout):,}")

# -- 5-Fold OOF ------------------------------------
    print("\n[1/4] Entrenamiento 5-Fold OOF y extracción de latentes ...")
    
    num_cols_pool = X_pool.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols_pool = X_pool.select_dtypes(include=["object"]).columns.tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))
    fold_iterations = []
    
    cb_params = {
        "iterations": 600, 
        "depth": 6, 
        "learning_rate": 0.05,
        "auto_class_weights": "Balanced", 
        "verbose": False, 
        "random_seed": RANDOM_STATE,
    }

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr, X_vl = X_pool.iloc[tr_idx].copy(), X_pool.iloc[vl_idx].copy()
        y_tr, y_vl = y_pool[tr_idx], y_pool[vl_idx]
        
        # 1. Medianas y Scaler ESTRICTAMENTE en el fold de entrenamiento
        num_cols = X_tr.select_dtypes(exclude=["object"]).columns.tolist()
        fold_medians = X_tr[num_cols].median()
        
        X_tr_num_filled = X_tr[num_cols].fillna(fold_medians)
        fold_scaler = StandardScaler()
        X_tr_sc = fold_scaler.fit_transform(X_tr_num_filled)
        
        # 2. Entrenar VAE ESTRICTAMENTE con el fold de entrenamiento
        fold_vae = train_vae(X_tr_sc, latent_dim=LATENT_DIM, epochs=VAE_EPOCHS, device=device)
        
        # 3. Aplicar FE completo usando los artefactos locales del fold
        X_tr_f, X_vl_f, _ = apply_fe(
            X_tr, X_vl, y_tr,
            vae=fold_vae, scaler=fold_scaler, train_medians=fold_medians, 
            fit=False, device=device # Pasamos fit=False porque ya ajustamos manual arriba, o pasamos None
        )
        
        mdl = cb.CatBoostClassifier(**cb_params)
        mdl.fit(
            X_tr_f, y_tr,
            cat_features=cat_cols_pool,
            eval_set=(X_vl_f, y_vl),
            early_stopping_rounds=40,
        )
        oof_proba[vl_idx] = mdl.predict_proba(X_vl_f)[:, 1]
        fold_iterations.append(mdl.get_best_iteration())
        print(f"    Fold {fold + 1}/5 OK  (best_iter={mdl.get_best_iteration()})")

    avg_iterations = int(np.mean(fold_iterations))
    print(f"\n    Iteraciones promedio en CV: {avg_iterations}")

    # -- Threshold tuning ------------------------------------------------------
    print("\n[2/4] Calibrando threshold OOF ...")
    best_t, best_f1_macro = 0.5, -1.0
    
    for t in np.linspace(0.2, 0.8, 100):
        preds = (oof_proba >= t).astype(int)
        score = f1_score(y_pool, preds, average="macro")
        if score > best_f1_macro:
            best_f1_macro, best_t = score, t
            
    print(f"    Threshold óptimo = {best_t:.4f} (Maximiza F1-Macro)")

    # -- Artefactos Globales y Modelo Final ------------------------------------
    print(f"\n[3/4] Generando artefactos globales en TODO el Pool ...")
    num_cols_pool = X_pool.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols_pool = X_pool.select_dtypes(include=["object"]).columns.tolist()
    
    # Usamos todo X_pool porque es el paso final antes del Holdout
    global_medians = X_pool[num_cols_pool].median()
    global_scaler = StandardScaler()
    X_pool_sc = global_scaler.fit_transform(X_pool[num_cols_pool].fillna(global_medians))
    
    print("    Entrenando VAE global ...")
    global_vae = train_vae(X_pool_sc, latent_dim=LATENT_DIM, epochs=VAE_EPOCHS, device=device)

    # -- Modelo final ----------------------------------------------------------
    print(f"\n[4/4] Entrenando modelo final ({avg_iterations} iteraciones) ...")
    X_pool_fe, X_holdout_fe, final_artifacts = apply_fe(
        X_pool, X_holdout, y_pool,
        vae=global_vae, scaler=global_scaler, train_medians=global_medians, 
        fit=False, device=device,
    )

    final_model = cb.CatBoostClassifier(
        iterations=avg_iterations,
        depth=cb_params["depth"],
        learning_rate=cb_params["learning_rate"],
        auto_class_weights="Balanced",
        verbose=False,
        random_seed=RANDOM_STATE,
    )
    final_model.fit(X_pool_fe, y_pool, cat_features=cat_cols_pool)

    # -- Evaluación sobre Holdout --------------------------------------
    print("\n" + "="*65)
    print("MÉTRICAS DE EVALUACIÓN SOBRE HOLDOUT")
    print("="*65)
    holdout_proba = final_model.predict_proba(X_holdout_fe)[:, 1]
    holdout_preds = (holdout_proba >= best_t).astype(int)
    
    # Mostramos el reporte completo para ver OK y NOK
    report = classification_report(
        y_holdout, holdout_preds, 
        target_names=["OK (0)", "NOK (1)"], 
        digits=4
    )
    print(report)
    
    roc_auc = roc_auc_score(y_holdout, holdout_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # -- Estadísticos y Serialización ------------------------------------------
    col_stats = {
        col: {"min": float(X_pool[col].min()), "max": float(X_pool[col].max()), "mean": float(X_pool[col].mean())}
        for col in num_cols_pool
    }
    categories = {
        col: [str(x) for x in sorted(X_pool[col].dropna().unique().tolist())]
        for col in cat_cols_pool
    }
    versions = {
        "python": __import__("sys").version, "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "catboost": cb.__version__, "joblib": joblib.__version__,
    }

    vae_buf = io.BytesIO()
    torch.save(final_artifacts["vae"].state_dict(), vae_buf)

    pkl_payload = {
        "model":          final_model,
        "vae_bytes":      vae_buf.getvalue(),
        "vae_input_dim":  len(num_cols_pool),
        "scaler":         final_artifacts["scaler"],
        "num_cols":       num_cols_pool,
        "cat_cols":       cat_cols_pool,
        "train_medians":  final_artifacts["train_medians"],
        "all_feats":      X_pool_fe.columns.tolist(),
        "threshold":      float(best_t),
        "latent_dim":     LATENT_DIM,
        "col_stats":      col_stats,
        "categories":     categories,
        "versions":       versions,
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkl_payload, MODEL_OUT)

    print(f"\n Artefacto limpio serializado en: {MODEL_OUT}")
    print(f" Tiempo total: {time.time() - t0:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()