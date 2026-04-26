#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_exp05_model.py
=====================
Entrena el pipeline completo del Experimento 05 (VAE + KBins + CatBoost)
sobre el 100% del dataset (pool de entrenamiento = 80%) y serializa el
artefacto final en:

    models/exp05_vae_catboost.pkl

Contenido del pkl:
    {
        "model"     : CatBoostClassifier entrenado,
        "vae"       : estado del VAE (state_dict como bytes serializados),
        "scaler"    : StandardScaler ajustado sobre num_cols del train pool,
        "kbd"       : KBinsDiscretizer ajustado sobre high_var del train pool,
        "oe"        : OrdinalEncoder ajustado sobre cat_cols del train pool,
        "high_var"  : lista de columnas con alta varianza (top-10),
        "num_cols"  : lista de columnas numéricas originales,
        "cat_cols"  : lista de columnas categóricas originales,
        "all_feats" : lista ordenada de TODAS las columnas del DataFrame
                      que entra al modelo (incluyendo vae_*, _bin),
        "threshold" : umbral OOF óptimo,
        "latent_dim": dimensión latente del VAE,
        "col_stats" : {col: {min, max, mean}} para rangos en la UI,
        "categories": {col: [valores]} para dropdowns en la UI,
    }

Uso:
    cd <raiz del proyecto PLUTO>
    python scripts/export_exp05_model.py
"""

import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OrdinalEncoder
import catboost as cb

warnings.filterwarnings("ignore")

# -- Rutas (pathlib - agnóstico de SO) ----------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "Dataset_01_Anonimizado.xlsx"
MODEL_OUT  = BASE_DIR / "models" / "exp05_vae_catboost.pkl"

RANDOM_STATE = 42
LATENT_DIM   = 12
VAE_EPOCHS   = 15


# ==============================================================================
# VAE (idéntico al Exp_05 original)
# ==============================================================================

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.enc1     = nn.Linear(input_dim, hidden_dim)
        self.enc2     = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu    = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar= nn.Linear(hidden_dim // 2, latent_dim)
        self.dec1     = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec2     = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec3     = nn.Linear(hidden_dim, input_dim)
        self.relu     = nn.ReLU()

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
    """Entrena el VAE sobre X_scaled y devuelve el modelo entrenado."""
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


def get_latents(vae: VAE, X_scaled: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (reconstruction_error, latent_means) para un array escalado."""
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

def load_raw_data(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Carga el Excel y devuelve (X_df, y). Sin transformaciones."""
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
    oe: OrdinalEncoder | None = None,
    kbd: KBinsDiscretizer | None = None,
    scaler: StandardScaler | None = None,
    vae: VAE | None = None,
    high_var: list[str] | None = None,
    device=None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Aplica el pipeline de Feature Engineering del Exp_05.

    Si fit=True  -> ajusta oe/kbd/scaler/vae sobre X_train_raw y transforma.
    Si fit=False -> solo transforma con artefactos pre-ajustados.

    Retorna (X_tr, X_vl, artefactos) donde artefactos contiene los objetos
    ajustados para su posterior serialización.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = X_train_raw.copy()
    X_vl = X_val_raw.copy()

    num_cols = X_tr.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=["object"]).columns.tolist()

    # 1. Imputación numérica
    X_tr[num_cols] = X_tr[num_cols].fillna(0)
    X_vl[num_cols] = X_vl[num_cols].fillna(0)

    # 2. Encoding categórico
    if cat_cols:
        if fit:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr[cat_cols] = oe.fit_transform(X_tr[cat_cols].fillna("missing").astype(str))
        X_vl[cat_cols] = oe.transform(X_vl[cat_cols].fillna("missing").astype(str))

    # 3. Binning de alta varianza (solo real_num_cols)
    if fit:
        variances = X_tr[num_cols].var().sort_values(ascending=False)
        high_var  = variances.head(10).index.tolist()
        kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
        X_tr_bin = kbd.fit_transform(X_tr[high_var])
    else:
        X_tr_bin = kbd.transform(X_tr[high_var])

    X_vl_bin = kbd.transform(X_vl[high_var])

    for i, col in enumerate(high_var):
        name = f"{col}_bin"
        X_tr[name] = X_tr_bin[:, i].astype(int)
        X_vl[name] = X_vl_bin[:, i].astype(int)

    # 4. VAE features
    if fit:
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr[num_cols])
    else:
        X_tr_sc = scaler.transform(X_tr[num_cols])

    X_vl_sc = scaler.transform(X_vl[num_cols])

    if fit:
        vae = train_vae(X_tr_sc, latent_dim=LATENT_DIM, epochs=VAE_EPOCHS, device=device)

    tr_err, tr_lat = get_latents(vae, X_tr_sc, device)
    vl_err, vl_lat = get_latents(vae, X_vl_sc, device)

    X_tr["vae_err"] = tr_err
    X_vl["vae_err"] = vl_err
    for i in range(tr_lat.shape[1]):
        X_tr[f"vae_l{i}"] = tr_lat[:, i]
        X_vl[f"vae_l{i}"] = vl_lat[:, i]

    artifacts = {
        "oe": oe, "kbd": kbd, "scaler": scaler, "vae": vae,
        "high_var": high_var, "num_cols": num_cols, "cat_cols": cat_cols,
    }
    return X_tr, X_vl, artifacts


# ==============================================================================
# Entrenamiento principal
# ==============================================================================

def main():
    t0 = time.time()
    print("=" * 62)
    print("  EXPORT EXP_05 - VAE + CatBoost -> pkl")
    print("=" * 62)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATA_PATH}")

    X_df, y = load_raw_data(DATA_PATH)

    # -- Holdout split (80 / 20) ----------------------------------------------─
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n  Train pool : {len(X_pool):,}  |  Holdout : {len(X_holdout):,}")

    # -- 5-Fold OOF para threshold ----------------------------------------------
    print("\n[1/3] Entrenamiento 5-Fold para calibración de threshold OOF ...")
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y_pool))
    cb_params = {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "verbose": False, "random_seed": RANDOM_STATE,
    }

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_pool, y_pool)):
        X_tr_f, X_vl_f, _ = apply_fe(
            X_pool.iloc[tr_idx], X_pool.iloc[vl_idx], y_pool[tr_idx], fit=True
        )
        mdl = cb.CatBoostClassifier(**cb_params)
        mdl.fit(X_tr_f, y_pool[tr_idx],
                eval_set=(X_vl_f, y_pool[vl_idx]),
                early_stopping_rounds=30)
        oof_proba[vl_idx] = mdl.predict_proba(X_vl_f)[:, 1]
        print(f"    Fold {fold + 1}/5 OK")

    # -- Threshold tuning (sobre OOF, nunca sobre holdout) ---------------------
    print("\n[2/3] Calibrando threshold OOF ...")
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 100):
        preds = (oof_proba >= t).astype(int)
        if recall_score(y_pool, preds) < 0.85:
            continue
        score = f1_score(y_pool, preds, average="weighted")
        if score > best_f1:
            best_f1, best_t = score, t
    print(f"    Threshold óptimo = {best_t:.4f}  (F1-weighted OOF = {best_f1:.4f})")

    # -- Modelo final sobre train pool completo --------------------------------─
    print("\n[3/3] Entrenando modelo final sobre train pool completo ...")
    X_pool_fe, X_holdout_fe, final_artifacts = apply_fe(
        X_pool, X_holdout, y_pool, fit=True
    )

    final_model = cb.CatBoostClassifier(**cb_params)
    final_model.fit(X_pool_fe, y_pool)

    all_feats = X_pool_fe.columns.tolist()
    print(f"    Features totales del modelo: {len(all_feats)}")

    # -- Estadísticos para UI (rangos y categorías) ----------------------------
    num_cols = final_artifacts["num_cols"]
    cat_cols = final_artifacts["cat_cols"]

    col_stats = {
        col: {
            "min":  float(X_pool[col].min()),
            "max":  float(X_pool[col].max()),
            "mean": float(X_pool[col].mean()),
        }
        for col in num_cols
    }
    categories = {
        col: sorted(X_pool[col].dropna().unique().tolist())
        for col in cat_cols
    }

    # -- Serialización --------------------------------------------------------─
    # Guardar VAE como bytes (torch.save en buffer) para incluirlo en el pkl
    vae_buf = io.BytesIO()
    torch.save(final_artifacts["vae"].state_dict(), vae_buf)
    vae_bytes = vae_buf.getvalue()

    pkl_payload = {
        "model":      final_model,
        "vae_bytes":  vae_bytes,          # VAE state_dict serializado
        "vae_input_dim": len(num_cols),   # Para reconstruir la arquitectura
        "scaler":     final_artifacts["scaler"],
        "kbd":        final_artifacts["kbd"],
        "oe":         final_artifacts["oe"],
        "high_var":   final_artifacts["high_var"],
        "num_cols":   num_cols,
        "cat_cols":   cat_cols,
        "all_feats":  all_feats,
        "threshold":  float(best_t),
        "latent_dim": LATENT_DIM,
        "col_stats":  col_stats,
        "categories": categories,
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkl_payload, MODEL_OUT)

    elapsed = time.time() - t0
    print(f"\nOK  Modelo serializado en: {MODEL_OUT}")
    print(f"    Tiempo total: {elapsed:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()

