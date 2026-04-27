"""
utils/ml_engine.py
==================
Motor de predicción del Experimento 05 (VAE + KBins + CatBoost).

Responsabilidades:
    - Cargar `models/exp05_vae_catboost.pkl` con rutas pathlib (agnóstico de SO).
    - Si el pkl no existe, mostrar instrucciones claras para generarlo.
    - Reconstruir el VAE (PyTorch) desde el state_dict almacenado en el pkl.
    - Exponer `predict(row_df) -> (label, proba, df_transformed)` con el pipeline
      completo: OrdinalEncoder -> KBins -> StandardScaler -> VAE -> CatBoost.
    - Exponer metadatos para la UI: num_cols, cat_cols, all_feats, col_stats,
      categories, threshold.
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Ruta del modelo (pathlib) ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "exp05_vae_catboost.pkl"
TRAIN_SCRIPT = BASE_DIR / "scripts" / "export_exp05_model.py"


# ══════════════════════════════════════════════════════════════════════════════
# Arquitectura VAE (debe coincidir con export_exp05_model.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# MLEngine - clase singleton
# ══════════════════════════════════════════════════════════════════════════════

class MLEngine:
    """
    Carga y encapsula el modelo Exp_05 (VAE + CatBoost) junto con
    todos sus artefactos de preprocesado.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self._model_path = model_path
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    # ── Carga ─────────────────────────────────────────────────────────────────

    def _load(self):
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"\nModelo Exp_05 no encontrado: {self._model_path}\n\n"
                "Para generarlo, ejecuta desde la raiz del proyecto:\n\n"
                f"    python {TRAIN_SCRIPT.relative_to(BASE_DIR)}\n\n"
                "El proceso tarda ~40 s en CPU."
            )

        print(f"[ml_engine] Cargando {self._model_path.name} ...")
        pkl = joblib.load(self._model_path)

        # Artefactos de preprocesado
        self.oe        = pkl["oe"]
        self.kbd       = pkl["kbd"]
        self.scaler    = pkl["scaler"]
        self.high_var  = pkl["high_var"]
        self.num_cols  = pkl["num_cols"]
        self.cat_cols  = pkl["cat_cols"]
        self.all_feats = pkl["all_feats"]
        self.threshold = float(pkl["threshold"])
        self.latent_dim= int(pkl["latent_dim"])

        # Metadatos para la UI
        self.col_stats  = pkl["col_stats"]
        self.categories = pkl["categories"]

        # Reconstrucción del VAE desde bytes
        vae_input_dim = int(pkl["vae_input_dim"])
        self._vae = VAE(
            input_dim=vae_input_dim,
            latent_dim=self.latent_dim,
        ).to(self._device)
        buf = io.BytesIO(pkl["vae_bytes"])
        self._vae.load_state_dict(torch.load(buf, map_location=self._device))
        self._vae.eval()

        # Modelo CatBoost
        self._model = pkl["model"]

        print(f"[ml_engine] OK  Cargado | threshold={self.threshold:.4f} "
              f"| features={len(self.all_feats)}")

    # ── Pipeline de transformación ────────────────────────────────────────────

    def _transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el pipeline de FE exactamente igual que en export_exp05_model.py,
        pero en modo transform (sin ajuste de artefactos).
        """
        X = X_raw.copy()

        # 1. Imputación numérica
        X[self.num_cols] = X[self.num_cols].fillna(0)

        # 2. Encoding categórico
        if self.cat_cols and self.oe is not None:
            X[self.cat_cols] = self.oe.transform(
                X[self.cat_cols].fillna("missing").astype(str)
            )

        # 3. Binning
        X_bin = self.kbd.transform(X[self.high_var])
        for i, col in enumerate(self.high_var):
            X[f"{col}_bin"] = X_bin[:, i].astype(int)

        # 4. VAE features
        X_sc = self.scaler.transform(X[self.num_cols])
        tensor = torch.tensor(X_sc, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            recon, mu, _ = self._vae(tensor)
            err = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
            lat = mu.cpu().numpy()

        X["vae_err"] = err
        for i in range(lat.shape[1]):
            X[f"vae_l{i}"] = lat[:, i]

        # Reordenar al orden estricto del entrenamiento
        return X[self.all_feats]

    # ── Predicción pública ────────────────────────────────────────────────────

    def predict(self, row: dict | pd.DataFrame) -> tuple[str, float, pd.DataFrame]:
        """
        Recibe un dict o DataFrame de 1 fila con las variables originales
        (sin ingeniería), aplica el pipeline y devuelve:

            (label, proba_nok, df_transformed)

        donde `df_transformed` tiene las columnas de `all_feats` para SHAP.
        """
        if isinstance(row, dict):
            df_raw = pd.DataFrame([row])
        else:
            df_raw = row.copy()

        # Alinear columnas originales (sin features de ingeniería)
        orig_cols = self.num_cols + self.cat_cols
        for col in orig_cols:
            if col not in df_raw.columns:
                df_raw[col] = np.nan

        df_fe = self._transform(df_raw[orig_cols])

        proba = float(self._model.predict_proba(df_fe)[0, 1])
        label = "NOK" if proba >= self.threshold else "OK"

        return label, proba, df_fe

    # ── Metadatos para la UI ──────────────────────────────────────────────────

    @property
    def orig_feats(self) -> list[str]:
        """Columnas originales (sin features de ingeniería VAE/_bin)."""
        return self.num_cols + self.cat_cols


# ── Instancia global (cargada una sola vez al importar el módulo) ─────────────
engine: MLEngine | None = None


def get_engine() -> MLEngine:
    """Devuelve la instancia singleton del MLEngine."""
    global engine
    if engine is None:
        engine = MLEngine()
    return engine

