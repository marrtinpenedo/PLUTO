import io
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Ruta del modelo (pathlib) ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "exp05_vae_catboost_v2.pkl" # Actualizado al v2
TRAIN_SCRIPT = BASE_DIR / "scripts" / "export_exp05_model_v2.py"


# ══════════════════════════════════════════════════════════════════════════════
# Arquitectura VAE (Idéntica al entrenamiento)
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
# MLEngine
# ══════════════════════════════════════════════════════════════════════════════
class MLEngine:
    """
    Motor alineado con VAE + CatBoost Nativo.
    NO usa OrdinalEncoder ni KBins. Pasa categóricas como strings.
    """
    def __init__(self, model_path: Path = MODEL_PATH):
        self._model_path = model_path
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"\nModelo no encontrado: {self._model_path}\n"
            )

        print(f"[ml_engine] Cargando {self._model_path.name} ...")
        pkl = joblib.load(self._model_path)

        # 1. Cargar artefactos (eliminados oe, kbd y high_var)
        self.scaler         = pkl["scaler"]
        self.num_cols       = pkl["num_cols"]
        self.cat_cols       = pkl["cat_cols"]
        self.all_feats      = pkl["all_feats"]
        self.threshold      = float(pkl["threshold"])
        self.latent_dim     = int(pkl["latent_dim"])
        self._train_medians = pkl["train_medians"] 
        self.col_stats      = pkl["col_stats"]
        self.categories     = pkl["categories"]

        # 2. Reconstruir VAE
        vae_input_dim = int(pkl["vae_input_dim"])
        self._vae = VAE(input_dim=vae_input_dim, latent_dim=self.latent_dim).to(self._device)
        buf = io.BytesIO(pkl["vae_bytes"])
        self._vae.load_state_dict(torch.load(buf, map_location=self._device))
        self._vae.eval()

        # 3. Modelo principal
        self._model = pkl["model"]

        print(f"[ml_engine] OK Cargado | threshold={self.threshold:.4f} | features={len(self.all_feats)}")

    def _transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Simetría estricta con apply_fe() del entrenamiento.
        """
        X = X_raw.copy()

        # 1. Imputación nativa
        medians_aligned = self._train_medians.reindex(self.num_cols)
        X[self.num_cols] = X[self.num_cols].fillna(medians_aligned)
        
        # Categóricas se rellenan y se fuerzan a string (nativas para CatBoost)
        X[self.cat_cols] = X[self.cat_cols].fillna("missing").astype(str)

        # 2. Escalado numérico
        X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        # 3. Features VAE
        tensor = torch.tensor(X[self.num_cols].values, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            recon, mu, _ = self._vae(tensor)
            err = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
            lat = mu.cpu().numpy()

        X["vae_err"] = err
        for i in range(lat.shape[1]):
            X[f"vae_l{i}"] = lat[:, i]

        # 4. Retornar en el orden exacto del entrenamiento
        return X[self.all_feats]

    def predict(self, row: Union[dict, pd.DataFrame]) -> tuple:
        if isinstance(row, dict):
            df_raw = pd.DataFrame([row])
        else:
            df_raw = row.copy()

        orig_cols = self.num_cols + self.cat_cols
        for col in orig_cols:
            if col not in df_raw.columns:
                df_raw[col] = np.nan

        df_fe = self._transform(df_raw[orig_cols])

        proba = float(self._model.predict_proba(df_fe)[0, 1])
        label = "NOK" if proba >= self.threshold else "OK"

        return label, proba, df_fe

    @property
    def orig_feats(self) -> list:
        return self.num_cols + self.cat_cols

_engine: Optional[MLEngine] = None

def get_engine() -> MLEngine:
    global _engine
    if _engine is None:
        _engine = MLEngine()
    return _engine