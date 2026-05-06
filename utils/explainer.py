"""
utils/explainer.py
==================
Modulo de explicabilidad SHAP para el modelo CatBoost del Exp_05.
Version 2.2 

    5.1  Agregacion algebraica: suma de variables derivadas (_bin) a su base fisica.
    5.2  Filtro de relevancia: descarte relativo al 15% del Top-1 (minimo 3 vars).
    5.3  Balanceo dinamico Real/VAE: min=3, max=10, siempre Reales >= VAE.
    5.4  Renombrado industrial: vae_err -> "Indice de Correlacion Global",
         vae_lN -> "Patron Estructural N" (N = numero de dimension latente).
"""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import catboost as cb  # <-- Usamos CatBoost nativo en lugar de la libreria shap externa

if TYPE_CHECKING:
    from utils.ml_engine import MLEngine

warnings.filterwarnings("ignore")


# ── Nombres industriales ──────────────────────────────────────────────────────
#   vae_err      -> "Indice de Correlacion Global"
#   vae_lN     -> "Patron Estructural N"  (N = numero de dimension latente)

_INDUSTRIAL_NAMES: dict[str, str] = {
    "vae_err": "Indice de Correlacion Global",
}

def _industrial_name(feat: str) -> str:
    if feat in _INDUSTRIAL_NAMES:
        return _INDUSTRIAL_NAMES[feat]
    if feat.startswith("vae_l"):
        suffix = feat[len("vae_l"):]
        n = suffix if suffix.isdigit() else suffix
        return f"Patron Estructural {n}"
    return feat


# ── Clasificacion de variables ────────────────────────────────────────────────

_VAE_PREFIXES = ("vae_err", "vae_l")

def _is_vae(feat: str) -> bool:
    return feat.startswith(_VAE_PREFIXES)

def _base_name(feat: str) -> str:
    if _is_vae(feat):
        return feat
    if feat.endswith("_bin"):
        return feat[:-4]
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# Clase principal
# ══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    Wrapper para el motor SHAP interno de CatBoost del Exp_05.
    Soporta variables categoricas nativas (strings).
    """

    def __init__(self, ml_engine: "MLEngine"):
        print("[explainer] Inicializando SHAP (CatBoost Nativo) ...")
        # Guardamos la referencia al motor para poder acceder a cat_cols y _model
        self._engine = ml_engine
        self._all_feats = ml_engine.all_feats
        print("[explainer] OK  Listo.")

    # ── API publica ───────────────────────────────────────────────────────────

    def explain(
        self,
        df_transformed: pd.DataFrame,
        top_k_range: tuple[int, int] = (3, 10),
    ) -> list[tuple[str, Any, float]]:
        min_k, max_k = top_k_range

        # ── 1. SHAP crudos (Via CatBoost Nativo) ──────────────────────────────
        # Construimos un Pool avisando a CatBoost de cuales son los strings
        pool = cb.Pool(df_transformed, cat_features=self._engine.cat_cols)
        shap_matrix = self._engine._model.get_feature_importance(pool, type='ShapValues')
        
        # La matriz devuelve (1, n_features + 1). La ultima columna es el base_value.
        # Excluimos el base_value para alinear con las features
        raw_vals = np.array(shap_matrix[0, :-1])

        feat_arr = np.array(self._all_feats)

        # ── 2. Agregacion algebraica (Sec 5.1) ────────────────────────────────
        aggregated: dict[str, float] = {}   
        feat_values: dict[str, Any] = {}         

        for i, feat in enumerate(feat_arr):
            base = _base_name(feat)
            shap_val = float(raw_vals[i])

            if base not in aggregated:
                aggregated[base] = 0.0
                # Extraemos el valor sin forzar a float() para que los strings no rompan
                if base in df_transformed.columns:
                    feat_values[base] = df_transformed.iloc[0][base]
                else:
                    feat_values[base] = df_transformed.iloc[0, i]

            aggregated[base] = aggregated[base] + shap_val

        ranked: list[tuple[str, Any, float]] = sorted(
            [(b, feat_values[b], aggregated[b]) for b in aggregated],
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        # ── 3. Filtro de caida relativa del 15% (Sec 5.2) ────────────────────
        if ranked:
            top1_abs = abs(ranked[0][2])
            threshold_15 = 0.15 * top1_abs if top1_abs > 0 else 0.0

            filtered: list[tuple[str, Any, float]] = []
            for item in ranked:
                if abs(item[2]) >= threshold_15 or len(filtered) < min_k:
                    filtered.append(item)
            ranked = filtered

        # ── 4. Balanceo dinamico Real/VAE (Sec 5.3) ──────────────────────────
        result = self._balance(ranked, min_k, max_k)

        # ── 5. Renombrado industrial (Sec 5.4) ───────────────────────────────
        result_named = [
            (_industrial_name(base), val, shap_net)
            for base, val, shap_net in result
        ]

        return result_named

    # ── Algoritmo de balanceo ─────────────────────────────────────────────────

    def _balance(
        self,
        ranked: list[tuple[str, Any, float]],
        min_k: int,
        max_k: int,
    ) -> list[tuple[str, Any, float]]:
        if not ranked:
            return []

        pool = ranked[: max_k]

        if len(pool) < min_k:
            pool = ranked[: min_k]  

        n_real = sum(1 for b, _, __ in pool if not _is_vae(b))
        n_vae  = sum(1 for b, _, __ in pool if _is_vae(b))

        if n_vae > n_real:
            pool_names = {b for b, _, __ in pool}
            candidates = [item for item in ranked if not _is_vae(item[0]) and item[0] not in pool_names]

            for candidate in candidates:
                if n_real >= n_vae:
                    break
                if len(pool) >= max_k:
                    break
                pool.append(candidate)
                n_real += 1

            pool.sort(key=lambda x: abs(x[2]), reverse=True)

        pool = pool[: max_k]
        return pool

    def shap_values_raw(self, df_transformed: pd.DataFrame) -> np.ndarray:
        """Devuelve los valores SHAP crudos (array 1D) para clase NOK."""
        pool = cb.Pool(df_transformed, cat_features=self._engine.cat_cols)
        shap_matrix = self._engine._model.get_feature_importance(pool, type='ShapValues')
        return np.array(shap_matrix[0, :-1])


# ── Instancia global ──────────────────────────────────────────────────────────
_explainer_instance: SHAPExplainer | None = None


def get_explainer(ml_engine: "MLEngine | None" = None) -> SHAPExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        if ml_engine is None:
            from utils.ml_engine import get_engine
            ml_engine = get_engine()
        _explainer_instance = SHAPExplainer(ml_engine)
    return _explainer_instance