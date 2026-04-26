"""
utils/explainer.py
==================
Módulo de explicabilidad SHAP para el modelo CatBoost del Exp_05.

Responsabilidades:
    - Inicializar `shap.TreeExplainer` sobre el modelo CatBoost cargado.
    - Exponer `explain(df_transformed, top_k_range=(3, 5)) -> list[tuple]`
      con selección dinámica del número de variables más influyentes.

El top-K dinámico selecciona entre 3 y 5 variables según la concentración
de la masa SHAP: si las 3 primeras acumulan más del 80% del impacto total,
se devuelven 3; si no, se devuelven 5.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap

if TYPE_CHECKING:
    from utils.ml_engine import MLEngine

warnings.filterwarnings("ignore")


class SHAPExplainer:
    """
    Wrapper de `shap.TreeExplainer` para el modelo CatBoost del Exp_05.
    """

    def __init__(self, ml_engine: "MLEngine"):
        print("[explainer] Inicializando SHAP TreeExplainer ...")
        self._explainer = shap.TreeExplainer(ml_engine._model)
        self._all_feats = ml_engine.all_feats
        print("[explainer] OK  Listo.")

    # ── API pública ───────────────────────────────────────────────────────────

    def explain(
        self,
        df_transformed: pd.DataFrame,
        top_k_range: tuple[int, int] = (3, 5),
    ) -> list[tuple[str, float, float]]:
        """
        Calcula los valores SHAP para una fila ya transformada y devuelve
        la lista de las top-K variables más influyentes.

        Args:
            df_transformed : DataFrame de 1 fila con `all_feats` columnas.
            top_k_range    : (min_k, max_k) para la selección dinámica.

        Returns:
            Lista de tuplas (feature_name, feature_value, shap_value)
            ordenada de mayor a menor impacto absoluto.
        """
        min_k, max_k = top_k_range

        sv = self._explainer.shap_values(df_transformed)

        # TreeExplainer con CatBoost devuelve:
        #   - lista [shap_class0, shap_class1]  ->  tomamos clase 1 (NOK)
        #   - o directamente un array 2D
        if isinstance(sv, list):
            vals = np.array(sv[1])[0]
        else:
            vals = np.array(sv)[0]

        feat_arr  = np.array(self._all_feats)
        abs_vals  = np.abs(vals)
        sorted_idx = np.argsort(abs_vals)[::-1]

        # Top-K dinámico
        total_mass = abs_vals.sum()
        if total_mass > 0:
            cum_mass = np.cumsum(abs_vals[sorted_idx])
            # Si los primeros min_k acumulan >= 80 % -> devolver min_k
            if cum_mass[min_k - 1] / total_mass >= 0.80:
                k = min_k
            else:
                k = max_k
        else:
            k = min_k

        top_idx = sorted_idx[:k]

        return [
            (
                str(feat_arr[i]),
                float(df_transformed.iloc[0, df_transformed.columns.get_loc(feat_arr[i])]),
                float(vals[i]),
            )
            for i in top_idx
        ]

    def shap_values_raw(self, df_transformed: pd.DataFrame) -> np.ndarray:
        """Devuelve los valores SHAP crudos (array 1D) para clase NOK."""
        sv = self._explainer.shap_values(df_transformed)
        if isinstance(sv, list):
            return np.array(sv[1])[0]
        return np.array(sv)[0]


# ── Instancia global ──────────────────────────────────────────────────────────
_explainer_instance: SHAPExplainer | None = None


def get_explainer(ml_engine: "MLEngine | None" = None) -> SHAPExplainer:
    """
    Devuelve la instancia singleton del SHAPExplainer.
    Si no existe, la crea usando `ml_engine`.
    """
    global _explainer_instance
    if _explainer_instance is None:
        if ml_engine is None:
            from utils.ml_engine import get_engine
            ml_engine = get_engine()
        _explainer_instance = SHAPExplainer(ml_engine)
    return _explainer_instance

