"""
utils/explainer.py
==================
Modulo de explicabilidad SHAP para el modelo CatBoost del Exp_05.
Version 2.2 - Implementacion de Seccion 5 GEMINI.md:

    5.1  Agregacion algebraica: suma de variables derivadas (_bin) a su base fisica.
    5.2  Filtro de relevancia: descarte relativo al 15% del Top-1 (minimo 3 vars).
    5.3  Balanceo dinamico Real/VAE: min=3, max=10, siempre Reales >= VAE.
    5.4  Renombrado industrial: vae_err -> "Indice de Correlacion Global",
         latent_N -> "Patron Estructural N" (N = numero de dimension latente).
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap

if TYPE_CHECKING:
    from utils.ml_engine import MLEngine

warnings.filterwarnings("ignore")


# ── Nombres industriales ──────────────────────────────────────────────────────
# Sec 5.4 GEMINI.md v2.2:
#   vae_err      -> "Indice de Correlacion Global"
#   latent_N     -> "Patron Estructural N"  (N = numero de dimension latente)

_INDUSTRIAL_NAMES: dict[str, str] = {
    "vae_err": "Indice de Correlacion Global",
}

def _industrial_name(feat: str) -> str:
    """
    Traduce un nombre de feature al nombre industrial estandarizado.

    Casos:
        vae_err     -> "Indice de Correlacion Global"
        latent_5    -> "Patron Estructural 5"
        latent_11   -> "Patron Estructural 11"
        <cualquier otro>  -> nombre original sin modificacion
    """
    if feat in _INDUSTRIAL_NAMES:
        return _INDUSTRIAL_NAMES[feat]
    if feat.startswith("latent_"):
        # Extraer el numero de dimension (ej. "latent_5" -> "5")
        suffix = feat[len("latent_"):]
        n = suffix if suffix.isdigit() else suffix
        return f"Patron Estructural {n}"
    return feat


# ── Clasificacion de variables ────────────────────────────────────────────────

_VAE_PREFIXES = ("vae_err", "latent_")

def _is_vae(feat: str) -> bool:
    """True si la variable es sintetica VAE (vae_err o latent_*)."""
    return feat.startswith(_VAE_PREFIXES)


def _base_name(feat: str) -> str:
    """
    Extrae el nombre base fisico de una feature derivada.
    Ejemplo: 'temperatura_bin'  ->  'temperatura'
             'temperatura'      ->  'temperatura'
    Variables VAE no tienen base fisica: se devuelven tal cual.
    """
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
    Wrapper de `shap.TreeExplainer` para el modelo CatBoost del Exp_05.
    Implementa la logica completa de la Seccion 5 de GEMINI.md v2.2.
    """

    def __init__(self, ml_engine: "MLEngine"):
        print("[explainer] Inicializando SHAP TreeExplainer ...")
        self._explainer = shap.TreeExplainer(ml_engine._model)
        self._all_feats = ml_engine.all_feats
        print("[explainer] OK  Listo.")

    # ── API publica ───────────────────────────────────────────────────────────

    def explain(
        self,
        df_transformed: pd.DataFrame,
        top_k_range: tuple[int, int] = (3, 10),
    ) -> list[tuple[str, float, float]]:
        """
        Calcula los valores SHAP para una fila ya transformada y devuelve la
        tabla procesada segun la Seccion 5 de GEMINI.md:

        1. Calcula SHAP crudos para la clase NOK.
        2. Agrega algebraicamente variables derivadas (_bin) a su base fisica.
        3. Aplica el filtro de caida relativa del 15% (minimo 3 vars).
        4. Balancea Real >= VAE (min 3, max 10) buscando en ranking descendente.
        5. Renombra segun nomenclatura industrial.

        Args:
            df_transformed : DataFrame de 1 fila con `all_feats` columnas.
            top_k_range    : (min_k, max_k) para las guardas del algoritmo.

        Returns:
            Lista de tuplas (nombre_industrial, valor_original, shap_neto)
            ordenada de mayor a menor impacto absoluto neto.
        """
        min_k, max_k = top_k_range

        # ── 1. SHAP crudos ────────────────────────────────────────────────────
        sv = self._explainer.shap_values(df_transformed)
        if isinstance(sv, list):
            raw_vals = np.array(sv[1])[0]
        else:
            raw_vals = np.array(sv)[0]

        feat_arr = np.array(self._all_feats)

        # ── 2. Agregacion algebraica (Sec 5.1) ────────────────────────────────
        # Construimos un dict: base_name -> (shap_neto, valor_fisico)
        aggregated: dict[str, list[float]] = {}   # base -> [shap_acumulado]
        feat_values: dict[str, float] = {}         # base -> valor de la fila

        for i, feat in enumerate(feat_arr):
            base = _base_name(feat)
            shap_val = float(raw_vals[i])

            if base not in aggregated:
                aggregated[base] = 0.0
                # Valor: usamos la feature base si existe, si no la derivada
                if base in df_transformed.columns:
                    feat_values[base] = float(
                        df_transformed.iloc[0][base]
                    )
                else:
                    feat_values[base] = float(
                        df_transformed.iloc[0, i]
                    )

            # Suma algebraica estricta (puede cambiar de signo => impacto neto)
            aggregated[base] = aggregated[base] + shap_val  # type: ignore[assignment]

        # Convertir a lista ordenada por |shap_neto| descendente
        ranked: list[tuple[str, float, float]] = sorted(
            [(b, feat_values[b], aggregated[b]) for b in aggregated],
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        # ── 3. Filtro de caida relativa del 15% (Sec 5.2) ────────────────────
        if ranked:
            top1_abs = abs(ranked[0][2])
            threshold_15 = 0.15 * top1_abs if top1_abs > 0 else 0.0

            filtered: list[tuple[str, float, float]] = []
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
        ranked: list[tuple[str, float, float]],
        min_k: int,
        max_k: int,
    ) -> list[tuple[str, float, float]]:
        """
        Sec 5.3 - Balanceo e Integridad:

        Prioridad 1 (Minimo): asegurar min_k variables priorizando mayor impacto.
        Prioridad 2 (Balanceo): n_reales >= n_vae; si domina VAE, escanear ranking
                                buscando reales adicionales.
        Gestion de Escasez: si se agotan las reales, conservar las encontradas.
        Prioridad 3 (Maximo): nunca superar max_k filas.
        """
        if not ranked:
            return []

        # Tomamos el top inicial: hasta max_k elementos del ranking
        pool = ranked[: max_k]

        # Garantizar minimo
        if len(pool) < min_k:
            pool = ranked[: min_k]  # puede ser menor si el ranking es corto

        # Contar reales y VAE en el pool inicial
        n_real = sum(1 for b, _, __ in pool if not _is_vae(b))
        n_vae  = sum(1 for b, _, __ in pool if _is_vae(b))

        # Prioridad 2: si VAE > real, buscar mas reales en el ranking completo
        if n_vae > n_real:
            # Indices ya en pool
            pool_names = {b for b, _, __ in pool}
            candidates = [item for item in ranked if not _is_vae(item[0]) and item[0] not in pool_names]

            for candidate in candidates:
                if n_real >= n_vae:
                    break
                if len(pool) >= max_k:
                    break
                pool.append(candidate)
                n_real += 1

            # Re-ordenar por impacto absoluto descendente
            pool.sort(key=lambda x: abs(x[2]), reverse=True)

        # Aplicar maximo final
        pool = pool[: max_k]

        # Aplicar minimo final (en caso de que el ranking original sea muy corto)
        return pool

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
