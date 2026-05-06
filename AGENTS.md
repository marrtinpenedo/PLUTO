# AGENTS.md — PLUTO (CTAG Quality Inspection System)

## Setup and startup

```bash
conda activate pluto
pip install -r requirements.txt

# Generate the model artifact ONLY if models/exp05_vae_catboost.pkl does not exist
# Must be run from the project root. Takes ~40 s on CPU.
python scripts/export_exp05_model.py

# Start the UI (http://localhost:7860)
python main.py

# Ollama must be running in a separate terminal (required for the LLM chatbot)
ollama pull llama3
ollama serve
```

No test runner, linter, or CI. No `setup.py` or `pyproject.toml`.

---

## Actual project structure

```
main.py                          # Entry point: calls build_app() and configures launch()
app/ui.py                        # All Gradio UI: layout, handlers, state
utils/ml_engine.py               # Singleton MLEngine: loads pkl, exposes predict()
utils/explainer.py               # Singleton SHAPExplainer: full SHAP pipeline
utils/llm_client.py              # stream_response(), build_prompt(), check_ollama()
scripts/export_exp05_model.py    # One-shot script: trains and serializes the pkl
models/exp05_vae_catboost.pkl    # Production artifact (gitignored but with exception)
data/raw/Dataset_01_Anonimizado.xlsx  # Only dataset; target column = "Variable de Salida"
src/models/                      # Research scripts, NOT imported by the app
```

`src/models/` contains comparative experiments (Exp_01..11) and a runner.
None of these files are imported by the application.

---

## Execution flow

```
main.py
  └─ build_app()  [app/ui.py]
       ├─ get_engine()      → MLEngine  (loads pkl once on import)
       ├─ get_explainer()   → SHAPExplainer  (initializes TreeExplainer on CatBoost)
       └─ check_ollama()    → Online/Offline badge in header

  on_predict(values):
       eng.predict(row)          → (label, proba, df_fe)  [123 features]
       explainer.explain(df_fe)  → top_k list[(industrial_name, value, net_shap)]
       → HTML banner + slider + SHAP DataFrame

  on_chat(user_msg, history, state):
       build_prompt(label, proba, threshold, top_k, user_msg)
       stream_response(prompt)   → streams tokens via Ollama /api/generate
```

---

## pkl artifact — exact contents

`joblib.load()` returns a dict with these keys (verified in `ml_engine.py` and `export_exp05_model.py`):

| Key | Type | Usage |
|---|---|---|
| `model` | CatBoostClassifier | Classification model |
| `vae_bytes` | bytes | VAE state_dict serialized with `torch.save` into BytesIO |
| `vae_input_dim` | int | Input dimension to reconstruct the VAE architecture |
| `scaler` | StandardScaler | Fitted only on num_cols from the train pool |
| `kbd` | KBinsDiscretizer | Fitted on `high_var` (top-10 columns by variance) |
| `oe` | OrdinalEncoder | For categorical columns |
| `high_var` | list[str] | The 10 numeric columns with highest variance |
| `num_cols` | list[str] | Original numeric columns |
| `cat_cols` | list[str] | Original categorical columns |
| `all_feats` | list[str] | Exact order of the 123 columns fed into the model |
| `threshold` | float | OOF-calibrated threshold (production value: 0.6818) |
| `latent_dim` | int | VAE latent dimension (value: 12) |
| `col_stats` | dict | `{col: {min, max, mean}}` for UI input ranges |
| `categories` | dict | `{col: [values]}` for UI dropdowns |

---

## Critical domain constants

These constants appear in the code — do not change them without updating both indicated files:

| Constant | Value | File(s) |
|---|---|---|
| NOK threshold | **0.6818** | `utils/llm_client.py → NOK_THRESHOLD`; also used in `ml_engine.predict` via `self.threshold` from the pkl |
| LATENT_DIM | **12** | `scripts/export_exp05_model.py`; must match the pkl |
| VAE_EPOCHS | **15** | `scripts/export_exp05_model.py` |
| SHAP 15% filter | `0.15 * abs(top1)` | `utils/explainer.py → explain()` |
| SHAP table range | min=3, max=10 | `utils/explainer.py → explain(top_k_range=(3,10))` |
| SECTION_SZ | 20 vars/accordion | `app/ui.py` |
| GROUP_SIZE | 4 inputs/row | `app/ui.py` |

---

## VAE architecture — sync trap

The `VAE(nn.Module)` class is **defined twice**, copied identically:
- `utils/ml_engine.py` (to load the pkl at inference time)
- `scripts/export_exp05_model.py` (for training)

Modifying one without modifying the other breaks `state_dict` loading. The architecture is:
`Linear(in,64) → Linear(64,32) → [fc_mu, fc_logvar](32,latent_dim)` in the encoder;
`Linear(latent_dim,32) → Linear(32,64) → Linear(64,in)` in the decoder. Activations: ReLU.

---

## Feature engineering pipeline (102 → 123 variables)

```
Input: 102 original variables (num_cols + cat_cols)
  1. Numeric imputation: fillna(0) for num_cols
  2. OrdinalEncoder on cat_cols (handle_unknown="use_encoded_value", unknown_value=-1)
  3. KBinsDiscretizer (n_bins=4, quantile) on high_var → adds <col>_bin columns
  4. StandardScaler on num_cols → fed into VAE
  5. VAE encoder → vae_err (reconstruction MSE) + vae_l0..vae_l11 (12 latents)
  6. Reorder according to all_feats → 123 columns → CatBoost
Output: (label, proba, df_fe)  where df_fe has exactly all_feats columns
```

VAE columns are named `vae_lN` (e.g. `vae_l0`, `vae_l11`), **not** `latent_N`.

---

## SHAP: post-processing logic in explainer.py

The `explain(df_transformed)` pipeline applies in order:

1. **Raw SHAP**: `TreeExplainer.shap_values(df_fe)` → values for class 1 (NOK).
   - If it returns a list: `sv[1][0]`. If it returns an array: `sv[0]`.

2. **Algebraic aggregation**: `<sensor>_bin` columns are summed into the SHAP value of `<sensor>`.
   - `_base_name()` handles the mapping: strips the `_bin` suffix. VAE features have no physical base.
   - If you extend with new derived suffixes, update `_base_name()`.

3. **15% filter**: discards features with `|net_shap| < 0.15 * |top1|`, but never drops below 3.

4. **Real/VAE balancing**:
   - Initial pool: top `max_k` (10) from the ranking.
   - If `n_vae > n_real`: scans the ranking looking for additional real features until balanced or exhausted.
   - No duplicates or invented features; if there are not enough real features, keeps what was found.
   - Final maximum: 10 rows.

5. **Industrial renaming**:
   - `vae_err` → `"Indice de Correlacion Global"`
   - `vae_lN` → `"Patron Estructural N"`
   - Everything else: original sensor name.

**Sign semantics**: `SHAP > 0` pushes toward NOK (defect); `SHAP < 0` toward OK (quality).
This is the opposite of common intuition — do not invert it.

---

## LLM — Ollama

- Endpoint: `http://localhost:11434/api/generate`
- Configurable model: `LLM_MODEL = "llama3"` in `utils/llm_client.py`
- Streaming timeout: 90 s (do not increase; switch to a smaller model instead)
- `check_ollama()` does GET to `/api/tags` with timeout=3 s on each chat submission
- The payload includes a `"system"` field with the System Prompt and `"stream": True`
- The LLM **never receives** technical names (`vae_l*`, `_bin`): the table is already renamed by `explainer.py`

---

## UI — non-negotiable rules (verified in ui.py)

- **Zero emojis** and **zero text tags** (`[PLUTO]`, `[wait]`, `[i]`, etc.) anywhere in the UI.
- `result_slider` has `interactive=False` — it is an output indicator, not an input.
- The slider displays the threshold in the label: `f"NOK Probability  (threshold = {eng.threshold:.4f})"`.
- Banners: CSS class `.ok-banner` (green `#34d399`) and `.nok-banner` (red `#f87171`).
- Global background: `#0c1629`. Primary text: `#f8fafc`.
- Form inputs use `elem_id=f"inp_{feat.replace(' ', '_')}"`.

---

## File autofill — matching logic (CSV and Excel)

`on_csv_upload()` in `ui.py` accepts `.csv`, `.xlsx`, and `.xls`.
Extension detection is done with `Path(file_obj.name).suffix.lower()`:
- `.xlsx` / `.xls` → `pd.read_excel(path, nrows=1)`
- anything else → `pd.read_csv(path, nrows=1)`

Headers are normalized (strip + lowercase) before comparing against `eng.orig_feats`.
Missing columns are imputed with the numeric mean or the first category.
`openpyxl>=3.1,<4` is already in `requirements.txt` — no new dependency needed.

---

## gitignore — important exception

```
models/*.pkl          # ignores all pkl files
!models/exp05_vae_catboost.pkl   # except the production artifact
```

Do not remove that exception or the pkl will disappear from the repository.

---

## Key dependencies (from requirements.txt)

```
gradio==5.50.0         # exact version
catboost>=1.2,<1.3     # narrow range — do not update without retraining
shap>=0.45,<0.49
torch>=2.2,<2.7
scikit-learn>=1.6,<1.9
```