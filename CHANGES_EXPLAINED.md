# Changes Explained

This document summarizes the files currently changed in the working tree. The changes fall into two groups: source-code improvements and regenerated model/data artifacts.

## 1. Source-code changes

### Import paths and package execution

The archived Python scripts now import one another through the `archive` package. This makes imports work when scripts are run from the repository root and avoids relying on the current working directory.

Updated files:

- `archive/continuous_learning_complete.py`
- `archive/fix_pickle.py`
- `archive/phase2_predictive_model.py`
- `archive/phase3_causal_modeling.py`
- `archive/phase4_continuous_learning.py`
- `archive/phase_c_fixes.py`
- `archive/phase_d_multi_model.py`

The affected imports now reference modules such as `archive.softpower_models`, `archive.phase2_predictive_model`, and `archive.kalman_softpower_complete`.

### Kalman filtering

`archive/kalman_softpower_complete.py` now allows callers to provide explicit `Q` and `R` noise parameters. When they are not supplied, the existing heuristic values are still used.

The normal forecasting and all-country processing paths now fit country-specific `Q` and `R` values with `optimize_kalman_params()` before running the forward Kalman filter. Previously, fitted values were used only by the smoother after the filter had already used the fixed heuristic. This reduces autocorrelation in the filter innovations and makes the diagnostic output more trustworthy.

### Prediction intervals

`archive/softpower_models.py` now uses a time-ordered calibration split inside temporal cross-validation:

- The earlier portion of training data fits the model.
- The most recent training slice calibrates residual error.
- The held-out test period remains strictly later than both.

The interval half-width is based on an empirical conformal residual quantile with a finite-sample correction. This replaces the models' previously miscalibrated individual interval estimates while preserving their point-prediction interface.

### Collinear feature handling

`archive/phase_c_complete.py` now excludes redundant subcomponents from the model feature set when their composite feature is already present. The excluded columns are:

- `fh_cl_score`
- `fh_pr_score`
- `fh_cl_rating`
- `fh_pr_rating`
- `unesco_cultural_sites`
- `unesco_natural_sites`
- `unesco_mixed_sites`

The raw columns remain in the master panel for dashboard display. They are removed only from model training and feature-importance generation so attribution is not arbitrarily split across duplicate information.

### Dashboard peer similarity

`dash/backend/app/file_data.py` now prefers `output/artifacts/embedding_matrix.npy` when calculating peer vectors. This keeps dashboard cosine similarity in the same normalized embedding space used by the model and FAISS index. It falls back to vectors from the parquet file if the artifact is missing or unusable.

Similarity values are also clamped to the valid range of `[-1, 1]`.

`dash/frontend/src/components/PeerPanel.jsx` now displays similarity percentages to one decimal place instead of rounding to whole percentages.

## 2. Model and data outputs regenerated

The following tracked files changed because the modeling pipeline was rerun after the source updates:

### Model binaries and embeddings

- `output/artifacts/embedding_matrix.npy`
- `output/artifacts/embedding_scaler.pkl`
- `output/artifacts/faiss_index.bin`
- `output/artifacts/xgb_model.pkl`
- `output/best_model.pkl`
- `output/ensemble_model.pkl`
- `output/country_embeddings.parquet`
- `output/ensemble_predictions.parquet`
- `output/forecasts.parquet`
- `output/soft_power_predictions.parquet`
- `output/trend_features.parquet`

### Model metrics and feature importance

- `output/best_model_name.txt`
- `output/cv_results.json`
- `output/multi_model_cv_results.json`
- `output/model_comparison.csv`
- `output/combined_feature_importance.csv`
- `output/feature_importance.csv`
- `output/lgbm_feature_importance.csv`
- `output/rf_feature_importance.csv`
- `output/xgb_feature_importance.csv`

### Predictions, forecasts, and Kalman results

- `output/delta_forecast_fixed.csv`
- `output/ensemble_predictions.csv`
- `output/soft_power_predictions.csv`
- `output/kalman_forecast_5yr.csv`
- `output/kalman_regimes.csv`
- `output/kalman_results.csv`
- `output/kalman_summary.csv`

These files are generated results, not independent hand-written changes. Their values, model serialization, row ordering, and binary contents can change when the updated pipeline is rerun.

## 3. Trust-analysis files added

The following files are untracked additions:

- `model_trust_suite.py`: runs diagnostic checks for cross-validation metrics, target-derived feature importance, naive baselines, generalization, multicollinearity, Kalman innovations, interval calibration, and ranking robustness. It writes a fresh report to `trust_report.md`.
- `MODEL_TRUST_GUIDE.md`: explains what each trust check measures and how to interpret the results in the capstone documentation.
- `trust_report.md`: generated report from the current output artifacts.

The current report contains 5 passes, 3 warnings, and 1 failure. In particular, level predictions are highly persistent and the delta/change model does not beat a zero-change baseline. The report therefore recommends presenting the model primarily as a persistence/smoothing model and treating causal or rise/fall claims cautiously.

## 4. Other untracked files

- `copy1.zip`: an untracked archive file; its contents were not part of the source diff.
- `__pycache__/`: generated Python bytecode directory.
- `archive/__pycache__/`: generated Python bytecode directory for archived modules.

These files are not source-code behavior changes. They are build/runtime or local workspace artifacts.

## 5. Verification notes

The working tree contains 39 modified tracked files and 6 untracked entries. No tracked files are deleted. The trust report can be regenerated from the repository root with:

```powershell
.venv/Scripts/python.exe model_trust_suite.py
```
