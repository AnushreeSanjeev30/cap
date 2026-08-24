# softpower_models.py
"""
Central model module for the Soft Power pipeline.
Contains four model classes + one ensemble:

  SoftPowerXGB      — Quantile XGBoost (point estimate + 80% CI)
  SoftPowerRF       — Random Forest with bootstrap CI
  SoftPowerLGBM     — LightGBM with quantile regression CI
  SoftPowerEnsemble — Weighted average of all three, auto-selects best
  
All classes share the same interface:
    model.fit(X_train, y_train)
    model.predict(X)  →  DataFrame with columns: score, ci_lower, ci_upper
    model.feature_importance()  →  DataFrame with feature, importance
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. XGBoost (your original — unchanged interface)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftPowerXGB:
    """
    Three XGBoost quantile models:
      q=0.10 → ci_lower (80% CI lower)
      q=0.50 → score   (point estimate)
      q=0.90 → ci_upper (80% CI upper)
    """
    QUANTILES = [0.10, 0.50, 0.90]
    MODEL_NAME = 'XGBoost'

    def __init__(self, n_estimators=500, lr=0.05, learning_rate=None, max_depth=6):
        # Accept both lr= and learning_rate= (phase_c uses lr=)
        lr = learning_rate if learning_rate is not None else lr
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42
        )
        self.models = {}
        self.feature_cols = None
        self.cv_metrics = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_cols = list(X_train.columns)
        for q in self.QUANTILES:
            model = XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                **self.params
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train)],
                      verbose=False)
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_in = self._align(X)
        return pd.DataFrame({
            'score':    self.models[0.50].predict(X_in),
            'ci_lower': self.models[0.10].predict(X_in),
            'ci_upper': self.models[0.90].predict(X_in),
        })

    def feature_importance(self) -> pd.DataFrame:
        fi = self.models[0.50].get_booster().get_score(importance_type='gain')
        return (pd.DataFrame({'feature': list(fi.keys()),
                              'importance': list(fi.values())})
                .sort_values('importance', ascending=False)
                .reset_index(drop=True))

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns match training, fill missing with 0."""
        X_out = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        return X_out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Random Forest  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftPowerRF:
    """
    Random Forest with bootstrap-based confidence intervals.
    CI is estimated from the std of individual tree predictions.
    
    Interface matches SoftPowerXGB exactly.
    """
    MODEL_NAME = 'RandomForest'

    def __init__(self, n_estimators=300, max_depth=12,
                 min_samples_leaf=5, max_features=0.5):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = None
        self.cv_metrics   = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_cols = list(X_train.columns)
        X_in = X_train.fillna(0).values
        self.rf.fit(X_in, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_in = self._align(X).values

        # Point estimate
        scores = self.rf.predict(X_in)

        # CI from individual tree predictions (std across forest)
        tree_preds = np.array([tree.predict(X_in)
                                for tree in self.rf.estimators_])   # (n_trees, n_samples)
        std = tree_preds.std(axis=0)

        return pd.DataFrame({
            'score':    np.clip(scores, 0, 100),
            'ci_lower': np.clip(scores - 1.28 * std, 0, 100),  # 80% CI
            'ci_upper': np.clip(scores + 1.28 * std, 0, 100),
        })

    def feature_importance(self) -> pd.DataFrame:
        return (pd.DataFrame({
                    'feature':    self.feature_cols,
                    'importance': self.rf.feature_importances_
                })
                .sort_values('importance', ascending=False)
                .reset_index(drop=True))

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LightGBM  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftPowerLGBM:
    """
    LightGBM with quantile regression for CI (same approach as XGBoost).
    Trains three models: q=0.10, 0.50, 0.90.

    Interface matches SoftPowerXGB exactly.
    """
    QUANTILES  = [0.10, 0.50, 0.90]
    MODEL_NAME = 'LightGBM'

    def __init__(self, n_estimators=500, learning_rate=0.05,
                 max_depth=6, num_leaves=31):
        self.base_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        self.models       = {}
        self.feature_cols = None
        self.cv_metrics   = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("pip install lightgbm")

        self.feature_cols = list(X_train.columns)
        X_in = X_train.fillna(0)

        for q in self.QUANTILES:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                **self.base_params
            )
            model.fit(X_in, y_train)
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_in = self._align(X).fillna(0)
        return pd.DataFrame({
            'score':    np.clip(self.models[0.50].predict(X_in), 0, 100),
            'ci_lower': np.clip(self.models[0.10].predict(X_in), 0, 100),
            'ci_upper': np.clip(self.models[0.90].predict(X_in), 0, 100),
        })

    def feature_importance(self) -> pd.DataFrame:
        import lightgbm as lgb
        return (pd.DataFrame({
                    'feature':    self.feature_cols,
                    'importance': self.models[0.50].feature_importances_
                })
                .sort_values('importance', ascending=False)
                .reset_index(drop=True))

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Ensemble  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftPowerEnsemble:
    """
    Weighted ensemble of XGBoost + RandomForest + LightGBM.

    Weights are set automatically after compare() is called:
      - Best model gets weight 0.5
      - Other two share 0.5 equally (0.25 each)
    
    Or pass custom weights to __init__.

    Usage:
        ensemble = SoftPowerEnsemble()
        metrics  = ensemble.fit_compare(X_train, y_train, X_val, y_val)
        preds    = ensemble.predict(X_test)
        best     = ensemble.best_model   # the single best model object
    """
    MODEL_NAME = 'Ensemble'

    def __init__(self, weights: dict = None):
        """
        weights: dict like {'XGBoost': 0.5, 'RandomForest': 0.25, 'LightGBM': 0.25}
                 If None, auto-set after compare().
        """
        self.xgb     = SoftPowerXGB()
        self.rf      = SoftPowerRF()
        self.lgbm    = SoftPowerLGBM()
        self.members = {
            'XGBoost':      self.xgb,
            'RandomForest': self.rf,
            'LightGBM':     self.lgbm,
        }
        self.weights      = weights   # None until set
        self.cv_metrics   = {}        # filled by fit_compare
        self.best_model   = None      # set after compare
        self.best_name    = None
        self.feature_cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit all three models. Weights default to equal if not set."""
        self.feature_cols = list(X_train.columns)
        print("  Training XGBoost...")
        self.xgb.fit(X_train, y_train)
        print("  Training Random Forest...")
        self.rf.fit(X_train, y_train)
        print("  Training LightGBM...")
        self.lgbm.fit(X_train, y_train)

        if self.weights is None:
            self.weights = {'XGBoost': 1/3, 'RandomForest': 1/3, 'LightGBM': 1/3}
        self.best_model = self.xgb   # default until compare() is called
        self.best_name  = 'XGBoost'
        return self

    def compare(self, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        """
        Evaluate all three models on a held-out validation set.
        Sets weights so best model has 0.5, others 0.25 each.
        Returns a comparison DataFrame.
        """
        rows = []
        scores_by_model = {}

        for name, model in self.members.items():
            preds = model.predict(X_val)
            mae   = mean_absolute_error(y_val, preds['score'])
            r2    = r2_score(y_val, preds['score'])
            ci_cov = (
                (y_val.values >= preds['ci_lower'].values) &
                (y_val.values <= preds['ci_upper'].values)
            ).mean()
            rows.append({
                'model':       name,
                'mae':         round(float(mae), 4),
                'r2':          round(float(r2),  4),
                'ci_coverage': round(float(ci_cov), 4),
            })
            scores_by_model[name] = preds['score'].values
            model.cv_metrics = {'mae': mae, 'r2': r2, 'ci_coverage': ci_cov}

        comparison = (pd.DataFrame(rows)
                      .sort_values('r2', ascending=False)
                      .reset_index(drop=True))

        # Best model = highest R²
        self.best_name  = comparison.iloc[0]['model']
        self.best_model = self.members[self.best_name]

        # Weights: best=0.5, others=0.25 each
        self.weights = {}
        for name in self.members:
            self.weights[name] = 0.5 if name == self.best_name else 0.25

        self.cv_metrics = {row['model']: row for _, row in comparison.iterrows()}

        print(f"\n  MODEL COMPARISON:")
        print(f"  {'Model':<18} {'MAE':>8} {'R²':>8} {'CI Cov':>8}  {'Weight':>8}")
        print(f"  {'-'*56}")
        for _, row in comparison.iterrows():
            star = ' ← BEST' if row['model'] == self.best_name else ''
            print(f"  {row['model']:<18} {row['mae']:>8.3f} {row['r2']:>8.3f} "
                  f"{row['ci_coverage']:>8.1%}  "
                  f"{self.weights[row['model']]:>8.2f}{star}")

        return comparison

    def fit_compare(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame,   y_val: pd.Series) -> pd.DataFrame:
        """Convenience: fit all models then compare on validation set."""
        self.fit(X_train, y_train)
        return self.compare(X_val, y_val)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Weighted ensemble prediction."""
        all_scores  = []
        all_lowers  = []
        all_uppers  = []
        all_weights = []

        for name, model in self.members.items():
            p = model.predict(X)
            w = self.weights.get(name, 1/3)
            all_scores.append(p['score'].values * w)
            all_lowers.append(p['ci_lower'].values * w)
            all_uppers.append(p['ci_upper'].values * w)
            all_weights.append(w)

        total_w = sum(all_weights)
        return pd.DataFrame({
            'score':    np.clip(sum(all_scores) / total_w, 0, 100),
            'ci_lower': np.clip(sum(all_lowers) / total_w, 0, 100),
            'ci_upper': np.clip(sum(all_uppers) / total_w, 0, 100),
        })

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns predictions from all models + ensemble in one DataFrame.
        Useful for comparison tables.
        """
        result = pd.DataFrame(index=range(len(X)))
        for name, model in self.members.items():
            p = model.predict(X)
            col = name.lower().replace(' ', '_')
            result[f'{col}_score']    = p['score'].values
            result[f'{col}_ci_lower'] = p['ci_lower'].values
            result[f'{col}_ci_upper'] = p['ci_upper'].values

        ens = self.predict(X)
        result['ensemble_score']    = ens['score'].values
        result['ensemble_ci_lower'] = ens['ci_lower'].values
        result['ensemble_ci_upper'] = ens['ci_upper'].values
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Weighted average feature importance across all three models."""
        dfs = []
        for name, model in self.members.items():
            fi = model.feature_importance().copy()
            fi['importance'] = fi['importance'] / fi['importance'].sum()  # normalise
            fi['model']  = name
            fi['weight'] = self.weights.get(name, 1/3)
            dfs.append(fi)

        combined = pd.concat(dfs)
        weighted = (combined.groupby('feature')
                    .apply(lambda g: (g['importance'] * g['weight']).sum())
                    .reset_index()
                    .rename(columns={0: 'importance'})
                    .sort_values('importance', ascending=False)
                    .reset_index(drop=True))
        return weighted

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Ensemble saved → {path}")

    @staticmethod
    def load(path: str) -> 'SoftPowerEnsemble':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal CV utility (shared across phase_c and phase_d)
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_cv(model_class, df: pd.DataFrame, feature_cols: list,
                target_col: str = 'target', year_col: str = 'year',
                n_splits: int = 4, model_kwargs: dict = None) -> dict:
    """
    Time-series cross-validation for any model class.
    Trains on past, tests on future — no data leakage.

    Args:
        model_class:  SoftPowerXGB | SoftPowerRF | SoftPowerLGBM
        df:           model DataFrame with features + target
        feature_cols: list of feature column names
        target_col:   column to predict
        n_splits:     number of folds
        model_kwargs: kwargs passed to model_class()

    Returns dict with mae_mean, mae_std, r2_mean, r2_std, ci_coverage
    """
    if model_kwargs is None:
        model_kwargs = {}

    years   = sorted(df[year_col].unique())
    fold_sz = max(1, len(years) // (n_splits + 1))
    maes, r2s, covs = [], [], []

    for fold in range(n_splits):
        cutoff   = years[min((fold + 1) * fold_sz, len(years) - 1)]
        next_cut = years[min((fold + 2) * fold_sz, len(years) - 1)]
        train = df[df[year_col] <  cutoff]
        test  = df[(df[year_col] >= cutoff) & (df[year_col] < next_cut)]
        if len(test) < 20 or len(train) < 50:
            continue

        # Split off the most recent slice of train as a conformal calibration
        # set (fit_set strictly precedes calib_set strictly precedes test).
        # Each model's own ci_lower/ci_upper (raw quantile regression, or
        # tree-variance for RF) is systematically miscalibrated -- see
        # trust_report.md section 1. Replacing it with an empirical residual
        # quantile from held-out-in-time data fixes that without touching
        # each model's point-prediction interface.
        train_years = sorted(train[year_col].unique())
        calib_cut = train_years[max(0, len(train_years) - max(1, len(train_years) // 5))]
        fit_set   = train[train[year_col] <  calib_cut]
        calib_set = train[train[year_col] >= calib_cut]
        if len(fit_set) < 50 or len(calib_set) < 20:
            fit_set, calib_set = train, train  # not enough history to split cleanly

        X_fit = fit_set[feature_cols].fillna(0)
        y_fit = fit_set[target_col]
        X_te  = test[feature_cols].fillna(0)
        y_te  = test[target_col]

        m = model_class(**model_kwargs)
        m.fit(X_fit, y_fit)

        calib_preds = m.predict(calib_set[feature_cols].fillna(0))
        calib_resid = np.abs(calib_set[target_col].values - calib_preds['score'].values)
        # Finite-sample-corrected conformal quantile (Vovk et al.): the plain
        # empirical 80th percentile undercovers on small calibration sets.
        # Using ceil((n+1)*0.80)/n as the quantile level instead guarantees
        # the intended marginal coverage on average, not just asymptotically.
        n_calib = len(calib_resid)
        q_level = min(1.0, np.ceil((n_calib + 1) * 0.80) / n_calib)
        half_width = np.quantile(calib_resid, q_level)

        preds = m.predict(X_te)
        preds['ci_lower'] = np.clip(preds['score'] - half_width, 0, 100)
        preds['ci_upper'] = np.clip(preds['score'] + half_width, 0, 100)

        mae = mean_absolute_error(y_te, preds['score'])
        r2  = r2_score(y_te, preds['score'])
        cov = ((y_te.values >= preds['ci_lower'].values) &
               (y_te.values <= preds['ci_upper'].values)).mean()

        maes.append(mae)
        r2s.append(r2)
        covs.append(cov)

    return {
        'mae_mean':    round(float(np.mean(maes)), 4),
        'mae_std':     round(float(np.std(maes)),  4),
        'r2_mean':     round(float(np.mean(r2s)),  4),
        'r2_std':      round(float(np.std(r2s)),   4),
        'ci_coverage': round(float(np.mean(covs)), 4),
        'n_folds':     len(maes),
    }
