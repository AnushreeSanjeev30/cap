# Model Trust Guide

How to check whether a number this pipeline produces (a score, a rank, a
forecast, a "driver") deserves to be trusted, and what to say about it in the
capstone write-up / defense if it doesn't fully deserve it.

Run the automated half of this first:

```bash
cd working
./.venv/Scripts/python.exe model_trust_suite.py
```

It prints a PASS/WARN/FAIL verdict per check to the console and writes the
same thing, with full detail, to `trust_report.md`. Re-run it any time
`output/` changes — it always reflects the artifacts currently on disk, never
a cached opinion. The rest of this document explains *why* each check exists
and what the current results actually mean, plus a few things worth checking
by hand that don't reduce to a script.

---

## 1. Overfitting

**The idea:** a model that memorizes its training rows will score great on a
metric computed from those same rows (or rows very similar to them) and much
worse on genuinely new data. The gap between "score on data like what it
trained on" and "score on data unlike what it trained on" *is* the amount of
overfitting.

**Why a random train/test split isn't enough here:** this dataset is a
country × year panel. A random 80/20 split puts, say, France-2015 in train
and France-2016 in test — the model has effectively already seen France. That
inflates the test score without proving the model generalizes to a country or
a future year it has never encountered.

**What the suite does (`check_generalization_splits`):** trains the same
LightGBM model three ways and compares:
- random row split (the overly optimistic baseline),
- temporal split (train on early years, test on the most recent ~20%),
- held-out-country split (train on 80% of countries, test on countries never
  seen at all).

A large drop from random → temporal/held-out is the signature of overfitting
to panel structure. **Current result: the drop is small (R² 0.997 → 0.987 →
0.962)** — this particular model generalizes reasonably across time and
across unseen countries. That's a genuinely good sign; it's the *other* two
issues below that undercut how much credit the KPIs deserve for it.

## 2. The R² is high — but is it measuring the right thing?

A composite score bounded to roughly 0–100 hitting R² ≈ 0.97 should always
raise an eyebrow before it raises confidence. Two follow-up questions decide
which:

**a) Is a big chunk of "importance" actually just the target's own history?**
`score_roll3_mean` (a rolling mean of the country's own past score) is lagged
by construction, so it isn't same-row leakage — but it's still the model
being told "here's roughly what this country scored recently," which is a
strong hint toward this year's answer. Check `check_feature_importance_leakage`
for the current split (currently ~16% of importance, not dominant, but it's
still the single largest individual feature at 28% on its own).

**b) Does the model actually beat a naive baseline that only ever sees the
target's own history?** This is the sharper test, and it's already been run
in `output/diagnostic_1_autocorrelation.json` /
`diagnostic_2_delta.json` (reused by `check_naive_baseline_gap`):

| variant | MAE | R² |
|---|---|---|
| naive: next year = this year | 1.660 | 0.970 |
| lag-only XGBoost (score history only) | 1.494 | 0.976 |
| full XGBoost (all KPIs) | 1.645 | 0.972 |

**The full KPI model does not beat the lag-only model.** All of the
independent, exogenous KPI indicators together add *nothing* once the
model already knows the country's recent score trajectory. Practically:
this is a well-behaved **persistence / smoothing model**, not evidence that
GDP, internet access, judicial effectiveness, etc. individually *cause* or
even *predict* soft power beyond "things don't change much year to year."

The delta (year-over-year change) model is worse than useless: every variant
has **R² ≤ 0 in cross-validation**, meaning "predict zero change" beats all
of them. Don't present rise/fall predictions from that model as a forecast —
present the Kalman trend slope instead, which is what the dashboard already
does.

**How to talk about this in the write-up:** don't claim "our KPI features
predict soft power with R²=0.97." Claim "soft power is highly persistent
year-over-year (R²=0.97 from history alone); once persistence is accounted
for, indicator X/Y/Z show up as the largest incremental drivers of that
persistence" — and back the second half with the SHAP/feature-importance
analysis run *after* looking at diagnostic_1's lag-only baseline, not before.

## 3. Multicollinearity

Highly correlated inputs (e.g. `unesco_cultural_sites` vs
`unesco_total_sites`, or the several Freedom House sub-ratings) don't
meaningfully hurt a tree ensemble's *predictions*, but they do make
per-feature attribution (SHAP values, `feature_importance.csv`,
`ridge_coefficients.csv`) unstable — the model can arbitrarily split credit
between two near-duplicate columns, and that split can flip between retrains.

`check_multicollinearity` computes VIF (variance inflation factor) by hand —
regress each feature on all the others and take `1/(1-R²)` — since
`statsmodels` isn't installed in this venv. **Current result: 30 of 54 raw
indicator columns have VIF > 10**, several effectively infinite (near-perfect
duplicates). Read single-feature explanations for anything in a correlated
cluster as "this cluster mattered," not as a precise ranking within it.

## 4. Kalman filter trustworthiness

Two independent things can go wrong with the state-space model in
`kalman_softpower.py`, and they're checked separately because they can fail
independently:

- **Is the noise model right?** (`check_kalman_residuals`) If `Q`/`R` are set
  well, the filter's one-step residuals ("innovations") should look like
  white noise — no leftover pattern for the filter to still be missing. The
  suite checks lag-1 autocorrelation of innovations per country. Current
  result: mean +0.18, with **34% of countries above |0.3|** — the fixed
  Q = 30%·Var(Δscore), R = 10%·Var(score) choice is a systematic mis-fit for
  a third of countries (usually ones with a real regime change the filter is
  slow to catch, or a single noisy year it over-reacts to). Countries flagged
  here deserve a wider error bar on their "stability class" than the ones
  with clean residuals.

- **Is the stated 95% band actually 95%?** (`check_kalman_ci_calibration`)
  Empirically, ~93.4% of observations fall inside their own `ci_lower_95`/
  `ci_upper_95`. Close to nominal — but note the caveat printed with that
  check: this compares the filter's own smoothed estimate to the very
  observation it used to produce that estimate, so it's an in-sample
  calibration check, not proof the *5-year forecast* (`kalman_forecast_5yr.csv`)
  is calibrated. The only way to check that honestly is to wait: backtest
  the forecast made from data through year *t* against what year *t+5*
  actually turned out to be, once enough years have passed.

## 5. Ranking robustness

`sensitivity_analysis.csv` already scores every country under four different
weighting schemes (XGBoost-implied weights, PCA-derived weights, equal
weights, literature-derived weights). `check_ranking_robustness` computes the
Spearman rank correlation between them. Current result: all pairs > 0.97 —
the ranking is not an artifact of the particular weighting choice, which is
good grounds for trusting the *ordering* claim, even where the exact score
value is more debatable.

## 6. Causal claims

`archive/phase3_causal_modeling.py` produces `ate` / `ci_lower` / `ci_upper` /
`p_value` for each indicator via DoWhy's backdoor linear-regression estimator.
This is not checked by the automated suite (there's no artifact CSV persisted
from it to read back), but the same skepticism applies double here:
- A significant ATE from an observational backdoor adjustment is only as good
  as the assumed causal graph — check `soft_power_cap.drawio` /
  `hld_cap.drawio` for whatever DAG was assumed, and whether an unmeasured
  confounder (e.g., regional stability, colonial history) could plausibly
  explain the same edge.
- Given the multicollinearity in section 3, treatment variables that are
  correlated with each other will produce backdoor estimates that trade off
  against each other in ways that are sensitive to which other variables were
  included in the adjustment set. Re-run with a couple of alternate adjustment
  sets before trusting a specific ATE number in a headline claim.

**Update:** `ridge_coefficients.csv`, `shap_global.csv`, and `shap_country.csv`
previously had no generating script anywhere in the repo and were static,
pre-multicollinearity-fix snapshots. `archive/regenerate_shap_outputs.py` now
regenerates all three against the currently deployed model
(`output/artifacts/xgb_model.pkl`) and the current (deduplicated —
see `archive/fix_duplicate_rows.py`) panel. Re-run it after retraining.

## 7. External validity: does the ranking agree with anything outside this pipeline?

Everything above proves the model is *internally* well-behaved — it doesn't
overfit, isn't dominated by leakage, is stable under re-weighting. None of
that proves the **score itself** corresponds to anything in the real world;
a model can pass every internal check and still be measuring something
nobody outside the pipeline would call "soft power."

`check_external_benchmark` compares this model's `global_rank` (from
`output/soft_power_predictions.csv`) against `external_benchmark_gspi2024.csv`
— a small hand-transcribed reference table (20 countries, spanning rank 1
through 193) from Brand Finance's **Global Soft Power Index 2024**, a
published survey of 170,000+ respondents across 100+ countries. It's an
independent, differently-constructed measure of the same underlying concept:
useful as a face-validity check precisely because it wasn't built from the
same KPI indicators or fit to the same data.

**Current result: Spearman ρ = 0.806 (n=20, PASS).** The two rankings agree
strongly at both ends (USA/UK/Germany/Japan/Switzerland cluster near the top
in both; Vanuatu/Nauru/Kiribati cluster near the bottom in both) even though
they're built from completely different inputs — one from public survey
perception, one from GDP/freedom/education/heritage KPIs. That's good
evidence this pipeline's ranking reflects something real, not just an
artifact of feature choice.

**Read the disagreements, not just the correlation.** China is the sharpest
outlier: GSPI rank 3, this model's rank 28. That's not a bug to "fix" —
GSPI is measuring global *perception* (media reach, cultural export,
diplomatic visibility), while this pipeline's KPI composite leans on
governance/freedom/institutional indicators where China scores much lower.
The gap is informative: it tells you what kind of "soft power" this model is
actually measuring (institutional + developmental capacity) versus what GSPI
measures (perceived global influence) — say that explicitly in the write-up
rather than picking whichever ranking is more flattering.

**Caveats on the check itself:**
- The reference table is 20 countries, not the full 193 — the complete index
  sits behind Brand Finance's PDF report, not a scrapable page. Extend
  `external_benchmark_gspi2024.csv` by hand from the report if a larger,
  more statistically powerful comparison is wanted.
- Years don't line up exactly: GSPI is 2024, most of the model's rows in
  `soft_power_predictions.csv` are 2023 data (`data_year` column). A country
  with a fast-moving score in that year will look like a bigger disagreement
  than it really is.
- Agreement is evidence of face validity, not proof either ranking is
  "correct" — GSPI itself is a perception survey, not ground truth either.

## Things worth checking by hand (not automated)

- **Data quality flags already computed for you**: the panel carries
  `is_outlier`, `is_extreme_outlier`, `outlier_score`,
  `outlier_feature_coverage` columns per row. Before trusting a country's
  trend or forecast, check whether it's flagged — a country with low
  `outlier_feature_coverage` (few indicators observed) is getting a score
  built mostly from imputation/normalization defaults, not real data.
- **Sample size per era**: `year` coverage in the panel goes from 165
  countries in 2000 to 199 in 2013+; earlier years and any country added mid-
  panel have less history for the Kalman filter and lag features to work
  with. Trend slopes for short histories are noisier than the CI suggests.
- **Rerun after any feature-engineering change**: the naive-baseline gap
  (section 2b) and the VIF table (section 3) can both shift when someone adds
  or removes a feature. Re-running `model_trust_suite.py` is cheap; make it
  part of the routine after retraining, not a one-off.

## Files this guide and the suite read

| File | Produced by | Used for |
|---|---|---|
| `output/master_soft_power_panel.csv` | `capstone/data/*.py` preprocessing | fresh generalization + VIF checks |
| `output/cv_results.json`, `multi_model_cv_results.json`, `model_comparison.csv` | `archive/phase_c_*.py` | CV metric sanity check |
| `output/feature_importance.csv` | `archive/phase2_predictive_model.py` | target-leakage scan |
| `output/diagnostic_1_autocorrelation.json`, `diagnostic_2_delta.json` | `archive/diagnostic_aftermodelout.py` | naive-baseline gap |
| `output/kalman_results.csv`, `kalman_summary.csv` | `archive/kalman_softpower.py` | residual + CI calibration checks |
| `output/sensitivity_analysis.csv` | `archive/sensitivity_analysis.py` | ranking robustness |
| `output/soft_power_predictions.csv` | dashboard prediction pipeline | external benchmark comparison |
| `external_benchmark_gspi2024.csv` | hand-transcribed from Brand Finance GSPI 2024 press coverage | external benchmark comparison |
