# Why a Kalman Filter State-Space Model

This pipeline uses a Kalman filter (`archive/kalman_softpower.py`,
extended in `archive/kalman_softpower_complete.py`) to turn each country's
noisy, sometimes-gappy yearly soft-power composite score into a smoothed
trajectory, a per-year confidence interval, a trend slope, a regime label
(Ascendant / Declining / Fragile / ...), and a 5-year-ahead forecast. This
document explains *why that model*, what problem it was actually solving,
and why the alternatives were rejected — for the write-up, the defense, or
just your own future reference.

---

## 1. What the model actually needs to do

Before comparing models, it's worth being precise about the shape of the
problem, because the right model choice falls out of these constraints
almost mechanically:

- **195 independent country-year series, each only ~15–25 points long**
  (`year` ranges 2000–2024; several countries only enter the panel partway
  through — see `MODEL_TRUST_GUIDE.md`'s "sample size per era" note). This
  is a *short-series* problem, 195 times over, not one long series.
- **The observed score is not the "true" quantity — it's a noisy proxy for
  it.** `soft_power_composite_raw` is itself built from ~50+ imputed,
  normalized, PCA/weighted indicators (see `check_multicollinearity` and
  the outlier-coverage flags in `MODEL_TRUST_GUIDE.md`). A country's real
  standing doesn't jump around year to year the way this composite number
  sometimes does — some of that year-to-year jitter is measurement noise
  in how the composite is built, not real change.
- **Irregular missingness.** Some countries have gaps (a missing survey
  year, a country that entered the UN mid-panel). Whatever method is used
  has to tolerate a missing year without needing to drop the country or
  interpolate by hand first.
- **The deliverable is a CI, not just a point estimate.** The dashboard's
  forecast panel and trajectory view show a shaded uncertainty band per
  country per year — this was a design requirement from the start (see the
  docstring in `kalman_softpower.py`: *"This is what your architecture
  diagram means by 'Latent Soft Power Score ± CI'"*), not something bolted
  on after the fact.
- **195 of these are needed, cheaply, without per-series hand-tuning.**
  Whatever the method, it has to run unattended across all 195 countries
  from one script, with at most one or two free parameters fit
  automatically per country — not a modeling choice made country-by-
  country by a human.
- **The trust suite already established there's little exploitable signal
  beyond persistence.** `model_trust_suite.py`'s naive-baseline check
  (section 3) shows the full KPI model doesn't beat a lag-only model, and
  the delta/change model has R² ≤ 0 in cross-validation. In plain terms:
  **year-over-year change in this score is close to unpredictable noise**,
  and the honest forecast is "carry the current smoothed level forward,
  with growing uncertainty" — not a model that claims to know *why* the
  score will move. That finding, established independently by the trust
  suite, is exactly the assumption a Kalman local-level model encodes on
  purpose. The model wasn't chosen and then happened to match that result —
  but the two are consistent, and that consistency is worth stating
  explicitly if asked.

## 2. What the current model actually is

The implementation (`kalman_filter_country` in both `kalman_softpower.py`
and `kalman_softpower_complete.py`) is a **1-dimensional local-level model**
— the simplest member of the Kalman filter family, also known as a
"random walk plus noise" model:

```
x_t = x_{t-1} + w_t        w_t ~ N(0, Q)   (state / process noise)
y_t = x_t + v_t             v_t ~ N(0, R)   (observation noise)
```

`x_t` is the country's latent "true" soft-power level; `y_t` is the
observed composite score. Two variances do all the work:

- **Q (process noise)** — how much the true latent level can genuinely
  shift in a year. Originally a fixed heuristic (`0.3 × Var(Δscore)`); now
  fit per country by maximum likelihood (`optimize_kalman_params`, added
  after `MODEL_TRUST_GUIDE.md` section 4 flagged that the fixed heuristic
  left 34% of countries with autocorrelated residuals — fitting Q/R per
  country brought that down to 22%).
- **R (observation noise)** — how much of the year-to-year wiggle is just
  measurement/construction noise in the composite score, not real change.

At each year, the filter does one of two things: if there's an
observation, it blends the prediction with the new data point weighted by
the Kalman gain (more weight to the data when R is small relative to
uncertainty, more weight to the prior prediction when R is large); if the
year is missing, it just propagates the prediction forward with growing
variance and waits for the next real observation. That's what makes it
tolerant of the panel's gaps for free, with no separate imputation step.

On top of the forward filter:
- An **RTS (Rauch–Tung–Striebel) smoother** runs backward through each
  country's history to produce the clean, smoothed historical trajectory
  the dashboard displays — this is what turns a jagged raw score into a
  presentable trend line, using information from *later* years to refine
  *earlier* estimates (legitimate for historical smoothing, not used for
  the forward forecast, which only ever looks backward — see
  `MODEL_TRUST_GUIDE.md` section 4's calibration caveat about this
  distinction).
- The **5-year forecast** carries the last smoothed level forward, adds a
  linear trend fit from the last 5 smoothed points, and grows the variance
  linearly with the horizon (`P_h = P_last + h·Q`) — the textbook
  random-walk-forecast uncertainty growth.
- **Regime labels** (`detect_regimes`) bucket each country by trend slope,
  volatility (innovation std), and level, using data-relative percentile
  thresholds rather than hardcoded cutoffs, so the six labels
  (Ascendant / Volatile Riser / Stable Power / Coasting / Declining /
  Fragile) stay meaningful regardless of the score's scale.

## 3. Why this shape of model fits the problem

| Requirement | Why the Kalman local-level model satisfies it |
|---|---|
| Short (~15–25 point) series, ×195 | Two scalar parameters (Q, R) fit per country by 1-D MLE — no order-selection, no large parameter search, cheap enough to refit for every country on every pipeline run |
| Noisy proxy for a slowly-changing latent quantity | This *is* the model's exact generative story — it doesn't need to be adapted to fit that structure, it already assumes it |
| Irregular missing years | Native — a missing observation just skips the update step and grows the variance; no separate imputation pass needed |
| Deliverable is a calibrated CI, not just a point forecast | The filter produces a variance at every step by construction, not as an add-on; `MODEL_TRUST_GUIDE.md` section 4's CI-calibration check exists *because* the model naturally emits one |
| Runs unattended across 195 countries | Closed-form filter updates (no iterative solver at filter time); the one thing that is fit (Q/R) is a 1-D bounded scalar optimization, fast and robust |
| Little exploitable signal beyond persistence (established separately by the trust suite) | A random-walk state-space model *is* the formal version of "persist the last good estimate, admit growing uncertainty" — it doesn't overclaim predictive skill the data doesn't support |
| Needs a principled way to separate "real change" from "measurement wobble" | Q vs. R is exactly that separation, and it's the one interpretable modeling decision a defense committee can ask about directly |

## 4. Alternatives considered, and why each was passed over

### ARIMA / SARIMA per country
The closest classical alternative — in fact, a local-level Kalman filter
*is* mathematically equivalent to an ARIMA(0,1,1) model with a specific
constraint on the MA coefficient, so this wasn't a rejection of the
statistical family so much as a choice of the simpler, more transparent
member of it. Full ARIMA(p,d,q) with order selection (AIC/BIC search over
p, q) per country adds real cost for no real benefit here: with ~15–25
points per series, order-selection is unstable and prone to overfitting a
particular country's idiosyncratic history; there's also no seasonality to
capture (annual data), so the extra generality ARIMA offers over a local-
level model mostly isn't needed. The Kalman formulation also makes the
two-noise-source interpretation (Q vs. R) explicit in a way a fitted
ARIMA(0,1,1) coefficient doesn't communicate as directly to a non-technical
reader of the write-up.

### Exponential smoothing / Holt-Winters
Very close in spirit — Holt's linear trend method is, under specific
parameter settings, a special case of a Kalman local-linear-trend model.
The reasons to prefer the explicit Kalman formulation instead: (1) no
native, principled uncertainty band — the standard error of an
exponential-smoothing forecast is usually bolted on after the fact from
residual variance, rather than propagating naturally step-by-step the way
`P_h = P_last + h·Q` does; (2) no natural handling of missing years without
a separate gap-filling step first; (3) no formal likelihood to fit the
smoothing parameter against, the way `optimize_kalman_params` does via
MLE — the smoothing constant in Holt-Winters is usually chosen by grid
search over one-step-ahead error, which is a weaker fitting criterion than
maximizing the full data likelihood.

### Gaussian Process regression, per country
Would give a properly calibrated, non-parametric uncertainty band and
handles irregular timing natively — genuinely a reasonable alternative.
Rejected mainly on grounds of complexity-for-no-payoff: fitting 195
independent GPs means choosing and validating a kernel (RBF vs. Matérn,
lengthscale priors) per country, which is a bigger, less transparent
modeling surface than two scalars — and a kernel that encodes "slow,
smooth drift" is essentially re-deriving the same local-level assumption
the Kalman filter already encodes, just with more machinery and a much
harder story to tell in a defense ("we chose a Matérn-3/2 kernel with an
empirical Bayes lengthscale" invites more follow-up questions than "process
noise vs. observation noise, fit by maximum likelihood").

### LSTM / RNN sequence models
Rejected primarily on data-volume grounds: 15–25 timesteps per country is
far too little to train a recurrent network per country without severe
overfitting, and a single shared network across all 195 countries
(effectively transfer learning across countries) is a much larger, harder-
to-interpret model for a problem the trust suite already showed has almost
no exploitable temporal structure beyond persistence (naive delta-model
R² ≤ 0). Reaching for a neural sequence model here would be materially more
complex machinery in service of a signal that, per the trust suite's own
diagnostics, isn't there to find. It would also forfeit the one thing this
project's committee can most easily audit: two named, interpretable noise
parameters instead of an opaque set of learned weights.

### Prophet (or similar business-forecasting libraries)
Built around detecting seasonality, holiday effects, and trend
changepoints in typically daily/weekly business time series — none of
which apply to an annual country-level panel. It would still require 195
independent fits, adds a heavier dependency, and its automatic changepoint
detection doesn't map onto a clean Q/R noise decomposition the way the
Kalman filter's does. Overkill in the same direction as the LSTM option,
just with different jargon.

### Naive persistence / simple moving average
This is close to the *baseline* the trust suite already benchmarks the
whole modeling pipeline against (`check_naive_baseline_gap`), not a
serious competitor to the Kalman approach for the dashboard's needs — a
raw moving average has no principled per-country noise decomposition, no
growing forecast-horizon uncertainty, and no backward-smoothing pass to
produce a clean historical trend line. It's a useful *sanity check*
(and is effectively what the Kalman filter degrades to when Q is small
relative to R), not a replacement.

### A single pooled/hierarchical model across all 195 countries
A Bayesian hierarchical state-space model (partial pooling of Q/R across
countries, sharing statistical strength from data-rich countries to help
data-poor, short-history ones) is a legitimate improvement direction and
probably the single best answer to "what would you do with more time."
It was passed over here for cost/complexity reasons appropriate to a
capstone: it needs MCMC or variational inference rather than a closed-form
filter, is much slower to fit and harder to diagnose, and the current
per-country-independent approach already has a documented, cheaper
mitigation for the same problem — `MODEL_TRUST_GUIDE.md` explicitly flags
short-history countries as needing wider error bars than the CI alone
suggests, rather than asking the model itself to fix it via pooling.

## 5. Honest limitations of the current choice

Worth stating in the write-up alongside the justification, not instead of
it:

- **Local *level*, not local *trend*.** The core filter has one state
  variable (the level) and treats the level as a random walk with no
  built-in drift term. The 5-year forecast's trend component is a
  separate, ad hoc linear fit over the last 5 smoothed points bolted onto
  the forecast step (`forecast_country` in `kalman_softpower_complete.py`),
  not part of the filter's own state vector. A local-*linear-trend* Kalman
  model (2D state: level + slope, both evolving with their own process
  noise) would fold trend estimation into the same principled
  filter/smoother machinery instead of handling it as a post-hoc OLS slope.
  This is the most defensible "why not a fancier Kalman variant"
  follow-up, and a good next-iteration item to name if asked.
- **Fixed Q/R was the original default; MLE fitting was a later fix**, and
  even after that fix, `MODEL_TRUST_GUIDE.md` section 4 still shows 22% of
  countries with innovation autocorrelation above the 0.3 threshold — the
  model is a good, not perfect, fit for every country's actual dynamics.
- **CI calibration is checked in-sample.** `MODEL_TRUST_GUIDE.md` section 4
  is explicit that the ~98.3% empirical coverage of the stated 95% band is
  measured against the same observations the filter used to produce that
  estimate — a real backtest of `kalman_forecast_5yr.csv` against years
  that have since become observable hasn't happened yet (there hasn't been
  time), and is the honest next validation step.
- **Countries are modeled independently.** No information is shared across
  countries, so a country with 3 years of history gets a forecast with the
  same *mechanical* treatment as one with 24 years, even though intuitively
  the short-history one deserves less confidence than its own CI alone
  implies (flagged explicitly in `MODEL_TRUST_GUIDE.md`'s "sample size per
  era" note).

## 6. How this choice is already being validated

Two of the nine checks in `model_trust_suite.py` exist specifically
*because* this is a Kalman filter, and wouldn't have a direct equivalent
for an opaque model like an LSTM:

- **`check_kalman_residuals`** (trust report section "Kalman diagnostics")
  — tests whether the filter's innovations are white noise, i.e. whether Q
  and R are well-specified. This diagnostic is a direct consequence of
  using a model with an explicit noise model in the first place.
- **`check_kalman_ci_calibration`** (trust report section "CI calibration")
  — tests whether the stated 95% band actually contains ~95% of
  observations. Same point: a model that emits a variance at every step,
  by construction, can be checked this way; a model that doesn't (e.g. a
  raw moving average, or most out-of-the-box neural forecasters) would
  need a separate, bolted-on uncertainty-quantification scheme just to make
  this question askable.

Run `.venv/Scripts/python.exe model_trust_suite.py` and read those two
sections in `trust_report.md` for the current numbers.
