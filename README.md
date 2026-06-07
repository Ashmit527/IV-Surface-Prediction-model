# Nifty50 IV Surface Completion

A machine learning pipeline to predict and fill missing implied volatility values across the Nifty50 options surface, built for the **Finance Club Open Project**.

---

## Overview

Options data in practice is always sparse — not every strike has a live quote at every timestamp. This pipeline takes a raw IV matrix (strikes × time) with missing values and fills it intelligently using a two-stage approach:

1. **PCHIP interpolation** — fits a smooth curve across strikes (the smile) and across time (temporal carry-forward), producing a physically reasonable baseline.
2. **XGBoost residual correction** — learns the systematic error of the PCHIP baseline from observed data points and corrects the prediction, without overfitting.

Separate models are trained for **Call (CE)** and **Put (PE)** options, since their IV surfaces exhibit structurally different behaviour.

**Kaggle private leaderboard RMSE: `0.0000463419`**

---

## Architecture

### Stage 1 — PCHIP Baseline

[PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) is used to interpolate across the smile for each timestamp:

- **Row PCHIP** (`row_pchip`): interpolates in the **moneyness dimension** (across strikes) using `log(K/S)` as the x-axis, preserving smile shape without oscillation.

The PCHIP baseline already handles most of the surface well. XGBoost only needs to learn the residual error.

### Stage 2 — XGBoost Residual Correction

The model is trained to predict `y_true − row_pchip` (the residual), not raw IV. This is a deliberate design choice:

- PCHIP is already a strong prior; training on residuals forces the model to focus only on systematic deviations.
- The target variance is much smaller → easier to fit, lower risk of overfitting.

The correction is then applied as:

```
prediction = row_pchip + XGBoost_residual × fraction
```

### The Fraction Factor

A damping factor (`fraction`) is applied to the XGBoost correction before adding it back to the PCHIP baseline. **CE uses 0.625, PE uses 0.675** (tuned empirically).

This matters because XGBoost, when correcting residuals, can over-correct on extreme points — particularly near expiry (low `tau`) where the IV surface is steep and noisy. The fraction factor shrinks the correction toward zero, acting as implicit regularisation that:

- Dampens over-confident corrections near expiry where interpolation error spikes.
- Reduces the impact of corrections on outlier points that the model hasn't seen enough of.
- Prevents the ensemble from introducing non-smooth artefacts into an otherwise smooth PCHIP surface.

The CE/PE asymmetry in fraction values reflects the fact that PE wings tend to have steeper skew and thus noisier residuals, requiring slightly more dampening.

### IDW Spatial Interpolation

An **Inverse Distance Weighted (IDW)** interpolator operates in a 2D space of `(time_index, log_moneyness)`, using the 20 nearest known points within a causal window of 20 rows. It provides a smooth spatial estimate that captures local structure in both dimensions simultaneously.

The window enforces a strict no-future-data constraint: only rows strictly before the target row (`start:row`, exclusive) are used from the time dimension. The current row is included in full (all columns, both left and right of the target strike), since same-timestamp strikes are contemporaneous observations, not future data. This is implemented as two separate index arrays that are concatenated before distance computation:

```python
# Past rows: all columns allowed
past_rows, past_cols = np.where(np.isfinite(self.train_matrix[start:row, :]))

# Current row: all columns allowed (right-side strikes are not future data)
cur_cols = np.where(np.isfinite(self.train_matrix[row, :]))[0]
```

---

## Feature Engineering

Features are built in `FeatureBuilder.build()` and cover five categories:

### Smile Shape Features
Capture the current cross-sectional structure of the IV smile:

| Feature | Description |
|---|---|
| `log_moneyness` | `log(K/S)` — signed distance from ATM |
| `abs_log_moneyness` | Distance from ATM (unsigned) |
| `log_moneyness_sq` | Quadratic moneyness (curvature sensitivity) |
| `strike_over_spot` | Simple moneyness ratio |
| `atm_iv` | ATM IV for the current row |
| `iv_minus_atm` | How far this strike's IV deviates from ATM |
| `iv_ratio_atm` | IV / ATM IV |
| `risk_reversal` | `OTM_put_IV − OTM_call_IV` (skew direction) |
| `butterfly` | `(OTM_put_IV + OTM_call_IV)/2 − ATM_IV` (curvature) |
| `skew_slope` | Linear slope from OLS fit across the smile |
| `skew_slope_change` | Change in skew slope vs previous row |
| `quad_coeff` | Quadratic coefficient from parabolic smile fit |
| `linear_coeff` | Linear coefficient from parabolic smile fit |
| `quad_fitted` | Parabolic prediction for this point |
| `smile_convexity` | `left_IV + right_IV − 2 × row_pchip` |

### Neighbourhood Features
Capture the local structure around the target point:

| Feature | Description |
|---|---|
| `left_iv`, `right_iv` | Nearest known IVs to the left and right in strike space |
| `left_dist`, `right_dist` | Strike-index distance to those neighbours |
| `row_neighbor_mean` | Mean of `left_iv` and `right_iv` |
| `left_slope`, `right_slope` | IV slope from target to each neighbour |
| `slope_asymmetry` | `left_slope − right_slope` (smile tilt) |
| `pchip_left_diff`, `pchip_right_diff` | PCHIP prediction vs each side neighbour |

### Temporal Features
Capture how IV at this strike has evolved over time:

| Feature | Description |
|---|---|
| `prev_iv`, `prev2_iv`, `prev3_iv` | Last 3 known IVs at this strike column |
| `prev_dist`, `prev2_dist`, `prev3_dist` | How many rows back those observations are |
| `time_neighbor_mean` | Mean of last 3 known IVs |
| `time_iv_trend` | `prev_iv − prev2_iv` (recent momentum) |
| `pchip_vs_prev`, `pchip_vs_prev2` | PCHIP prediction vs recent known values |
| `col_pchip_vs_prev` | Temporal PCHIP vs most recent known |
| `n_prev_known` | Count of known values in this strike's history |

### Market Context Features

| Feature | Description |
|---|---|
| `underlying_price` | Current spot |
| `log_spot_change` | `log(S_t / S_{t-1})` intraday spot move |
| `abs_spot_change` | Absolute spot move |
| `tau_years` | Time to expiry in years |
| `sqrt_tau` | Square root of time (Black-Scholes scaling) |
| `minutes_from_open` | Intraday time (market opens at 09:15) |
| `day_index` | Which trading day (factorised) |

### Data Density Features

| Feature | Description |
|---|---|
| `n_known` | How many strikes have known IV in this row |
| `is_interpolation` | 1 if target is between known strikes, 0 if extrapolation |
| `known_density` | `n_known / total_strikes` |
| `dist_to_nearest_known` | Moneyness distance to nearest observed point |

---

## No-Lookahead Design

All interpolators are designed to never use future timestamps:

- **`col_pchip`** uses `train_matrix[:row, col]` — strictly past rows only.
- **`previous_col_values`** walks backward from `row-1` — past only.
- **`nearest_row_values`** operates within the current row only — no time dimension, no future.
- **`IDW`** slices `train_matrix[start:row, :]` for past rows (exclusive upper bound), then includes the full current row separately. The original bug was `start : row + 1` — the `+1` included `row` inside the bulk slice, which risks future-row leakage if the window logic shifts. The fix separates past rows and current row explicitly.

Same-row right-side strikes are **not** considered future data — at any given timestamp all strikes are observed contemporaneously, so using them is valid.

The `skip_self=True` flag during training ensures that when computing features for a known point, that point's own value is excluded from all interpolation inputs, faithfully simulating the missing-data scenario.

---

## Separate CE / PE Models

Call and Put IV surfaces are fit independently for several reasons:

- **Structural asymmetry**: Put IV is systematically elevated at low strikes due to the demand for downside protection (negative skew). Call IV behaves differently — flatter or even inverted at high strikes.
- **Smile shape differences**: The put wing carries more skew; the call wing is more driven by carry.
- **Different residual distributions**: The XGBoost correction magnitude and direction differs between CE and PE, so shared training would introduce bias.

Both use the same feature set and hyperparameters but are trained and predicted independently.

---

## Ensemble & Seed Diversity

Ten models are trained per surface per run, each with a different random seed:

```python
seeds = [42, 7, 123, 999, 2024, 17, 256, 314, 88, 500]
```

Final prediction is the **mean** across all 10 models. This reduces variance without requiring additional data, smooths over lucky/unlucky initialisation of XGBoost's subsampling, and produces a more stable surface.

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 850 | Enough depth for the residual signal; stopped before overfitting |
| `learning_rate` | 0.022 | Low shrinkage — each tree contributes less, reducing variance |
| `max_depth` | 6 | Moderate depth; captures interaction effects without memorising |
| `min_child_weight` | 20 | Minimum 20 samples per leaf — prevents splits on outliers |
| `subsample` | 0.9 | Row sampling; adds stochasticity and reduces overfitting |
| `colsample_bytree` | 0.9 | Feature sampling per tree |
| `reg_alpha` | 0.02 | L1 regularisation (feature sparsity) |
| `reg_lambda` | 0.1 | L2 regularisation (weight shrinkage) |

`min_child_weight=20` is particularly important near expiry — with very few time-steps remaining, leaf nodes could be split on just 2-3 noisy points. Setting this high forces the model to generalise rather than memorise.

---

## Validation

`run_validation()` implements a proper held-out validation:

1. 20% of known values are randomly masked (`make_validation_mask`).
2. Features are built on the **masked** matrix (with `skip_self=False` for validation targets, simulating prediction on truly unseen points).
3. Model is trained on the remaining 80% with residuals as the target.
4. RMSE is reported on the held-out set.

Three seeds are used and results are averaged: `mean RMSE ± std`.

This is a **temporal cross-validation in disguise** — because known values are sparsely distributed, masking 20% effectively tests generalisation across both strike and time dimensions.

---

## File Structure

```
├── train.csv                   # Input: raw IV surface with missing values
├── main.py                     # Entry point (mode: "validate" or "predict")
├── filled_surfacebackfc2ff.csv # Output: completed IV surface
```

---

## Usage

**Validate** (evaluate model quality on held-out known points):
```bash
python main.py  # mode="validate" in __main__
```

**Predict** (fill all missing values and save):
```bash
python main.py  # mode="predict" in __main__
```

Switch mode in the `if __name__ == "__main__"` block at the bottom of `main.py`.

---

## Dependencies

```
numpy
pandas
scipy
xgboost
tqdm
```

Install with:
```bash
pip install numpy pandas scipy xgboost tqdm
```

---

## Approaches Tried Before Final Pipeline

Several approaches were explored and discarded before arriving at the current PCHIP + XGBoost residual design.

### RBF Interpolation

Radial Basis Function interpolation was the first attempt. RBF operates in the full 2D `(log_moneyness, time)` space and fits a globally smooth surface through all known points. While it is mathematically elegant, it performed poorly on this dataset for a few reasons:

- RBF assumes a globally uniform smoothness — it applies the same kernel bandwidth everywhere. The IV surface is not globally uniform: it is steep and noisy near expiry (low `tau`) and flatter far from expiry. RBF cannot adapt to this.
- It does not encode any structure of the smile (symmetry around ATM, wing behaviour, skew direction). It just interpolates numbers in 2D without any financial intuition.
- As the number of known points grows, RBF requires solving a dense linear system whose cost scales as O(n³), making it slow for large datasets.

PCHIP respects the smile's local structure in the moneyness dimension and handles sparse data per row independently, which is a much better fit for this problem.

### Neural Network to Predict PCHIP Coefficients

The idea here was: instead of directly interpolating missing IV values, predict the PCHIP spline coefficients at missing points using a neural network trained on the known coefficients. The model would learn the temporal evolution of the spline shape.

This turned out to be a fundamental misunderstanding of what interpolation does. PCHIP's job is precisely to enforce smoothness constraints — it guarantees monotonicity and continuity by construction. By asking a neural network to predict the coefficients, the approach was effectively bypassing the smoothness guarantee and replacing it with a learned approximation that had no such guarantee. The outputs were rough and physically unreasonable. The network was making the predictions worse, not better, because it was taking over the one thing interpolation already does correctly. The right role for a learned model is to correct the residuals after interpolation, not to replace the interpolation itself — which is exactly what the final design does.

### SVI (Stochastic Volatility Inspired) Parametric Model

SVI was implemented and tested as both a baseline and a fallback for PCHIP. The raw SVI parametrisation fits a 5-parameter model to the smile:

```
w(k) = a + b * (ρ(k − m) + √((k − m)² + σ²))
```

where `w` is total variance and `k = log(K/S)`. The appeal is that SVI respects the no-arbitrage structure of the smile and extrapolates wings more faithfully than PCHIP.

In practice two problems prevented it from being included:

- **Speed**: fitting SVI requires running `scipy.optimize.curve_fit` per row, which is a nonlinear least-squares solve. With thousands of rows and many iterations, this added significant runtime that made the pipeline impractical for the dataset size.
- **Convergence failures**: for rows with few known points, sparse wing data, or noisy quotes near expiry, the SVI optimiser frequently failed to converge or returned degenerate parameters. Robust fallback logic mitigated this partially, but not enough to trust it across all rows.

SVI is retained as a future upgrade rather than a current component — see the section below.

---

## Future Upgrades

### SVI as Baseline (Adaptive Blend)

The current baseline is `row_pchip`. A stronger baseline would blend PCHIP with a fitted **Stochastic Volatility Inspired (SVI)** parametric smile:

```
w(k) = a + b * (ρ(k − m) + √((k − m)² + σ²))
```

where `w` is total variance and `k = log(K/S)`. SVI is grounded in the no-arbitrage structure of the smile, so it extrapolates the wings more faithfully than PCHIP, which can oscillate or flatten at the extremes.

The blend weight between PCHIP and SVI would be adaptive based on data availability:

```python
# rough idea
if n_known < 5:
    baseline = pchip          # not enough points to trust SVI fit
elif is_interpolation:
    baseline = 0.4 * pchip + 0.6 * svi   # SVI dominates in the interior
else:
    baseline = 0.2 * pchip + 0.8 * svi   # SVI dominates in the wings
```

More known points → higher trust in SVI. Extrapolation zones (OTM wings) → even higher SVI weight since PCHIP has no structural anchor there. This would shrink the residuals that XGBoost needs to correct, making the overall pipeline more accurate and stable.

SVI was implemented and tested in an earlier version of this pipeline but removed due to per-row `scipy.optimize.curve_fit` calls being too slow for a dataset with thousands of rows. The practical path to re-enabling it is vectorised fitting or pre-fitting SVI parameters in a batched loop with early stopping, then caching the results.

### SVI Parameters as XGBoost Features

Even without using SVI as the baseline, the 5 fitted SVI parameters `(a, b, ρ, m, σ)` are extremely informative features for XGBoost:

| Parameter | What it captures |
|---|---|
| `a` | Overall ATM variance level |
| `b` | Smile width / vega sensitivity |
| `ρ` | Skew direction (negative = put skew) |
| `m` | ATM shift / smile centre |
| `σ` | Smile curvature / butterfly |

These directly parameterise the shape that the current smile is trying to fit. Passing them to XGBoost would let the model condition its residual correction on the global smile regime, not just local neighbourhood features. For example, a high `|ρ|` signals a strongly skewed surface near expiry — the model could learn to apply a different correction magnitude in that regime versus a flat smile with low `|ρ|`.

The SVI prediction itself (`svi_fitted`) would also be a strong feature — essentially a parametric alternative to `quad_fitted` that respects no-arbitrage constraints.

### Other Potential Improvements

**LightGBM as an alternative to XGBoost** — LightGBM is generally faster on tabular data with many features and can match or exceed XGBoost accuracy with less tuning. Worth benchmarking on the same feature set.

**Separate wing models** — OTM wings (high `|log_moneyness|`) and ATM region behave differently. Training a dedicated model for each zone (split by `is_interpolation` and `abs_log_moneyness` threshold) could reduce systematic bias at the extremes.

**Temporal ensemble** — instead of averaging 10 seeds of the same model, train models on different time-window splits (e.g. first 60% of rows, first 80%, all rows) and average. This provides diversity in what each model has seen, which is more meaningful than seed diversity alone.
