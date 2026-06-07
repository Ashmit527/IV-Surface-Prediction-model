from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit, minimize
import warnings
from tqdm import tqdm


class IVSurface:
    def __init__(self, name , type: str):
        self.name = name
        self.df = pd.read_csv(name)
        self._cols = [c for c in self.df.columns if c.endswith(type)]
        self.raw = self.df[["datetime", "underlying_price"] + self._cols]
        self.option_cols = [c for c in self.raw.columns if c not in ("datetime", "underlying_price")]
        self.datetimes = pd.to_datetime(self.raw["datetime"], dayfirst=True, errors="raise")
        self.spot = self.raw["underlying_price"].to_numpy(float)
        self.matrix = self.raw[self.option_cols].to_numpy(float)
        self.known_mask = np.isfinite(self.matrix)
        self.missing_mask = ~self.known_mask
        OPTION_RE = re.compile(r"(?P<strike>\d{5})(?P<type>CE|PE)$")

        strikes: list[int] = []
        for col in self.option_cols:
            match = OPTION_RE.search(col)
            if not match:
                raise ValueError(f"Cannot parse option column: {col}")
            strikes.append(int(match.group("strike")))
        self.strikes = np.asarray(strikes, dtype=float)
        self.log_m = np.log(self.strikes[None, :] / self.spot[:, None])

        expiry = pd.Timestamp(2026, 1, 27, 15, 30)
        tau = (expiry - self.datetimes).dt.total_seconds().to_numpy() / (365.0 * 24.0 * 3600.0)
        self.tau = np.maximum(tau, 1.0 / (365.0 * 24.0 * 60.0))
        market_open = self.datetimes.dt.normalize() + pd.Timedelta(hours=9, minutes=15)
        self.minutes = ((self.datetimes - market_open).dt.total_seconds() / 60.0).to_numpy()
        self.day_index = pd.factorize(self.datetimes.dt.date)[0].astype(float)
        self.time_index = np.arange(len(self.raw), dtype=float)


def pchip_predict(x_values: np.ndarray, y_values: np.ndarray, x_target: float) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() == 0:
        return np.nan
    x = x_values[valid]
    y = y_values[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if len(x) == 1:
        return float(y[0])
    if x_target <= x[0] and len(x) >= 3:
        # fit quadratic through first 3 points
        coeffs = np.polyfit(x[:3], y[:3], 2)
        return float(np.polyval(coeffs, x_target))

    if x_target >= x[-1] and len(x) >= 3:
        # fit quadratic through last 3 points
        coeffs = np.polyfit(x[-3:], y[-3:], 2)
        return float(np.polyval(coeffs, x_target))
    return float(PchipInterpolator(x, y, extrapolate=False)(x_target))     


def safe_nanmean(values: list[float] | np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return default
    return float(np.mean(arr[valid]))


def safe_nanstd(values: list[float] | np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return default
    return float(np.std(arr[valid]))


def nearest_row_values(matrix: np.ndarray, row: int, col: int, skip_self: bool) -> tuple[float, float, int, int]:
    left_value = np.nan
    right_value = np.nan
    left_dist = 99
    right_dist = 99
    for j in range(col - 1, -1, -1):
        if np.isfinite(matrix[row, j]) and not (skip_self and j == col):
            left_value = matrix[row, j]
            left_dist = col - j
            break
    for j in range(col + 1, matrix.shape[1]):
        if np.isfinite(matrix[row, j]) and not (skip_self and j == col):
            right_value = matrix[row, j]
            right_dist = j - col
            break
    return left_value, right_value, left_dist, right_dist


def previous_col_values(matrix: np.ndarray, row: int, col: int, count: int = 3) -> tuple[list[float], list[int]]:
    values: list[float] = []
    distances: list[int] = []
    for i in range(row - 1, -1, -1):
        if np.isfinite(matrix[i, col]):
            values.append(float(matrix[i, col]))
            distances.append(row - i)
            if len(values) == count:
                break
    while len(values) < count:
        values.append(np.nan)
        distances.append(999)
    return values, distances


def make_validation_mask(surface: IVSurface, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid_mask = np.zeros_like(surface.known_mask, dtype=bool)
    known_targets = np.argwhere(surface.known_mask)
    n_valid = int(round(len(known_targets) * fraction))
    chosen = rng.choice(len(known_targets), size=n_valid, replace=False)
    valid_mask[known_targets[chosen, 0], known_targets[chosen, 1]] = True
    return valid_mask


class FeatureBuilder:
    def __init__(self, surface: IVSurface, train_matrix: np.ndarray):
        self.surface = surface
        self.train_matrix = train_matrix
        self.idw_window = 20
        self._svi_cache = {}

    def idw_predict(self, row: int, col: int, skip_self: bool) -> float:
        start = max(0, row - self.idw_window)
        rows, cols = np.where(np.isfinite(self.train_matrix[start : row + 1, :]))
        if len(rows) == 0:
            return np.nan
        rows = rows + start
        if skip_self:
            keep = ~((rows == row) & (cols == col))
            rows = rows[keep]
            cols = cols[keep]
        else:
            keep = ~((rows == row) & (cols == col) & ~np.isfinite(self.train_matrix[row, col]))
            rows = rows[keep]
            cols = cols[keep]
        if len(rows) == 0:
            return np.nan

        dt = (row - rows) / 10.0
        dx = (self.surface.log_m[row, col] - self.surface.log_m[rows, cols]) / 0.01
        distances = np.sqrt(dt * dt + dx * dx)
        order = np.argsort(distances)[:20]
        distances = distances[order]
        values = self.train_matrix[rows[order], cols[order]]
        weights = 1.0 / np.maximum(distances, 1e-6) ** 2
        return float(np.sum(weights * values) / np.sum(weights))
    

    
    def row_pchip(self, row: int, col: int, skip_self: bool) -> float:
        x_values = self.surface.log_m[row, :][self.surface.known_mask[row, :]]
        y_values = self.surface.matrix[row, :][self.surface.known_mask[row, :]]
        x_target = self.surface.log_m[row, col]
        if skip_self:
            known_cols = np.where(self.surface.known_mask[row, :])[0]
            exclude_mask = known_cols != col 
            x = x_values[exclude_mask]
            y = y_values[exclude_mask]
        else:
            x = x_values
            y = y_values
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        x, unique_idx = np.unique(x, return_index=True)
        y = y[unique_idx]
        if len(x) == 1:
            return float(y[0])
        if x_target <= x[0] and len(x) >= 3:
    # fit quadratic through first 3 points
            coeffs = np.polyfit(x[:3], y[:3], 2)
            return float(np.polyval(coeffs, x_target))
        if x_target >= x[-1] and len(x) >= 3:
            # fit quadratic through last 3 points
            coeffs = np.polyfit(x[-3:], y[-3:], 2)
            return float(np.polyval(coeffs, x_target))
        return float(PchipInterpolator(x, y, extrapolate=False)(x_target))     

    def col_pchip(self, row: int, col: int, skip_self: bool) -> tuple[float, float]:
        row_values = self.train_matrix[row, :]
        if skip_self:
            row_values = row_values.copy()
            row_values[col] = np.nan
        col_pchip = pchip_predict(
            self.surface.time_index[:row],
            self.train_matrix[:row, col],
            self.surface.time_index[row],
        )
        return col_pchip
    

    def build(self, targets: np.ndarray, skip_self: bool) -> pd.DataFrame:
        rows: list[list[float]] = []
        for row, col in targets.astype(int):
            row_values = self.train_matrix[row, :]
            col_values = self.train_matrix[:row, col]
            if skip_self:
                row_values = row_values.copy()
                row_values[col] = np.nan
            row_pchip =self.row_pchip(row, col, skip_self=skip_self)
            col_pchip = self.col_pchip(row, col, skip_self=skip_self)
            row_time_blend = 0.75 * row_pchip + 0.25 * col_pchip
            idw = self.idw_predict(row, col, skip_self=skip_self)

            left_iv, right_iv, left_dist, right_dist = nearest_row_values(self.train_matrix, row, col, skip_self)
            prev_values, prev_distances = previous_col_values(self.train_matrix, row, col, count=3)
            prev_iv, prev2_iv, prev3_iv = prev_values
            prev_dist, prev2_dist, prev3_dist = prev_distances

            row_neighbor_mean = safe_nanmean([left_iv, right_iv])
            time_neighbor_mean = safe_nanmean(prev_values)
            row_mean = safe_nanmean(row_values)
            row_std = safe_nanstd(row_values)
            col_mean = safe_nanmean(col_values)
            col_std = safe_nanstd(col_values)

            known_cols   = np.where(self.surface.known_mask[row, :])[0]
            n_known      = len(known_cols)
            x_target     = self.surface.log_m[row, col]
            x_known      = self.surface.log_m[row, known_cols]
            is_interp    = float(
                len(x_known) > 0
                and x_known.min() <= x_target <= x_known.max()
            )
            # compute
            spot_prev        = self.surface.spot[row-1] if row > 0 else self.surface.spot[row]
            log_spot_change  = np.log(self.surface.spot[row] / spot_prev)
            smile_convexity  = left_iv + right_iv - 2 * row_pchip
            pchip_vs_prev    = row_pchip - prev_iv
            pchip_vs_prev2   = row_pchip - prev2_iv
            pchip_left_diff  = row_pchip - left_iv
            pchip_right_diff = row_pchip - right_iv
            left_slope       = (row_pchip - left_iv)  / max(left_dist,  1)
            right_slope      = (right_iv  - row_pchip) / max(right_dist, 1)
            slope_asymmetry  = left_slope - right_slope
            time_iv_trend    = prev_iv - prev2_iv
            known_x          = self.surface.log_m[row, known_cols]
            dist_to_nearest  = float(np.min(np.abs(x_target - known_x))) if len(known_x) > 0 else 99.0
            known_density    = n_known / self.train_matrix.shape[1]
            n_prev_known     = float(np.sum(np.isfinite(self.train_matrix[:row, col])))
            col_pchip_vs_prev = col_pchip - prev_iv
            # find the strike closest to ATM (log_moneyness closest to 0)
            atm_col = int(np.argmin(np.abs(self.surface.log_m[row, :])))
            atm_iv = self.train_matrix[row, atm_col]
            if np.isnan(atm_iv):
                # fallback to row mean
                atm_iv = row_mean
            # how far is this strike's IV from ATM IV
            iv_minus_atm = row_pchip - atm_iv

            # ratio version
            iv_ratio_atm = row_pchip / atm_iv if atm_iv > 0 else 1.0
            known_cols_arr = np.where(np.isfinite(row_values))[0]
            known_ivs_arr  = row_values[known_cols_arr]
            known_logm_arr = self.surface.log_m[row, known_cols_arr]

            # left wing = most OTM put (lowest log_m)
            # right wing = most OTM call (highest log_m)
            otm_put_iv  = float(known_ivs_arr[np.argmin(known_logm_arr)])
            otm_call_iv = float(known_ivs_arr[np.argmax(known_logm_arr)])

            risk_reversal = otm_put_iv - otm_call_iv      # skew direction
            butterfly     = (otm_put_iv + otm_call_iv) / 2 - atm_iv  # curvature
            x = self.surface.log_m[row, col]
            if len(known_logm_arr) >= 2:
                skew_slope = float(np.polyfit(known_logm_arr, known_ivs_arr, 1)[0])
            else:
                skew_slope = 0.0
            if row > 0:
                prev_known = np.where(np.isfinite(self.train_matrix[row-1, :]))[0]
                if len(prev_known) >= 2:
                    prev_ivs  = self.train_matrix[row-1, prev_known]
                    prev_logm = self.surface.log_m[row-1, prev_known]
                    prev_skew_slope = float(np.polyfit(prev_logm, prev_ivs, 1)[0])
                    skew_slope_change = skew_slope - prev_skew_slope
                else:
                    skew_slope_change = 0.0
            else:
                skew_slope_change = 0.0
            if len(known_logm_arr) >= 3:
                coeffs = np.polyfit(known_logm_arr, known_ivs_arr, 2)
                quad_coeff   = float(coeffs[0])  # curvature (positive = smile, negative = frown)
                linear_coeff = float(coeffs[1])  # skew direction
                quad_fitted  = float(np.polyval(coeffs, self.surface.log_m[row, col]))
            else:
                quad_coeff   = 0.0
                linear_coeff = 0.0
                quad_fitted  = row_pchip
            rows.append(
                [
                    row_pchip,
                    col_pchip,
                    row_time_blend,
                    idw,
                    left_iv,
                    right_iv,
                    left_dist,
                    right_dist,
                    prev_iv,
                    prev2_iv,
                    prev3_iv,
                    prev_dist,
                    prev2_dist,
                    prev3_dist,
                    row_neighbor_mean,
                    time_neighbor_mean,
                    row_mean,
                    row_std,
                    col_mean,
                    col_std,
                    x,
                    abs(x),
                    x * x,
                    self.surface.strikes[col] / self.surface.spot[row],
                    self.surface.spot[row],
                    self.surface.strikes[col],
                    self.surface.tau[row],
                    np.sqrt(self.surface.tau[row]),
                    self.surface.minutes[row],
                    self.surface.day_index[row],
                    n_known,
                    is_interp,
                    # append
                    smile_convexity,
                    pchip_vs_prev,
                    pchip_vs_prev2,
                    pchip_left_diff,
                    pchip_right_diff,
                    log_spot_change,
                    abs(log_spot_change),
                    left_slope,
                    right_slope,
                    slope_asymmetry,
                    time_iv_trend,
                    dist_to_nearest,
                    known_density,
                    n_prev_known,
                    col_pchip_vs_prev,
                    # rows.append
                    atm_iv,
                    iv_minus_atm,
                    iv_ratio_atm,
                    risk_reversal,
                    butterfly,
                    skew_slope,
                    skew_slope_change,
                    quad_coeff,
                    linear_coeff,
                    quad_fitted,
                    
                ]
            )

        columns = [
            "row_pchip",
            "col_pchip",
            "row_time_blend",
            "idw",
            "left_iv",
            "right_iv",
            "left_dist",
            "right_dist",
            "prev_iv",
            "prev2_iv",
            "prev3_iv",
            "prev_dist",
            "prev2_dist",
            "prev3_dist",
            "row_neighbor_mean",
            "time_neighbor_mean",
            "row_mean",
            "row_std",
            "col_mean",
            "col_std",
            "log_moneyness",
            "abs_log_moneyness",
            "log_moneyness_sq",
            "strike_over_spot",
            "underlying_price",
            "strike",
            "tau_years",
            "sqrt_tau",
            "minutes_from_open",
            "day_index",
            "n_known",
            "is_interpolation",
            # columns
            "smile_convexity",
            "pchip_vs_prev",
            "pchip_vs_prev2",
            "pchip_left_diff",
            "pchip_right_diff",
            "log_spot_change",
            "abs_spot_change",
            "left_slope",
            "right_slope",
            "slope_asymmetry",
            "time_iv_trend",
            "dist_to_nearest_known",
            "known_density",
            "n_prev_known",
            "col_pchip_vs_prev",
            # columns
            "atm_iv",
            "iv_minus_atm",
            "iv_ratio_atm",
            "risk_reversal",
            "butterfly",
            "skew_slope",
            "skew_slope_change",
            "quad_coeff",
            "linear_coeff",
            "quad_fitted",
            
        ]
        
        return pd.DataFrame(rows, columns=columns).fillna(-1.0)
    def adaptive_blend_from_df(self, df: pd.DataFrame) -> np.ndarray:
        pchip_val = df["row_pchip"].to_numpy()
        svi_val   = df["row_svi"].to_numpy()
        n_known   = df["n_known"].to_numpy()
        is_interp = df["is_interpolation"].to_numpy().astype(bool)

        w_pchip = np.full(len(df), 0.1)
        w_svi   = np.full(len(df), 0.9)

        few    = is_interp & (n_known < 5)
        medium = is_interp & (n_known >= 5) & (n_known < 8)
        many   = is_interp & (n_known >= 8)

        w_pchip[few]    = 0.2;  w_svi[few]    = 0.8
        w_pchip[medium] = 0.4;  w_svi[medium] = 0.6
        w_pchip[many]   = 0.65; w_svi[many]   = 0.35

        return w_pchip * pchip_val + w_svi * svi_val
    


def fit_model(x_train: pd.DataFrame, y_train: np.ndarray, seed: int, type: str, learning_rate: float, min_child_weight: int, max_depth: int, n_estimators: int) -> xgb.XGBRegressor:
    if type == "CE":
        modelce = xgb.XGBRegressor(
            objective="reg:squarederror",  
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,                    
            min_child_weight=min_child_weight,            
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=0.1,
            random_state=seed,
            verbosity=0,                    
        )
        modelce.fit(x_train, y_train)
        return modelce
    else:
        modelpe = xgb.XGBRegressor(
            objective="reg:squarederror",  
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,                    
            min_child_weight=min_child_weight,            
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=0.1,
            random_state=seed,
            verbosity=0,                    
        )
        modelpe.fit(x_train, y_train)
        return modelpe


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "count": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "median_ae": float(np.median(np.abs(error))),
    }



def complete_matrix(surface: IVSurface, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    completed = surface.matrix.copy()
    for (row, col), prediction in zip(targets.astype(int), predictions):
        completed[row, col] = prediction
    return completed


def run_validation(surface: IVSurface, seed: int, type: str, learning_rate: float,
                   min_child_weight: int, max_depth: int, n_estimators: int) -> dict[str, float]:
    valid_mask   = make_validation_mask(surface, seed=seed, fraction=0.2)
    train_matrix = surface.matrix.copy()
    train_matrix[valid_mask] = np.nan

    train_targets = np.argwhere(np.isfinite(train_matrix))
    valid_targets = np.argwhere(valid_mask)
    y_train = train_matrix[train_targets[:, 0], train_targets[:, 1]]
    y_valid = surface.matrix[valid_targets[:, 0], valid_targets[:, 1]]

    print("Building train features...")
    builder = FeatureBuilder(surface, train_matrix)
    x_train = builder.build(train_targets, skip_self=True)

    print("Building validation features...")
    x_valid = builder.build(valid_targets, skip_self=False)
    residual_target = y_train - x_train["row_pchip"].to_numpy()  # ← blend baseline for train

    predictions = []
    seeds = [42, 7, 123, 999, 2024, 17, 256, 314, 88, 500]

    if type=="CE":
        fraction = 0.425
    else:
        fraction = 0.475


    for model_seed in tqdm(seeds, desc="Validation models"):
        model = fit_model(
            x_train, residual_target,
            seed=model_seed, type=type,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        predictions.append(
            x_valid["row_pchip"].to_numpy() + model.predict(x_valid) * fraction
        )

    prediction = np.maximum(np.mean(predictions, axis=0), 1e-5)
    return score(y_valid, prediction)


def fit_and_predict_missing(surface: IVSurface, type: str, fraction: float) -> None:
    train_targets   = np.argwhere(surface.known_mask)
    missing_targets = np.argwhere(surface.missing_mask)
    y_train         = surface.matrix[train_targets[:, 0], train_targets[:, 1]]

    print("Building train features...")
    builder   = FeatureBuilder(surface, surface.matrix)
    x_train   = builder.build(train_targets, skip_self=True)

    print("Building missing features...")
    x_missing = builder.build(missing_targets, skip_self=False)

    residual_target = y_train - x_train["row_pchip"].to_numpy()  # ← blend baseline for train

    predictions = []
    seeds = [42, 7, 123, 999, 2024, 17, 256, 314, 88, 500]

    for model_seed in tqdm(seeds, desc="Prediction models"):
        model = fit_model(
            x_train, residual_target,
            seed=model_seed,
            type=type,
            learning_rate=0.022,
            min_child_weight=20,
            max_depth=6,
            n_estimators=850
        )
        predictions.append(
            x_missing["row_pchip"].to_numpy() + model.predict(x_missing) * fraction)

    missing_pred = np.maximum(np.mean(predictions, axis=0), 1e-5)
    completed    = complete_matrix(surface, missing_targets, missing_pred)
    return completed
    

def main(mode: str) -> None:
    if mode == "validate":
        for type in ["CE", "PE"]:
            surface = IVSurface("train.csv", type=type)
            results = []
            for seed in [42, 7, 123]:
                # plug into your run_validation or fit_and_predict
                score = run_validation(surface, n_estimators=850, learning_rate=0.022,
                                    max_depth=6, min_child_weight=20,type=type, seed=seed)
                results.append(score['rmse'])
            avg_rmse = np.mean(results)
            std_rmse = np.std(results)
            print(f"rmse={avg_rmse:.8f} ± {std_rmse:.8f}")
    else:
        suf = {}
        comp = {}
        for type in ["CE", "PE"]:
            if type=="CE":
                fraction = 0.625
            else:
                fraction = 0.675
            surface = IVSurface("train.csv", type=type)
            suf[type] = surface
            comp[type] = fit_and_predict_missing(surface,type=type,fraction=fraction)
        df = pd.read_csv("train.csv")
        for type in ["CE", "PE"]:
            for col_idx, col_name in enumerate(suf[type].option_cols):
                df[col_name] = comp[type][:, col_idx]

        df.to_csv("filled_surfacebackfc2ff.csv", index=False)
        
if __name__ == "__main__":
    main(mode="predict")
