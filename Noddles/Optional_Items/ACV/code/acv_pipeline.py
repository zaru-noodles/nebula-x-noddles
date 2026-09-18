"""ACV refrigerant-leak localisation: peer-based anomaly ranking.

Implements the approach described in ``PS3/05_Implementation_Plan.md`` (section
"3. ACV"): with only six labelled fault cases, this avoids a conventional
supervised classifier. Instead, each of the 8 cars on a train is compared
against its 7 peers at every timestamp, and a car that persistently,
increasingly, or unrecoverably deviates from its peers is scored as more
likely to be the one with the refrigerant leak.

Every function here reads each workbook's own column headers dynamically —
one training file records ~8 ACV parameters per car, another records ~60 —
so nothing assumes a fixed column layout.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CAR_COL_RE = re.compile(r"^Car (\d+) - (.+)$")
METADATA_COLS = ("Car model", "Train number", "Time")

# Canonical feature slot -> ranked list of parameter-name substrings that may
# denote it in a given file's own headers, checked case-insensitively in
# priority order. Different case files name the same physical quantity
# differently (see the "basic" 8-parameter schema vs. the ~60-parameter
# schema in acv_case_04.xlsx), so this is a fuzzy lookup, not an exact map.
PARAM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "indoor_temp": (
        "Indoor Average Temperature",
        "Passenger Cabin Temperature Detected Value",
        "Passenger Cabin Temperature",
    ),
    "outdoor_temp": (
        "Outdoor Average Temperature",
        "Fresh Air Temperature Detected Value",
        "Fresh Air Temperature",
    ),
    "target_temp_cooling": (
        "Control Temperature (Cooling)",
        "Target Temperature Value",
    ),
    "target_temp_heating": ("Control Temperature (Heating)",),
    "running_mode": ("ACV Running Mode",),
    "setting_mode": ("ACV Setting Mode", "ACV Operating Mode", "ACV Control Mode"),
    "load_halved": ("Load Halved", "Load Shedding"),
    "info_valid": ("ACV Information Valid",),
}

# Substrings that mark a running-mode value as "actively cooling", used to
# scope cooling-target comparisons and to detect cooling-transition events.
COOLING_MODE_KEYWORDS = ("cool",)


def list_case_files(path: str) -> list[str]:
    """Return every real .xlsx case file at ``path`` (a file or a directory).

    Excel lock files (``~$case.xlsx``, created while the workbook is open
    elsewhere) are excluded — they aren't real data files and can't be read.
    """
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.xlsx")))
    else:
        candidates = [path]
    return [f for f in candidates if not os.path.basename(f).startswith("~$")]


def load_case(file_path: str) -> pd.DataFrame:
    """Read one case workbook, keeping its own column headers as-is."""
    return pd.read_excel(file_path)


def get_car_ids(df: pd.DataFrame) -> list[str]:
    """Return every car identifier found in the header, exactly as written.

    E.g. ``"03"`` — matching the identifier ``ranked_cars`` must use, per the
    ACV info kit — not a normalised/int form.
    """
    ids = set()
    for col in df.columns:
        m = CAR_COL_RE.match(col)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def _car_param_columns(df: pd.DataFrame, car_id: str) -> dict[str, str]:
    """Map this car's own parameter names -> full column name, for one car."""
    prefix = f"Car {car_id} - "
    return {
        col[len(prefix):]: col
        for col in df.columns
        if col.startswith(prefix)
    }


def _find_param_column(car_params: dict[str, str], canonical_key: str) -> str | None:
    """Fuzzy-match a canonical feature slot to one of this car's own columns."""
    keywords = PARAM_KEYWORDS[canonical_key]
    lower_params = {name.lower(): full_col for name, full_col in car_params.items()}
    for keyword in keywords:
        kw = keyword.lower()
        # Prefer an exact name match, then fall back to substring containment
        # (needed for the richer schema, e.g. "Target Temperature Value" is
        # an exact match, but other slots only appear as substrings).
        if kw in lower_params:
            return lower_params[kw]
        for name, full_col in lower_params.items():
            if kw in name:
                return full_col
    return None


@dataclass
class CarSeries:
    """One car's standardised telemetry for a case, aligned on ``time``."""

    car_id: str
    time: pd.Series
    indoor_temp: pd.Series
    outdoor_temp: pd.Series
    target_temp_cooling: pd.Series
    target_temp_heating: pd.Series
    running_mode: pd.Series
    setting_mode: pd.Series
    load_halved: pd.Series
    info_valid: pd.Series
    available: set[str] = field(default_factory=set)

    @property
    def valid_mask(self) -> pd.Series:
        """Rows this car's own information-valid flag marks as trustworthy.

        Step 3 of the plan: "Respect information-valid flags and exclude
        invalid observations." When a case's schema has no such flag for a
        car, every row is treated as valid rather than guessing.
        """
        if self.info_valid.isna().all():
            return pd.Series(True, index=self.time.index)
        return self.info_valid.fillna(False).astype(bool)


def _as_bool_flag(series: pd.Series) -> pd.Series:
    """Coerce a categorical/text/numeric column into a nullable boolean."""
    if series.dtype == bool:
        return series
    mapped = series.astype("string").str.strip().str.lower()
    truthy = {"true", "1", "yes", "valid", "y", "halved", "on"}
    falsy = {"false", "0", "no", "invalid", "n", "off"}
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[mapped.isin(truthy)] = True
    out[mapped.isin(falsy)] = False
    return out


def build_car_series(df: pd.DataFrame, car_id: str) -> CarSeries:
    """Extract one car's standardised fields from the raw case DataFrame."""
    params = _car_param_columns(df, car_id)
    available: set[str] = set()

    def numeric(key: str) -> pd.Series:
        col = _find_param_column(params, key)
        if col is None:
            return pd.Series(np.nan, index=df.index)
        available.add(key)
        return pd.to_numeric(df[col], errors="coerce")

    def categorical(key: str) -> pd.Series:
        col = _find_param_column(params, key)
        if col is None:
            return pd.Series(pd.NA, index=df.index, dtype="string")
        available.add(key)
        return df[col].astype("string")

    info_valid_col = _find_param_column(params, "info_valid")
    if info_valid_col is not None:
        available.add("info_valid")
        info_valid = _as_bool_flag(df[info_valid_col])
    else:
        info_valid = pd.Series(pd.NA, index=df.index, dtype="boolean")

    load_halved_col = _find_param_column(params, "load_halved")
    if load_halved_col is not None:
        available.add("load_halved")
        load_halved = _as_bool_flag(df[load_halved_col])
    else:
        load_halved = pd.Series(pd.NA, index=df.index, dtype="boolean")

    return CarSeries(
        car_id=car_id,
        time=df["Time"],
        indoor_temp=numeric("indoor_temp"),
        outdoor_temp=numeric("outdoor_temp"),
        target_temp_cooling=numeric("target_temp_cooling"),
        target_temp_heating=numeric("target_temp_heating"),
        running_mode=categorical("running_mode"),
        setting_mode=categorical("setting_mode"),
        load_halved=load_halved,
        info_valid=info_valid,
        available=available,
    )


def load_case_cars(file_path: str) -> tuple[pd.DataFrame, dict[str, CarSeries]]:
    """Load a case file and build every car's standardised series."""
    df = load_case(file_path)
    car_ids = get_car_ids(df)
    cars = {car_id: build_car_series(df, car_id) for car_id in car_ids}
    return df, cars


def _peer_median_excluding(values: pd.DataFrame, car_id: str) -> pd.Series:
    """At every timestamp, the median across every car except ``car_id``.

    This is the "robust residual from the other cars" of plan step 4: using
    the median (not the mean) keeps one faulty car from dragging its own
    peer baseline toward itself, and excluding the car itself avoids
    circularity.
    """
    peers = values.drop(columns=[car_id], errors="ignore")
    return peers.median(axis=1, skipna=True)


def _robust_z(x: pd.Series) -> pd.Series:
    """Median/MAD z-score, robust to the one genuinely faulty car."""
    median = x.median()
    mad = (x - median).abs().median()
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale < 1e-9:
        scale = x.std(ddof=0) or 1.0
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (x - median) / scale


def _linear_trend_slope(y: pd.Series) -> float:
    """Slope of y vs. sample index, ignoring NaNs; 0 if not enough points."""
    y = y.dropna()
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    slope, _ = np.polyfit(x, y.to_numpy(dtype=float), 1)
    return float(slope)


def _duty_deviation(this_series: pd.Series, peer_series_by_car: dict[str, pd.Series]) -> float:
    """Total-variation distance between this car's mode mix and its peers'.

    Captures step 5's "running-mode and load-halving duty cycles" as a
    peer-relative anomaly rather than a raw duty cycle, since a normal duty
    cycle can legitimately differ car-to-car (e.g. by position in train).
    """
    this_counts = this_series.dropna().value_counts(normalize=True)
    peer_frames = [s.dropna().value_counts(normalize=True) for s in peer_series_by_car.values()]
    if this_counts.empty or not peer_frames:
        return np.nan
    peer_avg = pd.concat(peer_frames, axis=1).mean(axis=1, skipna=True)
    all_categories = this_counts.index.union(peer_avg.index)
    a = this_counts.reindex(all_categories, fill_value=0.0)
    b = peer_avg.reindex(all_categories, fill_value=0.0)
    return float(0.5 * (a - b).abs().sum())


def _cooling_recovery_badness(residual: pd.Series, running_mode: pd.Series, window: int = 10) -> float:
    """How poorly the residual decays after entering an active-cooling mode.

    Step 5's "cooling recovery after control-mode transitions": at each
    transition into a cooling-related running mode, fit the residual's slope
    over the following ``window`` samples. A healthy car's temperature
    residual should fall back toward the peer baseline (negative slope); a
    leaking car recovers slowly or not at all (near-zero or positive slope).
    Returns the average of ``-slope`` across every detected transition
    (higher = worse recovery = more anomalous), or NaN if no transitions or
    no running-mode data are available.
    """
    if running_mode.isna().all():
        return np.nan
    is_cooling = running_mode.fillna("").str.lower().str.contains("|".join(COOLING_MODE_KEYWORDS))
    transitions = is_cooling & ~is_cooling.shift(1, fill_value=False)
    transition_idx = np.flatnonzero(transitions.to_numpy())
    if len(transition_idx) == 0:
        return np.nan
    badness = []
    values = residual.to_numpy(dtype=float)
    for start in transition_idx:
        end = min(start + window, len(values))
        segment = values[start:end]
        if np.count_nonzero(~np.isnan(segment)) < 3:
            continue
        x = np.arange(len(segment), dtype=float)
        mask = ~np.isnan(segment)
        slope, _ = np.polyfit(x[mask], segment[mask], 1)
        badness.append(-slope)
    if not badness:
        return np.nan
    return float(np.mean(badness))


def compute_case_features(file_path: str) -> pd.DataFrame:
    """Build the per-car engineered-feature table for one case file.

    Every feature is oriented so that a larger value means "more anomalous /
    more consistent with a refrigerant leak"; :func:`score_cars` relies on
    that convention when combining them.
    """
    _, cars = load_case_cars(file_path)
    car_ids = sorted(cars)

    indoor = pd.DataFrame({cid: cars[cid].indoor_temp.where(cars[cid].valid_mask) for cid in car_ids})
    features: dict[str, dict[str, float]] = {cid: {} for cid in car_ids}

    for cid in car_ids:
        car = cars[cid]
        valid = car.valid_mask
        this_indoor = indoor[cid]
        peer_indoor = _peer_median_excluding(indoor, cid)
        residual = (this_indoor - peer_indoor).where(valid)

        features[cid]["resid_mean_abs"] = float(residual.abs().mean(skipna=True))
        features[cid]["resid_persistent"] = float(
            residual.rolling(window=10, min_periods=5).mean().max(skipna=True)
        )
        features[cid]["resid_trend_slope"] = _linear_trend_slope(residual) or np.nan

        pooled_abs = pd.concat(
            [
                (indoor[c] - _peer_median_excluding(indoor, c)).where(cars[c].valid_mask)
                for c in car_ids
            ]
        ).abs()
        threshold = pooled_abs.median(skipna=True) + 3 * (pooled_abs - pooled_abs.median(skipna=True)).abs().median(skipna=True)
        threshold = threshold if np.isfinite(threshold) and threshold > 0 else 1.0
        denom = valid.sum()
        features[cid]["resid_time_above_thresh"] = (
            float((residual.abs() > threshold).sum() / denom) if denom else np.nan
        )

        if "target_temp_cooling" in car.available:
            is_cooling = car.running_mode.fillna("").str.lower().str.contains("|".join(COOLING_MODE_KEYWORDS))
            scope = is_cooling if is_cooling.any() else pd.Series(True, index=car.time.index)
            offset = (car.indoor_temp - car.target_temp_cooling).where(valid & scope)
            features[cid]["indoor_minus_target_mean"] = float(offset.mean(skipna=True))
        else:
            features[cid]["indoor_minus_target_mean"] = np.nan

        features[cid]["cooling_recovery_badness"] = _cooling_recovery_badness(residual, car.running_mode)

        peer_running = {c: cars[c].running_mode.where(cars[c].valid_mask) for c in car_ids if c != cid}
        features[cid]["running_mode_duty_dev"] = _duty_deviation(car.running_mode.where(valid), peer_running)

        peer_load = {c: cars[c].load_halved.astype("string").where(cars[c].valid_mask) for c in car_ids if c != cid}
        features[cid]["load_halved_duty_dev"] = _duty_deviation(
            car.load_halved.astype("string").where(valid), peer_load
        )

        n_rows = len(car.time)
        invalid_rate = float((~valid).sum() / n_rows) if n_rows else np.nan
        missing_numeric_rate = float(car.indoor_temp.isna().sum() / n_rows) if n_rows else np.nan
        features[cid]["missing_rate"] = np.nanmean([invalid_rate, missing_numeric_rate])

        if "indoor_minus_target_mean" in features[cid] and not np.isnan(features[cid]["indoor_minus_target_mean"]):
            peer_offsets = []
            for c in car_ids:
                if c == cid or "target_temp_cooling" not in cars[c].available:
                    continue
                other = cars[c]
                other_valid = other.valid_mask
                other_is_cooling = other.running_mode.fillna("").str.lower().str.contains("|".join(COOLING_MODE_KEYWORDS))
                other_scope = other_is_cooling if other_is_cooling.any() else pd.Series(True, index=other.time.index)
                other_offset = (other.indoor_temp - other.target_temp_cooling).where(other_valid & other_scope)
                peer_offsets.append(other_offset.mean(skipna=True))
            peer_offset_median = np.nanmedian(peer_offsets) if peer_offsets else np.nan
            features[cid]["target_offset_vs_peers"] = features[cid]["indoor_minus_target_mean"] - peer_offset_median
        else:
            features[cid]["target_offset_vs_peers"] = np.nan

    return pd.DataFrame.from_dict(features, orient="index").sort_index()


FEATURE_COLUMNS = (
    "resid_mean_abs",
    "resid_persistent",
    "resid_trend_slope",
    "resid_time_above_thresh",
    "target_offset_vs_peers",
    "cooling_recovery_badness",
    "running_mode_duty_dev",
    "load_halved_duty_dev",
    "missing_rate",
)

DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 / len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS}


def zscore_features(features: pd.DataFrame) -> pd.DataFrame:
    """Robust-z-score every feature column across the cars within one case.

    This does not depend on feature weights, so it's the expensive part of
    scoring (per-column median/MAD over the case's cars) and should be done
    once per case and reused — see :func:`score_cars` for the single-shot
    convenience path, and ``evaluate.py`` for why a weight search precomputes
    this instead of recomputing it on every candidate weight vector.
    """
    z = pd.DataFrame(index=features.index, columns=list(FEATURE_COLUMNS), dtype=float)
    for col in FEATURE_COLUMNS:
        if col in features.columns:
            z[col] = _robust_z(features[col]).fillna(0.0)
        else:
            z[col] = 0.0
    return z


def combine_scores(z: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted sum of already-z-scored features -> one anomaly score per car.

    Cheap and weight-dependent only — the part of scoring that a weight
    search should repeat on every trial, once :func:`zscore_features` has
    been computed a single time per case.
    """
    weights = weights or DEFAULT_WEIGHTS
    w = np.array([weights.get(col, 0.0) for col in z.columns])
    return pd.Series(z.to_numpy() @ w, index=z.index)


def score_cars(features: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Combine per-car features into one weighted anomaly score.

    Each feature column is robust-z-scored across the cars *within this
    case* (so scores are comparable regardless of a feature's raw units),
    missing values are filled with 0 (neutral / no evidence either way), and
    the result is a non-negative-weighted sum — higher score = more likely
    to be the faulty car. Convenience wrapper around
    :func:`zscore_features` + :func:`combine_scores` for one-off use (e.g.
    ``predict.py``); prefer calling them separately when scoring the same
    case under many different weight vectors.
    """
    return combine_scores(zscore_features(features), weights)


def rank_cars(scores: pd.Series) -> list[str]:
    """Car identifiers ordered from most- to least-likely faulty."""
    return list(scores.sort_values(ascending=False).index)


def predict_case(file_path: str, weights: dict[str, float] | None = None) -> list[str]:
    """End-to-end: one case file -> a full ranked list of car identifiers."""
    features = compute_case_features(file_path)
    scores = score_cars(features, weights)
    return rank_cars(scores)
