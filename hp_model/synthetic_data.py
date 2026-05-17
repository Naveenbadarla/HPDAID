"""
synthetic_data.py — Fallback data generators.

Used when the user has not uploaded real CSVs.  Produces:
  - Day-Ahead price time series
  - Intraday price time series (derived from DA with a configurable spread)
  - German-flavoured outdoor temperature time series
  - DHW daily draw profile

All series are returned as `pandas.DataFrame` objects indexed by timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _timestamps(start: pd.Timestamp, n_steps: int, timestep_h: float) -> pd.DatetimeIndex:
    freq = pd.Timedelta(hours=timestep_h)
    return pd.date_range(start=start, periods=n_steps, freq=freq)


# --------------------------------------------------------------------------
# Day-Ahead prices
# --------------------------------------------------------------------------
def synthetic_da_prices(
    start: pd.Timestamp,
    n_steps: int,
    timestep_h: float = C.DEFAULT_TIMESTEP_HOURS,
    mean: float = C.DA_PRICE_MEAN_EUR_MWH,
    std: float = C.DA_PRICE_STD_EUR_MWH,
    neg_fraction: float = C.NEG_PRICE_FRACTION,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate Germany-like 15-min DA prices.

    Shape:  baseline mean + daily double-peak + weekly weekend dip + noise.
    Negative-price events are injected at random hours.
    """
    rng = np.random.default_rng(seed)
    idx = _timestamps(start, n_steps, timestep_h)

    hours = np.asarray(idx.hour, dtype=float) + np.asarray(idx.minute, dtype=float) / 60.0
    weekday = np.asarray(idx.weekday, dtype=int)
    day_of_year = np.asarray(idx.dayofyear, dtype=float)

    # Daily double-peak (morning ~8h, evening ~19h)
    daily = (
        18 * np.sin((hours - 8) / 24 * 2 * np.pi) ** 2
        + 25 * np.exp(-((hours - 19) ** 2) / 6.0)
        - 10 * np.exp(-((hours - 13) ** 2) / 6.0)        # midday solar dip
    )
    # Weekly: weekends cheaper
    weekly = np.where(weekday >= 5, -12.0, 0.0)
    # Seasonal: winter expensive, summer cheap
    seasonal = 20 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)

    noise = rng.normal(0, std * 0.5, n_steps)

    prices = mean + daily + weekly + seasonal + noise
    prices = np.asarray(prices, dtype=float)

    # Inject negative-price events (rare, mostly midday weekends)
    n_neg = int(neg_fraction * n_steps)
    neg_candidates = np.where(
        ((hours >= 11) & (hours <= 15)) | (weekday >= 5)
    )[0]
    if len(neg_candidates) > 0 and n_neg > 0:
        idxs = rng.choice(neg_candidates, size=min(n_neg, len(neg_candidates)), replace=False)
        prices[idxs] = rng.uniform(-50, -2, size=len(idxs))

    return pd.DataFrame({"timestamp": idx, "da_price_eur_per_mwh": prices})


# --------------------------------------------------------------------------
# Intraday prices
# --------------------------------------------------------------------------
def synthetic_id_prices(
    da_df: pd.DataFrame,
    spread_std: float = C.ID_SPREAD_STD_EUR_MWH,
    vol_burst_prob: float = 0.05,
    seed: int = 43,
) -> pd.DataFrame:
    """Derive ID prices from DA prices.

    id_price = da_price + spread,  where `spread ~ N(0, spread_std)`.
    Volatility bursts widen the spread to 3x for a few hours at a time.
    """
    rng = np.random.default_rng(seed)
    n = len(da_df)
    spread = rng.normal(0.0, spread_std, n)

    # Volatility bursts (e.g. wind ramps, outages)
    in_burst = False
    burst_remaining = 0
    multiplier = np.ones(n)
    for i in range(n):
        if in_burst:
            multiplier[i] = 3.0
            burst_remaining -= 1
            if burst_remaining <= 0:
                in_burst = False
        elif rng.random() < vol_burst_prob / 96:        # rare
            in_burst = True
            burst_remaining = rng.integers(8, 24)        # 2–6 h bursts
    spread = spread * multiplier

    id_price = da_df["da_price_eur_per_mwh"].values + spread
    return pd.DataFrame(
        {"timestamp": da_df["timestamp"].values, "id_price_eur_per_mwh": id_price}
    )


# --------------------------------------------------------------------------
# Outdoor temperature
# --------------------------------------------------------------------------
def synthetic_weather(
    start: pd.Timestamp,
    n_steps: int,
    timestep_h: float = C.DEFAULT_TIMESTEP_HOURS,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate a Germany-like outdoor temperature profile.

    Annual mean ~10 °C, summer peak ~25 °C, winter trough ~-5 °C, daily swing
    ~6 °C, plus mild AR(1) noise to mimic weather persistence.
    """
    rng = np.random.default_rng(seed)
    idx = _timestamps(start, n_steps, timestep_h)

    day_of_year = np.asarray(idx.dayofyear, dtype=float) + np.asarray(idx.hour, dtype=float) / 24.0
    hour = np.asarray(idx.hour, dtype=float) + np.asarray(idx.minute, dtype=float) / 60.0

    seasonal = 10 + 15 * np.cos(2 * np.pi * (day_of_year - 200) / 365.0)
    # warmest at ~day 200 (≈ 19 July), coldest at day ~17 (≈ 17 Jan)
    daily = 3.0 * np.sin(2 * np.pi * (hour - 6) / 24.0)

    # AR(1) weather noise
    noise = np.zeros(n_steps)
    eps = rng.normal(0, 1.2, n_steps)
    rho = 0.95
    for i in range(1, n_steps):
        noise[i] = rho * noise[i - 1] + eps[i]

    temp = seasonal + daily + noise
    return pd.DataFrame({"timestamp": idx, "outdoor_temp_c": temp})


# --------------------------------------------------------------------------
# DHW draw profile
# --------------------------------------------------------------------------
def synthetic_dhw_draw(
    timestamps: pd.DatetimeIndex,
    annual_dhw_kwh: float,
    timestep_h: float = C.DEFAULT_TIMESTEP_HOURS,
    seed: int = 45,
) -> pd.DataFrame:
    """Generate a kWh-thermal DHW draw per timestep.

    Twin morning + evening peaks, low overnight.  Normalised so that
    `sum(draw) * (365 / horizon_days) == annual_dhw_kwh` on average.
    """
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    hours = np.asarray(timestamps.hour, dtype=float) + np.asarray(timestamps.minute, dtype=float) / 60.0

    # base hourly shape, will be normalised
    shape = (
        2.5 * np.exp(-((hours - 7.0) ** 2) / 2.0)        # morning
        + 3.0 * np.exp(-((hours - 19.0) ** 2) / 2.5)     # evening
        + 0.4 * np.exp(-((hours - 12.0) ** 2) / 6.0)     # midday
        + 0.1
    )
    shape = np.asarray(shape, dtype=float) * (1 + rng.normal(0, 0.15, n))
    shape = np.maximum(shape, 0.0)

    horizon_days = n * timestep_h / 24.0
    daily_target = annual_dhw_kwh / 365.0
    total_target = daily_target * horizon_days
    shape = shape / shape.sum() * total_target

    return pd.DataFrame({"timestamp": timestamps, "dhw_draw_kwh": shape})
