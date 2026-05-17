"""
data_loader.py — Load CSV inputs and align them to a common timestamp grid.

If a CSV is missing or invalid, the loader falls back to the synthetic
generator and surfaces a warning so the user knows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from . import synthetic_data as syn
from . import config as C


@dataclass
class MarketData:
    """Container for time-aligned market & weather data."""
    timestamps: pd.DatetimeIndex
    da_price_eur_mwh: np.ndarray             # €/MWh
    id_price_eur_mwh: np.ndarray             # €/MWh
    outdoor_temp_c: np.ndarray               # °C
    dhw_draw_kwh: np.ndarray                 # kWh per step
    notes: List[str]                         # warnings / fallbacks

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)


def _read_csv(file) -> Optional[pd.DataFrame]:
    """Try to read a Streamlit UploadedFile / path / DataFrame; return None on failure."""
    if file is None:
        return None
    if isinstance(file, pd.DataFrame):
        return file.copy()
    try:
        df = pd.read_csv(file)
        if "timestamp" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception:
        return None


def load_market_data(
    start: pd.Timestamp,
    n_steps: int,
    timestep_h: float,
    da_file=None,
    id_file=None,
    weather_file=None,
    dhw_file=None,
    annual_dhw_kwh: float = 2_000.0,
    id_spread_std: float = C.ID_SPREAD_STD_EUR_MWH,
    seed: int = 42,
) -> MarketData:
    """Load all input series, falling back to synthetic data where needed.

    The output is guaranteed to have `n_steps` rows aligned to a regular
    `timestep_h` grid starting at `start`.
    """
    notes: List[str] = []
    idx = pd.date_range(start=start, periods=n_steps, freq=pd.Timedelta(hours=timestep_h))

    # ----- DA prices -----
    da_df = _read_csv(da_file)
    if da_df is not None and "da_price_eur_per_mwh" in da_df.columns:
        da_aligned = (
            da_df.set_index("timestamp")["da_price_eur_per_mwh"]
            .reindex(idx, method="ffill")
            .bfill()
        )
        if da_aligned.isna().any():
            notes.append("DA CSV did not cover the full horizon, gaps were filled forward.")
        da = da_aligned.values
    else:
        if da_file is not None:
            notes.append("DA CSV could not be parsed — using synthetic DA prices.")
        else:
            notes.append("No DA CSV uploaded — using synthetic DA prices.")
        da = syn.synthetic_da_prices(start, n_steps, timestep_h, seed=seed)[
            "da_price_eur_per_mwh"
        ].values

    # ----- ID prices -----
    id_df = _read_csv(id_file)
    if id_df is not None and "id_price_eur_per_mwh" in id_df.columns:
        id_aligned = (
            id_df.set_index("timestamp")["id_price_eur_per_mwh"]
            .reindex(idx, method="ffill")
            .bfill()
        )
        if id_aligned.isna().any():
            notes.append("ID CSV did not cover the full horizon, gaps were filled forward.")
        idp = id_aligned.values
    else:
        if id_file is not None:
            notes.append("ID CSV could not be parsed — deriving ID from DA + synthetic spread.")
        else:
            notes.append(
                "No ID CSV uploaded — deriving ID from DA + synthetic spread. "
                "ID-stage value should be replaced with real EPEX continuous data for a final business case."
            )
        idp = syn.synthetic_id_prices(
            pd.DataFrame({"timestamp": idx, "da_price_eur_per_mwh": da}),
            spread_std=id_spread_std,
            seed=seed + 1,
        )["id_price_eur_per_mwh"].values

    # ----- Weather -----
    w_df = _read_csv(weather_file)
    if w_df is not None and "outdoor_temp_c" in w_df.columns:
        w_aligned = (
            w_df.set_index("timestamp")["outdoor_temp_c"]
            .reindex(idx, method="ffill")
            .bfill()
        )
        if w_aligned.isna().any():
            notes.append("Weather CSV did not cover the full horizon, gaps were filled forward.")
        temp = w_aligned.values
    else:
        if weather_file is not None:
            notes.append("Weather CSV could not be parsed — using synthetic temperatures.")
        else:
            notes.append("No weather CSV uploaded — using synthetic German temperatures.")
        temp = syn.synthetic_weather(start, n_steps, timestep_h, seed=seed + 2)[
            "outdoor_temp_c"
        ].values

    # ----- DHW -----
    dhw_df = _read_csv(dhw_file)
    if dhw_df is not None and "dhw_draw_kwh" in dhw_df.columns:
        dhw_aligned = (
            dhw_df.set_index("timestamp")["dhw_draw_kwh"]
            .reindex(idx, method="ffill")
            .fillna(0.0)
        )
        dhw_draw = dhw_aligned.values
    else:
        dhw_draw = syn.synthetic_dhw_draw(idx, annual_dhw_kwh, timestep_h, seed=seed + 3)[
            "dhw_draw_kwh"
        ].values

    return MarketData(
        timestamps=idx,
        da_price_eur_mwh=np.array(da, dtype=float),
        id_price_eur_mwh=np.array(idp, dtype=float),
        outdoor_temp_c=np.array(temp, dtype=float),
        dhw_draw_kwh=np.array(dhw_draw, dtype=float),
        notes=notes,
    )


def write_sample_csvs(out_dir: str = "data") -> Tuple[str, str, str]:
    """Write sample DA / ID / weather CSVs covering one week."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    start = pd.Timestamp("2025-01-13 00:00:00")
    n = C.STEPS_PER_DAY * 7
    da = syn.synthetic_da_prices(start, n)
    idp = syn.synthetic_id_prices(da)
    w = syn.synthetic_weather(start, n)

    da_path = f"{out_dir}/sample_da_prices.csv"
    id_path = f"{out_dir}/sample_id_prices.csv"
    w_path = f"{out_dir}/sample_weather.csv"
    da.to_csv(da_path, index=False)
    idp.to_csv(id_path, index=False)
    w.to_csv(w_path, index=False)
    return da_path, id_path, w_path
