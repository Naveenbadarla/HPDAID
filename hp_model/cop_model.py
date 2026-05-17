"""
cop_model.py — Simplified Coefficient-of-Performance model.

The COP curve is a linear function of the outdoor temperature:

    COP(T_out) = a + b * T_out                  (clamped to [cop_min, cop_max])

This model deliberately keeps the optimisation linear: COP is treated as a
known, time-varying coefficient.  This matches industry practice for first-
order flexibility studies and avoids the bilinearity that would appear if
the supply temperature was a decision variable.

A separate (lower) COP curve is used for DHW because DHW requires a higher
supply temperature than space heating.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from . import config as C


def cop_curve_params(emitter: str) -> Dict[str, float]:
    """Return COP curve parameters for the requested space-heating emitter."""
    emitter = emitter.lower()
    if emitter not in C.COP_DEFAULTS:
        raise KeyError(f"Unknown emitter type: {emitter}")
    return dict(C.COP_DEFAULTS[emitter])


def cop_series(outdoor_temp_c: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Vectorised COP(T_out)."""
    raw = params["a"] + params["b"] * np.asarray(outdoor_temp_c, dtype=float)
    return np.clip(raw, params["cop_min"], params["cop_max"])


def cop_space_heating(outdoor_temp_c: np.ndarray, emitter: str) -> np.ndarray:
    """Convenience wrapper for space-heating COP."""
    return cop_series(outdoor_temp_c, cop_curve_params(emitter))


def cop_dhw(outdoor_temp_c: np.ndarray) -> np.ndarray:
    """COP for DHW heating (uses the dedicated dhw curve)."""
    return cop_series(outdoor_temp_c, C.COP_DEFAULTS["dhw"])


def average_cop_from_series(thermal_kwh: np.ndarray, electricity_kwh: np.ndarray) -> float:
    """Energy-weighted seasonal COP (SCOP)."""
    el = float(np.sum(electricity_kwh))
    if el <= 1e-9:
        return float("nan")
    return float(np.sum(thermal_kwh) / el)
