"""
thermal_model.py — Single-zone first-order RC building model.

Discrete-time update used in both the baseline simulator and the LP:

    T_in[t+1] = T_in[t]
                + dt / C_th  * (
                      q_sh[t]/dt              # heat delivered (kW)
                    - UA * (T_in[t] - T_out[t])
                    + Q_gain[t]
                )

`q_sh[t]` is in kWh (energy delivered during the step), so dividing by `dt`
converts it back to an average power.  The equation is linear in
`T_in[t]` and `q_sh[t]` — perfect for the LP.

Heating demand is assumed to only occur when `T_out < HEATING_THRESHOLD_C`,
but the simulator allows heat input at any time (this matters for pre-cool
flexibility in summer / shoulder season — not used in MVP).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import config as C
from .archetypes import Archetype


@dataclass
class BuildingState:
    """Lightweight container for time-series of the building state."""
    t_in_c: np.ndarray
    q_sh_kwh: np.ndarray


def update_temperature(
    t_in: float,
    t_out: float,
    q_sh_kwh: float,
    arch: Archetype,
    timestep_h: float,
    q_gain_kw: float = C.INTERNAL_GAIN_KW,
) -> float:
    """One-step Euler update of indoor temperature."""
    avg_power_kw = q_sh_kwh / timestep_h
    dTdt = (avg_power_kw - arch.ua_kw_per_k * (t_in - t_out) + q_gain_kw) / arch.c_th_kwh_per_k
    return t_in + dTdt * timestep_h


def coefficients(
    arch: Archetype, timestep_h: float
):
    """Pre-compute LP coefficients for the discrete RC update.

    Returns (alpha_T, alpha_q, alpha_out, alpha_gain) such that

        T_in[t+1] = alpha_T * T_in[t]
                  + alpha_q * q_sh[t]
                  + alpha_out * T_out[t]
                  + alpha_gain * Q_gain[t]

    where q_sh is in kWh, Q_gain in kW, T's in °C.
    """
    dt = timestep_h
    ua = arch.ua_kw_per_k
    c = arch.c_th_kwh_per_k
    alpha_T = 1.0 - dt * ua / c
    alpha_q = 1.0 / c                            # kWh -> K
    alpha_out = dt * ua / c
    alpha_gain = dt / c
    return alpha_T, alpha_q, alpha_out, alpha_gain
