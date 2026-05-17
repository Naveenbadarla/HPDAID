"""
dhw_model.py — Domestic hot-water tank as a thermal store.

We track the *usable* thermal energy in the tank in kWh.  Bounds are
expressed as fractions of the tank size and translated into kWh.

    E_dhw[t+1] = E_dhw[t] + q_dhw[t] - draw[t] - loss

`loss` is a constant standing-loss per step (linear, easy to keep in LP).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .archetypes import Archetype


@dataclass
class DHWBounds:
    e_min_kwh: float       # minimum usable energy (~ 45 °C threshold)
    e_max_kwh: float       # maximum usable energy (~ 55 °C threshold)
    e_init_kwh: float
    loss_kwh_per_step: float


def dhw_bounds(arch: Archetype, frac_min: float = 0.30, frac_max: float = 1.0) -> DHWBounds:
    """Translate the tank capacity into LP bounds (kWh)."""
    e_max = arch.dhw_tank_kwh * frac_max
    e_min = arch.dhw_tank_kwh * frac_min
    e_init = 0.7 * e_max
    return DHWBounds(
        e_min_kwh=e_min,
        e_max_kwh=e_max,
        e_init_kwh=e_init,
        loss_kwh_per_step=arch.dhw_tank_loss_kwh_per_step,
    )


def dhw_step(e: float, q_in: float, draw: float, loss: float) -> float:
    """One-step Euler update of tank energy."""
    return e + q_in - draw - loss
