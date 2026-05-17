"""
optimizer_id.py — Intraday re-optimisation wrapper.

Conceptually this is the SAME LP as the DA optimiser, but called with the
ID price series.  The DA schedule is treated as a financial commitment —
the physical re-optimisation against ID prices is what determines the
"actual" delivered consumption.

Total DA+ID settlement is then handled in `settlement.py`:
    cost_t = da_volume_t * da_price_t
           + (actual_t - da_volume_t) * id_price_t

Mathematically:
    cost_t = actual_t * id_price_t + da_volume_t * (da_price_t - id_price_t)

The second term is fixed at the ID-stage (because da_volume is locked in
from DA stage), so minimising "actual_t * id_price_t" is equivalent to
minimising the total bill at ID stage.  We therefore reuse `optimise()`
with ID prices.

This wrapper exists so the calling code makes its intent explicit, and so
we have a natural place to plug in rolling-horizon / receding-horizon
logic later (the MVP uses perfect ID foresight).
"""
from __future__ import annotations

import numpy as np

from .optimizer_da import optimise, OptResult
from .archetypes import Archetype
from . import config as C


def reoptimise_id(
    arch: Archetype,
    outdoor_temp_c: np.ndarray,
    dhw_draw_kwh: np.ndarray,
    id_price_eur_per_mwh: np.ndarray,
    timestep_h: float,
    comfort_band: tuple[float, float] = (C.T_MIN, C.T_MAX),
    enable_dhw_flex: bool = True,
    t_in_init: float | None = None,
    e_dhw_init: float | None = None,
) -> OptResult:
    """Run the LP against ID prices.

    The result represents the *actual* physical delivery on the day,
    after intraday re-optimisation.
    """
    return optimise(
        arch=arch,
        outdoor_temp_c=outdoor_temp_c,
        dhw_draw_kwh=dhw_draw_kwh,
        price_eur_per_mwh=id_price_eur_per_mwh,
        timestep_h=timestep_h,
        comfort_band=comfort_band,
        enable_dhw_flex=enable_dhw_flex,
        t_in_init=t_in_init,
        e_dhw_init=e_dhw_init,
    )
