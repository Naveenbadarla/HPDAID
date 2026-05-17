"""
fleet.py — Scale single-household results to a portfolio.

Simple but explicit:
    aggregated_value_eur_year     = n_households * single_household_value
    aggregated_mwh_year           = n_households * single_household_kwh / 1000
    customer_bonus_pool_eur_year  = customer_share * aggregated_value
    eon_retained_value_eur_year   = (1-customer_share) * aggregated_value
    flexible_mw_capacity          = n_households * arch.hp_electrical_kw / 1000

Heterogeneous portfolios are supported by passing a list of weighted
archetype shares.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .archetypes import Archetype, ARCHETYPES


@dataclass
class FleetParams:
    n_households: int = int(C.PORTFOLIO_DEFAULTS["n_households"])
    customer_share: float = C.PORTFOLIO_DEFAULTS["customer_share"]
    adoption_rate: float = 1.0
    archetype_mix: Dict[str, float] = None       # {name: share}, sums to 1


def default_fleet(n_households: int = None) -> FleetParams:
    return FleetParams(
        n_households=n_households or int(C.PORTFOLIO_DEFAULTS["n_households"]),
        archetype_mix={"new_underfloor": 0.30, "renovated_sfh": 0.50, "old_radiators": 0.20},
    )


def scale_results(
    single_value_eur_year: float,
    single_elec_kwh_year: float,
    arch: Archetype,
    fleet: FleetParams,
) -> Dict[str, float]:
    """Simple homogeneous scaling using a single representative household."""
    n = fleet.n_households * fleet.adoption_rate
    agg_value = single_value_eur_year * n
    agg_mwh = single_elec_kwh_year * n / 1000.0
    flex_mw = arch.hp_electrical_kw * n / 1000.0
    return {
        "n_households_active": n,
        "agg_annual_value_eur": agg_value,
        "agg_annual_mwh": agg_mwh,
        "flex_mw_capacity": flex_mw,
        "value_per_household_eur": single_value_eur_year,
        "customer_bonus_pool_eur": agg_value * fleet.customer_share,
        "eon_retained_value_eur": agg_value * (1.0 - fleet.customer_share),
    }


def heterogeneous_fleet(
    per_arch_value_year_eur: Dict[str, float],
    per_arch_elec_year_kwh: Dict[str, float],
    fleet: FleetParams,
) -> Dict[str, float]:
    """Mixed archetype portfolio scaling.

    Inputs:
        per_arch_value_year_eur  : {archetype_name: single-household annual saving €}
        per_arch_elec_year_kwh   : {archetype_name: single-household annual elec kWh}
    """
    mix = fleet.archetype_mix or {"renovated_sfh": 1.0}
    n_total = fleet.n_households * fleet.adoption_rate

    agg_value = 0.0
    agg_mwh = 0.0
    flex_mw = 0.0
    for name, share in mix.items():
        n_a = n_total * share
        agg_value += per_arch_value_year_eur.get(name, 0.0) * n_a
        agg_mwh += per_arch_elec_year_kwh.get(name, 0.0) * n_a / 1000.0
        flex_mw += ARCHETYPES[name].hp_electrical_kw * n_a / 1000.0

    return {
        "n_households_active": n_total,
        "agg_annual_value_eur": agg_value,
        "agg_annual_mwh": agg_mwh,
        "flex_mw_capacity": flex_mw,
        "value_per_household_eur": agg_value / n_total if n_total > 0 else 0.0,
        "customer_bonus_pool_eur": agg_value * fleet.customer_share,
        "eon_retained_value_eur": agg_value * (1.0 - fleet.customer_share),
    }


def scaling_curve(
    value_per_household_eur: float,
    arch_hp_electrical_kw: float,
    customer_share: float,
    n_max: int = 100_000,
    n_points: int = 25,
) -> pd.DataFrame:
    """Tabulate value and flex-MW versus number of homes."""
    ns = np.linspace(0, n_max, n_points)
    rows = []
    for n in ns:
        agg = n * value_per_household_eur
        rows.append({
            "n_households": n,
            "agg_annual_value_eur": agg,
            "customer_bonus_pool_eur": agg * customer_share,
            "eon_retained_value_eur": agg * (1.0 - customer_share),
            "flex_mw_capacity": n * arch_hp_electrical_kw / 1000.0,
        })
    return pd.DataFrame(rows)
