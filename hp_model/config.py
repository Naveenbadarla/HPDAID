"""
config.py — Default configuration and global constants.

All defaults are user-editable from the Streamlit UI; this module
centralises numbers so that assumptions are easy to audit.

Unit convention used throughout the project:
    energy        : kWh
    power         : kW
    temperature   : °C
    time step     : hours (default 0.25 = 15 min)
    wholesale     : €/MWh
    retail        : ct/kWh   (1 €/MWh = 0.1 ct/kWh ; 1 ct/kWh = 10 €/MWh)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


# --------------------------------------------------------------------------
# Time resolution
# --------------------------------------------------------------------------
DEFAULT_TIMESTEP_HOURS: float = 0.25            # 15 minutes (German MTU)
STEPS_PER_DAY: int = int(24 / DEFAULT_TIMESTEP_HOURS)


# --------------------------------------------------------------------------
# Heating physics defaults
# --------------------------------------------------------------------------
HEATING_THRESHOLD_C: float = 15.0               # below this, building needs heat
INTERNAL_GAIN_KW: float = 0.3                   # average internal gains


# --------------------------------------------------------------------------
# COP defaults  (linear-in-T_out model: COP = a + b*T_out, clamped)
# Underfloor outperforms radiators because of lower supply temperature.
# --------------------------------------------------------------------------
COP_DEFAULTS: Dict[str, Dict[str, float]] = {
    "underfloor": {"a": 3.5, "b": 0.10, "cop_min": 2.0, "cop_max": 5.5},
    "radiator":   {"a": 2.8, "b": 0.10, "cop_min": 1.8, "cop_max": 4.5},
    "dhw":        {"a": 2.2, "b": 0.08, "cop_min": 1.5, "cop_max": 3.8},
}
# Approximate COP penalty when indoor T rises above target (higher supply T).
COP_PREHEAT_PENALTY_PER_K: float = 0.10


# --------------------------------------------------------------------------
# Comfort and DHW defaults
# --------------------------------------------------------------------------
T_TARGET: float = 21.0
T_MIN: float = 20.0
T_MAX: float = 22.0
T_MIN_FLEX: float = 19.5
T_MAX_FLEX: float = 22.5

# Comfort violation penalty is intentionally high so that the optimiser
# does NOT use comfort slack as a cheap way to lower cost.
COMFORT_PENALTY_EUR_PER_KWH_DEG: float = 5.0
DHW_PENALTY_EUR_PER_KWH: float = 2.0


# --------------------------------------------------------------------------
# Market defaults (German DA wholesale, recent ballpark)
# --------------------------------------------------------------------------
DA_PRICE_MEAN_EUR_MWH: float = 95.0
DA_PRICE_STD_EUR_MWH: float = 35.0
ID_SPREAD_STD_EUR_MWH: float = 18.0             # std-dev of (ID − DA) spread
NEG_PRICE_FRACTION: float = 0.02                # share of hours with neg prices

# Retail tariff stack (ct/kWh) — all editable in the app
RETAIL_MARKUP_CT_KWH: float = 2.0
GRID_FEE_CT_KWH: float = 9.5
TAXES_LEVIES_CT_KWH: float = 3.0
VAT_RATE: float = 0.19


# --------------------------------------------------------------------------
# Imbalance / settlement
# --------------------------------------------------------------------------
IMBALANCE_PENALTY_EUR_MWH: float = 50.0


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
SOLVER_TIME_LIMIT_SEC: int = 120
SOLVER_MSG: bool = False


# --------------------------------------------------------------------------
# Portfolio defaults
# --------------------------------------------------------------------------
PORTFOLIO_DEFAULTS: Dict[str, float] = {
    "n_households": 10_000,
    "customer_share": 0.50,
    "eon_share": 0.50,
}


@dataclass
class RunConfig:
    """Run-time configuration assembled from UI inputs."""
    timestep_h: float = DEFAULT_TIMESTEP_HOURS
    n_steps: int = STEPS_PER_DAY * 7                # default 1 week
    comfort_band: Tuple[float, float] = (T_MIN, T_MAX)
    comfort_band_flex: Tuple[float, float] = (T_MIN_FLEX, T_MAX_FLEX)
    enable_dhw: bool = True
    enable_id: bool = True
    enable_wider_comfort: bool = False
    use_synthetic_id: bool = True
    seed: int = 42
    archetype_name: str = "renovated_sfh"
    extra: Dict[str, Any] = field(default_factory=dict)
