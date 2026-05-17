"""
archetypes.py — Predefined household / building archetypes.

Each archetype bundles:
    - building thermal parameters (floor area, UA, thermal capacitance)
    - heat-pump capacity and emitter type
    - DHW tank configuration
    - default comfort band

The defaults below are deliberately ballpark numbers for a German market
context. They are not engineering-grade values; the app exposes every
field for the user to override.
"""
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class Archetype:
    name: str
    label: str
    floor_area_m2: float
    annual_heat_demand_kwh: float        # space-heating only
    annual_dhw_demand_kwh: float
    ua_kw_per_k: float                   # building heat loss coefficient
    c_th_kwh_per_k: float                # building thermal capacitance
    hp_thermal_kw: float                 # nominal thermal capacity
    hp_electrical_kw: float              # nominal electrical capacity
    hp_min_modulation: float             # 0-1, fraction of nominal
    emitter: str                         # "underfloor" or "radiator"
    dhw_tank_kwh: float                  # usable energy in tank
    dhw_tank_loss_kwh_per_step: float    # standing loss per 15-min step
    t_target: float = 21.0
    t_min: float = 20.0
    t_max: float = 22.0
    flexibility_rating: str = "medium"

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Helper: rough sizing.  Tank capacity of 200 l of water heated from 10 °C
# to 55 °C is ~10.5 kWh thermal. We use 10 kWh as a reasonable default for
# the usable energy of a 200-litre tank.
# --------------------------------------------------------------------------
ARCHETYPES: Dict[str, Archetype] = {
    "new_underfloor": Archetype(
        name="new_underfloor",
        label="New build + underfloor heating + DHW tank",
        floor_area_m2=140,
        annual_heat_demand_kwh=7_000,
        annual_dhw_demand_kwh=2_000,
        ua_kw_per_k=0.12,
        c_th_kwh_per_k=18.0,             # well-insulated, lots of slab mass
        hp_thermal_kw=6.0,
        hp_electrical_kw=2.2,
        hp_min_modulation=0.30,
        emitter="underfloor",
        dhw_tank_kwh=10.0,
        dhw_tank_loss_kwh_per_step=0.04,
        flexibility_rating="high",
    ),
    "renovated_sfh": Archetype(
        name="renovated_sfh",
        label="Renovated single-family home + radiators + DHW tank",
        floor_area_m2=150,
        annual_heat_demand_kwh=10_000,
        annual_dhw_demand_kwh=2_000,
        ua_kw_per_k=0.20,
        c_th_kwh_per_k=12.0,
        hp_thermal_kw=8.0,
        hp_electrical_kw=3.0,
        hp_min_modulation=0.35,
        emitter="radiator",
        dhw_tank_kwh=10.0,
        dhw_tank_loss_kwh_per_step=0.05,
        flexibility_rating="medium",
    ),
    "old_radiators": Archetype(
        name="old_radiators",
        label="Older home + radiators + limited thermal flexibility",
        floor_area_m2=160,
        annual_heat_demand_kwh=16_000,
        annual_dhw_demand_kwh=2_200,
        ua_kw_per_k=0.35,
        c_th_kwh_per_k=8.0,
        hp_thermal_kw=10.0,
        hp_electrical_kw=4.0,
        hp_min_modulation=0.40,
        emitter="radiator",
        dhw_tank_kwh=10.0,
        dhw_tank_loss_kwh_per_step=0.06,
        flexibility_rating="low",
    ),
    "custom": Archetype(
        name="custom",
        label="User-defined custom archetype",
        floor_area_m2=150,
        annual_heat_demand_kwh=10_000,
        annual_dhw_demand_kwh=2_000,
        ua_kw_per_k=0.20,
        c_th_kwh_per_k=12.0,
        hp_thermal_kw=8.0,
        hp_electrical_kw=3.0,
        hp_min_modulation=0.30,
        emitter="radiator",
        dhw_tank_kwh=10.0,
        dhw_tank_loss_kwh_per_step=0.05,
        flexibility_rating="medium",
    ),
}


def get_archetype(name: str) -> Archetype:
    """Return a copy of the requested archetype so the caller can edit safely."""
    a = ARCHETYPES[name]
    return Archetype(**a.to_dict())


def archetype_options() -> Dict[str, str]:
    """Mapping {name: human label} for the UI selectbox."""
    return {k: v.label for k, v in ARCHETYPES.items()}
