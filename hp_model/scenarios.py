"""
scenarios.py — Run S0..S4 in one call and compute headline KPIs.

This is the orchestration layer:
    S0  Baseline (dumb thermostat)              -> baseline_controller
    S1  DA only                                 -> optimizer_da with DA prices
    S2  DA + ID re-optimisation                 -> DA -> ID re-opt -> DA+ID settlement
    S3  S2 with wider comfort band              -> same but (T_min_flex, T_max_flex)
    S4  S2 with DHW flexibility enabled         -> same but enable_dhw_flex=True

A single `run_all_scenarios` call returns a dict of `ScenarioResult` plus a
KPI summary table.  All economics are reported in two views:
    - wholesale  (€/MWh, used for the EEM / aggregator perspective)
    - retail     (ct/kWh inc VAT, used for the customer perspective)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from . import baseline_controller as bc
from . import optimizer_da as od
from . import optimizer_id as oi
from . import settlement as stl
from . import cop_model as cm
from . import config as C
from .archetypes import Archetype
from .data_loader import MarketData


@dataclass
class ScenarioResult:
    name: str
    label: str
    timestamps: pd.DatetimeIndex
    t_in_c: np.ndarray
    q_sh_kwh: np.ndarray
    q_dhw_kwh: np.ndarray
    e_dhw_kwh: np.ndarray
    elec_kwh: np.ndarray
    elec_sh_kwh: np.ndarray
    elec_dhw_kwh: np.ndarray
    cop_sh: np.ndarray
    cop_dhw: np.ndarray
    comfort_violation_kdeg: np.ndarray
    dhw_violation_kwh: np.ndarray
    # Settlement
    da_volume_kwh: np.ndarray
    id_volume_kwh: np.ndarray
    id_adjustment_kwh: np.ndarray
    wholesale_cost_eur: float
    da_cost_component_eur: float
    id_adjustment_cost_eur: float
    retail_cost_eur: float
    status: str
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------
def _scenario_from_baseline(
    name: str, label: str, md: MarketData, br: bc.BaselineResult,
    retail_kwargs: dict
) -> ScenarioResult:
    settlement = stl.settle_da_only(br.elec_kwh, md.da_price_eur_mwh)
    retail = stl.retail_cost_eur(br.elec_kwh, md.da_price_eur_mwh, **retail_kwargs)
    return ScenarioResult(
        name=name, label=label, timestamps=md.timestamps,
        t_in_c=br.t_in_c, q_sh_kwh=br.q_sh_kwh, q_dhw_kwh=br.q_dhw_kwh,
        e_dhw_kwh=br.e_dhw_kwh, elec_kwh=br.elec_kwh,
        elec_sh_kwh=br.elec_sh_kwh, elec_dhw_kwh=br.elec_dhw_kwh,
        cop_sh=br.cop_sh, cop_dhw=br.cop_dhw,
        comfort_violation_kdeg=np.zeros_like(br.elec_kwh),
        dhw_violation_kwh=np.zeros_like(br.elec_kwh),
        da_volume_kwh=settlement.da_volume_kwh,
        id_volume_kwh=settlement.id_volume_kwh,
        id_adjustment_kwh=settlement.id_adjustment_kwh,
        wholesale_cost_eur=settlement.total_wholesale_cost_eur,
        da_cost_component_eur=settlement.da_cost_eur,
        id_adjustment_cost_eur=0.0,
        retail_cost_eur=retail,
        status="baseline",
    )


def _scenario_from_da_only(
    name: str, label: str, md: MarketData, da_res: od.OptResult,
    retail_kwargs: dict
) -> ScenarioResult:
    settlement = stl.settle_da_only(da_res.elec_kwh, md.da_price_eur_mwh)
    retail = stl.retail_cost_eur(da_res.elec_kwh, md.da_price_eur_mwh, **retail_kwargs)
    return ScenarioResult(
        name=name, label=label, timestamps=md.timestamps,
        t_in_c=da_res.t_in_c, q_sh_kwh=da_res.q_sh_kwh, q_dhw_kwh=da_res.q_dhw_kwh,
        e_dhw_kwh=da_res.e_dhw_kwh, elec_kwh=da_res.elec_kwh,
        elec_sh_kwh=da_res.elec_sh_kwh, elec_dhw_kwh=da_res.elec_dhw_kwh,
        cop_sh=da_res.cop_sh, cop_dhw=da_res.cop_dhw,
        comfort_violation_kdeg=da_res.comfort_violation_kdeg,
        dhw_violation_kwh=da_res.dhw_violation_kwh,
        da_volume_kwh=settlement.da_volume_kwh,
        id_volume_kwh=settlement.id_volume_kwh,
        id_adjustment_kwh=settlement.id_adjustment_kwh,
        wholesale_cost_eur=settlement.total_wholesale_cost_eur,
        da_cost_component_eur=settlement.da_cost_eur,
        id_adjustment_cost_eur=0.0,
        retail_cost_eur=retail,
        status=da_res.status,
    )


def _scenario_from_da_id(
    name: str, label: str, md: MarketData,
    da_res: od.OptResult, id_res: od.OptResult, retail_kwargs: dict
) -> ScenarioResult:
    settlement = stl.settle_da_id(
        da_volume_kwh=da_res.elec_kwh,
        id_volume_kwh=id_res.elec_kwh,
        da_price_eur_mwh=md.da_price_eur_mwh,
        id_price_eur_mwh=md.id_price_eur_mwh,
    )
    # The customer pays grid fees + VAT on the actual delivered energy.
    # We use the DA price as the wholesale spine for retail conversion to
    # ensure consistent comparison against the baseline.
    retail = stl.retail_cost_eur(id_res.elec_kwh, md.da_price_eur_mwh, **retail_kwargs)
    return ScenarioResult(
        name=name, label=label, timestamps=md.timestamps,
        t_in_c=id_res.t_in_c, q_sh_kwh=id_res.q_sh_kwh, q_dhw_kwh=id_res.q_dhw_kwh,
        e_dhw_kwh=id_res.e_dhw_kwh, elec_kwh=id_res.elec_kwh,
        elec_sh_kwh=id_res.elec_sh_kwh, elec_dhw_kwh=id_res.elec_dhw_kwh,
        cop_sh=id_res.cop_sh, cop_dhw=id_res.cop_dhw,
        comfort_violation_kdeg=id_res.comfort_violation_kdeg,
        dhw_violation_kwh=id_res.dhw_violation_kwh,
        da_volume_kwh=settlement.da_volume_kwh,
        id_volume_kwh=settlement.id_volume_kwh,
        id_adjustment_kwh=settlement.id_adjustment_kwh,
        wholesale_cost_eur=settlement.total_wholesale_cost_eur,
        da_cost_component_eur=settlement.da_cost_eur,
        id_adjustment_cost_eur=settlement.id_adjustment_cost_eur,
        retail_cost_eur=retail,
        status=id_res.status,
        meta={"da_only_cost_eur": settlement.da_only_cost_eur,
              "pure_id_cost_eur": settlement.pure_id_cost_eur},
    )


# --------------------------------------------------------------------------
def run_all_scenarios(
    arch: Archetype,
    md: MarketData,
    timestep_h: float = C.DEFAULT_TIMESTEP_HOURS,
    comfort_band: Tuple[float, float] = (C.T_MIN, C.T_MAX),
    comfort_band_flex: Tuple[float, float] = (C.T_MIN_FLEX, C.T_MAX_FLEX),
    retail_kwargs: Optional[dict] = None,
    scenarios: Optional[List[str]] = None,
) -> Dict[str, ScenarioResult]:
    """Run S0..S4 and return a dict keyed by scenario name."""
    retail_kwargs = retail_kwargs or {}
    selected = scenarios or ["S0", "S1", "S2", "S3", "S4"]
    results: Dict[str, ScenarioResult] = {}

    if "S0" in selected:
        baseline = bc.run_baseline(arch, md.outdoor_temp_c, md.dhw_draw_kwh, timestep_h)
        results["S0"] = _scenario_from_baseline("S0", "Baseline thermostat", md, baseline, retail_kwargs)

    if "S1" in selected:
        da_res = od.optimise(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.da_price_eur_mwh, timestep_h,
            comfort_band=comfort_band, enable_dhw_flex=False,
        )
        results["S1"] = _scenario_from_da_only("S1", "DA optimised", md, da_res, retail_kwargs)

    if "S2" in selected:
        da_res = od.optimise(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.da_price_eur_mwh, timestep_h,
            comfort_band=comfort_band, enable_dhw_flex=False,
        )
        id_res = oi.reoptimise_id(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.id_price_eur_mwh, timestep_h,
            comfort_band=comfort_band, enable_dhw_flex=False,
        )
        results["S2"] = _scenario_from_da_id("S2", "DA + ID re-optimised", md, da_res, id_res, retail_kwargs)

    if "S3" in selected:
        da_res = od.optimise(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.da_price_eur_mwh, timestep_h,
            comfort_band=comfort_band_flex, enable_dhw_flex=False,
        )
        id_res = oi.reoptimise_id(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.id_price_eur_mwh, timestep_h,
            comfort_band=comfort_band_flex, enable_dhw_flex=False,
        )
        results["S3"] = _scenario_from_da_id("S3", "DA + ID + wider comfort", md, da_res, id_res, retail_kwargs)

    if "S4" in selected:
        da_res = od.optimise(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.da_price_eur_mwh, timestep_h,
            comfort_band=comfort_band, enable_dhw_flex=True,
        )
        id_res = oi.reoptimise_id(
            arch, md.outdoor_temp_c, md.dhw_draw_kwh, md.id_price_eur_mwh, timestep_h,
            comfort_band=comfort_band, enable_dhw_flex=True,
        )
        results["S4"] = _scenario_from_da_id("S4", "DA + ID + DHW flexibility", md, da_res, id_res, retail_kwargs)

    return results


# --------------------------------------------------------------------------
def kpi_table(results: Dict[str, ScenarioResult], horizon_days: float) -> pd.DataFrame:
    """Aggregate per-scenario KPIs into a DataFrame for the UI."""
    if "S0" in results:
        baseline_wholesale = results["S0"].wholesale_cost_eur
        baseline_retail = results["S0"].retail_cost_eur
        baseline_elec = float(results["S0"].elec_kwh.sum())
    else:
        baseline_wholesale = baseline_retail = baseline_elec = float("nan")

    scale_to_year = 365.0 / horizon_days if horizon_days > 0 else 1.0
    rows = []
    for s_name, r in results.items():
        elec = float(r.elec_kwh.sum())
        wholesale_save = baseline_wholesale - r.wholesale_cost_eur
        retail_save = baseline_retail - r.retail_cost_eur
        total_th = float((r.q_sh_kwh + r.q_dhw_kwh).sum())
        scop = total_th / elec if elec > 1e-6 else float("nan")
        comfort_hours = float(np.sum(r.comfort_violation_kdeg > 1e-4))
        dhw_hours = float(np.sum(r.dhw_violation_kwh > 1e-4))

        if "S0" in results:
            shifted = 0.5 * float(np.sum(np.abs(r.elec_kwh - results["S0"].elec_kwh)))
        else:
            shifted = float("nan")
        shifted_share = (shifted / elec * 100) if elec > 1e-6 else float("nan")

        rows.append({
            "scenario": s_name,
            "label": r.label,
            "horizon_elec_kwh": elec,
            "annualised_elec_kwh": elec * scale_to_year,
            "wholesale_cost_eur_horizon": r.wholesale_cost_eur,
            "wholesale_cost_eur_year": r.wholesale_cost_eur * scale_to_year,
            "retail_cost_eur_horizon": r.retail_cost_eur,
            "retail_cost_eur_year": r.retail_cost_eur * scale_to_year,
            "wholesale_saving_eur_year": wholesale_save * scale_to_year,
            "retail_saving_eur_year": retail_save * scale_to_year,
            "saving_pct_retail": (retail_save / baseline_retail * 100) if baseline_retail else float("nan"),
            "saving_ct_per_kwh_hp_load": (retail_save / baseline_elec * 100) if baseline_elec else float("nan"),
            "shifted_kwh_horizon": shifted,
            "shifted_share_pct": shifted_share,
            "scop": scop,
            "avg_cop_sh": cm.average_cop_from_series(r.q_sh_kwh, r.elec_sh_kwh),
            "avg_cop_dhw": cm.average_cop_from_series(r.q_dhw_kwh, r.elec_dhw_kwh),
            "comfort_violation_hours": comfort_hours,
            "dhw_violation_hours": dhw_hours,
            "da_cost_eur": r.da_cost_component_eur,
            "id_adjustment_cost_eur": r.id_adjustment_cost_eur,
            "status": r.status,
        })
    return pd.DataFrame(rows)


def value_stack(results: Dict[str, ScenarioResult], horizon_days: float) -> Dict[str, float]:
    """Decompose wholesale savings into contributions (annualised €)."""
    scale = 365.0 / horizon_days if horizon_days > 0 else 1.0
    cost = {k: r.wholesale_cost_eur for k, r in results.items()}
    out: Dict[str, float] = {}
    if "S0" in cost and "S1" in cost:
        out["da_value_eur_year"] = (cost["S0"] - cost["S1"]) * scale
    if "S1" in cost and "S2" in cost:
        out["id_incremental_eur_year"] = (cost["S1"] - cost["S2"]) * scale
    if "S2" in cost and "S3" in cost:
        out["wider_comfort_eur_year"] = (cost["S2"] - cost["S3"]) * scale
    if "S2" in cost and "S4" in cost:
        out["dhw_flex_eur_year"] = (cost["S2"] - cost["S4"]) * scale
    if "S0" in cost:
        best = min(v for k, v in cost.items() if k != "S0")
        out["total_value_eur_year"] = (cost["S0"] - best) * scale
    return out


def build_interpretation(
    results: Dict[str, ScenarioResult],
    value: Dict[str, float],
    customer_share: float,
    horizon_days: float,
) -> str:
    """Plain-English summary of the result for the executive view."""
    if "S0" not in results:
        return "Baseline scenario was not run; cannot interpret value."

    best_scenario = max(
        (k for k in results if k != "S0"),
        key=lambda k: results["S0"].wholesale_cost_eur - results[k].wholesale_cost_eur,
        default=None,
    )
    if best_scenario is None:
        return "Only the baseline scenario was run."

    r0 = results["S0"]
    rb = results[best_scenario]
    scale = 365.0 / horizon_days if horizon_days > 0 else 1.0
    gross = (r0.wholesale_cost_eur - rb.wholesale_cost_eur) * scale
    cust = gross * customer_share
    eon = gross * (1.0 - customer_share)

    da_val = value.get("da_value_eur_year", 0.0)
    id_val = value.get("id_incremental_eur_year", 0.0)
    dhw_val = value.get("dhw_flex_eur_year", 0.0)
    comfort_hours = float(np.sum(rb.comfort_violation_kdeg > 1e-4))
    shifted = 0.5 * float(np.sum(np.abs(rb.elec_kwh - r0.elec_kwh)))
    shifted_share = (shifted / float(rb.elec_kwh.sum()) * 100) if rb.elec_kwh.sum() > 0 else 0.0

    txt = (
        f"In the best scenario ({best_scenario}: {rb.label}), the heat pump unlocks "
        f"€{gross:,.0f}/year of gross wholesale-flexibility value. "
        f"Roughly €{da_val:,.0f} comes from Day-Ahead price shifting and €{id_val:,.0f} from "
        f"Intraday re-optimisation. "
    )
    if dhw_val:
        txt += f"Adding DHW tank flexibility contributes a further €{dhw_val:,.0f}/year. "
    txt += (
        f"The optimiser shifts {shifted:,.0f} kWh of heat-pump electricity over the horizon "
        f"({shifted_share:.0f}% of the optimised load) while keeping comfort violations to "
        f"{comfort_hours:.0f} time-steps. "
        f"At a {customer_share*100:.0f}% customer share, a FlexHeat bonus of ~€{cust:,.0f}/year "
        f"could be funded while retaining €{eon:,.0f}/year in portfolio value."
    )
    return txt
