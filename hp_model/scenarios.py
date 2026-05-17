"""
scenarios.py — Orchestration of S0–S4 scenarios + KPI / value-stack helpers.

Normalisation
-------------
Short-horizon runs (a winter week) cannot be naively multiplied by 52 to get
an annual customer benefit. Winter weeks have 3–4× the heating load and 2–3×
the price volatility of a year average, so simple annualisation **overstates**
both electricity demand and flexibility value.

This module therefore distinguishes:

    * horizon_*       — value over the simulated period only (no extrapolation)
    * annualised_*    — extrapolation under a chosen normalisation policy
    * mode            — which policy was used; surfaced in every output

Supported normalisation modes:
    "none"             : no extrapolation; annualised columns = horizon values.
    "simple"           : ×(365 / horizon_days). Cleanest only when the horizon
                         is a representative full-year sample.
    "heating_season"   : assumes the run represents the heating season; scales
                         heating-related quantities by (heating_days / horizon_days),
                         where heating_days defaults to 180 (~Oct–Mar).
    "full_year"        : pass-through; only meaningful when horizon_days >= 360.

Every result carries a `normalisation_mode` and a `season_label` so downstream
UI / reports can warn appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from . import cop_model as cm
from . import baseline_controller as bc
from . import optimizer_da as od
from . import optimizer_id as oi
from . import settlement as stl
from .archetypes import Archetype
from .data_loader import MarketData


# Heating season assumed to be Oct 1 – Mar 31 inclusive
HEATING_SEASON_DAYS: int = 182
HEATING_SEASON_MONTHS: Tuple[int, ...] = (10, 11, 12, 1, 2, 3)


# --------------------------------------------------------------------------
# Scenario result dataclass
# --------------------------------------------------------------------------
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
# Season detection — flags whether the horizon is winter-heavy
# --------------------------------------------------------------------------
def detect_season(
    timestamps: pd.DatetimeIndex,
    outdoor_temp_c: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Return a dict describing seasonality of the simulated horizon.

    Output keys:
        season_label       : 'winter', 'shoulder', 'summer', or 'mixed'
        heating_share      : fraction of timesteps falling in heating months
        avg_outdoor_c      : mean outdoor temp (if provided), else NaN
        horizon_days       : duration in days
        spans_full_year    : True if horizon covers >= 360 days
        is_short_horizon   : True if horizon_days < 30
        winter_heavy       : True if heating_share > 0.7
    """
    if len(timestamps) == 0:
        return {
            "season_label": "unknown",
            "heating_share": float("nan"),
            "avg_outdoor_c": float("nan"),
            "horizon_days": 0.0,
            "spans_full_year": False,
            "is_short_horizon": True,
            "winter_heavy": False,
        }

    horizon_days = float(
        (pd.Timestamp(timestamps[-1]) - pd.Timestamp(timestamps[0])).total_seconds() / 86400.0
    )
    months = pd.DatetimeIndex(timestamps).month
    heating_share = float(np.mean([m in HEATING_SEASON_MONTHS for m in months]))
    avg_outdoor = float(np.nanmean(outdoor_temp_c)) if outdoor_temp_c is not None else float("nan")
    spans_full_year = horizon_days >= 360.0

    if spans_full_year:
        label = "full_year"
    elif heating_share > 0.85:
        label = "winter"
    elif heating_share < 0.15:
        label = "summer"
    elif 0.4 <= heating_share <= 0.7:
        label = "mixed"
    else:
        label = "shoulder"

    return {
        "season_label": label,
        "heating_share": heating_share,
        "avg_outdoor_c": avg_outdoor,
        "horizon_days": horizon_days,
        "spans_full_year": spans_full_year,
        "is_short_horizon": horizon_days < 30.0,
        "winter_heavy": heating_share > 0.7 and not spans_full_year,
    }


# --------------------------------------------------------------------------
# Normalisation helper — single source of truth for horizon → annual scaling
# --------------------------------------------------------------------------
def annualisation_factor(
    mode: str,
    horizon_days: float,
    season: Dict[str, object],
) -> Tuple[float, str]:
    """Return (multiplier, explanation) for converting horizon → annual.

    Modes:
        none           : factor = 1.0
        simple         : factor = 365 / horizon_days
        heating_season : factor = HEATING_SEASON_DAYS / horizon_days
                         (assumes the run is a representative heating-season sample)
        full_year      : factor = 365 / horizon_days
                         (only honest when horizon >= 360 days)
    """
    if horizon_days <= 0:
        return 1.0, "horizon_days <= 0; no scaling applied."

    mode = (mode or "simple").lower()

    if mode == "none":
        return 1.0, "No annualisation — values shown are for the horizon only."
    if mode == "simple":
        f = 365.0 / horizon_days
        note = (
            f"Simple annualisation ×{f:.1f}. Only honest if the simulated horizon is a "
            "representative full-year sample. Winter-only weeks will overstate annual savings."
        )
        return f, note
    if mode == "heating_season":
        f = HEATING_SEASON_DAYS / horizon_days
        note = (
            f"Heating-season annualisation ×{f:.1f} (assumes horizon represents the "
            f"{HEATING_SEASON_DAYS}-day heating season; outside-season HP load assumed ~0)."
        )
        return f, note
    if mode == "full_year":
        if not season.get("spans_full_year", False):
            f = 365.0 / horizon_days
            note = (
                f"Full-year mode selected but horizon is only {horizon_days:.1f} days; "
                f"falling back to ×{f:.1f}. Re-run with a full year of data for a validated number."
            )
            return f, note
        return 1.0, "Full-year simulation — no extrapolation needed."

    # Unknown mode
    return 365.0 / horizon_days, f"Unknown mode '{mode}'; defaulted to simple annualisation."


# --------------------------------------------------------------------------
# (Per-stage scenario builders unchanged from original)
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
    retail = stl.retail_cost_eur(id_res.elec_kwh, md.id_price_eur_mwh, **retail_kwargs)
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
    )


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
# KPI table — horizon AND annualised columns side-by-side
# --------------------------------------------------------------------------
def kpi_table(
    results: Dict[str, ScenarioResult],
    horizon_days: float,
    normalisation_mode: str = "simple",
    season: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Aggregate per-scenario KPIs. Always returns horizon AND annualised columns.

    Parameters
    ----------
    results : dict of ScenarioResult
    horizon_days : actual duration of the simulated horizon
    normalisation_mode : "none" | "simple" | "heating_season" | "full_year"
    season : optional pre-computed season dict (from detect_season)
    """
    if "S0" in results:
        baseline_wholesale = results["S0"].wholesale_cost_eur
        baseline_retail = results["S0"].retail_cost_eur
        baseline_elec = float(results["S0"].elec_kwh.sum())
    else:
        baseline_wholesale = baseline_retail = baseline_elec = float("nan")

    if season is None:
        first = next(iter(results.values()))
        season = detect_season(first.timestamps)
    factor, _ = annualisation_factor(normalisation_mode, horizon_days, season)

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
            # ---- HORIZON (no extrapolation) ----
            "horizon_elec_kwh": elec,
            "horizon_wholesale_cost_eur": r.wholesale_cost_eur,
            "horizon_retail_cost_eur": r.retail_cost_eur,
            "horizon_wholesale_saving_eur": wholesale_save,
            "horizon_retail_saving_eur": retail_save,
            "horizon_shifted_kwh": shifted,
            # ---- ANNUALISED (with disclosure) ----
            "annualised_elec_kwh": elec * factor,
            "annualised_wholesale_cost_eur": r.wholesale_cost_eur * factor,
            "annualised_retail_cost_eur": r.retail_cost_eur * factor,
            "annualised_wholesale_saving_eur": wholesale_save * factor,
            "annualised_retail_saving_eur": retail_save * factor,
            # ---- Ratios that are scale-invariant ----
            "saving_pct_retail": (retail_save / baseline_retail * 100) if baseline_retail else float("nan"),
            "saving_ct_per_kwh_hp_load": (retail_save / baseline_elec * 100) if baseline_elec else float("nan"),
            "shifted_share_pct": shifted_share,
            "scop": scop,
            "avg_cop_sh": cm.average_cop_from_series(r.q_sh_kwh, r.elec_sh_kwh),
            "avg_cop_dhw": cm.average_cop_from_series(r.q_dhw_kwh, r.elec_dhw_kwh),
            "comfort_violation_hours": comfort_hours,
            "dhw_violation_hours": dhw_hours,
            "da_cost_eur": r.da_cost_component_eur,
            "id_adjustment_cost_eur": r.id_adjustment_cost_eur,
            "status": r.status,
            # ---- Metadata ----
            "normalisation_mode": normalisation_mode,
            "annualisation_factor": factor,
            "season_label": season.get("season_label", "unknown"),
        })

    # Backwards-compat aliases for any callers that still expect the old names
    df = pd.DataFrame(rows)
    df["wholesale_cost_eur_year"] = df["annualised_wholesale_cost_eur"]
    df["retail_cost_eur_year"] = df["annualised_retail_cost_eur"]
    df["wholesale_saving_eur_year"] = df["annualised_wholesale_saving_eur"]
    df["retail_saving_eur_year"] = df["annualised_retail_saving_eur"]
    df["wholesale_cost_eur_horizon"] = df["horizon_wholesale_cost_eur"]
    df["retail_cost_eur_horizon"] = df["horizon_retail_cost_eur"]
    df["shifted_kwh_horizon"] = df["horizon_shifted_kwh"]
    return df


# --------------------------------------------------------------------------
# Value stack — both horizon and annualised, mode disclosed
# --------------------------------------------------------------------------
def value_stack(
    results: Dict[str, ScenarioResult],
    horizon_days: float,
    normalisation_mode: str = "simple",
    season: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Decompose wholesale savings into contributions.

    Returns both horizon and annualised components plus a `_metadata` entry
    describing the normalisation policy used.
    """
    if season is None:
        first = next(iter(results.values()))
        season = detect_season(first.timestamps)
    factor, note = annualisation_factor(normalisation_mode, horizon_days, season)
    cost = {k: r.wholesale_cost_eur for k, r in results.items()}
    out: Dict[str, float] = {}

    # Horizon values (no extrapolation)
    if "S0" in cost and "S1" in cost:
        out["da_value_eur_horizon"] = cost["S0"] - cost["S1"]
        out["da_value_eur_year"] = (cost["S0"] - cost["S1"]) * factor
    if "S1" in cost and "S2" in cost:
        out["id_incremental_eur_horizon"] = cost["S1"] - cost["S2"]
        out["id_incremental_eur_year"] = (cost["S1"] - cost["S2"]) * factor
    if "S2" in cost and "S3" in cost:
        out["wider_comfort_eur_horizon"] = cost["S2"] - cost["S3"]
        out["wider_comfort_eur_year"] = (cost["S2"] - cost["S3"]) * factor
    if "S2" in cost and "S4" in cost:
        out["dhw_flex_eur_horizon"] = cost["S2"] - cost["S4"]
        out["dhw_flex_eur_year"] = (cost["S2"] - cost["S4"]) * factor
    if "S0" in cost and len(cost) > 1:
        best = min(v for k, v in cost.items() if k != "S0")
        out["total_value_eur_horizon"] = cost["S0"] - best
        out["total_value_eur_year"] = (cost["S0"] - best) * factor

    out["_normalisation_mode"] = normalisation_mode
    out["_annualisation_factor"] = factor
    out["_annualisation_note"] = note
    out["_season_label"] = season.get("season_label", "unknown")
    out["_horizon_days"] = horizon_days
    return out


# --------------------------------------------------------------------------
# Recommended board-safe interpretation text
# --------------------------------------------------------------------------
def build_interpretation(
    results: Dict[str, ScenarioResult],
    value: Dict[str, float],
    customer_share: float,
    horizon_days: float,
    normalisation_mode: str = "simple",
    season: Optional[Dict[str, object]] = None,
) -> str:
    """Plain-English summary that respects horizon/annual distinction."""
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
    if season is None:
        season = detect_season(r0.timestamps)
    factor, factor_note = annualisation_factor(normalisation_mode, horizon_days, season)

    horizon_value = r0.wholesale_cost_eur - rb.wholesale_cost_eur
    annual_value = horizon_value * factor
    cust = annual_value * customer_share
    eon = annual_value * (1.0 - customer_share)

    da_val_h = value.get("da_value_eur_horizon", 0.0)
    id_val_h = value.get("id_incremental_eur_horizon", 0.0)
    dhw_val_h = value.get("dhw_flex_eur_horizon", 0.0)
    comfort_hours = float(np.sum(rb.comfort_violation_kdeg > 1e-4))
    shifted = 0.5 * float(np.sum(np.abs(rb.elec_kwh - r0.elec_kwh)))
    shifted_share = (shifted / float(rb.elec_kwh.sum()) * 100) if rb.elec_kwh.sum() > 0 else 0.0

    season_label = season.get("season_label", "unknown")
    is_full_year = bool(season.get("spans_full_year", False))
    is_winter_heavy = bool(season.get("winter_heavy", False))

    # ---- Opening line: explicitly horizon-anchored ----
    txt = (
        f"In this **{horizon_days:.0f}-day {season_label}-period** model run, the best scenario "
        f"({best_scenario}: {rb.label}) shows **€{horizon_value:,.0f} of horizon wholesale value**, "
    )
    if normalisation_mode == "none" or factor == 1.0:
        txt += "with no annualisation applied. "
    else:
        txt += (
            f"equivalent to **€{annual_value:,.0f}/year** under "
            f"**{normalisation_mode.replace('_', ' ')} annualisation (×{factor:.1f})**. "
        )

    # ---- Caveat tier ----
    if not is_full_year:
        if is_winter_heavy:
            txt += (
                "\n\n⚠️ **Winter-heavy short horizon.** Heat-pump electricity demand and DA/ID price "
                "volatility are highest in winter; multiplying a winter week by 52 materially "
                "**overstates** annual savings. Treat this as an **indicative upside scenario**, "
                "not a validated customer benefit. Run a full-year simulation with real EPEX + "
                "DWD data before quoting €/year figures externally."
            )
        elif season.get("is_short_horizon", False):
            txt += (
                "\n\n⚠️ **Short horizon.** The annualised figure assumes the simulated window is "
                "representative of the full year. Verify with at least one shoulder-season and "
                "one summer week, or a full-year run, before treating this as a customer benefit."
            )

    # ---- Value-stack breakdown — in horizon terms (factor-independent) ----
    txt += (
        f"\n\nOver the horizon, ~€{da_val_h:,.0f} comes from Day-Ahead price shifting and "
        f"€{id_val_h:,.0f} from Intraday re-optimisation."
    )
    if dhw_val_h:
        txt += f" Adding DHW tank flexibility contributes a further €{dhw_val_h:,.0f}."

    txt += (
        f"\n\nThe optimiser shifts {shifted:,.0f} kWh of HP electricity over the horizon "
        f"({shifted_share:.0f}% of optimised load) while keeping comfort violations to "
        f"{comfort_hours:.0f} time-steps."
    )

    # ---- Customer / E.ON split — only quoted if annualised at all ----
    if factor != 1.0 or is_full_year:
        suffix = "/year" if (is_full_year or normalisation_mode != "none") else " (horizon)"
        txt += (
            f"\n\nAt a {customer_share*100:.0f}% customer share, a FlexHeat bonus of "
            f"~€{cust:,.0f}{suffix} could be funded while retaining €{eon:,.0f}{suffix} "
            f"in portfolio value."
        )
        if not is_full_year:
            txt += (
                "\n\n**Recommended board-slide wording:** "
                "*\"Indicative winter-week annualisation only. Full-year simulation required "
                "before treating this as a customer benefit estimate.\"*"
            )
    return txt
