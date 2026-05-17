"""
validation.py — Sanity checks on optimisation results.

Validation categories:
    A. Thermal sanity        — temperature inside bounds, no unrealistic ramps
    B. Energy sanity         — elec = thermal / COP, DHW balance closes
    C. Optimisation sanity   — end-of-horizon not gamed, comfort penalty not abused
    D. Market sanity         — DA-only vs DA+ID handled correctly; no min(DA,ID) hack
    E. Business sanity       — savings within plausible range
    F. Annualisation sanity  — short / winter-heavy horizons flagged when
                               extrapolated to a full year

Each check returns a `Finding`; the app renders them as colour-coded
banners.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .scenarios import ScenarioResult, detect_season, annualisation_factor
from .archetypes import Archetype


@dataclass
class Finding:
    code: str
    level: str          # "ok" | "info" | "warning" | "error"
    message: str


def _ok(code: str, msg: str) -> Finding: return Finding(code, "ok", msg)
def _info(code: str, msg: str) -> Finding: return Finding(code, "info", msg)
def _warn(code: str, msg: str) -> Finding: return Finding(code, "warning", msg)
def _err(code: str, msg: str) -> Finding: return Finding(code, "error", msg)


# --------------------------------------------------------------------------
# A. Thermal sanity
# --------------------------------------------------------------------------
def check_thermal(r: ScenarioResult, arch: Archetype) -> List[Finding]:
    out: List[Finding] = []
    if len(r.t_in_c) == 0:
        return [_err("thermal.empty", "No temperature series available.")]

    t = r.t_in_c
    if np.any(np.isnan(t)):
        out.append(_err("thermal.nan", "Indoor temperature contains NaN — solver likely failed."))
    if np.any(t < 10) or np.any(t > 30):
        out.append(_warn(
            "thermal.bounds",
            f"Indoor temperature went outside physically plausible 10-30 °C window "
            f"(min {t.min():.1f} °C, max {t.max():.1f} °C).",
        ))
    else:
        out.append(_ok("thermal.bounds", f"Indoor temperature within 10-30 °C "
                                          f"(min {t.min():.1f}, max {t.max():.1f}, mean {t.mean():.1f})."))

    ramp = np.diff(t)
    max_ramp = np.max(np.abs(ramp)) if len(ramp) else 0.0
    if max_ramp > 2.5:
        out.append(_warn(
            "thermal.ramp",
            f"Indoor temperature changes by up to {max_ramp:.2f} K per step — "
            "check thermal capacitance.",
        ))
    return out


# --------------------------------------------------------------------------
# B. Energy sanity
# --------------------------------------------------------------------------
def check_energy(r: ScenarioResult, arch: Archetype) -> List[Finding]:
    out: List[Finding] = []
    expected_el = r.q_sh_kwh / r.cop_sh + r.q_dhw_kwh / r.cop_dhw
    diff = np.max(np.abs(expected_el - r.elec_kwh))
    if diff > 1e-3:
        out.append(_err("energy.elec_consistency",
                        f"elec != thermal/COP, max deviation {diff:.4f} kWh."))
    else:
        out.append(_ok("energy.elec_consistency", "elec = thermal/COP holds for every step."))

    q_max = arch.hp_thermal_kw * (r.timestamps[1] - r.timestamps[0]).total_seconds() / 3600.0
    if np.any(r.q_sh_kwh + r.q_dhw_kwh > q_max * 1.01):
        out.append(_err("energy.capacity", "Heat-pump thermal capacity breached."))
    else:
        out.append(_ok("energy.capacity", f"HP thermal output stays within {q_max:.2f} kWh / step."))

    if np.any(r.cop_sh < 1.5) or np.any(r.cop_sh > 6.0):
        out.append(_warn("energy.cop_range", "Space-heating COP outside [1.5, 6.0]."))
    return out


# --------------------------------------------------------------------------
# C. Optimisation sanity
# --------------------------------------------------------------------------
def check_optimisation(r: ScenarioResult, arch: Archetype) -> List[Finding]:
    out: List[Finding] = []
    if r.status not in ("baseline", "Optimal", "1"):
        out.append(_warn("opt.status", f"Solver status: {r.status}."))
    else:
        out.append(_ok("opt.status", f"Solver status: {r.status}."))

    if len(r.t_in_c) > 0:
        t_end = r.t_in_c[-1]
        if t_end < arch.t_target - 1.5:
            out.append(_warn(
                "opt.end_horizon",
                f"Final indoor temperature {t_end:.1f} °C is well below target {arch.t_target} °C — "
                "possible end-of-horizon gaming despite terminal constraint.",
            ))
        else:
            out.append(_ok("opt.end_horizon",
                           f"Final indoor temperature {t_end:.1f} °C close to target."))
    if len(r.e_dhw_kwh) > 0:
        e_end = r.e_dhw_kwh[-1]
        if e_end < 0.3 * arch.dhw_tank_kwh:
            out.append(_warn("opt.end_dhw", f"Final DHW state {e_end:.1f} kWh is low."))
        else:
            out.append(_ok("opt.end_dhw", f"Final DHW state {e_end:.1f} kWh."))

    cv_total = float(np.sum(r.comfort_violation_kdeg))
    if cv_total > 1.0:
        out.append(_warn(
            "opt.comfort_abuse",
            f"Cumulative comfort violation {cv_total:.2f} K·step — check penalty calibration.",
        ))
    else:
        out.append(_ok("opt.comfort_abuse", f"Comfort violation {cv_total:.2f} K·step (near zero)."))
    return out


# --------------------------------------------------------------------------
# D. Market sanity
# --------------------------------------------------------------------------
def check_market(r: ScenarioResult) -> List[Finding]:
    out: List[Finding] = []
    if r.name == "S0":
        return out
    if r.name in ("S2", "S3", "S4"):
        adj = float(np.sum(np.abs(r.id_adjustment_kwh)))
        if adj < 1e-6:
            out.append(_info("market.id_zero", "ID adjustment volume is zero — ID prices may equal DA prices."))
        else:
            out.append(_ok("market.id_nonzero",
                           f"DA+ID settled correctly with {adj:.1f} kWh of ID adjustment volume."))

    expected_total = r.da_cost_component_eur + r.id_adjustment_cost_eur
    if abs(expected_total - r.wholesale_cost_eur) > 0.01:
        out.append(_err("market.settlement",
                        f"Settlement mismatch: total ≠ DA + ID adjustment "
                        f"({r.wholesale_cost_eur:.2f} vs {expected_total:.2f})."))
    return out


# --------------------------------------------------------------------------
# E. Business sanity (uses simple annualisation for screening only)
# --------------------------------------------------------------------------
def check_business(results: Dict[str, ScenarioResult], horizon_days: float) -> List[Finding]:
    out: List[Finding] = []
    if "S0" not in results:
        return out
    baseline = results["S0"].wholesale_cost_eur
    if baseline <= 0:
        return out
    scale_to_year = 365.0 / horizon_days if horizon_days > 0 else 1.0
    for k, r in results.items():
        if k == "S0":
            continue
        annual_save = (baseline - r.wholesale_cost_eur) * scale_to_year
        annual_baseline = baseline * scale_to_year
        save_pct = (annual_save / annual_baseline) * 100.0 if annual_baseline > 0 else 0.0
        if save_pct > 35:
            out.append(_warn(
                f"biz.high_save_{k}",
                f"{k} shows {save_pct:.0f}% wholesale saving — very high, "
                "likely reflects synthetic price volatility or wide comfort band.",
            ))
        elif save_pct < 0:
            out.append(_warn(
                f"biz.neg_save_{k}",
                f"{k} produces a NEGATIVE saving ({save_pct:.1f}%) — check inputs.",
            ))
        else:
            out.append(_ok(
                f"biz.save_{k}",
                f"{k}: {save_pct:.1f}% horizon-scaled wholesale saving "
                f"(naive annualisation €{annual_save:,.0f}/year — verify horizon).",
            ))
    return out


# --------------------------------------------------------------------------
# F. Annualisation / horizon sanity — NEW
# --------------------------------------------------------------------------
def check_annualisation(
    results: Dict[str, ScenarioResult],
    horizon_days: float,
    normalisation_mode: str,
) -> List[Finding]:
    """Flag short, winter-heavy, or otherwise non-representative horizons.

    Returns findings with codes prefixed `annual.*`. These are the warnings
    that should be loudly surfaced anywhere a €/year figure is shown.
    """
    out: List[Finding] = []
    if not results:
        return out

    first = next(iter(results.values()))
    season = detect_season(first.timestamps, getattr(first, "meta", {}).get("outdoor_temp_c"))
    factor, note = annualisation_factor(normalisation_mode, horizon_days, season)
    label = season["season_label"]
    spans = season["spans_full_year"]
    winter_heavy = season["winter_heavy"]
    short = season["is_short_horizon"]

    # 1. Mode disclosure — always shown
    out.append(_info(
        "annual.mode",
        f"Normalisation mode = '{normalisation_mode}', factor ×{factor:.2f}. {note}",
    ))

    # 2. Horizon length
    if spans:
        out.append(_ok(
            "annual.full_year",
            f"Horizon is {horizon_days:.0f} days — covers a full year. "
            "Annualised values are validated.",
        ))
    elif short:
        out.append(_warn(
            "annual.short_horizon",
            f"Horizon is only {horizon_days:.1f} days — annualised €/year values are "
            "extrapolations and may not be representative. "
            "Run a full year of weather and price data for board-grade estimates.",
        ))
    else:
        out.append(_warn(
            "annual.medium_horizon",
            f"Horizon is {horizon_days:.0f} days (< 1 year) — annualised values are "
            "extrapolations. Verify with a full-year run before external quoting.",
        ))

    # 3. Winter-heavy bias
    if winter_heavy:
        out.append(_warn(
            "annual.winter_heavy",
            f"Horizon is winter-heavy ({season['heating_share']*100:.0f}% of timesteps fall in "
            "heating months). Multiplying winter weeks by 52 OVERSTATES annual savings: "
            "winter loads are 3–4× the annual average and price volatility is 2–3× higher. "
            "Treat annualised €/year figures as INDICATIVE UPSIDE, not validated benefit.",
        ))

    # 4. Recommended board-slide wording
    if not spans:
        out.append(_info(
            "annual.board_wording",
            "Recommended board-slide caveat: \"Indicative winter-week annualisation only. "
            "Full-year simulation required before treating this as a customer benefit estimate.\"",
        ))

    # 5. Hard ceiling: simple-annualised €/year > €600/household is implausible
    if "S0" in results and len(results) > 1:
        baseline = results["S0"].wholesale_cost_eur
        best_cost = min(r.wholesale_cost_eur for k, r in results.items() if k != "S0")
        annual_value = (baseline - best_cost) * factor
        # Literature central case is €100-350/yr customer-side, €200-600 wholesale-side
        if annual_value > 600:
            out.append(_warn(
                "annual.implausibly_high",
                f"Annualised wholesale value €{annual_value:,.0f}/year exceeds typical German "
                "literature range (€100–600/yr/household). Likely caused by winter-week × 52 "
                "extrapolation. Re-run with a representative full-year horizon.",
            ))

    return out


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
def run_all_validations(
    results: Dict[str, ScenarioResult],
    arch: Archetype,
    horizon_days: float,
    normalisation_mode: str = "simple",
) -> Dict[str, List[Finding]]:
    """Run all validation suites; return findings grouped per scenario.

    Special keys:
        'business'      — cross-scenario findings (% savings, etc.)
        'annualisation' — horizon / extrapolation flags
    """
    out: Dict[str, List[Finding]] = {}
    for name, r in results.items():
        findings = []
        findings += check_thermal(r, arch)
        findings += check_energy(r, arch)
        findings += check_optimisation(r, arch)
        findings += check_market(r)
        out[name] = findings
    out["business"] = check_business(results, horizon_days)
    out["annualisation"] = check_annualisation(results, horizon_days, normalisation_mode)
    return out
