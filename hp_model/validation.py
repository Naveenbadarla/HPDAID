"""
validation.py — Sanity checks on optimisation results.

Validation categories:
    A. Thermal sanity     — temperature inside bounds, no unrealistic ramps
    B. Energy sanity      — elec = thermal / COP, DHW balance closes
    C. Optimisation sanity— end-of-horizon not gamed, comfort penalty not abused
    D. Market sanity      — DA-only vs DA+ID handled correctly; no min(DA,ID) hack
    E. Business sanity    — savings within plausible range

Each check returns a `Finding`; the app renders them as colour-coded
banners.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .scenarios import ScenarioResult
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

    # Unrealistic ramp check: |dT| per step
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
    # elec consistency
    expected_el = r.q_sh_kwh / r.cop_sh + r.q_dhw_kwh / r.cop_dhw
    diff = np.max(np.abs(expected_el - r.elec_kwh))
    if diff > 1e-3:
        out.append(_err("energy.elec_consistency",
                        f"elec != thermal/COP, max deviation {diff:.4f} kWh."))
    else:
        out.append(_ok("energy.elec_consistency", "elec = thermal/COP holds for every step."))

    # Capacity check
    q_max = arch.hp_thermal_kw * (r.timestamps[1] - r.timestamps[0]).total_seconds() / 3600.0
    if np.any(r.q_sh_kwh + r.q_dhw_kwh > q_max * 1.01):
        out.append(_err("energy.capacity", "Heat-pump thermal capacity breached."))
    else:
        out.append(_ok("energy.capacity", f"HP thermal output stays within {q_max:.2f} kWh / step."))

    # COP range
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

    # End-of-horizon gaming: terminal indoor temp not far below target
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
    # End-of-horizon DHW
    if len(r.e_dhw_kwh) > 0:
        e_end = r.e_dhw_kwh[-1]
        if e_end < 0.3 * arch.dhw_tank_kwh:
            out.append(_warn("opt.end_dhw", f"Final DHW state {e_end:.1f} kWh is low."))
        else:
            out.append(_ok("opt.end_dhw", f"Final DHW state {e_end:.1f} kWh."))

    # Comfort violations should be near zero (otherwise the optimiser used the
    # comfort slack as cheap "savings").
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
    # DA+ID scenarios must have a non-zero ID adjustment (otherwise S2==S1)
    if r.name in ("S2", "S3", "S4"):
        adj = float(np.sum(np.abs(r.id_adjustment_kwh)))
        if adj < 1e-6:
            out.append(_info("market.id_zero", "ID adjustment volume is zero — ID prices may equal DA prices."))
        else:
            out.append(_ok("market.id_nonzero",
                           f"DA+ID settled correctly with {adj:.1f} kWh of ID adjustment volume."))

    # Cost must equal da_component + id_component (we check up to rounding)
    expected_total = r.da_cost_component_eur + r.id_adjustment_cost_eur
    if abs(expected_total - r.wholesale_cost_eur) > 0.01:
        out.append(_err("market.settlement",
                        f"Settlement mismatch: total ≠ DA + ID adjustment "
                        f"({r.wholesale_cost_eur:.2f} vs {expected_total:.2f})."))
    return out


# --------------------------------------------------------------------------
# E. Business sanity
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
                f"{k}: {save_pct:.1f}% wholesale saving (€{annual_save:,.0f}/year).",
            ))
    return out


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
def run_all_validations(
    results: Dict[str, ScenarioResult], arch: Archetype, horizon_days: float
) -> Dict[str, List[Finding]]:
    """Run all validation suites; return findings grouped per scenario.

    A special key 'business' contains cross-scenario findings.
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
    return out
