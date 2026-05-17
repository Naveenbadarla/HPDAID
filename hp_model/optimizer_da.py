"""
optimizer_da.py — Core linear-programming optimiser.

This module solves the price-aware dispatch of a residential heat pump with
optional DHW storage and configurable comfort band.  It is used for:
    - Day-Ahead optimisation (using DA prices)
    - Intraday re-optimisation (using ID prices, same physics)

Solver
------
We use ``scipy.optimize.linprog`` with the HiGHS backend (ships inside
scipy).  HiGHS is a high-quality open-source LP solver that requires no
external install.  This makes the project trivially deployable inside
Streamlit Cloud, Docker images, or any standard Python environment.

OBJECTIVE
---------
Minimise:
       Σ_t  price[t]/1000 * elec[t]              # wholesale €/MWh -> €/kWh
     + Σ_t  λ_comfort * (v_pos[t] + v_neg[t])    # comfort violation €
     + Σ_t  λ_dhw     * (d_pos[t] + d_neg[t])    # dhw violation €

We divide the price by 1000 because prices are in €/MWh and energies in
kWh.  Cost units come out in €.

CONSTRAINTS
-----------
(a) Indoor temperature dynamics (RC):
        T_in[t+1] = α_T * T_in[t]
                  + α_q * q_sh[t]
                  + α_out * T_out[t]
                  + α_gain * Q_gain[t]

(b) Comfort with slack:
        T_in[t] >= T_min - v_neg[t]
        T_in[t] <= T_max + v_pos[t]
        v_pos, v_neg >= 0

(c) DHW dynamics:
        E_dhw[t+1] = E_dhw[t] + q_dhw[t] - draw[t] - loss

(d) DHW bounds with slack:
        E_dhw[t] >= E_min - d_neg[t]
        E_dhw[t] <= E_max + d_pos[t]

(e) Heat-pump capacity (thermal AND electrical):
        q_sh[t] + q_dhw[t] <= Q_th_max * dt
        elec[t] = q_sh[t]/COP_sh[t] + q_dhw[t]/COP_dhw[t]
        elec[t] <= P_el_max * dt

(f) Terminal anti-gaming constraints:
        T_in[N] >= T_target - 0.3
        E_dhw[N] >= 0.6 * E_max
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

from . import thermal_model as tm
from . import cop_model as cm
from . import dhw_model as dhw_mod
from . import config as C
from .archetypes import Archetype


@dataclass
class OptResult:
    """Dispatch + economics from a single optimisation."""
    t_in_c: np.ndarray
    q_sh_kwh: np.ndarray
    q_dhw_kwh: np.ndarray
    e_dhw_kwh: np.ndarray
    elec_sh_kwh: np.ndarray
    elec_dhw_kwh: np.ndarray
    elec_kwh: np.ndarray
    cop_sh: np.ndarray
    cop_dhw: np.ndarray
    comfort_violation_kdeg: np.ndarray
    dhw_violation_kwh: np.ndarray
    objective_eur: float
    wholesale_cost_eur: float
    status: str


# ---------------------------------------------------------------------------
# Variable layout (per horizon of length n)
# ---------------------------------------------------------------------------
def _index_offsets(n: int) -> dict:
    return {
        "q_sh":   0,
        "q_dhw":  n,
        "t_in":   2 * n,
        "e_dhw":  3 * n + 1,
        "v_pos":  4 * n + 2,
        "v_neg":  5 * n + 2,
        "d_pos":  6 * n + 2,
        "d_neg":  7 * n + 2,
        "n_vars": 8 * n + 2,
    }


def optimise(
    arch: Archetype,
    outdoor_temp_c: np.ndarray,
    dhw_draw_kwh: np.ndarray,
    price_eur_per_mwh: np.ndarray,
    timestep_h: float,
    comfort_band: Tuple[float, float] = (C.T_MIN, C.T_MAX),
    enable_dhw_flex: bool = True,
    t_in_init: Optional[float] = None,
    e_dhw_init: Optional[float] = None,
    t_in_terminal: Optional[float] = None,
    e_dhw_terminal_frac: float = 0.6,
    comfort_penalty: float = C.COMFORT_PENALTY_EUR_PER_KWH_DEG,
    dhw_penalty: float = C.DHW_PENALTY_EUR_PER_KWH,
    solver_msg: bool = C.SOLVER_MSG,
    solver_time_limit: int = C.SOLVER_TIME_LIMIT_SEC,
) -> OptResult:
    """Solve the LP for one horizon and one price series."""
    n = len(price_eur_per_mwh)
    assert len(outdoor_temp_c) == n and len(dhw_draw_kwh) == n, (
        f"Length mismatch: prices={n}, T_out={len(outdoor_temp_c)}, "
        f"DHW={len(dhw_draw_kwh)}"
    )

    # ---------- Pre-compute coefficients ----------
    cop_sh      = cm.cop_space_heating(outdoor_temp_c, arch.emitter)
    cop_dhw_arr = cm.cop_dhw(outdoor_temp_c)
    cop_sh      = np.maximum(cop_sh,      1e-3)
    cop_dhw_arr = np.maximum(cop_dhw_arr, 1e-3)

    alpha_T, alpha_q, alpha_out, alpha_gain = tm.coefficients(arch, timestep_h)
    bounds_dhw = dhw_mod.dhw_bounds(arch)

    t_min, t_max = comfort_band
    t0 = arch.t_target if t_in_init  is None else t_in_init
    e0 = bounds_dhw.e_init_kwh if e_dhw_init is None else e_dhw_init
    t_term = (arch.t_target - 0.3) if t_in_terminal is None else t_in_terminal
    e_term = e_dhw_terminal_frac * bounds_dhw.e_max_kwh

    q_th_step_max = arch.hp_thermal_kw    * timestep_h
    e_el_step_max = arch.hp_electrical_kw * timestep_h

    # Narrow DHW bounds when flex disabled
    if enable_dhw_flex:
        e_dhw_lo, e_dhw_hi = bounds_dhw.e_min_kwh, bounds_dhw.e_max_kwh
    else:
        e_dhw_lo = bounds_dhw.e_min_kwh
        e_dhw_hi = min(bounds_dhw.e_max_kwh, e0 + 0.5)
        e_dhw_hi = max(e_dhw_hi, e_dhw_lo + 0.1)

    off = _index_offsets(n)
    nv = off["n_vars"]

    # ---------- Variable bounds ----------
    var_lb = np.full(nv, -np.inf)
    var_ub = np.full(nv,  np.inf)

    var_lb[off["q_sh"]   : off["q_sh"]   + n] = 0.0
    var_ub[off["q_sh"]   : off["q_sh"]   + n] = q_th_step_max
    var_lb[off["q_dhw"]  : off["q_dhw"]  + n] = 0.0
    var_ub[off["q_dhw"]  : off["q_dhw"]  + n] = q_th_step_max

    var_lb[off["t_in"]   : off["t_in"]   + n + 1] = t_min - 5.0
    var_ub[off["t_in"]   : off["t_in"]   + n + 1] = t_max + 5.0
    var_lb[off["e_dhw"]  : off["e_dhw"]  + n + 1] = 0.0
    var_ub[off["e_dhw"]  : off["e_dhw"]  + n + 1] = bounds_dhw.e_max_kwh + 5.0

    var_lb[off["v_pos"]  : off["v_pos"]  + n] = 0.0
    var_lb[off["v_neg"]  : off["v_neg"]  + n] = 0.0
    var_lb[off["d_pos"]  : off["d_pos"]  + n] = 0.0
    var_lb[off["d_neg"]  : off["d_neg"]  + n] = 0.0

    # ---------- Objective ----------
    c = np.zeros(nv)
    price_kwh = np.asarray(price_eur_per_mwh, dtype=float) / 1000.0
    c[off["q_sh"]   : off["q_sh"]   + n] = price_kwh / cop_sh
    c[off["q_dhw"]  : off["q_dhw"]  + n] = price_kwh / cop_dhw_arr
    c[off["v_pos"]  : off["v_pos"]  + n] = comfort_penalty
    c[off["v_neg"]  : off["v_neg"]  + n] = comfort_penalty
    c[off["d_pos"]  : off["d_pos"]  + n] = dhw_penalty
    c[off["d_neg"]  : off["d_neg"]  + n] = dhw_penalty

    # ---------- Equality constraints ----------
    n_eq = 2 + 2 * n
    A_eq = lil_matrix((n_eq, nv))
    b_eq = np.zeros(n_eq)

    A_eq[0, off["t_in"]]  = 1.0
    b_eq[0] = t0
    A_eq[1, off["e_dhw"]] = 1.0
    b_eq[1] = e0

    row = 2
    for t in range(n):
        # Thermal dynamics
        A_eq[row, off["t_in"] + (t + 1)] = 1.0
        A_eq[row, off["t_in"] + t]       = -alpha_T
        A_eq[row, off["q_sh"] + t]       = -alpha_q
        b_eq[row] = alpha_out * float(outdoor_temp_c[t]) + alpha_gain * float(C.INTERNAL_GAIN_KW)
        row += 1

        # DHW dynamics
        A_eq[row, off["e_dhw"] + (t + 1)] = 1.0
        A_eq[row, off["e_dhw"] + t]       = -1.0
        A_eq[row, off["q_dhw"] + t]       = -1.0
        b_eq[row] = -float(dhw_draw_kwh[t]) - bounds_dhw.loss_kwh_per_step
        row += 1

    # ---------- Inequality constraints ----------
    n_ineq = 6 * n + 5
    A_ub = lil_matrix((n_ineq, nv))
    b_ub = np.zeros(n_ineq)

    r = 0
    for t in range(n):
        # comfort lower
        A_ub[r, off["t_in"] + t]   = -1.0
        A_ub[r, off["v_neg"] + t]  = -1.0
        b_ub[r] = -t_min
        r += 1
        # comfort upper
        A_ub[r, off["t_in"] + t]   =  1.0
        A_ub[r, off["v_pos"] + t]  = -1.0
        b_ub[r] =  t_max
        r += 1
        # DHW lower
        A_ub[r, off["e_dhw"] + t]  = -1.0
        A_ub[r, off["d_neg"] + t]  = -1.0
        b_ub[r] = -e_dhw_lo
        r += 1
        # DHW upper
        A_ub[r, off["e_dhw"] + t]  =  1.0
        A_ub[r, off["d_pos"] + t]  = -1.0
        b_ub[r] =  e_dhw_hi
        r += 1
        # thermal cap
        A_ub[r, off["q_sh"]  + t]  = 1.0
        A_ub[r, off["q_dhw"] + t]  = 1.0
        b_ub[r] = q_th_step_max
        r += 1
        # electrical cap
        A_ub[r, off["q_sh"]  + t]  = 1.0 / cop_sh[t]
        A_ub[r, off["q_dhw"] + t]  = 1.0 / cop_dhw_arr[t]
        b_ub[r] = e_el_step_max
        r += 1

    # Terminal
    A_ub[r, off["t_in"] + n] = -1.0;  b_ub[r] = -t_term;        r += 1
    A_ub[r, off["t_in"] + n] =  1.0;  b_ub[r] =  t_max + 1.0;   r += 1
    A_ub[r, off["e_dhw"] + n] = -1.0; b_ub[r] = -e_term;        r += 1
    # final-step comfort (slacked) to keep LP feasible if T must dip at end
    A_ub[r, off["t_in"] + n]        = -1.0
    A_ub[r, off["v_neg"] + (n - 1)] = -1.0
    b_ub[r] = -t_min
    r += 1
    A_ub[r, off["t_in"] + n]        =  1.0
    A_ub[r, off["v_pos"] + (n - 1)] = -1.0
    b_ub[r] = t_max
    r += 1

    A_eq = csr_matrix(A_eq)
    A_ub = csr_matrix(A_ub)

    # ---------- Solve ----------
    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=list(zip(var_lb, var_ub)),
        method="highs",
        options={
            "disp": solver_msg,
            "time_limit": float(solver_time_limit),
        },
    )

    if not res.success or res.x is None:
        zero = np.zeros(n)
        return OptResult(
            t_in_c=np.full(n, t0), q_sh_kwh=zero, q_dhw_kwh=zero,
            e_dhw_kwh=np.full(n, e0),
            elec_sh_kwh=zero, elec_dhw_kwh=zero, elec_kwh=zero,
            cop_sh=cop_sh, cop_dhw=cop_dhw_arr,
            comfort_violation_kdeg=zero, dhw_violation_kwh=zero,
            objective_eur=0.0, wholesale_cost_eur=0.0,
            status=f"INFEASIBLE: {res.message}",
        )

    x = res.x

    def slice_(name, length):
        return x[off[name] : off[name] + length].copy()

    q_sh_v   = slice_("q_sh",  n)
    q_dhw_v  = slice_("q_dhw", n)
    t_in_v   = slice_("t_in",  n + 1)[1:]
    e_dhw_v  = slice_("e_dhw", n + 1)[1:]
    v_pos_v  = slice_("v_pos", n)
    v_neg_v  = slice_("v_neg", n)
    d_pos_v  = slice_("d_pos", n)
    d_neg_v  = slice_("d_neg", n)

    elec_sh  = q_sh_v  / cop_sh
    elec_dhw = q_dhw_v / cop_dhw_arr
    elec     = elec_sh + elec_dhw

    wholesale = float(np.sum(elec * price_kwh))

    return OptResult(
        t_in_c=t_in_v,
        q_sh_kwh=q_sh_v,
        q_dhw_kwh=q_dhw_v,
        e_dhw_kwh=e_dhw_v,
        elec_sh_kwh=elec_sh,
        elec_dhw_kwh=elec_dhw,
        elec_kwh=elec,
        cop_sh=cop_sh,
        cop_dhw=cop_dhw_arr,
        comfort_violation_kdeg=v_pos_v + v_neg_v,
        dhw_violation_kwh=d_pos_v + d_neg_v,
        objective_eur=float(res.fun) if res.fun is not None else 0.0,
        wholesale_cost_eur=wholesale,
        status=f"OK: {res.message}",
    )
