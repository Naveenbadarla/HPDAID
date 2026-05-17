"""
baseline_controller.py — Baseline "dumb" thermostat (Scenario S0).

This is *not* an optimisation.  It is a forward simulation of a household
that uses heat only to maintain comfort, with no awareness of prices.

Logic per timestep:
    - If T_in < T_target − deadband: turn on space-heating, output full thermal capacity.
    - If T_in > T_target + deadband: turn off space-heating.
    - DHW: if tank energy < (E_min + reserve): reheat to E_max, else nothing.
    - Heat-pump shared capacity: DHW has priority (real-world behaviour).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import thermal_model as tm
from . import cop_model as cm
from . import dhw_model as dhw
from .archetypes import Archetype
from . import config as C


@dataclass
class BaselineResult:
    t_in_c: np.ndarray
    q_sh_kwh: np.ndarray
    q_dhw_kwh: np.ndarray
    e_dhw_kwh: np.ndarray
    elec_kwh: np.ndarray
    elec_sh_kwh: np.ndarray
    elec_dhw_kwh: np.ndarray
    cop_sh: np.ndarray
    cop_dhw: np.ndarray


def run_baseline(
    arch: Archetype,
    outdoor_temp_c: np.ndarray,
    dhw_draw_kwh: np.ndarray,
    timestep_h: float,
    t_in_init: float | None = None,
    deadband_k: float = 0.4,
) -> BaselineResult:
    """Simulate a thermostat-controlled household for the given horizon."""
    n = len(outdoor_temp_c)

    cop_sh = cm.cop_space_heating(outdoor_temp_c, arch.emitter)
    cop_dhw_arr = cm.cop_dhw(outdoor_temp_c)

    t_in = np.zeros(n + 1)
    t_in[0] = arch.t_target if t_in_init is None else t_in_init
    q_sh = np.zeros(n)
    q_dhw = np.zeros(n)
    e_dhw = np.zeros(n + 1)

    bounds = dhw.dhw_bounds(arch)
    e_dhw[0] = bounds.e_init_kwh

    # Capacities expressed in kWh per step
    q_th_step_max = arch.hp_thermal_kw * timestep_h
    e_el_step_max = arch.hp_electrical_kw * timestep_h
    q_dhw_step_max = q_th_step_max                      # DHW can use full capacity

    # Reserve margin in DHW tank
    dhw_reserve_kwh = 0.15 * arch.dhw_tank_kwh

    for t in range(n):
        # --- Decide DHW heating first (priority for the heat pump) ---
        need_dhw = e_dhw[t] < (bounds.e_min_kwh + dhw_reserve_kwh)
        if need_dhw:
            # ramp to full tank — limited by capacity & ceiling
            room_in_tank = bounds.e_max_kwh - e_dhw[t]
            q_dhw_t = min(q_dhw_step_max, room_in_tank)
            # also limit by electrical capacity
            elec_dhw_t = q_dhw_t / cop_dhw_arr[t]
            if elec_dhw_t > e_el_step_max:
                q_dhw_t = e_el_step_max * cop_dhw_arr[t]
        else:
            q_dhw_t = 0.0
        q_dhw[t] = q_dhw_t

        # --- Decide space heating with remaining capacity ---
        q_remaining = max(0.0, q_th_step_max - q_dhw_t)
        elec_remaining = max(0.0, e_el_step_max - q_dhw_t / cop_dhw_arr[t])
        max_sh_thermal_step = min(q_remaining, elec_remaining * cop_sh[t])

        if t_in[t] < arch.t_target - deadband_k:
            q_sh[t] = max_sh_thermal_step
        elif t_in[t] > arch.t_target + deadband_k:
            q_sh[t] = 0.0
        else:
            # Inside the deadband — keep previous state-ish:
            # simple rule: heat just enough to compensate losses to outside,
            # if positive and within capacity.
            loss_kw = max(0.0, arch.ua_kw_per_k * (t_in[t] - outdoor_temp_c[t]))
            q_sh[t] = min(max_sh_thermal_step, max(0.0, loss_kw * timestep_h - C.INTERNAL_GAIN_KW * timestep_h))

        # --- Apply dynamics ---
        t_in[t + 1] = tm.update_temperature(
            t_in[t], outdoor_temp_c[t], q_sh[t], arch, timestep_h
        )
        e_dhw[t + 1] = dhw.dhw_step(
            e_dhw[t], q_dhw[t], dhw_draw_kwh[t], dhw.dhw_bounds(arch).loss_kwh_per_step
        )
        e_dhw[t + 1] = max(0.0, e_dhw[t + 1])

    elec_sh = q_sh / np.where(cop_sh > 0, cop_sh, 1.0)
    elec_dhw = q_dhw / np.where(cop_dhw_arr > 0, cop_dhw_arr, 1.0)
    return BaselineResult(
        t_in_c=t_in[1:],
        q_sh_kwh=q_sh,
        q_dhw_kwh=q_dhw,
        e_dhw_kwh=e_dhw[1:],
        elec_kwh=elec_sh + elec_dhw,
        elec_sh_kwh=elec_sh,
        elec_dhw_kwh=elec_dhw,
        cop_sh=cop_sh,
        cop_dhw=cop_dhw_arr,
    )
