"""
FlexHeat Optimiser — Streamlit application.

A residential heat-pump flexibility valuation tool for the German market.
Compares baseline thermostat operation against four market-aware
optimisation scenarios (Day-Ahead, DA+Intraday, wider comfort band, DHW
flexibility) and aggregates the result to a portfolio (E.ON view).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from hp_model import config as C
from hp_model import archetypes as A
from hp_model import data_loader as dl
from hp_model import scenarios as sc
from hp_model import fleet as fl
from hp_model import validation as val
from hp_model import plotting as pl


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FlexHeat Optimiser",
    page_icon="🔥",
    layout="wide",
)


# --------------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "results": None,           # Dict[str, ScenarioResult]
        "kpi_df": None,            # pd.DataFrame
        "value": None,             # Dict[str, float]
        "interpretation": None,    # str
        "md": None,                # MarketData
        "arch": None,              # Archetype
        "run_cfg": None,           # Dict
        "validations": None,       # Dict[str, List[Finding]]
        "fleet_summary": None,
        "scaling_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# --------------------------------------------------------------------------
# Sidebar — global run controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️  FlexHeat Optimiser")
st.sidebar.caption("Residential heat-pump flex valuation · Germany")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("Run window")
    horizon_days = st.slider("Horizon (days)", 1, 14, 7, 1)
    timestep_min = st.selectbox("Time step (minutes)", [15, 30, 60], index=0)
    timestep_h = timestep_min / 60.0
    n_steps = int(round(horizon_days * 24 / timestep_h))
    start_date = pd.Timestamp("2025-01-13")  # a winter Monday

    st.subheader("Archetype")
    arch_options = A.archetype_options()
    arch_name = st.selectbox(
        "Building / heat-pump archetype",
        list(arch_options.keys()),
        format_func=lambda k: arch_options[k],
        index=list(arch_options.keys()).index("renovated_sfh"),
    )

    st.subheader("Comfort & flexibility")
    t_min = st.number_input("Comfort min (°C)", 16.0, 22.0, float(C.T_MIN), 0.5)
    t_max = st.number_input("Comfort max (°C)", 20.0, 26.0, float(C.T_MAX), 0.5)
    t_min_flex = st.number_input("Flex min (°C)", 16.0, 22.0, float(C.T_MIN_FLEX), 0.5)
    t_max_flex = st.number_input("Flex max (°C)", 20.0, 26.0, float(C.T_MAX_FLEX), 0.5)

    st.subheader("Markets")
    id_spread_std = st.slider(
        "Synthetic ID spread std (€/MWh)", 0.0, 50.0, float(C.ID_SPREAD_STD_EUR_MWH), 1.0,
        help="Volatility of intraday vs. day-ahead. Real EPEX continuous data typically 15-25.",
    )

    st.subheader("Scenarios")
    selected_scenarios = st.multiselect(
        "Scenarios to run",
        ["S0", "S1", "S2", "S3", "S4"],
        default=["S0", "S1", "S2", "S3", "S4"],
        help=(
            "S0 = Baseline thermostat · S1 = DA-only optimisation · "
            "S2 = DA + Intraday re-optimisation · S3 = S2 + wider comfort · "
            "S4 = S2 + DHW flexibility"
        ),
    )

    st.markdown("---")
    run_btn = st.button("▶  Run optimisation", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Main tabs
# --------------------------------------------------------------------------
st.title("🔥 FlexHeat Optimiser")
st.caption(
    "Quantify Day-Ahead + Intraday market value, comfort-band flexibility, and DHW shifting "
    "for a residential heat pump — then scale to an E.ON-style aggregator portfolio."
)

tabs = st.tabs([
    "1 · Executive Summary",
    "2 · Household Setup",
    "3 · Market Data",
    "4 · Optimisation",
    "5 · Thermal Behaviour",
    "6 · Market Value",
    "7 · Fleet Scaling",
    "8 · Validation",
    "9 · Export",
])


# --------------------------------------------------------------------------
# TAB 2 — Household Setup (rendered first so users can tweak before run)
# --------------------------------------------------------------------------
with tabs[1]:
    st.header("Household & heat-pump configuration")
    st.caption("Edit any field; the LP uses these numbers verbatim.")

    base = A.get_archetype(arch_name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Building**")
        floor_area = st.number_input(
            "Floor area (m²)", 30.0, 500.0, float(base.floor_area_m2), 5.0
        )
        annual_heat = st.number_input(
            "Annual SH demand (kWh)", 1_000.0, 50_000.0,
            float(base.annual_heat_demand_kwh), 100.0,
        )
        annual_dhw = st.number_input(
            "Annual DHW demand (kWh)", 500.0, 8_000.0,
            float(base.annual_dhw_demand_kwh), 50.0,
        )
        ua = st.number_input(
            "UA value (kW/K)", 0.05, 1.0, float(base.ua_kw_per_k), 0.01, format="%.3f"
        )
        c_th = st.number_input(
            "Thermal capacitance (kWh/K)", 5.0, 100.0, float(base.c_th_kwh_per_k), 1.0
        )
    with col2:
        st.markdown("**Heat pump**")
        hp_thermal = st.number_input(
            "HP thermal capacity (kW)", 2.0, 30.0, float(base.hp_thermal_kw), 0.5
        )
        hp_elec = st.number_input(
            "HP electrical capacity (kW)", 1.0, 15.0, float(base.hp_electrical_kw), 0.5
        )
        hp_min_mod = st.number_input(
            "Min modulation (0-1)", 0.0, 1.0, float(base.hp_min_modulation), 0.05
        )
        emitter = st.selectbox(
            "Emitter type", ["underfloor", "radiator"],
            index=["underfloor", "radiator"].index(base.emitter),
        )
    with col3:
        st.markdown("**DHW tank**")
        dhw_kwh = st.number_input(
            "Tank usable energy (kWh)", 2.0, 30.0, float(base.dhw_tank_kwh), 0.5
        )
        dhw_loss = st.number_input(
            "Standing loss (kWh/step)", 0.0, 0.5,
            float(base.dhw_tank_loss_kwh_per_step), 0.005, format="%.4f",
        )
        st.markdown("**Comfort defaults**")
        t_target = st.number_input(
            "Target T_in (°C)", 18.0, 24.0, float(base.t_target), 0.5
        )

    arch = A.Archetype(
        name=base.name + ("_custom" if base.name == "custom" else ""),
        label=base.label,
        floor_area_m2=float(floor_area),
        annual_heat_demand_kwh=float(annual_heat),
        annual_dhw_demand_kwh=float(annual_dhw),
        ua_kw_per_k=float(ua),
        c_th_kwh_per_k=float(c_th),
        hp_thermal_kw=float(hp_thermal),
        hp_electrical_kw=float(hp_elec),
        hp_min_modulation=float(hp_min_mod),
        emitter=emitter,
        dhw_tank_kwh=float(dhw_kwh),
        dhw_tank_loss_kwh_per_step=float(dhw_loss),
        t_target=float(t_target),
        t_min=float(t_min),
        t_max=float(t_max),
        flexibility_rating=base.flexibility_rating,
    )
    st.session_state["arch"] = arch

    st.markdown("---")
    st.markdown("**Resolved archetype passed to optimiser**")
    st.json(arch.to_dict(), expanded=False)


# --------------------------------------------------------------------------
# TAB 3 — Market Data
# --------------------------------------------------------------------------
with tabs[2]:
    st.header("Market data inputs")
    st.caption(
        "Upload CSVs with a `timestamp` column and one of: "
        "`da_price_eur_per_mwh`, `id_price_eur_per_mwh`, `outdoor_temp_c`, `dhw_draw_kwh`. "
        "Anything you don't upload will fall back to synthetic data."
    )

    col1, col2 = st.columns(2)
    with col1:
        da_file = st.file_uploader("Day-Ahead prices (CSV)", type="csv", key="da_csv")
        id_file = st.file_uploader("Intraday prices (CSV)", type="csv", key="id_csv")
    with col2:
        weather_file = st.file_uploader("Outdoor temperature (CSV)", type="csv", key="w_csv")
        dhw_file = st.file_uploader("DHW draw profile (CSV)", type="csv", key="dhw_csv")

    if st.button("Load market data", use_container_width=True):
        md = dl.load_market_data(
            start=start_date,
            n_steps=n_steps,
            timestep_h=timestep_h,
            da_file=da_file,
            id_file=id_file,
            weather_file=weather_file,
            dhw_file=dhw_file,
            annual_dhw_kwh=arch.annual_dhw_demand_kwh,
            id_spread_std=id_spread_std,
        )
        st.session_state["md"] = md
        st.success("Market data loaded.")

    md = st.session_state["md"]
    if md is not None:
        for note in md.notes:
            if "synthetic" in note.lower() or "could not" in note.lower():
                st.warning(note)
            else:
                st.info(note)

        st.plotly_chart(
            pl.plot_prices(md.timestamps, md.da_price_eur_mwh, md.id_price_eur_mwh),
            use_container_width=True,
            key="md_prices",
        )

        with st.expander("Preview market data table"):
            preview = pd.DataFrame({
                "timestamp": md.timestamps,
                "DA €/MWh": md.da_price_eur_mwh,
                "ID €/MWh": md.id_price_eur_mwh,
                "T_out °C": md.outdoor_temp_c,
                "DHW draw kWh": md.dhw_draw_kwh,
            })
            st.dataframe(preview.head(96), use_container_width=True)
    else:
        st.info(
            "Click **Load market data** above, or hit **Run optimisation** in the sidebar — "
            "synthetic data will be generated automatically."
        )


# --------------------------------------------------------------------------
# Run handler
# --------------------------------------------------------------------------
if run_btn:
    with st.spinner("Loading market data and solving LP for each scenario…"):
        # Always (re)load market data with current settings if missing
        md = st.session_state["md"]
        if md is None:
            md = dl.load_market_data(
                start=start_date,
                n_steps=n_steps,
                timestep_h=timestep_h,
                annual_dhw_kwh=arch.annual_dhw_demand_kwh,
                id_spread_std=id_spread_std,
            )
            st.session_state["md"] = md

        # Run scenarios
        results = sc.run_all_scenarios(
            arch=arch,
            md=md,
            timestep_h=timestep_h,
            comfort_band=(float(t_min), float(t_max)),
            comfort_band_flex=(float(t_min_flex), float(t_max_flex)),
            scenarios=selected_scenarios,
        )
        kpi_df = sc.kpi_table(results, horizon_days=float(horizon_days))
        value = sc.value_stack(results, horizon_days=float(horizon_days))
        interp = sc.build_interpretation(
            results, value,
            customer_share=float(C.PORTFOLIO_DEFAULTS["customer_share"]),
            horizon_days=float(horizon_days),
        )
        validations = val.run_all_validations(results, arch, float(horizon_days))

        st.session_state["results"] = results
        st.session_state["kpi_df"] = kpi_df
        st.session_state["value"] = value
        st.session_state["interpretation"] = interp
        st.session_state["validations"] = validations
        st.session_state["run_cfg"] = {
            "horizon_days": horizon_days,
            "timestep_h": timestep_h,
            "n_steps": n_steps,
            "comfort_band": (float(t_min), float(t_max)),
            "comfort_band_flex": (float(t_min_flex), float(t_max_flex)),
            "archetype": arch_name,
            "scenarios": selected_scenarios,
        }
    st.success(f"Solved {len(results)} scenario(s).")


# --------------------------------------------------------------------------
# TAB 1 — Executive Summary
# --------------------------------------------------------------------------
with tabs[0]:
    results = st.session_state["results"]
    if results is None:
        st.info("Run an optimisation from the sidebar to populate the executive view.")
    else:
        kpi_df = st.session_state["kpi_df"]
        value = st.session_state["value"]
        interp = st.session_state["interpretation"]

        st.subheader("Headline numbers (annualised)")
        if "S0" in kpi_df.scenario.values:
            baseline_retail = float(
                kpi_df.loc[kpi_df.scenario == "S0", "retail_cost_eur_year"].iloc[0]
            )
        else:
            baseline_retail = float("nan")
        best_row = (
            kpi_df.loc[kpi_df.scenario != "S0"]
            .sort_values("wholesale_cost_eur_year").head(1)
        )
        if not best_row.empty:
            best = best_row.iloc[0]
            saving_eur = float(best["wholesale_saving_eur_year"])
            saving_retail = float(best["retail_saving_eur_year"])
            saving_pct = float(best["saving_pct_retail"])
        else:
            best = None
            saving_eur = saving_retail = saving_pct = float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Baseline retail bill",
            f"€{baseline_retail:,.0f}/yr" if np.isfinite(baseline_retail) else "—",
        )
        m2.metric(
            "Best-case retail saving",
            f"€{saving_retail:,.0f}/yr" if np.isfinite(saving_retail) else "—",
            delta=f"−{saving_pct:.1f}%" if np.isfinite(saving_pct) else None,
        )
        m3.metric(
            "Wholesale value (E.ON view)",
            f"€{saving_eur:,.0f}/yr" if np.isfinite(saving_eur) else "—",
        )
        m4.metric(
            "Best scenario",
            best["scenario"] if best is not None else "—",
            help=best["label"] if best is not None else None,
        )

        st.markdown("### Interpretation")
        st.markdown(interp)

        st.markdown("### Value waterfall — baseline to optimised")
        st.plotly_chart(pl.plot_waterfall(kpi_df, value), use_container_width=True, key="exec_waterfall")

        st.markdown("### Value stack — where the savings come from")
        st.plotly_chart(pl.plot_value_stack(value), use_container_width=True, key="exec_value_stack")


# --------------------------------------------------------------------------
# TAB 4 — Optimisation results detail
# --------------------------------------------------------------------------
with tabs[3]:
    st.header("Optimisation results")
    results = st.session_state["results"]
    kpi_df = st.session_state["kpi_df"]
    if results is None:
        st.info("Run an optimisation to see results.")
    else:
        st.markdown("#### KPI table — per scenario")
        display = kpi_df.copy()
        currency_cols = [c for c in display.columns if "eur" in c]
        for c in currency_cols:
            display[c] = display[c].map(lambda x: f"€{x:,.0f}" if np.isfinite(x) else "—")
        for c in ["scop", "avg_cop_sh", "avg_cop_dhw"]:
            display[c] = display[c].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
        for c in ["saving_pct_retail", "shifted_share_pct"]:
            display[c] = display[c].map(lambda x: f"{x:.1f}%" if np.isfinite(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("#### Heat-pump electrical load — all scenarios")
        st.plotly_chart(pl.plot_hp_load(results), use_container_width=True, key="opt_hp_load")

        st.markdown("#### Cost comparison")
        mode = st.radio("View costs as", ["wholesale", "retail"], horizontal=True, index=0)
        st.plotly_chart(pl.plot_cost_bars(kpi_df, mode=mode), use_container_width=True, key="opt_cost_bars")

        with st.expander("Settlement breakdown (DA vs ID cost components)"):
            settle = kpi_df[["scenario", "label", "da_cost_eur", "id_adjustment_cost_eur"]].copy()
            settle["total_eur"] = settle["da_cost_eur"] + settle["id_adjustment_cost_eur"]
            st.dataframe(settle, use_container_width=True, hide_index=True)
            st.caption(
                "Two-stage settlement: DA volume is sold the day before at DA price; "
                "intraday adjustments are settled at ID price. We do NOT take min(DA, ID) — "
                "that would assume perfect foresight of which market would clear cheaper."
            )


# --------------------------------------------------------------------------
# TAB 5 — Thermal behaviour
# --------------------------------------------------------------------------
with tabs[4]:
    st.header("Thermal behaviour")
    results = st.session_state["results"]
    md = st.session_state["md"]
    if results is None or md is None:
        st.info("Run an optimisation to see thermal trajectories.")
    else:
        st.markdown("#### Indoor temperature over the horizon")
        scen_for_temp = st.multiselect(
            "Scenarios shown",
            list(results.keys()),
            default=list(results.keys()),
            key="temp_scen",
        )
        comfort = (float(t_min), float(t_max))
        if scen_for_temp:
            sub = {k: results[k] for k in scen_for_temp}
            st.plotly_chart(
                pl.plot_indoor_temperature(
                    results=sub,
                    comfort_band=comfort,
                    outdoor=md.outdoor_temp_c,
                    focus=list(sub.keys()),
                ),
                use_container_width=True,
                key="th_indoor_t",
            )

        st.markdown("#### DHW tank state of charge")
        if scen_for_temp:
            # DHW tank bounds: lower ≈ 30% SOC, upper = full capacity
            dhw_full = float(arch.dhw_tank_kwh)
            st.plotly_chart(
                pl.plot_dhw_tank(
                    results={k: results[k] for k in scen_for_temp},
                    e_min=0.3 * dhw_full,
                    e_max=dhw_full,
                    focus=list(scen_for_temp),
                ),
                use_container_width=True,
                key="th_dhw_tank",
            )

        st.markdown("#### Single-day dispatch view")
        day_choice = st.selectbox(
            "Day to inspect", list(range(1, int(horizon_days) + 1)), index=0, key="day_pick"
        )
        scen_for_dispatch = st.selectbox(
            "Scenario", list(results.keys()),
            index=min(2, len(results) - 1),
            key="dispatch_scen",
        )
        # Day timestamp from the market data index
        day_ts = pd.Timestamp(md.timestamps[0]).normalize() + pd.Timedelta(days=int(day_choice) - 1)
        dhw_full = float(arch.dhw_tank_kwh)
        st.plotly_chart(
            pl.plot_daily_dispatch(
                md_ts=md.timestamps,
                da=md.da_price_eur_mwh,
                idp=md.id_price_eur_mwh,
                result=results[scen_for_dispatch],
                day=day_ts,
                e_min=0.3 * dhw_full,
                e_max=dhw_full,
                comfort_band=(float(t_min), float(t_max)),
            ),
            use_container_width=True,
            key="th_daily_dispatch",
        )


# --------------------------------------------------------------------------
# TAB 6 — Market Value
# --------------------------------------------------------------------------
with tabs[5]:
    st.header("Market value decomposition")
    results = st.session_state["results"]
    value = st.session_state["value"]
    kpi_df = st.session_state["kpi_df"]
    if results is None:
        st.info("Run an optimisation to see value decomposition.")
    else:
        st.plotly_chart(pl.plot_value_stack(value), use_container_width=True, key="mv_value_stack")

        st.markdown("#### Value table (annualised €)")
        vdf = pd.DataFrame(
            [{"component": k, "eur_per_year": v} for k, v in value.items()]
        )
        vdf["eur_per_year"] = vdf["eur_per_year"].map(lambda x: f"€{x:,.0f}")
        st.dataframe(vdf, use_container_width=True, hide_index=True)

        st.markdown("#### Cost components per scenario (€/year)")
        cost_view = kpi_df[[
            "scenario", "label", "wholesale_cost_eur_year",
            "retail_cost_eur_year", "wholesale_saving_eur_year", "retail_saving_eur_year",
        ]].copy()
        for c in cost_view.columns:
            if "eur" in c:
                cost_view[c] = cost_view[c].map(lambda x: f"€{x:,.0f}")
        st.dataframe(cost_view, use_container_width=True, hide_index=True)

        st.caption(
            "**Two-stage settlement, not min(DA, ID).**  "
            "The DA leg is a firm financial position priced at DA. Intraday re-optimisation "
            "moves *physical* energy against the ID curve; the cash flow is "
            "`DA_vol × DA_price + (actual − DA_vol) × ID_price`."
        )


# --------------------------------------------------------------------------
# TAB 7 — Fleet scaling
# --------------------------------------------------------------------------
with tabs[6]:
    st.header("Portfolio scaling — E.ON aggregator view")
    results = st.session_state["results"]
    value = st.session_state["value"]
    if results is None or value is None:
        st.info("Run an optimisation to scale results to a portfolio.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_house = st.number_input(
                "Households in portfolio", 100, 5_000_000,
                int(C.PORTFOLIO_DEFAULTS["n_households"]), 1000,
            )
        with col2:
            adoption = st.slider("Adoption rate (active share)", 0.1, 1.0, 1.0, 0.05)
        with col3:
            cust_share = st.slider(
                "Customer share of saving",
                0.0, 1.0, float(C.PORTFOLIO_DEFAULTS["customer_share"]), 0.05,
                help="Split of the wholesale saving between customer bonus and E.ON retained.",
            )

        # Use best non-baseline scenario as the portfolio basis
        kpi_df = st.session_state["kpi_df"]
        non_baseline = kpi_df.loc[kpi_df.scenario != "S0"]
        if non_baseline.empty:
            st.warning("Need at least one optimised scenario to scale.")
        else:
            best = non_baseline.sort_values("wholesale_cost_eur_year").iloc[0]
            val_per_hh = float(best["wholesale_saving_eur_year"])
            elec_per_hh = float(best["annualised_elec_kwh"])

            fp = fl.FleetParams(
                n_households=int(n_house),
                adoption_rate=float(adoption),
                customer_share=float(cust_share),
            )
            summary = fl.scale_results(val_per_hh, elec_per_hh, arch, fp)
            st.session_state["fleet_summary"] = summary

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Aggregate value (€/yr)", f"€{summary['agg_annual_value_eur']/1e6:,.2f}M")
            m2.metric("Flex capacity (MW)", f"{summary['flex_mw_capacity']:,.0f}")
            m3.metric("Customer bonus pool (€/yr)", f"€{summary['customer_bonus_pool_eur']/1e6:,.2f}M")
            m4.metric("E.ON retained value (€/yr)", f"€{summary['eon_retained_value_eur']/1e6:,.2f}M")

            with st.expander("Full portfolio summary"):
                st.json({k: round(v, 2) for k, v in summary.items()})

            st.markdown("#### Value vs. portfolio size")
            scaling_df = fl.scaling_curve(
                value_per_household_eur=val_per_hh,
                arch_hp_electrical_kw=float(arch.hp_electrical_kw),
                customer_share=float(cust_share),
            )
            st.session_state["scaling_df"] = scaling_df
            st.plotly_chart(pl.plot_fleet_scaling(scaling_df), use_container_width=True, key="fleet_scaling")


# --------------------------------------------------------------------------
# TAB 8 — Validation & diagnostics
# --------------------------------------------------------------------------
with tabs[7]:
    st.header("Validation & diagnostics")
    validations = st.session_state["validations"]
    results = st.session_state["results"]
    if validations is None or results is None:
        st.info("Run an optimisation to populate validation checks.")
    else:
        n_err = sum(1 for k, fs in validations.items() for f in fs if f.level == "error")
        n_warn = sum(1 for k, fs in validations.items() for f in fs if f.level == "warning")
        n_ok = sum(1 for k, fs in validations.items() for f in fs if f.level == "ok")
        n_info = sum(1 for k, fs in validations.items() for f in fs if f.level == "info")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Passed", n_ok)
        c2.metric("ℹ️ Info", n_info)
        c3.metric("⚠️ Warnings", n_warn)
        c4.metric("❌ Errors", n_err)

        for key, findings in validations.items():
            with st.expander(
                f"{key} — {len(findings)} finding(s)",
                expanded=(n_err > 0 and key != "business"),
            ):
                for f in findings:
                    label = f"`{f.code}` · {f.message}"
                    if f.level == "error":
                        st.error(label)
                    elif f.level == "warning":
                        st.warning(label)
                    elif f.level == "info":
                        st.info(label)
                    else:
                        st.success(label)

        with st.expander("Solver status per scenario"):
            for k, r in results.items():
                st.write(f"**{k}** ({r.label}): `{r.status}`")


# --------------------------------------------------------------------------
# TAB 9 — Export
# --------------------------------------------------------------------------
with tabs[8]:
    st.header("Export results")
    results = st.session_state["results"]
    kpi_df = st.session_state["kpi_df"]
    if results is None:
        st.info("Run an optimisation first.")
    else:
        st.markdown(
            "Download a single comprehensive report bundle covering every scenario, "
            "or grab individual files below."
        )

        # ------------------------------------------------------------------
        # Helper: build a self-contained HTML report
        # ------------------------------------------------------------------
        def _build_html_report() -> str:
            run_cfg = st.session_state.get("run_cfg") or {}
            arch_obj = st.session_state.get("arch")
            validations = st.session_state.get("validations") or {}
            value = st.session_state.get("value") or {}
            interp = st.session_state.get("interpretation") or ""

            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            # ---- Section 1: Run summary
            cfg_rows = "".join(
                f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in run_cfg.items()
            )
            arch_rows = ""
            if arch_obj is not None:
                arch_rows = "".join(
                    f"<tr><th>{k}</th><td>{v}</td></tr>"
                    for k, v in arch_obj.to_dict().items()
                )

            # ---- Section 2: KPI table
            kpi_html = kpi_df.to_html(
                index=False, float_format=lambda x: f"{x:,.2f}", classes="kpi"
            )

            # ---- Section 3: Value stack
            value_rows = "".join(
                f"<tr><th>{k.replace('_', ' ')}</th><td>€{v:,.0f}/yr</td></tr>"
                for k, v in value.items()
            )

            # ---- Section 4: Validation findings
            valid_html_parts = []
            for key, findings in validations.items():
                rows = "".join(
                    f'<tr><td class="lvl-{f.level}">{f.level.upper()}</td>'
                    f'<td><code>{f.code}</code></td><td>{f.message}</td></tr>'
                    for f in findings
                )
                if rows:
                    valid_html_parts.append(
                        f"<h3>{key}</h3>"
                        f"<table class='findings'><thead><tr>"
                        f"<th>Level</th><th>Code</th><th>Message</th>"
                        f"</tr></thead><tbody>{rows}</tbody></table>"
                    )
            valid_html = "".join(valid_html_parts) if valid_html_parts else "<p>No findings.</p>"

            # ---- Section 5: Per-scenario time series (first 96 rows preview)
            preview_html_parts = []
            for k, r in results.items():
                df = pd.DataFrame({
                    "timestamp": r.timestamps[:96],
                    "t_in_c": r.t_in_c[:96],
                    "elec_kwh": r.elec_kwh[:96],
                    "q_sh_kwh": r.q_sh_kwh[:96],
                    "q_dhw_kwh": r.q_dhw_kwh[:96],
                    "e_dhw_kwh": r.e_dhw_kwh[:96],
                    "cop_sh": r.cop_sh[:96],
                    "da_volume_kwh": r.da_volume_kwh[:96],
                    "id_adjustment_kwh": r.id_adjustment_kwh[:96],
                })
                preview_html_parts.append(
                    f"<h3>{k} — {r.label}</h3>"
                    f"<p><em>First 96 steps (one day at 15-min). Full series in the CSV/Excel exports.</em></p>"
                    + df.to_html(index=False, float_format=lambda x: f"{x:.3f}")
                )

            # ---- Assemble
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>FlexHeat Optimiser — Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 3px solid #ff6900; padding-bottom: 0.3em; }}
  h2 {{ color: #ff6900; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  h3 {{ margin-top: 1.5em; }}
  table {{ border-collapse: collapse; margin: 1em 0; width: 100%; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  .kpi tbody tr:nth-child(odd) {{ background: #fafafa; }}
  .lvl-ok {{ color: #2e7d32; font-weight: 600; }}
  .lvl-info {{ color: #1565c0; font-weight: 600; }}
  .lvl-warning {{ color: #ef6c00; font-weight: 600; }}
  .lvl-error {{ color: #c62828; font-weight: 600; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  .interp {{ background: #fff3e0; border-left: 4px solid #ff6900; padding: 1em; }}
  footer {{ margin-top: 3em; color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>🔥 FlexHeat Optimiser — Results Report</h1>
<p><strong>Generated:</strong> {now}</p>

<h2>1. Executive interpretation</h2>
<div class="interp">{interp.replace(chr(10), "<br>") if interp else "No interpretation available."}</div>

<h2>2. Run configuration</h2>
<table><tbody>{cfg_rows}</tbody></table>

<h2>3. Archetype parameters</h2>
<table><tbody>{arch_rows}</tbody></table>

<h2>4. KPI table — all scenarios</h2>
{kpi_html}

<h2>5. Annualised value stack (€/year)</h2>
<table><tbody>{value_rows}</tbody></table>

<h2>6. Validation findings</h2>
{valid_html}

<h2>7. Per-scenario time series (preview)</h2>
{"".join(preview_html_parts)}

<footer>
FlexHeat Optimiser — prototype model. Wholesale prices in €/MWh; energies in kWh per time step;
temperatures in °C. Synthetic-data runs are illustrative only.
</footer>
</body>
</html>"""
            return html

        # ------------------------------------------------------------------
        # Helper: build a multi-sheet Excel workbook
        # ------------------------------------------------------------------
        def _build_excel_bundle() -> bytes:
            buf = io.BytesIO()
            try:
                writer = pd.ExcelWriter(buf, engine="openpyxl")
            except ModuleNotFoundError:
                # Fallback: use xlsxwriter if openpyxl is missing
                writer = pd.ExcelWriter(buf, engine="xlsxwriter")

            with writer:
                # KPI sheet
                kpi_df.to_excel(writer, sheet_name="KPI", index=False)

                # Value stack sheet
                value = st.session_state.get("value") or {}
                pd.DataFrame(
                    [{"component": k, "eur_per_year": v} for k, v in value.items()]
                ).to_excel(writer, sheet_name="Value_stack", index=False)

                # Run config sheet
                run_cfg = st.session_state.get("run_cfg") or {}
                pd.DataFrame(
                    [{"key": k, "value": str(v)} for k, v in run_cfg.items()]
                ).to_excel(writer, sheet_name="Run_config", index=False)

                # Archetype sheet
                arch_obj = st.session_state.get("arch")
                if arch_obj is not None:
                    pd.DataFrame(
                        [{"key": k, "value": v} for k, v in arch_obj.to_dict().items()]
                    ).to_excel(writer, sheet_name="Archetype", index=False)

                # Validation sheet
                validations = st.session_state.get("validations") or {}
                vrows = []
                for category, findings in validations.items():
                    for f in findings:
                        vrows.append({
                            "category": category, "level": f.level,
                            "code": f.code, "message": f.message,
                        })
                if vrows:
                    pd.DataFrame(vrows).to_excel(writer, sheet_name="Validation", index=False)

                # One sheet per scenario time series
                for k, r in results.items():
                    df = pd.DataFrame({
                        "timestamp": r.timestamps,
                        "t_in_c": r.t_in_c,
                        "q_sh_kwh": r.q_sh_kwh,
                        "q_dhw_kwh": r.q_dhw_kwh,
                        "e_dhw_kwh": r.e_dhw_kwh,
                        "elec_kwh": r.elec_kwh,
                        "elec_sh_kwh": r.elec_sh_kwh,
                        "elec_dhw_kwh": r.elec_dhw_kwh,
                        "cop_sh": r.cop_sh,
                        "cop_dhw": r.cop_dhw,
                        "da_volume_kwh": r.da_volume_kwh,
                        "id_volume_kwh": r.id_volume_kwh,
                        "id_adjustment_kwh": r.id_adjustment_kwh,
                    })
                    df.to_excel(writer, sheet_name=f"TS_{k}", index=False)

            return buf.getvalue()

        # ------------------------------------------------------------------
        # Helper: build a ZIP of all CSVs
        # ------------------------------------------------------------------
        def _build_csv_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # KPI
                zf.writestr("kpi_table.csv", kpi_df.to_csv(index=False))

                # Value stack
                value = st.session_state.get("value") or {}
                vdf = pd.DataFrame(
                    [{"component": k, "eur_per_year": v} for k, v in value.items()]
                )
                zf.writestr("value_stack.csv", vdf.to_csv(index=False))

                # Run config
                run_cfg = st.session_state.get("run_cfg") or {}
                zf.writestr("run_config.json", json.dumps(run_cfg, default=str, indent=2))

                # Archetype
                arch_obj = st.session_state.get("arch")
                if arch_obj is not None:
                    zf.writestr(
                        "archetype.json",
                        json.dumps(arch_obj.to_dict(), default=str, indent=2),
                    )

                # Validation
                validations = st.session_state.get("validations") or {}
                vrows = []
                for category, findings in validations.items():
                    for f in findings:
                        vrows.append({
                            "category": category, "level": f.level,
                            "code": f.code, "message": f.message,
                        })
                if vrows:
                    zf.writestr("validation.csv", pd.DataFrame(vrows).to_csv(index=False))

                # Per-scenario time series
                for k, r in results.items():
                    df = pd.DataFrame({
                        "timestamp": r.timestamps,
                        "t_in_c": r.t_in_c,
                        "q_sh_kwh": r.q_sh_kwh,
                        "q_dhw_kwh": r.q_dhw_kwh,
                        "e_dhw_kwh": r.e_dhw_kwh,
                        "elec_kwh": r.elec_kwh,
                        "elec_sh_kwh": r.elec_sh_kwh,
                        "elec_dhw_kwh": r.elec_dhw_kwh,
                        "cop_sh": r.cop_sh,
                        "cop_dhw": r.cop_dhw,
                        "da_volume_kwh": r.da_volume_kwh,
                        "id_volume_kwh": r.id_volume_kwh,
                        "id_adjustment_kwh": r.id_adjustment_kwh,
                    })
                    zf.writestr(f"timeseries_{k}.csv", df.to_csv(index=False))

                # HTML report inside the zip too
                zf.writestr("report.html", _build_html_report())

            return buf.getvalue()

        # ------------------------------------------------------------------
        # Render the three bundled download buttons up top
        # ------------------------------------------------------------------
        st.subheader("📦 Full report bundles")
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📄 HTML report (open in browser)",
                data=_build_html_report().encode("utf-8"),
                file_name=f"flexheat_report_{stamp}.html",
                mime="text/html",
                use_container_width=True,
                key="dl_html_report",
            )
            st.caption("Single self-contained HTML file — opens in any browser, no software needed.")
        with c2:
            st.download_button(
                "📊 Excel workbook (.xlsx)",
                data=_build_excel_bundle(),
                file_name=f"flexheat_report_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_xlsx",
            )
            st.caption("Multi-sheet workbook: KPIs, value stack, validation, every scenario.")
        with c3:
            st.download_button(
                "🗜️ Full ZIP (CSVs + HTML)",
                data=_build_csv_zip(),
                file_name=f"flexheat_report_{stamp}.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip",
            )
            st.caption("Everything packaged: CSVs for each scenario plus the HTML report.")

        st.markdown("---")
        st.subheader("Individual files")

        st.markdown("#### KPI table")
        st.download_button(
            "Download KPI CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name="flexheat_kpi.csv",
            mime="text/csv",
            key="dl_kpi_single",
        )

        st.markdown("#### Per-scenario time series")
        for k, r in results.items():
            df = pd.DataFrame({
                "timestamp": r.timestamps,
                "t_in_c": r.t_in_c,
                "q_sh_kwh": r.q_sh_kwh,
                "q_dhw_kwh": r.q_dhw_kwh,
                "e_dhw_kwh": r.e_dhw_kwh,
                "elec_kwh": r.elec_kwh,
                "elec_sh_kwh": r.elec_sh_kwh,
                "elec_dhw_kwh": r.elec_dhw_kwh,
                "cop_sh": r.cop_sh,
                "cop_dhw": r.cop_dhw,
                "da_volume_kwh": r.da_volume_kwh,
                "id_volume_kwh": r.id_volume_kwh,
                "id_adjustment_kwh": r.id_adjustment_kwh,
            })
            st.download_button(
                f"Download {k} time series ({r.label})",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"flexheat_{k}_timeseries.csv",
                mime="text/csv",
                key=f"dl_{k}",
            )

        st.markdown("#### Run configuration")
        st.download_button(
            "Download run config JSON",
            data=json.dumps(st.session_state["run_cfg"], default=str, indent=2),
            file_name="flexheat_run_config.json",
            mime="application/json",
            key="dl_runcfg",
        )

        st.markdown("---")
        st.caption(
            "These exports are intended for offline analysis. Wholesale prices are €/MWh; "
            "energies are kWh per time step; temperatures in °C."
        )


# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "FlexHeat Optimiser · prototype model for assessing residential heat-pump flex value. "
    "Synthetic-data results are illustrative only; replace with EPEX DA/ID and real weather + "
    "metering data for a production business case."
)
