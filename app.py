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
        floor_area = st.number_input("Floor area (m²)", 30.0, 500.0, base.floor_area_m2, 5.0)
        annual_heat = st.number_input(
            "Annual SH demand (kWh)", 1_000.0, 50_000.0, base.annual_heat_demand_kwh, 100.0
        )
        annual_dhw = st.number_input(
            "Annual DHW demand (kWh)", 500.0, 8_000.0, base.annual_dhw_demand_kwh, 50.0
        )
        ua = st.number_input(
            "UA value (kW/K)", 0.05, 1.0, base.ua_kw_per_k, 0.01, format="%.3f"
        )
        c_th = st.number_input(
            "Thermal capacitance (kWh/K)", 5.0, 100.0, base.c_th_kwh_per_k, 1.0
        )
    with col2:
        st.markdown("**Heat pump**")
        hp_thermal = st.number_input("HP thermal capacity (kW)", 2.0, 30.0, base.hp_thermal_kw, 0.5)
        hp_elec = st.number_input(
            "HP electrical capacity (kW)", 1.0, 15.0, base.hp_electrical_kw, 0.5
        )
        hp_min_mod = st.number_input(
            "Min modulation (0-1)", 0.0, 1.0, base.hp_min_modulation, 0.05
        )
        emitter = st.selectbox(
            "Emitter type", ["underfloor", "radiator"],
            index=["underfloor", "radiator"].index(base.emitter),
        )
    with col3:
        st.markdown("**DHW tank**")
        dhw_kwh = st.number_input("Tank usable energy (kWh)", 2.0, 30.0, base.dhw_tank_kwh, 0.5)
        dhw_loss = st.number_input(
            "Standing loss (kWh/step)", 0.0, 0.5, base.dhw_tank_loss_kwh_per_step, 0.005,
            format="%.4f",
        )
        st.markdown("**Comfort defaults**")
        t_target = st.number_input("Target T_in (°C)", 18.0, 24.0, base.t_target, 0.5)

    arch = A.Archetype(
        name=base.name + ("_custom" if base.name == "custom" else ""),
        label=base.label,
        floor_area_m2=floor_area,
        annual_heat_demand_kwh=annual_heat,
        annual_dhw_demand_kwh=annual_dhw,
        ua_kw_per_k=ua,
        c_th_kwh_per_k=c_th,
        hp_thermal_kw=hp_thermal,
        hp_electrical_kw=hp_elec,
        hp_min_modulation=hp_min_mod,
        emitter=emitter,
        dhw_tank_kwh=dhw_kwh,
        dhw_tank_loss_kwh_per_step=dhw_loss,
        t_target=t_target,
        t_min=t_min,
        t_max=t_max,
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
        st.info("Click **Load market data** above, or hit **Run optimisation** in the sidebar — synthetic data will be generated automatically.")


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
            comfort_band=(t_min, t_max),
            comfort_band_flex=(t_min_flex, t_max_flex),
            scenarios=selected_scenarios,
        )
        kpi_df = sc.kpi_table(results, horizon_days=horizon_days)
        value = sc.value_stack(results, horizon_days=horizon_days)
        interp = sc.build_interpretation(
            results, value,
            customer_share=C.PORTFOLIO_DEFAULTS["customer_share"],
            horizon_days=horizon_days,
        )
        validations = val.run_all_validations(results, arch, horizon_days)

        st.session_state["results"] = results
        st.session_state["kpi_df"] = kpi_df
        st.session_state["value"] = value
        st.session_state["interpretation"] = interp
        st.session_state["validations"] = validations
        st.session_state["run_cfg"] = {
            "horizon_days": horizon_days,
            "timestep_h": timestep_h,
            "n_steps": n_steps,
            "comfort_band": (t_min, t_max),
            "comfort_band_flex": (t_min_flex, t_max_flex),
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
        baseline_retail = float(kpi_df.loc[kpi_df.scenario == "S0", "retail_cost_eur_year"].iloc[0]) if "S0" in kpi_df.scenario.values else float("nan")
        best_row = kpi_df.loc[kpi_df.scenario != "S0"].sort_values("wholesale_cost_eur_year").head(1)
        if not best_row.empty:
            best = best_row.iloc[0]
            saving_eur = best["wholesale_saving_eur_year"]
            saving_retail = best["retail_saving_eur_year"]
            saving_pct = best["saving_pct_retail"]
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
        st.plotly_chart(pl.plot_waterfall(kpi_df, value), use_container_width=True)

        st.markdown("### Value stack — where the savings come from")
        st.plotly_chart(pl.plot_value_stack(value), use_container_width=True)


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
        st.plotly_chart(pl.plot_hp_load(results), use_container_width=True)

        st.markdown("#### Cost comparison")
        mode = st.radio("View costs as", ["wholesale", "retail"], horizontal=True, index=0)
        st.plotly_chart(pl.plot_cost_bars(kpi_df, mode=mode), use_container_width=True)

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
        comfort = (t_min, t_max)
        if scen_for_temp:
            sub = {k: results[k] for k in scen_for_temp}
            st.plotly_chart(
                pl.plot_indoor_temperature(sub, md.outdoor_temp_c, comfort),
                use_container_width=True,
            )

        st.markdown("#### DHW tank state of charge")
        if scen_for_temp:
            st.plotly_chart(
                pl.plot_dhw_tank({k: results[k] for k in scen_for_temp}, arch.dhw_tank_kwh),
                use_container_width=True,
            )

        st.markdown("#### Single-day dispatch view")
        day_choice = st.selectbox(
            "Day to inspect", list(range(1, horizon_days + 1)), index=0, key="day_pick"
        )
        steps_per_day = int(round(24 / timestep_h))
        i0 = (day_choice - 1) * steps_per_day
        i1 = i0 + steps_per_day
        scen_for_dispatch = st.selectbox(
            "Scenario", list(results.keys()),
            index=min(2, len(results) - 1),
            key="dispatch_scen",
        )
        st.plotly_chart(
            pl.plot_daily_dispatch(
                results[scen_for_dispatch],
                md.da_price_eur_mwh,
                md.id_price_eur_mwh,
                start_idx=i0, end_idx=i1,
            ),
            use_container_width=True,
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
        st.plotly_chart(pl.plot_value_stack(value), use_container_width=True)

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
                adoption_rate=adoption,
                customer_share=cust_share,
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
                arch_hp_electrical_kw=arch.hp_electrical_kw,
                customer_share=cust_share,
            )
            st.session_state["scaling_df"] = scaling_df
            st.plotly_chart(pl.plot_fleet_scaling(scaling_df), use_container_width=True)


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
            with st.expander(f"{key} — {len(findings)} finding(s)", expanded=(n_err > 0 and key != "business")):
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
        st.markdown("#### KPI table")
        st.download_button(
            "Download KPI CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name="flexheat_kpi.csv",
            mime="text/csv",
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
