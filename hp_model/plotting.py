"""
plotting.py — Plotly figures for the Streamlit UI.

Every function returns a `plotly.graph_objects.Figure`; the app just
embeds them.  We use a consistent colour scheme so the eye can map
quickly across charts.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .scenarios import ScenarioResult


# --------------------------------------------------------------------------
# Colour palette
# --------------------------------------------------------------------------
COLORS = {
    "S0": "#7d7d7d",       # grey — baseline
    "S1": "#1f77b4",       # blue — DA
    "S2": "#2ca02c",       # green — DA+ID
    "S3": "#ff7f0e",       # orange — wider comfort
    "S4": "#9467bd",       # purple — DHW flex
    "da_price": "#1f77b4",
    "id_price": "#ff9933",
    "indoor": "#d62728",
    "outdoor": "#17becf",
    "comfort_band": "rgba(214,39,40,0.10)",
    "dhw": "#9467bd",
    "value": "#2ca02c",
}


def _common_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title, height=height, hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --------------------------------------------------------------------------
# 1. Prices
# --------------------------------------------------------------------------
def plot_prices(ts: pd.DatetimeIndex, da: np.ndarray, idp: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=da, name="DA price",
                              line=dict(color=COLORS["da_price"], width=2)))
    fig.add_trace(go.Scatter(x=ts, y=idp, name="ID price",
                              line=dict(color=COLORS["id_price"], width=1, dash="dot")))
    fig.add_hline(y=0, line=dict(color="black", width=0.7))
    fig.update_yaxes(title="€ / MWh")
    return _common_layout(fig, "Day-Ahead and Intraday prices")


# --------------------------------------------------------------------------
# 2. HP load (baseline vs optimised)
# --------------------------------------------------------------------------
def plot_hp_load(results: Dict[str, ScenarioResult], focus: List[str] = None) -> go.Figure:
    focus = focus or ["S0", "S2", "S4"]
    fig = go.Figure()
    for k in focus:
        if k not in results:
            continue
        r = results[k]
        fig.add_trace(go.Scatter(
            x=r.timestamps, y=r.elec_kwh, name=f"{k} {r.label}",
            line=dict(color=COLORS.get(k, "#000"), width=1.5),
        ))
    fig.update_yaxes(title="HP electricity per step (kWh)")
    return _common_layout(fig, "Heat-pump electricity profile")


# --------------------------------------------------------------------------
# 3. Indoor temperature with comfort band
# --------------------------------------------------------------------------
def plot_indoor_temperature(
    results: Dict[str, ScenarioResult],
    comfort_band: tuple[float, float],
    outdoor: Optional[np.ndarray] = None,
    focus: List[str] = None,
) -> go.Figure:
    focus = focus or ["S0", "S2"]
    fig = go.Figure()
    ts = results[next(iter(results))].timestamps
    t_min, t_max = comfort_band
    # comfort band shading
    fig.add_trace(go.Scatter(
        x=list(ts) + list(ts)[::-1],
        y=[t_max] * len(ts) + [t_min] * len(ts),
        fill="toself", fillcolor=COLORS["comfort_band"],
        line=dict(color="rgba(0,0,0,0)"), name="Comfort band", hoverinfo="skip",
    ))
    if outdoor is not None:
        fig.add_trace(go.Scatter(
            x=ts, y=outdoor, name="Outdoor T",
            line=dict(color=COLORS["outdoor"], width=1, dash="dot"), yaxis="y2"
        ))
    for k in focus:
        if k not in results:
            continue
        r = results[k]
        fig.add_trace(go.Scatter(
            x=r.timestamps, y=r.t_in_c, name=f"{k} indoor T",
            line=dict(color=COLORS.get(k, "#000"), width=1.5),
        ))
    fig.update_yaxes(title="Indoor temperature (°C)")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Outdoor T (°C)"))
    return _common_layout(fig, "Indoor temperature vs comfort band")


# --------------------------------------------------------------------------
# 4. DHW tank
# --------------------------------------------------------------------------
def plot_dhw_tank(
    results: Dict[str, ScenarioResult],
    e_min: float, e_max: float,
    focus: List[str] = None,
) -> go.Figure:
    focus = focus or ["S0", "S4"]
    fig = go.Figure()
    ts = results[next(iter(results))].timestamps
    fig.add_hline(y=e_min, line=dict(color="red", dash="dot"), annotation_text="Min")
    fig.add_hline(y=e_max, line=dict(color="green", dash="dot"), annotation_text="Max")
    for k in focus:
        if k not in results:
            continue
        r = results[k]
        fig.add_trace(go.Scatter(
            x=r.timestamps, y=r.e_dhw_kwh, name=f"{k} DHW tank",
            line=dict(color=COLORS.get(k, "#000"), width=1.5),
        ))
    fig.update_yaxes(title="DHW tank energy (kWh)")
    return _common_layout(fig, "DHW tank state")


# --------------------------------------------------------------------------
# 5. Cost comparison bar chart
# --------------------------------------------------------------------------
def plot_cost_bars(kpi_df: pd.DataFrame, mode: str = "wholesale") -> go.Figure:
    fig = go.Figure()
    col = "wholesale_cost_eur_year" if mode == "wholesale" else "retail_cost_eur_year"
    title_mode = "Wholesale" if mode == "wholesale" else "Retail"
    fig.add_trace(go.Bar(
        x=kpi_df["scenario"], y=kpi_df[col],
        marker_color=[COLORS.get(s, "#666") for s in kpi_df["scenario"]],
        text=[f"€{v:,.0f}" for v in kpi_df[col]], textposition="outside",
    ))
    fig.update_yaxes(title=f"{title_mode} cost (€/year)")
    return _common_layout(fig, f"Annualised {title_mode.lower()} cost by scenario")


# --------------------------------------------------------------------------
# 6. Waterfall
# --------------------------------------------------------------------------
def plot_waterfall(kpi_df: pd.DataFrame, value: Dict[str, float]) -> go.Figure:
    if "S0" not in kpi_df["scenario"].values:
        return go.Figure()
    baseline = float(kpi_df.loc[kpi_df["scenario"] == "S0", "wholesale_cost_eur_year"].iloc[0])
    labels = ["Baseline cost"]
    values_l = [baseline]
    measures = ["absolute"]

    for k, lab in [
        ("da_value_eur_year", "DA shifting"),
        ("id_incremental_eur_year", "ID re-opt"),
        ("dhw_flex_eur_year", "DHW flex"),
        ("wider_comfort_eur_year", "Wider comfort"),
    ]:
        if k in value:
            labels.append(lab)
            values_l.append(-value[k])
            measures.append("relative")
    labels.append("Final optimised cost")
    values_l.append(0)
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        x=labels, y=values_l, measure=measures,
        text=[f"€{v:,.0f}" for v in values_l],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#d62728"}},
        decreasing={"marker": {"color": "#2ca02c"}},
        totals={"marker": {"color": "#7f7f7f"}},
    ))
    fig.update_yaxes(title="€ / year")
    return _common_layout(fig, "Wholesale cost waterfall (annualised)")


# --------------------------------------------------------------------------
# 7. Value stack
# --------------------------------------------------------------------------
def plot_value_stack(value: Dict[str, float]) -> go.Figure:
    pretty = {
        "da_value_eur_year": "DA shifting",
        "id_incremental_eur_year": "ID re-optimisation",
        "dhw_flex_eur_year": "DHW flexibility",
        "wider_comfort_eur_year": "Wider comfort band",
    }
    rows = [(pretty[k], v) for k, v in value.items() if k in pretty and v > 0]
    if not rows:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
        marker_color=["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"][: len(rows)],
        text=[f"€{r[1]:,.0f}" for r in rows], textposition="outside",
    ))
    fig.update_xaxes(title="€ / year")
    return _common_layout(fig, "Value stack — components of total saving")


# --------------------------------------------------------------------------
# 8. Fleet scaling
# --------------------------------------------------------------------------
def plot_fleet_scaling(scaling_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=scaling_df["n_households"], y=scaling_df["agg_annual_value_eur"],
        name="Aggregated value (€)", line=dict(color="#2ca02c", width=3),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=scaling_df["n_households"], y=scaling_df["flex_mw_capacity"],
        name="Flexible MW capacity", line=dict(color="#1f77b4", width=2, dash="dot"),
    ), secondary_y=True)
    fig.update_xaxes(title="Number of households")
    fig.update_yaxes(title="€ / year", secondary_y=False)
    fig.update_yaxes(title="MW (HP electrical)", secondary_y=True)
    return _common_layout(fig, "Fleet scaling: aggregated value vs flex capacity")


# --------------------------------------------------------------------------
# 9. Heatmap (sensitivity)
# --------------------------------------------------------------------------
def plot_sensitivity_heatmap(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str
) -> go.Figure:
    piv = df.pivot(index=y_col, columns=x_col, values=z_col)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index,
        colorscale="Viridis", colorbar=dict(title=z_col),
        text=np.round(piv.values, 1), texttemplate="%{text}",
    ))
    fig.update_xaxes(title=x_col)
    fig.update_yaxes(title=y_col)
    return _common_layout(fig, title)


# --------------------------------------------------------------------------
# 10. Daily dispatch — multi-panel
# --------------------------------------------------------------------------
def plot_daily_dispatch(
    md_ts: pd.DatetimeIndex, da: np.ndarray, idp: np.ndarray,
    result: ScenarioResult, day: pd.Timestamp, e_min: float, e_max: float,
    comfort_band: tuple[float, float],
) -> go.Figure:
    day_start = pd.Timestamp(day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    mask = (md_ts >= day_start) & (md_ts < day_end)
    if not mask.any():
        return go.Figure()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=("Prices", "HP electricity",
                                         "Indoor temperature", "DHW tank"))
    # Prices
    fig.add_trace(go.Scatter(x=md_ts[mask], y=da[mask], name="DA",
                              line=dict(color=COLORS["da_price"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=md_ts[mask], y=idp[mask], name="ID",
                              line=dict(color=COLORS["id_price"], dash="dot")), row=1, col=1)
    # HP electricity
    fig.add_trace(go.Bar(x=result.timestamps[mask], y=result.elec_kwh[mask],
                          name="HP elec", marker_color="#1f77b4"), row=2, col=1)
    # Indoor T
    fig.add_hrect(y0=comfort_band[0], y1=comfort_band[1],
                  fillcolor=COLORS["comfort_band"], line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(x=result.timestamps[mask], y=result.t_in_c[mask],
                              name="Indoor T", line=dict(color=COLORS["indoor"])),
                   row=3, col=1)
    # DHW
    fig.add_hline(y=e_min, line=dict(color="red", dash="dot"), row=4, col=1)
    fig.add_hline(y=e_max, line=dict(color="green", dash="dot"), row=4, col=1)
    fig.add_trace(go.Scatter(x=result.timestamps[mask], y=result.e_dhw_kwh[mask],
                              name="DHW", line=dict(color=COLORS["dhw"])),
                   row=4, col=1)
    fig.update_yaxes(title="€/MWh", row=1, col=1)
    fig.update_yaxes(title="kWh", row=2, col=1)
    fig.update_yaxes(title="°C", row=3, col=1)
    fig.update_yaxes(title="kWh", row=4, col=1)
    fig.update_layout(height=900, showlegend=True,
                      title=f"Daily dispatch — {result.name} {result.label} — {day_start.date()}")
    return fig


# --------------------------------------------------------------------------
# Helper for archetype × saving heatmap (used in Tab 5)
# --------------------------------------------------------------------------
def archetype_sensitivity_df(per_arch_value: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "archetype": list(per_arch_value.keys()),
        "saving_eur_year": list(per_arch_value.values()),
    })
