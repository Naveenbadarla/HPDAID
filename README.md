# FlexHeat Optimiser

A residential heat-pump flexibility valuation model for the German market, packaged as a Streamlit
application. The tool quantifies the value of operating a heat pump against wholesale spot markets
(Day-Ahead + Intraday) while respecting thermal comfort and domestic-hot-water (DHW) needs, and
scales the result to a portfolio (E.ON-style aggregator view).

## What it answers

1. How much money does a household save (€/year) if the heat pump dispatches against EPEX
   Day-Ahead prices instead of a fixed-setpoint thermostat?
2. How much additional value is captured if the dispatch is re-optimised against the
   Intraday continuous market once shorter-horizon prices are known?
3. How much extra comes from widening the comfort band, vs. shifting DHW heating in time?
4. Aggregated across a portfolio of *N* households, what does this look like as a business
   opportunity for the aggregator (E.ON), and what share can be passed back to the customer as
   a FlexHeat bonus?

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Without any uploaded files it generates synthetic
German market data (prices, weather, DHW draws) automatically — useful for demos, but synthetic
DA/ID/weather data is **illustrative only** and must be replaced with real EPEX and DWD series for
a production business case.

## Scenarios

| ID  | Name                          | Description                                                                 |
|-----|-------------------------------|-----------------------------------------------------------------------------|
| S0  | Baseline thermostat           | Bang-bang controller around `T_target`; DHW priority. No market awareness.  |
| S1  | DA-only optimisation          | LP minimises DA cost. Volume is "locked" once DA closes.                    |
| S2  | DA + Intraday re-optimisation | Same DA commitment as S1, plus ID re-dispatch against actual ID prices.     |
| S3  | S2 + wider comfort band       | Same as S2 with a wider (e.g. 19.5–22.5 °C) comfort envelope.               |
| S4  | S2 + DHW flexibility          | Same as S2; the DHW tank is treated as a thermal battery (frac_min lowered). |

The value stack is computed by differencing wholesale costs:

```
DA value          = S0 − S1
ID incremental    = S1 − S2
Wider comfort     = S2 − S3
DHW flexibility   = S2 − S4
Total value       = S0 − min(S1..S4)
```

## DA / Intraday settlement — *not* `min(DA, ID)`

This is the single most-important modelling detail. A naive comparison of "DA cost vs ID cost"
uses `min(DA, ID)` and overstates value by 2–4× because it assumes perfect foresight of which
market clears cheaper for every quarter-hour.

The correct two-stage settlement is:

```
cost = DA_volume * DA_price   +   (actual_volume − DA_volume) * ID_price
```

The DA volume is a firm financial position locked at gate closure. Intraday re-optimisation moves
*physical* energy against the ID curve and the resulting imbalance is settled at ID. See
`hp_model/settlement.py` and the comments at the top of `hp_model/optimizer_id.py` for the full
derivation.

## Architecture

```
heatpump_flex_model/
├── app.py                  Streamlit UI — 9 tabs
├── requirements.txt
├── README.md
├── data/                   Sample CSVs (one-week extracts)
├── outputs/                Auto-created for run artefacts
└── hp_model/
    ├── __init__.py
    ├── config.py           All default constants + RunConfig dataclass
    ├── archetypes.py       4 building/HP archetypes
    ├── synthetic_data.py   Fallback price / weather / DHW generators
    ├── data_loader.py      CSV ingestion + MarketData container
    ├── cop_model.py        COP curves (underfloor vs radiator vs DHW)
    ├── thermal_model.py    Single-zone RC building dynamics
    ├── dhw_model.py        DHW tank state of charge dynamics
    ├── baseline_controller.py   S0 thermostat
    ├── optimizer_da.py     LP for DA-stage dispatch (also used for ID)
    ├── optimizer_id.py     Intraday re-optimisation wrapper
    ├── settlement.py       DA + ID two-stage settlement + retail stack
    ├── scenarios.py        Orchestration + KPI table + value stack
    ├── fleet.py            Portfolio scaling
    ├── validation.py       Sanity checks on every result
    └── plotting.py         All Plotly charts used by the UI
```

## Key modelling assumptions

- 15-minute time step (German MTU).
- Single-zone first-order RC thermal model. Inputs: outdoor temperature, internal gains, heat
  injection from the HP. State: indoor air temperature.
- COP modelled as a linear function of outdoor temperature, distinct curves for underfloor / radiator
  emitters and a third curve for DHW (higher supply temperature → lower COP).
- DHW tank: one-state energy bucket with constant standing loss per step; lower bound on
  state-of-charge enforced as a soft constraint to keep the LP feasible.
- LP formulation only (PuLP + CBC). HP modulation is continuous; min-modulation is not enforced
  as a binary in this version. The architecture is MILP-ready — see `optimizer_da.py`.
- Comfort and DHW violations are penalised in the objective with high cost (`€/kWh·K`), not
  forbidden. This avoids infeasibility and exposes "would have violated by X" diagnostics.
- Terminal constraints prevent end-of-horizon gaming (the LP cannot end the run with a cold tank
  and a cold house to lower cost).
- Retail tariff stack (€-ct/kWh): wholesale + markup + grid + taxes/levies, then VAT.
- Portfolio scaling is linear (homogeneous fleet). The heterogeneous-fleet helper in `fleet.py`
  supports a mix of archetypes if you compute per-archetype values separately.

## Limitations / what this model does **not** do

- No imbalance / control-energy revenue stack — only DA + ID arbitrage.
- No grid-fee dynamic component (e.g. §14a EnWG dynamic grid tariffs). Easy to add to the
  retail stack in `settlement.py`.
- No HP cycling / start-up losses. Continuous modulation only.
- No stochastic optimisation — perfect foresight within each stage. The DA→ID two-stage structure
  partly mitigates this; a true rolling-horizon stochastic model is future work.
- No degradation, no operational reserves, no V2G / EV interaction.
- Synthetic price data is *not* a substitute for real EPEX data — magnitudes are realistic but
  hour-by-hour shapes are stylised.

## Input data formats

If you upload CSVs (Market Data tab), they must have a `timestamp` column plus one of:

- `da_price_eur_per_mwh`
- `id_price_eur_per_mwh`
- `outdoor_temp_c`
- `dhw_draw_kwh`

Timestamps can be any pandas-parsable format. Gaps are forward-filled with a warning.

## Validation

Every run is checked across 5 categories — thermal, energy balance, optimisation health,
market settlement, and business sanity. Findings are surfaced in the **Validation** tab as
✅/ℹ️/⚠️/❌ and you should not present a number from this tool externally if there are any
unresolved ❌ errors.

## License & scope

Prototype model — not for trading decisions. Numbers are calibrated for plausibility, not
for engineering accuracy. Use as a structured business-case scoping tool.
