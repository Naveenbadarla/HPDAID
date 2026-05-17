"""
settlement.py — Wholesale and retail settlement.

WHOLESALE
---------
For S1 (DA-only):
    cost = Σ actual_t * da_price_t / 1000

For S2..S4 (DA + ID re-optimisation):
    cost_t = da_volume_t * da_price_t
           + (actual_t - da_volume_t) * id_price_t      (all per kWh, /1000 for €)

`da_volume_t` is the DA-stage scheduled volume (in kWh).
`actual_t` is the post-ID-stage delivered volume (in kWh).

If actual < da_volume:  ID adjustment is negative -> household effectively
"sells back" surplus DA volume at the ID price.  If ID price < DA price,
that creates value.  If ID price > DA price, it's a loss.

We never replace DA with min(DA, ID).  That would be wrong because the DA
commitment cannot be unwound except via the ID market at the ID price.

OPTIONAL IMBALANCE
------------------
If a small mismatch (forecast error) is kept between the ID-stage schedule
and the physical metered consumption, the residual is charged at an
imbalance price.  In the MVP we set this residual to zero because the LP
is perfect-foresight.

RETAIL
------
For the customer view, we convert wholesale €/MWh to retail ct/kWh:

    retail_ct_kwh = wholesale_ct_kwh + markup + grid_fee + taxes_levies
    retail_inc_vat = retail * (1 + VAT)
    retail €/kWh   = retail_inc_vat / 100
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import config as C


@dataclass
class SettlementResult:
    da_volume_kwh: np.ndarray
    id_volume_kwh: np.ndarray          # actual physical delivered
    id_adjustment_kwh: np.ndarray      # = id_volume - da_volume
    da_cost_eur: float                 # Σ da_volume * da_price / 1000
    id_adjustment_cost_eur: float      # Σ (id_volume - da_volume) * id_price / 1000
    total_wholesale_cost_eur: float
    da_only_cost_eur: float            # Σ da_volume * da_price / 1000 (S1 cost)
    pure_id_cost_eur: float            # Σ id_volume * id_price / 1000 (for diagnostics)


def settle_da_only(da_volume_kwh: np.ndarray, da_price_eur_mwh: np.ndarray) -> SettlementResult:
    """Settlement for S1 (no ID stage)."""
    da_cost = float(np.sum(da_volume_kwh * da_price_eur_mwh) / 1000.0)
    zeros = np.zeros_like(da_volume_kwh)
    return SettlementResult(
        da_volume_kwh=da_volume_kwh,
        id_volume_kwh=da_volume_kwh.copy(),
        id_adjustment_kwh=zeros,
        da_cost_eur=da_cost,
        id_adjustment_cost_eur=0.0,
        total_wholesale_cost_eur=da_cost,
        da_only_cost_eur=da_cost,
        pure_id_cost_eur=da_cost,
    )


def settle_da_id(
    da_volume_kwh: np.ndarray,
    id_volume_kwh: np.ndarray,
    da_price_eur_mwh: np.ndarray,
    id_price_eur_mwh: np.ndarray,
) -> SettlementResult:
    """Proper DA + ID two-stage settlement."""
    da_cost = float(np.sum(da_volume_kwh * da_price_eur_mwh) / 1000.0)
    id_adj = id_volume_kwh - da_volume_kwh
    id_adj_cost = float(np.sum(id_adj * id_price_eur_mwh) / 1000.0)
    total = da_cost + id_adj_cost

    pure_id = float(np.sum(id_volume_kwh * id_price_eur_mwh) / 1000.0)
    return SettlementResult(
        da_volume_kwh=da_volume_kwh,
        id_volume_kwh=id_volume_kwh,
        id_adjustment_kwh=id_adj,
        da_cost_eur=da_cost,
        id_adjustment_cost_eur=id_adj_cost,
        total_wholesale_cost_eur=total,
        da_only_cost_eur=da_cost,
        pure_id_cost_eur=pure_id,
    )


# --------------------------------------------------------------------------
# Retail conversion
# --------------------------------------------------------------------------
def wholesale_to_retail_ct_kwh(
    wholesale_eur_mwh: np.ndarray,
    markup_ct_kwh: float = C.RETAIL_MARKUP_CT_KWH,
    grid_fee_ct_kwh: float = C.GRID_FEE_CT_KWH,
    taxes_levies_ct_kwh: float = C.TAXES_LEVIES_CT_KWH,
    vat_rate: float = C.VAT_RATE,
) -> np.ndarray:
    """Convert wholesale €/MWh to consumer retail ct/kWh (incl. VAT)."""
    wholesale_ct_kwh = np.asarray(wholesale_eur_mwh) * 0.1     # €/MWh -> ct/kWh
    pre_vat = wholesale_ct_kwh + markup_ct_kwh + grid_fee_ct_kwh + taxes_levies_ct_kwh
    return pre_vat * (1.0 + vat_rate)


def retail_cost_eur(
    consumption_kwh: np.ndarray,
    price_eur_mwh: np.ndarray,
    **retail_kwargs,
) -> float:
    """Customer bill in € given consumption (kWh) and a wholesale price track."""
    retail_ct_kwh = wholesale_to_retail_ct_kwh(price_eur_mwh, **retail_kwargs)
    return float(np.sum(consumption_kwh * retail_ct_kwh) / 100.0)
