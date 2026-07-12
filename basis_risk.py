"""
Basis-risk analysis for parametric index insurance.

Basis risk is the gap between the index-triggered payout and the farmer's actual
loss. It is the single biggest threat to smallholder value and to product
credibility. We quantify it four ways:

  * correlation of payout vs actual loss (higher = tighter product)
  * downside basis risk: seasons where the farmer had a real loss but the index
    paid too little (false negatives) -- the reputational killer
  * upside basis risk: seasons where the index paid but there was little loss
    (false positives) -- the cost-efficiency drain
  * a single Basis Risk Ratio (mean absolute gap / mean actual loss)

All computed on realised or simulated season pairs, no external stats deps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BasisRiskReport:
    correlation: float
    downside_frequency: float      # P(actual loss materially unpaid)
    upside_frequency: float        # P(paid with little actual loss)
    basis_risk_ratio: float        # mean |payout-loss| / mean loss
    n_seasons: int

    def as_dict(self) -> dict:
        return {
            "correlation": round(self.correlation, 3),
            "downside_frequency": round(self.downside_frequency, 3),
            "upside_frequency": round(self.upside_frequency, 3),
            "basis_risk_ratio": round(self.basis_risk_ratio, 3),
            "n_seasons": self.n_seasons,
        }


def analyse(
    payouts: np.ndarray,
    actual_losses: np.ndarray,
    material_threshold: float = 0.10,
) -> BasisRiskReport:
    """Compare payouts against actual losses (same currency units).

    material_threshold : loss/payout below this fraction of TSI is 'immaterial'.
    """
    payouts = np.asarray(payouts, dtype=float)
    actual_losses = np.asarray(actual_losses, dtype=float)
    n = len(payouts)
    scale = max(actual_losses.max(), payouts.max(), 1e-9)
    thr = material_threshold * scale

    if payouts.std() < 1e-9 or actual_losses.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(payouts, actual_losses)[0, 1])

    real_loss = actual_losses > thr
    underpaid = payouts < (actual_losses - thr)
    downside = float(np.mean(real_loss & underpaid))

    overpaid = (payouts > thr) & (actual_losses <= thr)
    upside = float(np.mean(overpaid))

    brr = float(np.mean(np.abs(payouts - actual_losses)) / max(actual_losses.mean(), 1e-9))

    return BasisRiskReport(
        correlation=corr,
        downside_frequency=downside,
        upside_frequency=upside,
        basis_risk_ratio=brr,
        n_seasons=n,
    )
