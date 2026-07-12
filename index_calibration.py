"""
Index calibration for parametric weather-index crop insurance.

Given historical weather (e.g. cumulative growing-season rainfall) and a proxy
for realised farm loss, calibrate a piecewise-linear payout function:

    payout(index) = 0                       if index >= trigger
                  = TSI * (trigger-index)/(trigger-exit)   if exit < index < trigger
                  = TSI                      if index <= exit

where TSI = total sum insured, `trigger` is the strike (payouts begin), and
`exit` is the point of maximum payout. Calibration picks (trigger, exit) that
maximise the fit between modelled payouts and observed losses on the history.

No external calibration libraries: a transparent grid search over candidate
(trigger, exit) pairs, scored by RMSE against observed loss ratios. Transparent
by design because regulators and reinsurers must be able to audit the triggers.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class IndexContract:
    trigger: float      # index level where payouts begin (the strike)
    exit: float         # index level of maximum payout (full TSI)
    tsi: float          # total sum insured (currency per unit insured)

    def payout(self, index: np.ndarray | float) -> np.ndarray | float:
        index = np.asarray(index, dtype=float)
        span = self.trigger - self.exit
        if span <= 0:
            raise ValueError("trigger must be strictly greater than exit")
        frac = np.clip((self.trigger - index) / span, 0.0, 1.0)
        return frac * self.tsi


def calibrate(
    index_history: np.ndarray,
    observed_loss_ratio: np.ndarray,
    tsi: float,
    n_grid: int = 40,
) -> tuple[IndexContract, float]:
    """Grid-search (trigger, exit) minimising RMSE vs observed loss ratios.

    index_history        : weather index per season (e.g. mm cumulative rainfall)
    observed_loss_ratio  : realised loss as a fraction of TSI in [0, 1] per season
    Returns the fitted contract and the achieved RMSE (in loss-ratio units).
    """
    index_history = np.asarray(index_history, dtype=float)
    observed_loss_ratio = np.asarray(observed_loss_ratio, dtype=float)
    lo, hi = np.percentile(index_history, [5, 95])
    candidates = np.linspace(lo, hi, n_grid)

    best: tuple[IndexContract, float] | None = None
    for trigger in candidates:
        for exit_ in candidates:
            if exit_ >= trigger:
                continue
            contract = IndexContract(trigger=trigger, exit=exit_, tsi=tsi)
            modelled_ratio = contract.payout(index_history) / tsi
            rmse = float(np.sqrt(np.mean((modelled_ratio - observed_loss_ratio) ** 2)))
            if best is None or rmse < best[1]:
                best = (contract, rmse)

    assert best is not None
    return best
