"""
Guarantee-fund / first-loss mechanism sizing.

A first-loss guarantee fund absorbs the first tranche of aggregate claims before
a reinsurer is tapped, which is what lets a young index-insurance scheme reach
smallholders at an affordable premium. The design questions are:

  * how large must the fund be to survive a bad year at a target confidence?
  * what leverage ratio (portfolio TSI / fund capital) is prudent?
  * what is the probability of ruin (fund exhausted) over the horizon?

We answer with Monte-Carlo on the aggregate portfolio payout distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from index_calibration import IndexContract
from premium import _simulate_index


@dataclass
class FundDesign:
    fund_capital: float          # recommended first-loss capital
    leverage_ratio: float        # total portfolio TSI / fund_capital
    ruin_probability: float      # P(aggregate claims > fund) per season
    expected_recovery_years: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "fund_capital": round(self.fund_capital, 0),
            "leverage_ratio": round(self.leverage_ratio, 1),
            "ruin_probability": round(self.ruin_probability, 4),
            "expected_recovery_years": round(self.expected_recovery_years, 1),
            "confidence": self.confidence,
        }


def size_fund(
    contract: IndexContract,
    index_history: np.ndarray,
    n_policies: int,
    confidence: float = 0.99,
    n_sims: int = 50_000,
    seed: int = 11,
) -> FundDesign:
    """Size a first-loss fund for a portfolio of `n_policies` identical contracts.

    Correlated weather is modelled by drawing ONE index per season for the whole
    book (systemic risk), which is the realistic and conservative assumption for
    area-yield / rainfall index products in a single agro-climatic zone.
    """
    rng = np.random.default_rng(seed)
    season_index = _simulate_index(index_history, n_sims, rng, method="bootstrap")
    per_policy_payout = np.asarray(contract.payout(season_index), dtype=float)
    aggregate = per_policy_payout * n_policies

    fund = float(np.percentile(aggregate, confidence * 100))
    portfolio_tsi = contract.tsi * n_policies
    leverage = portfolio_tsi / max(fund, 1e-9)
    ruin = float(np.mean(aggregate > fund))
    exp_recovery = 1.0 / ruin if ruin > 0 else float("inf")

    return FundDesign(
        fund_capital=fund,
        leverage_ratio=leverage,
        ruin_probability=ruin,
        expected_recovery_years=exp_recovery,
        confidence=confidence,
    )
