"""
Actuarial premium structuring for a weather-index contract.

Pure Monte-Carlo pricing built from the calibrated payout function:

    pure_premium      = E[payout]                       (expected loss cost)
    risk_load         = k * std(payout)                 (volatility loading)
    commercial_premium = (pure + risk_load) / (1 - expense_ratio)
    rate_on_line      = commercial_premium / TSI

The weather index is resampled either by bootstrapping the historical series or
from a fitted lognormal (rainfall-like, non-negative, right-skewed). We report
the full payout distribution so the reinsurer sees the tail (VaR / TVaR), not
just the mean.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from index_calibration import IndexContract


@dataclass
class PremiumQuote:
    pure_premium: float
    risk_load: float
    commercial_premium: float
    rate_on_line: float
    var_99: float          # 99% payout Value-at-Risk
    tvar_99: float         # tail conditional expectation beyond VaR99

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


def _simulate_index(index_history: np.ndarray, n: int, rng: np.random.Generator,
                    method: str = "bootstrap") -> np.ndarray:
    index_history = np.asarray(index_history, dtype=float)
    if method == "bootstrap":
        return rng.choice(index_history, size=n, replace=True)
    if method == "lognormal":
        pos = np.clip(index_history, 1e-6, None)
        mu, sigma = np.mean(np.log(pos)), np.std(np.log(pos))
        return rng.lognormal(mean=mu, sigma=sigma, size=n)
    raise ValueError(f"unknown method {method!r}")


def price(
    contract: IndexContract,
    index_history: np.ndarray,
    expense_ratio: float = 0.25,
    risk_load_k: float = 0.30,
    n_sims: int = 100_000,
    method: str = "bootstrap",
    seed: int = 7,
) -> PremiumQuote:
    rng = np.random.default_rng(seed)
    sims = _simulate_index(index_history, n_sims, rng, method=method)
    payouts = np.asarray(contract.payout(sims), dtype=float)

    pure = float(payouts.mean())
    load = float(risk_load_k * payouts.std())
    commercial = (pure + load) / (1.0 - expense_ratio)
    var99 = float(np.percentile(payouts, 99))
    tail = payouts[payouts >= var99]
    tvar99 = float(tail.mean()) if tail.size else var99

    return PremiumQuote(
        pure_premium=pure,
        risk_load=load,
        commercial_premium=commercial,
        rate_on_line=commercial / contract.tsi,
        var_99=var99,
        tvar_99=tvar99,
    )
