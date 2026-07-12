"""
End-to-end demo: rainfall-index drought cover for a smallholder rice zone.

Runnable with only numpy. Generates a synthetic-but-realistic 30-season history
(cumulative growing-window rainfall vs realised loss), then runs the full
quantitative pipeline an insurance-product lead would need from a quant:

    1. calibrate the payout triggers to the loss history
    2. quantify basis risk (the smallholder-value question)
    3. price the actuarial premium with tail metrics
    4. size the first-loss guarantee fund

Numbers below are illustrative (synthetic data), not a real Lao product.
"""
from __future__ import annotations

import numpy as np

from index_calibration import calibrate, IndexContract
from basis_risk import analyse
from premium import price
from guarantee_fund import size_fund

TSI = 1_000.0          # sum insured per policy (USD)
N_POLICIES = 5_000     # cooperative-distributed book


def synthetic_history(n_seasons: int = 30, seed: int = 42):
    """Rainfall (mm) over the growing window and realised loss ratio.

    Loss is driven by drought (low rainfall) with noise, so the index is
    informative but imperfect -- exactly the setting where basis risk matters.
    """
    rng = np.random.default_rng(seed)
    rainfall = rng.normal(650, 120, n_seasons).clip(200, 1100)
    # drought loss: kicks in below ~550mm, saturates near 350mm, plus idiosyncratic noise
    drought = np.clip((580 - rainfall) / 230, 0, 1)
    loss_ratio = np.clip(drought + rng.normal(0, 0.08, n_seasons), 0, 1)
    return rainfall, loss_ratio


def main() -> None:
    rainfall, loss_ratio = synthetic_history()
    actual_losses = loss_ratio * TSI

    print("=" * 64)
    print("PARAMETRIC RAINFALL-INDEX DROUGHT COVER  (illustrative demo)")
    print("=" * 64)
    print(f"Seasons: {len(rainfall)} | TSI/policy: ${TSI:,.0f} | Book: {N_POLICIES:,} policies\n")

    # 1. calibrate
    contract, rmse = calibrate(rainfall, loss_ratio, tsi=TSI)
    print("[1] CALIBRATED TRIGGERS")
    print(f"    trigger (strike): {contract.trigger:6.1f} mm  -> payouts begin")
    print(f"    exit  (full pay): {contract.exit:6.1f} mm  -> full TSI")
    print(f"    fit RMSE (loss-ratio units): {rmse:.3f}\n")

    # 2. basis risk
    modelled_payouts = np.asarray(contract.payout(rainfall), dtype=float)
    br = analyse(modelled_payouts, actual_losses)
    print("[2] BASIS RISK")
    for k, v in br.as_dict().items():
        print(f"    {k:22s}: {v}")
    print()

    # 3. premium
    quote = price(contract, rainfall)
    print("[3] ACTUARIAL PREMIUM (per policy)")
    for k, v in quote.as_dict().items():
        unit = "" if k == "rate_on_line" else " USD"
        val = f"{v:.4f}" if k == "rate_on_line" else f"{v:,.2f}{unit}"
        print(f"    {k:22s}: {val}")
    print()

    # 4. guarantee fund
    fund = size_fund(contract, rainfall, n_policies=N_POLICIES)
    print("[4] FIRST-LOSS GUARANTEE FUND (portfolio)")
    d = fund.as_dict()
    print(f"    fund_capital           : ${d['fund_capital']:,.0f}")
    print(f"    leverage_ratio         : {d['leverage_ratio']:.1f}x  (TSI / fund)")
    print(f"    ruin_probability       : {d['ruin_probability']:.4f}  (~1 in "
          f"{d['expected_recovery_years']:.0f} seasons)")
    print(f"    confidence             : {d['confidence']:.0%}")
    print("=" * 64)


if __name__ == "__main__":
    main()
