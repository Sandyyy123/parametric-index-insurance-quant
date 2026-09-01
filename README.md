> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# parametric-index-insurance-quant

Transparent, auditable **quantitative engine** for weather-index / area-yield crop
insurance aimed at smallholder de-risking (rainfall / temperature / drought triggers).

> **Scope note.** This repo is the *quantitative modelling layer* — index calibration,
> basis-risk analysis, actuarial premium simulation, and first-loss guarantee-fund
> sizing. It is **not** an insurance-product, distribution, or regulatory package. It is
> the maths a product lead / actuary needs under those decisions. All numbers in the demo
> are synthetic and illustrative.

## What it does

| Module | Question it answers |
|---|---|
| `index_calibration.py` | Where do the payout **triggers** sit? Grid-searches a piecewise-linear payout (`trigger`→`exit`) to best fit the loss history. |
| `basis_risk.py` | How well does the index track **actual farm loss**? Correlation, downside (false-negative) and upside (false-positive) frequency, and a single Basis-Risk Ratio. |
| `premium.py` | What is the **actuarial premium**? Monte-Carlo pure premium + volatility loading + expense loading, with tail metrics (VaR99 / TVaR99) for the reinsurer. |
| `guarantee_fund.py` | How big must the **first-loss fund** be? Sizes capital at a target confidence under *systemic* (fully-correlated) weather, reports leverage and ruin probability. |

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Produces a full worked example for a rainfall-index drought cover on a
5,000-policy cooperative book (synthetic 30-season history).

## Design choices worth reading

- **Transparent by construction.** Calibration is a visible grid search, not a black box —
  regulators, reinsurers and cooperatives must be able to audit why a trigger sits where it does.
- **Basis risk is treated as the headline metric**, not a footnote. Downside basis risk
  (real loss, index underpays) is the reputational killer for smallholder products, so it is
  reported separately from upside basis risk.
- **Systemic risk is not diversified away.** Weather in one agro-climatic zone hits the whole
  book at once, so the fund is sized on a single per-season draw applied to every policy.
  A consequence you can see in the demo: at 99% confidence a pure first-loss fund can't be
  leveraged far past ~1× — which is exactly why a reinsurance layer belongs *above* the fund.

## Not included (deliberately)

Product specification, peril legal definitions, distribution/channel design, claims
operations, and regulatory filing are insurance-domain deliverables owned by the product
lead. This engine plugs underneath them.
