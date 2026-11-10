# Decision Lab — data-backed risk and business research

**First implementation, not a validated investment-advice system.** This isolated
module extends the existing research lab; it does not replace the paper-trading
engine, change strategy mandates, promote a strategy, or connect to a broker.

Three questions must stay separate:

| Research lane | Question | Evidence required |
| --- | --- | --- |
| Market risk | How often do defined losses occur under a fitted historical distribution? | Aligned returns, currency and distribution treatment, robust validation |
| Housing | How have a specified region's prices and affordability changed? | Native-frequency price, income, rent and financing series with release dates |
| Project economics | Which explicit commercial assumptions produce positive contribution? | Fees, cash costs, founder hours, collections and separately justified sales assumptions |

A market loss probability is **not** a probability of overvaluation, a housing
bubble, business success, or an instruction to buy or sell. Unreproduced numbers
from informal analyses are not input data or validated model outputs.

## Architecture

```text
Explicit download / licensed export
               |
      raw source hash + metadata
               |
      immutable JSON snapshot
               |
  units / currency / dates / freshness gate
          /                  \
 joint historical scenarios  native-frequency housing indicator
          |
 separate terminal loss / temporary loss / peak-to-trough drawdown
          |
 retrospective rolling-origin diagnostic + naive baseline
          |
 model report: assumptions, hashes, versions, limitations

Private project inputs -> assumption-driven economics -> local-only result
```

The new code has no dependency on `trading`, `brain`, broker SDKs, wallet keys,
or Claude/API calls. LLMs may assist research and explanations, but deterministic
code calculates the numbers. Monetary units are explicit; the new AUD lane never
silently changes the existing USD paper book.

## Run an offline example

From the repository root in the existing development environment:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab demo
PYTHONPATH=src pytest -q tests/test_decision_lab.py
PYTHONPATH=src python -m llm_quant.decision_lab project \
  config/decision_lab/project.example.json
```

The demo uses **fictitious, seeded observations**, not downloaded market data.
Its status is `SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE`. All values in the example
project file are illustrative, not a real client's fees or a measured win rate.

Default data and outputs go to `research-local/decision-lab/`. The committed
`research-local/.gitignore` excludes everything there except that ignore file.
Gitignore is an accident-prevention measure, not access control: never force-add
private files and never publish restricted datasets without permission.

## Data sources and explicit download

| Source | Implemented coverage | Important boundary |
| --- | --- | --- |
| BIS via FRED | `QAUR628BIS` (real) and `QAUN628BIS` (nominal), quarterly | Eight large Australian cities, **not Sydney specifically**; revised/latest vintage |
| Yahoo via existing yfinance dependency | `VAS.AX`, `VGS.AX`, `NDQ.AX`, `VAF.AX`, `VAP.AX`, `GOLD.AX` | Provider must confirm AUD; adjusted close is a proxy, not independently verified net total return |
| Licensed Sydney index | No adapter enabled | Obtain a permitted, consistent history; do not scrape or splice headline numbers |
| RBA / ABS / ALFRED | Source references and requirements only | No claim that these additional adapters or point-in-time archive are implemented |

Network access is disabled unless the fetch command explicitly receives `--online`.
There is no paid-feed subscription, scheduled download, or API-key requirement in
this increment. Yahoo's terms must be reviewed before any non-personal use.
Downloaded vendor observations are **not** committed to this public repository.

```sh
PYTHONPATH=src python -m llm_quant.decision_lab fetch fred QAUR628BIS --online
PYTHONPATH=src python -m llm_quant.decision_lab fetch yahoo VAS.AX \
  --start 2010-01-01 --end 2026-11-11 --online
```

Yahoo's `--end` is exclusive. Both adapters fail rather than substitute fabricated
observations. FRED's convenience CSV endpoint can change: its schema is checked.
A canonical production integration should use a governed data vendor and/or the
FRED API, with credentials supplied outside Git. Each FRED snapshot retains the
raw response hash; yfinance's raw cache is explicitly a provider-frame CSV export,
not an HTTP-body capture. Snapshots also have a normalized-content hash.

## Run a market or housing scenario

Substitute the actual path printed by `fetch` for `SNAPSHOT.json`:

```sh
# Quarterly housing index: four native quarters, NOT four days.
PYTHONPATH=src python -m llm_quant.decision_lab risk \
  --snapshot SNAPSHOT.json --weights 1 --indicator \
  --as-of 2026-11-11 --max-age-days 200 \
  --horizon 4 --block-length 4 --paths 10000

# Daily AUD ETF. For a portfolio, repeat --snapshot and supply aligned weights.
PYTHONPATH=src python -m llm_quant.decision_lab risk \
  --snapshot SNAPSHOT.json --weights 1 \
  --as-of 2026-11-11 --max-age-days 10 \
  --horizon 252 --block-length 20 --paths 10000
```

The first command may correctly **refuse stale quarterly data**. Freshness is
conservatively measured from the source observation label, which for FRED
quarterly data is the quarter's first day, not its release date. Raising the age
limit is an explicit analyst decision, not a repair for missing current evidence.

The model requires at least 24 observations and three forecast horizons of history.
These are computational minimums, **not sufficient statistical validation**.
Histories must have identical dates, frequency, currency, value basis and evidence
type. The first version intentionally does not automatically align calendars,
forward-fill prices, convert FX, or interpolate a quarterly index into daily data.
Daily exchange-calendar completeness still requires a separate data-quality check.

`usd_to_aud` is a tested numerical helper using `USD price / (USD per AUD)`;
production timestamp-aware FX alignment is not implemented. Housing indicator
changes exclude rent, transaction costs, maintenance, leverage and sale liquidity.
They cannot be presented as an investable total-return portfolio.

## Model card: historical risk

- Method: joint stationary bootstrap, geometric blocks, synchronized asset rows.
- Allocation: fully invested, long-only, unleveraged **buy-and-hold**. Weights drift.
- Outputs: return quantiles, lower-tail expected shortfall, terminal-loss frequency,
  ever-below-initial frequency, and peak-to-trough drawdown frequency.
- Reproducibility: source hash, snapshot hashes, seed, settings, code hash, Python
  and NumPy versions. Matching inputs and environment reproduce the run.
- No costs, tax, transaction execution, liquidity model or deposit insurance model.
- Monte Carlo standard error describes finite simulation noise only. It is not
  parameter uncertainty, model uncertainty or a guarantee about real-world odds.
- A bootstrap cannot reproduce shocks absent from its observations. Regime shifts,
  history length, survivorship, corporate actions and block length matter.

The report remains `UNCALIBRATED_HISTORICAL_SCENARIOS`. More simulations alone do
not make that status decision-ready, and a favourable backtest never promotes it.

## Retrospective validation

```sh
PYTHONPATH=src python -m llm_quant.decision_lab validate \
  --snapshot SNAPSHOT.json --weights 1 \
  --as-of 2026-11-11 --max-age-days 10 \
  --horizon 63 --block-length 20 --paths 2000 --min-train 756
```

Every fit uses observations strictly before its origin. Origins advance by the
horizon, so realized labels do not overlap. A training-only historical base rate
is scored alongside the model using the Brier score. Scores with fewer than 20
folds are explicitly flagged. Non-overlapping labels do not imply independence.

This is a **retrospective diagnostic**, not genuine point-in-time trading evidence:
current adjusted prices and revised macro data may differ from what was known at
the historical origin. No current adapter provides historical availability dates.
ALFRED vintages, exchange-close timestamps and release-aware joins are separate
acceptance requirements. No automated portfolio recommendation is generated.

## Model card: project economics

The project calculator separates expected cash contribution from economic surplus
after valuing founder time. Presales and fixed experimental costs are incurred even
when no sale occurs; delivery hours/costs are conditional on winning. It reports
an economic break-even win probability and identifies cases that are unprofitable
even at a 100% close rate. Ratios of expected contribution to expected hours are
not guaranteed realized hourly rates. Prior sunk costs are not automatically added.

The win rate is an explicit assumption. GitHub commits, stars, pull requests and
synthetic analytics **cannot establish demand or a probability of commercial success**.
Collected cash, genuine customers, retention, support obligations and attributable
costs belong in a separately permissioned private dataset. The public module uses
no real client names, business accounts or project-specific financial records.

`mortgage_payment` and `affordability_change` provide deterministic scenario
arithmetic. They do not estimate intrinsic property value or crash probabilities.

## Evidence standard for subsequent releases

| Stage | Acceptance evidence | Current status |
| --- | --- | --- |
| Engineering foundation | Data contracts, reproducible scenarios, offline tests | Implemented; repository-wide checks still required |
| Governed datasets | Licensed sources, calendar/FX/corporate-action checks, release timestamps | Incomplete; current two adapters are research-only |
| Validated forecasting | Untouched holdout; calibration/coverage; benchmark skill; window/block/regime sensitivity | Not established |
| Independent replication | Same conclusions from a second implementation and a versioned dataset | Not established |
| Forward research | Append-only dated forecasts evaluated only after outcomes mature | Not established |
| Capital use | Existing governance, independent review and explicit human authorization | Not authorized by this module |

A sensible extension sequence is data integrity first, empirical baselines second,
then calibrated volatility/regime models, and only then optimization or ML models.
A model that cannot beat a simple benchmark out of sample is not improved by a
more elaborate dashboard or a larger Monte Carlo path count.

## Source references

- BIS housing coverage and real/nominal series: https://data.bis.org/topics/RPP
- FRED real series: https://fred.stlouisfed.org/series/QAUR628BIS
- FRED nominal series: https://fred.stlouisfed.org/series/QAUN628BIS
- FRED API vintages: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- FRED real-time periods: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- yfinance usage/terms notice: https://ranaroussi.github.io/yfinance/
- RBA financial-data conditions: https://www.rba.gov.au/copyright/
- ABS discontinued index: https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/residential-property-price-indexes-eight-capital-cities
- Politis & Romano, *The Stationary Bootstrap* (1994): https://doi.org/10.1080/01621459.1994.10476870

Source catalogue checked 11 November 2026. Provider terms and series metadata must
be rechecked before operational use; source references are not endorsements.
