# Niche Arbitrage Research — Combined Summary

## Date: 2026-03-27
## Purpose: Reference document for Track C (Niche Arbitrage) research track

## Shortlisted Strategies (Both Reports Agree)

### Tier 1 — Highest Conviction
1. **Crypto Perpetual Funding Rate Arbitrage**
   - Mechanism: Long spot + short perp, collect funding payments (positive 71.4% of time)
   - Returns: 10-30% annualized gross (Gate.io avg 19.26% in 2025)
   - Sharpe: 1.5-3.0+ (regime dependent)
   - Capital: $5K min, $50K+ optimal (VIP fee tiers)
   - Delta-neutral, 8-hour frequency — no HFT needed
   - Cross-exchange enhancement: +3-5% from funding rate differentials
   - Risk: exchange counterparty (FTX), liquidation during squeezes, funding reversal in bear markets
   - Stack: CCXT, WebSocket feeds, pre-funded accounts on 3-4 exchanges

2. **Closed-End Fund (CEF) Discount Mean-Reversion**
   - Mechanism: Buy deeply discounted CEFs, sell when discount narrows
   - Academic validation: CUNY study — 14.9% annualized, Sharpe 1.519 (quintile strategy)
   - Enhanced model: 18.2% annualized, Sharpe 1.918
   - Calamos CCEF ETF: 17.04% annualized real-world validation
   - Current opportunity: avg discount -6.89% (Q4 2025), widened 3pp during 2025
   - Universe: 837 funds, $1.12T assets
   - Capital: $50K+ for diversification across 10-20 CEFs
   - Data: CEFConnect.com, CEFData.com, Fidelity screener (free)
   - Risk: discounts can persist/widen indefinitely (no AP mechanism like ETFs)

### Tier 2 — Strong Supplementary
3. **Crypto Cash-and-Carry Basis Trade**
   - Mechanism: Long spot, short quarterly futures — guaranteed convergence at expiry
   - Returns: 5-25% variable (25% peak Feb 2024, ~5% by Dec 2025)
   - Implementation: Micro BTC futures (MBT) on CME via IB, 0.1 BTC per contract
   - Capital: $20K+ for futures margin
   - Risk: basis compression as market matures, capital locked until expiry

4. **Merger Arbitrage / Deal Spread Capture**
   - Mechanism: Buy target stock after announcement, short acquirer (stock-for-stock deals)
   - Returns: 6-10% annualized, Sharpe 0.9-1.5
   - Eurekahedge index: 6.7% return, 3.2% vol, Sharpe 1.46, only 2 down years in 25
   - Current spreads: 10-11.9% annualized (550bps above T-bills)
   - Data: SEC EDGAR (free), InsideArbitrage
   - Risk: asymmetric — small gains vs 20-40% loss on deal breaks
   - Capital: $50K+ for 10+ deal diversification

5. **VIX Contango Harvesting (Overlay Only)**
   - Mechanism: Short VIX futures in contango (~80% of time)
   - Returns: 15-40% in calm markets
   - CRITICAL: tail risk is existential (XIV lost 96% in one day, Feb 2018)
   - Max allocation: 5-10% of portfolio, strict conditional entry
   - Academic Sharpe: 0.36-0.60

## Strategy Graveyard (Do NOT Implement)
- Leveraged ETF decay capture (TQQQ/SQQQ short both) — mathematically zero-sum (Elm Wealth 2025)
- Triangle crypto arbitrage — 4,879 opportunities, zero profitable after costs (Muck et al 2025)
- Put-call parity violations — HFT captures in microseconds
- Cross-exchange crypto on majors — $50-200 gaps, below retail fees
- S&P 500 index inclusion effect — declined from 7.4% (1990s) to 1.0% (2010-2020)
- Kimchi premium — requires Korean residency and capital controls block foreigners
- Mutual fund 4PM NAV — regulated away post-2003 Spitzer investigation

## ETF/NAV Arbitrage (Stress-Event Only)
- March 2020: bond ETFs >5% discount to NAV
- Normal times: 0.1-0.5% — requires AP access or large capital
- Not a systematic strategy; more of an opportunistic overlay during crises

## Key Academic Findings
- McLean & Pontiff (2016): anomaly returns 26% lower OOS, 58% lower post-publication
- Xiu et al (2024, Chicago Booth): statistical uncertainty creates fundamental Sharpe ceiling
- Jacobs & Müller (2020): post-publication decay mostly US phenomenon — international anomalies persist
- Small capital is an advantage: sub-$10M strategies systematically ignored by institutions

## Portfolio Construction (Proposed $100K Allocation)
| Strategy | Allocation | Expected Return | Expected Sharpe |
|---|---|---|---|
| Crypto funding rate arb | 30% ($30K) | 15-25% | 1.5-3.0 |
| CEF discount arb | 25% ($25K) | 8-12% | 1.5-1.9 |
| Merger arbitrage | 20% ($20K) | 6-10% | 0.9-1.5 |
| Crypto basis trade | 15% ($15K) | 5-20% | 1.0-2.0 |
| VIX contango overlay | 5% ($5K) | 15-40% | 0.4-0.6 |
| Cash reserve (events) | 5% ($5K) | Variable | N/A |

Combined portfolio expected: 12-18% annualized, Sharpe 1.5-2.5
Strategies are genuinely decorrelated (crypto sentiment, rate cycles, deal activity, vol regimes)

## Implementation Stack
- Interactive Brokers (equities, ETFs, futures, options)
- CCXT (crypto exchange connectivity)
- ib_insync (Python IB wrapper)
- DuckDB (local storage — already in our stack)
- Polars (dataframes — already in our stack)
- Free data: yfinance, CoinGlass, CEFConnect, SEC EDGAR, Polygon.io free tier
