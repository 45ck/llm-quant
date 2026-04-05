# Polymarket systematic trading: a comprehensive research landscape

**Polymarket has become the dominant prediction market platform with ~$3.7B monthly volume and an $11.6B valuation, but systematic alpha remains scarce, risks are severe, and the most promising edge — LLM-assisted forecasting — is rapidly commoditizing.** This report synthesizes platform mechanics, microstructure research, strategy viability, data infrastructure, and risk analysis for an open-source systematic trading research project (llm-quant). The core finding is that while structural mispricings exist (favorite-longshot bias, cross-platform discrepancies, calibration gaps), they are smaller and more contested than commonly claimed. Oracle manipulation, regulatory flux, and liquidity constraints represent existential risks that no backtest can capture. Paper-trading-first operation is not merely advisable — it is essential.

---

## 1. Platform architecture and API landscape

### The CLOB: hybrid-decentralized order matching

Polymarket operates a **hybrid-decentralized Central Limit Order Book** — orders are matched off-chain by the Polymarket operator, but settlement occurs non-custodially on the Polygon blockchain via the CTF Exchange smart contract. This design eliminates gas costs for order placement while preserving on-chain auditability. The architecture differs fundamentally from traditional CLOBs in one critical way: **supply is dynamically created**. When opposing limit orders match (YES at $0.65 + NO at $0.35 = $1.00), a new token pair is atomically minted from USDC.e collateral via the Gnosis Conditional Tokens Framework (ERC-1155). Supply is theoretically unlimited and entirely demand-driven — a feature with no equity market equivalent.

Four distinct API services support programmatic access:

| Service | Base URL | Auth | Rate Limit (per 10s) |
|---------|----------|------|---------------------|
| **CLOB API** | `https://clob.polymarket.com` | Read: No; Write: EIP-712 L2 keys | 9,000 general; 3,500 order placement |
| **Gamma API** | `https://gamma-api.polymarket.com` | None | 4,000 general |
| **Data API** | `https://data-api.polymarket.com` | Yes (user data) | 1,000 general |
| **WebSocket** | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Market: No; User: Yes | Real-time streaming |

Order types include **GTC limit orders** (the workhorse), **FOK market orders**, and **post-only orders** that reject if they would immediately match. The default tick size is $0.01, with dynamic narrowing to $0.001 when prices exceed $0.96 or fall below $0.04 — a design choice that improves granularity where it matters most for near-certain outcomes. Maximum batch size is **15 orders per request**, with sustained order placement capped at **60/second**.

### Fee structure has evolved rapidly

Polymarket's fee model changed dramatically in early 2026, abandoning its zero-fee model:

- **Makers pay zero fees** and receive daily USDC rebates (20–50% of taker fees, category-dependent).
- **Taker fees** are nonlinear, calculated as `feeRateBps × min(price, 1−price) × size`, peaking at 50% probability and approaching zero at extremes. Maximum effective rates range from **~1.5% (economics/mentions) to ~1.8% (crypto markets)**. Geopolitics markets remain fee-free.
- **Polymarket US** (CFTC-regulated, launched December 2025 as invite-only): **30 bps taker fee, 20 bps maker rebate** on Total Contract Premium.
- No deposit, withdrawal, or settlement fees from Polymarket itself. Polygon gas costs are negligible (~$0.002/tx).

The `feeRateBps` value must be fetched dynamically per market — hardcoding will produce incorrect calculations. This is a critical implementation detail for any backtesting framework.

### Market creation and resolution mechanics

Only Polymarket's Markets Team creates markets. Users can suggest ideas via Discord or Twitter. Each market is assigned a `questionId` (IPFS-stored), an `outcomeSlotCount`, and an oracle address. Three oracle systems are used:

- **UMA Optimistic Oracle** (primary, international platform): Proposers post a **$750 USDC bond** and a 2-hour challenge period follows. ~98.5% of requests resolve without dispute. If escalated, UMA token holders vote via the Data Verification Mechanism (DVM) over 48–72 hours.
- **Chainlink**: Automated resolution for objective data (crypto prices, sports scores) via Data Streams and Automation.
- **Markets Team**: Direct resolution on Polymarket US.

Contract types include simple **binary (YES/NO)** markets and **NegRisk multi-outcome** markets (e.g., "Who will win the presidential election?" with 17+ candidates), where capital efficiency allows trading all outcomes with only $1.00 in collateral.

### Geographic restrictions and compliance

The international platform is geoblocked for US IP addresses with no KYC. **OFAC-sanctioned countries** are fully blocked (Cuba, Iran, North Korea, Syria, Russia, Belarus). An expanding list of countries has restricted or banned access: France, Belgium, Germany, Italy, Singapore, Switzerland, Poland, Portugal, Hungary, and others. Polymarket US requires **full KYC** (government ID, live selfie, SSN) and operates through registered Futures Commission Merchants. It remains waitlist-only as of April 2026, with full public access estimated Q3–Q4 2026.

---

## 2. Prediction market microstructure and what the academic literature reveals

### Liquidity patterns show extreme concentration

The most comprehensive academic study to date — Tsang & Yang (2026, arXiv:2603.03136), "The Anatomy of Polymarket" — documents pronounced characteristics of Polymarket's microstructure using complete on-chain Polygon data from the 2024 presidential election:

**Bid-ask spreads** in liquid markets (major elections, NBA Championship) run **0.5–2%** (1–4 cents), while illiquid markets show spreads as wide as **$0.34** — buying and immediately selling could cost ~50%. Becker's analysis of 72.1M trades on Kalshi found average spreads compressed ~43% as volume grew, narrowing from ~4.2¢ to ~2.4¢ in 2025. Finance markets show the tightest effective spreads (maker-taker gap of only **0.17 percentage points**), while World Events and Media exceed **7 pp**.

**Kyle's λ** (price impact coefficient) in the 2024 presidential market declined from approximately **0.518 to 0.01** as the election approached — a dramatic deepening of liquidity. Trading activity peaks during U.S. business hours, confirming a primarily U.S.-centric participant base despite the offshore platform.

No formal designated market makers exist, but Polymarket's **Liquidity Rewards Program** pays traders who place orders within a "max spread" of the midpoint, scored quadratically and sampled every minute. Practitioner reports claim professional market makers earn **$150–300/day per market** with $100K+ daily volume, but these figures come from marketing-adjacent sources, not peer-reviewed research.

### The academic consensus on prediction market efficiency

The literature broadly supports prediction market efficiency for liquid, high-profile markets, with important caveats:

**Foundational work** by Wolfers & Zitzewitz (2004, *Journal of Economic Perspectives*) established that "market-generated forecasts are typically fairly accurate" and outperform "most moderately sophisticated benchmarks." Snowberg, Wolfers & Zitzewitz (2012) reinforced this: "prediction markets quickly incorporate new information, are largely efficient, and impervious to manipulation." Arrow et al. (2008, *Science*), signed by 22 prominent scholars, recommended regulatory safe harbors based on the information aggregation properties. Manski (2006) offered an important corrective: prices reflect *budget-weighted beliefs*, not mean beliefs, and may only bound true probabilities.

**The favorite-longshot bias (FLB)** is the best-documented anomaly. Snowberg & Wolfers (2010, *JPE*) attributed it primarily to **probability misperception** rather than risk-love. Bürgi, Deng & Whelan (2025, CESifo) confirmed the bias using 300,000+ Kalshi contracts. Becker (2026) found contracts at 5¢ win only 4.18% of the time versus the implied 5% — a **-16.36% mispricing** — with all contracts below 20¢ underperforming and above 80¢ outperforming. The bias is robust across platforms and categories but is often too small to exploit after transaction costs.

**Markets vs. alternatives**: Berg, Nelson & Rietz (2008) found IEM markets closer to election outcomes than polls on **74% of days** across five elections. But Erikson & Wlezien (2008/2012) argued that "when both polls and markets exist, market prices add nothing beyond what polls already contain." Clinton & Huang (2024/2025, Vanderbilt) analyzed 2024 election markets and found **PredictIt achieved 93% accuracy, Kalshi 78%, Polymarket 67%** — though methodology differences complicate comparison. The 2024 election was widely cited as a vindication for prediction markets (Polymarket priced Trump at ~57% while polls showed a toss-up), but selection bias in citing successes is a concern.

**On manipulation**: the literature has shifted dramatically. Rhode & Strumpf (2004/2006) found "little evidence that political stock markets can be systematically manipulated beyond short time periods." But Rasooly & Rozzi (2025, arXiv:2503.03312), in a large randomized field experiment on Manifold, found **substantial manipulability**: while prices partially revert (~25% after one week), effects persisted **60 days later** — contradicting the earlier optimistic findings. Mitts & Ofir (2026, Columbia/Harvard Law) screened 93,000+ Polymarket markets and found **210,718 suspicious wallet-market pairs with a 69.9% win rate, exceeding random chance by >60 standard deviations**. Wash trading accounts for an estimated **25% of all Polymarket volume** per Kanoria, Ma, Sethi & Sirolly (2025, Columbia).

---

## 3. Strategy mechanism families: an honest assessment

### LLM-assisted forecasting is the most promising but rapidly commoditizing

Eight strategy families were evaluated against evidence quality, practical feasibility, and edge decay. The most important finding is that **no strategy has robust, peer-reviewed evidence of persistent alpha in prediction markets after transaction costs**. What follows is a severity-ranked assessment:

**LLM-assisted calibration arbitrage** (Strong evidence, rapidly decaying edge): ForecastBench data shows GPT-4.5 achieving a Brier score of **0.101** versus superforecasters' **0.081**, with projected parity by **late 2026** (95% CI: Dec 2025 – Jan 2028). Halawi et al. (2024, NeurIPS) demonstrated that retrieval-augmented LLM systems approach human-level forecasting on a 914-question dataset. The ensemble approach is critical: Schoenegger et al. (2024, *Science Advances*) found a 12-LLM ensemble was "statistically indistinguishable from the human crowd." Key limitation: **LLMs have particular deficits in uncertainty quantification** (calibration ECE ~0.10–0.15 vs. superforecasters ~0.03–0.05), requiring post-hoc calibration before use as trading signals. The scalability advantage — an LLM monitoring thousands of markets at low marginal cost — is real, but edge will decay as LLM-based trading proliferates.

**Cross-market arbitrage** (Strong evidence, rapid decay): Price discrepancies of **1–2.5%** between Polymarket, Kalshi, PredictIt, and Robinhood persist, with occasional 3–5% gaps during news events. Automated bots captured ~**$40M in "risk-free" profits** in a recent year. But settlement mismatch (different resolution criteria make "identical" markets non-identical), capital lockup (positions locked weeks/months), and competitive crowding make this increasingly difficult. Latency requirements are sub-second. The ecosystem is already crowded with commercial services like ArbBets claiming "100+ arbitrage opportunities daily."

**Market making / liquidity provision** (Moderate evidence): Polymarket's explicit incentive programs (maker rebates + liquidity rewards) create a structural return on top of spread capture. Typical achievable spreads are **2–4 cents** on liquid markets. The critical risk is **adverse selection**: informed traders "pick off" stale quotes during news events, sometimes causing 40–50 point jumps. Binary settlement means inventory errors are catastrophic ($0 or $1 resolution). Becker (2026) documented that makers earn systematically (+1.12% average excess return) versus takers (-1.12%), but this "Optimism Tax" emerged only after professional market makers entered post-Q2 2024.

**Calibration arbitrage without LLMs** (Moderate evidence): The FLB is well-documented but after transaction costs (now 1–1.8% on Polymarket), net profitability is uncertain. Only **~16.8% of Polymarket wallets** show a net gain. Top earners (>$22M lifetime) succeeded through domain expertise in political markets, not systematic approaches.

**Portfolio/Kelly approaches** (Moderate as framework, not an alpha source): Kelly criterion applies directly to binary outcomes: f* = (bp − q)/b. A December 2024 arXiv paper specifically addresses application to prediction markets, noting that "the identification of prices in prediction markets with probabilities is popular but incorrect." Fractional Kelly (25–50%) is standard practice to account for estimation errors. This is position-sizing methodology, not independent alpha.

**Mean reversion** (Speculative): Angelini et al. (2021) documented overreaction persisting for minutes in sports betting exchanges, but **no robust study demonstrates profitable mean reversion in prediction markets specifically**. The bounded [0,1] price range and deterministic terminal values complicate traditional mean reversion analysis.

**News-driven event trading** (Speculative): Requires sub-10ms latency to compete. A large study of 1.86M headlines found that "sentiment scores lack robust predictive power" even in equity markets. In prediction markets, the window is even narrower. Practically inaccessible to most participants.

**Momentum** (Speculative/Unlikely): No academic evidence for exploitable momentum in prediction markets. What appears to be trending behavior is likely efficient information incorporation. The bounded price dynamics fundamentally differ from equities.

---

## 4. Data infrastructure and open-source tooling

### Historical data is the binding constraint for backtesting

The most critical gap for systematic research is **historical tick-level data**. The CLOB API's `/prices-history` endpoint supports configurable resolution but community reports indicate resolved markets may return only 12+ hour granularity, severely limiting backtesting fidelity. Real-time WebSocket data is comprehensive (order book changes, price updates, trades), but building a historical archive requires continuous collection from the start.

The best free historical data sources are:

- **Dune Analytics**: Transaction-level on-chain data queryable via SQL. Key dashboards include `dune.com/rchen8/polymarket` and `dune.com/alexmccullough/how-accurate-is-polymarket`. Refreshes weekly.
- **TheGraph subgraphs**: Block-level indexing of on-chain trades, volumes, positions. Free tier provides 100K queries/month. Multiple community subgraphs cover core data, orderbook analytics, and market names.
- **Goldsky**: Real-time streaming pipelines from on-chain data into your own database, partnered with ClickHouse.

Paid services include **PolymarketData.co** (bulk S3 exports, REST API, Python SDK) and **Telonex.io** (tick-level trades and orderbook snapshots).

### External signal data sources worth integrating

For news-driven signals, **GDELT** is the best free option (global coverage, 65 languages, ~15-min delay, Python client available). **NewsAPI.ai** (Event Registry) offers the richest paid option with sentiment and entity extraction. **AskNews** has proven utility specifically in LLM forecasting pipelines per Törnquist and Caulk (2024).

For calibration and base rates, the **Metaculus API** provides 12+ years of prediction data across **23,400+ questions** with community prediction timeseries and resolution outcomes. The official `Metaculus/forecasting-tools` framework (Python) includes a bot framework, benchmarking system, and question decomposer — the most production-ready forecasting toolkit available. The **FRED API** (`fredapi` Python wrapper) covers 800K+ economic time series for macro-event markets. **FiveThirtyEight data** is available as CC-BY 4.0 CSVs on GitHub for polling data.

### Open-source tools: maturity varies enormously

The official Polymarket client libraries are the foundation:

- **py-clob-client** (~783 GitHub stars, 339 commits, Python 3.9+): Full CLOB API coverage including order management and WebSocket. Active community but WebSocket stability issues reported (silent freezes when subscribing to 250+ tokens).
- **clob-client** (TypeScript/npm `@polymarket/clob-client`): Supports ethers v5 + viem.
- **rs-clob-client** (Rust): Zero-cost abstractions, auto-heartbeat WebSocket.

**`Polymarket/agents`** (1.7K stars, 440 forks) is Polymarket's official AI trading agent — but with only **7 commits**, it is a reference implementation, not production software. For community tools, **polybot** (Java, ent0n29) is the most architecturally sophisticated, featuring a full microservice system with ClickHouse + Redpanda pipeline and Grafana monitoring. **poly-maker** (Python, warproxxx) is the best dedicated market-making tool. Most other "trading bots" on GitHub should be treated as marketing vehicles or learning resources until independently verified.

For LLM forecasting research specifically, the key benchmark is **ForecastBench** by the Forecasting Research Institute, which tests frontier models on 500+ questions per round. GPT-4.5's Brier score of **0.101** versus superforecasters' **0.081** establishes the current frontier. The `Metaculus/forecasting-tools` framework is the most practical starting point for building an LLM forecasting pipeline.

---

## 5. Risk landscape: three existential threats

### Oracle manipulation is demonstrated, repeatable, and uncompensated

The single most important risk finding: **UMA oracle manipulation has been demonstrated multiple times with millions of dollars at stake, and Polymarket has declined to issue refunds.** The March 2025 Ukraine mineral deal market ($7M at stake) saw a UMA whale with ~5 million tokens (~25% of votes) force through a factually incorrect resolution. The July 2025 Zelensky suit market ($150–160M at stake) generated similar controversy. UMA's economic security model is fundamentally insufficient: with a market cap of ~$85M, only **~5M tokens needed to control a vote**, and attack costs can be lower than profits from manipulated Polymarket positions. The August 2025 MOOV2 upgrade restricted proposal submissions to a 37-address whitelist, but dispute escalation to DVM voting remains manipulable. **Any systematic strategy must model resolution risk as non-negligible, particularly for subjective or politically contentious markets.**

### Regulatory uncertainty is actively litigating with no stable equilibrium

The regulatory landscape is in genuine flux. The Kalshi v. CFTC ruling (September 2024) declared election contracts are not "gaming," enabling industry expansion. But **10+ Congressional bills have been introduced in 2026** to restrict prediction markets, state attorneys general in Arizona, Nevada, New Jersey, and Maryland are actively challenging federal preemption, and the CFTC in April 2026 took the extraordinary step of suing three states to assert exclusive jurisdiction. Meanwhile, **federal prosecutors (SDNY) are actively examining** whether prediction market trades constitute insider trading — with documented cases including Israeli military personnel using classified information, a Google insider profiting >$1M on Year in Search rankings, and traders earning $553K betting on Iran strikes 71 minutes before news broke. The current favorable regulatory stance depends on the Trump administration's CFTC leadership; a change in administration could reverse course entirely.

### Liquidity risk makes systematic strategies impractical outside top markets

Of Polymarket's 295,000+ markets, liquidity is brutally concentrated. Major markets handle billions in volume, but long-tail markets may show only hundreds of dollars in daily activity with spreads exceeding 10%. **In thin markets, there is no reliable way to exit large positions** — the standard advice is "if you cannot answer 'who am I going to sell this to?', assume the position is hold-to-expiry." This fundamentally constrains portfolio-based strategies that require dynamic rebalancing. Near resolution, liquidity can evaporate as the outcome becomes clear.

Additional risks that warrant serious attention include **Polygon network outages** (documented 11-hour outage in March 2022, additional incidents in 2024–2025), **bridge vulnerabilities** (two near-catastrophic Plasma Bridge bugs discovered, each threatening $800M+), and **smart contract risk** (ChainSecurity audits found the NegRiskAdapter's emergency resolution mechanism "possibly not sufficient" with only a 1-hour emergency window).

### Consolidated risk matrix for systematic trading

| Risk | Severity | Residual After Mitigation | Paper Trading Impact |
|------|----------|--------------------------|---------------------|
| Oracle manipulation | **Critical** | High | Cannot be simulated; must monitor live |
| Regulatory disruption | **High** | High | Existential; monitor legislation continuously |
| Liquidity/exit risk | **High** | High (structural) | Paper trading overestimates fill quality |
| Insider trading (adverse selection) | **High** | High | Real-time risk impossible to backtest |
| Polygon infrastructure | **Medium-High** | Medium | Can simulate with historical outage data |
| Smart contract | **High** | Medium-High | Model as tail risk |
| Fee model changes | **Medium** | Medium | Must dynamically fetch `feeRateBps` |

---

## 6. Practical implications for the llm-quant project

### What the 5-gate robustness filter means for prediction markets

The project's existing robustness framework (Sharpe thresholds, max drawdown, Deflated Sharpe Ratio ≥ 0.95, CPCV out-of-sample ratio, perturbation stability) will require adaptation for prediction markets. Binary outcomes violate the continuous-return assumptions underlying Sharpe ratios. Resolution timing introduces path-dependency that CPCV cannot cleanly handle. Perturbation stability testing must account for the fact that small changes in probability estimates near 0.50 produce large position-sizing changes under Kelly criterion. A prediction-market-specific robustness filter should add: **resolution risk adjustment** (discount expected value by historical incorrect-resolution rate), **liquidity-adjusted Sharpe** (penalize strategies that require fills exceeding observable book depth), and **oracle manipulation stress tests** (model scenarios where winning positions resolve incorrectly).

### Recommended architecture for paper trading

The most productive path forward combines:

1. **Data pipeline**: Deploy py-clob-client WebSocket connections to continuously archive order book and trade data. Supplement with Dune Analytics SQL queries for historical on-chain data and Gamma API for market metadata.
2. **Signal generation**: Build a retrieval-augmented LLM ensemble (3–5 frontier models) using the Metaculus `forecasting-tools` framework as scaffolding. Calibrate outputs against Metaculus historical data before deployment. Post-hoc calibration is essential — raw LLM probabilities are systematically overconfident.
3. **Paper trading engine**: Simulate execution against live order book data with realistic slippage models (walk the book, don't assume midpoint fills). Track resolution outcomes. Use fractional Kelly (25%) sizing. Exclude markets with <$50K in 24h volume.
4. **Cross-platform monitoring**: Track Kalshi, PredictIt, and Robinhood prediction markets for arbitrage signals, even if not trading them, to validate model calibration.
5. **Risk monitoring**: Build real-time alerts for UMA dispute escalations (monitor on-chain `DisputePrice` events), Polygon network health, and regulatory developments.

---

## Conclusion: edges exist but are narrow, contested, and fragile

The prediction market landscape in 2026 is both more developed and more dangerous than commonly presented. Polymarket's infrastructure is sophisticated — the hybrid CLOB, dynamic token minting, and comprehensive API make systematic trading technically feasible. But the evidence for persistent, exploitable alpha is thin. The favorite-longshot bias is real but small. Cross-market arbitrage is documented but crowded. LLM-assisted forecasting is the most promising frontier, with projected superforecaster parity by late 2026, but the same capability available to every participant eliminates the edge once adopted.

The risks are more concrete than the opportunities. **Oracle manipulation is not theoretical — it has been demonstrated repeatedly for millions of dollars, with no recourse for affected traders.** Regulatory flux could shut down or fundamentally alter the market within a single legislative session. Insider trading is documented, widespread, and actively under federal investigation. These are not risks that show up in backtests.

For the llm-quant project, the highest-value work is likely not finding alpha but **building infrastructure**: robust data pipelines, calibrated forecasting models validated against Metaculus benchmarks, realistic paper-trading simulators that account for liquidity constraints and resolution risk. If an edge exists, it will emerge from disciplined, reproducible research — not from assuming one exists and optimizing to find it. The 5-gate robustness filter is the right instinct, but must be adapted for binary outcomes, resolution risk, and the structural differences between prediction markets and traditional financial markets.