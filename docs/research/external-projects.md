# Comparable projects, pinned in `external/`

[Research index](README.md) · [Thesis](frontier-ai-edge-thesis.md) ·
[Results](edge-search-2026-11-17-results.md) · [Attribution](../../THIRD_PARTY_RESEARCH.md)

**Research only. Listing a project is not an endorsement of its claims.**

Eight public projects that try to trade or research markets with language models
are attached as Git submodules, each pinned to the commit that was its default
branch head on 17 November 2026. One recursive clone gives a single tree holding
all of them beside this repository's own code.

```sh
git submodule update --init --depth 1 external/stockbench   # one project
git submodule update --init --depth 1                       # all eight
```

They are reference checkouts. Nothing in `src/`, `scripts/` or `tests/` imports,
installs or runs them, and no upstream file is copied into this repository. Two of
the eight cannot legally be merged into MIT-licensed code (see the licence column),
which is one reason they are linked, not copied. Read each upstream licence before
running or redistributing anything.

## What each one is and what happened to its claims

Stars are from the GitHub search API on 17 November 2026.

| Path | Upstream | Licence | Stars | What it is | What independent evidence says |
| --- | --- | --- | ---: | --- | --- |
| `external/tradingagents` | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | Apache-2.0 | 122,867 | Multi-agent LLM "trading desk": analysts, debating researchers, trader, risk | A reproduction ("The Alpha Illusion") fell below buy-and-hold after costs; open issues report future news leaking into backtests; FINSABER notes its original three-month window |
| `external/ai-hedge-fund` | [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | MIT | 64,711 | LLM agents styled on famous investors voting on trades | States it is educational; makes no performance claim |
| `external/rd-agent` | [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | MIT | 15,398 | Automated hypothesis, code and backtest loop for factors and models | The RD-Agent(Q) paper reports up to twice the annualised return of classical factor libraries on CSI 300, in the authors' own backtests |
| `external/qlib` | [microsoft/qlib](https://github.com/microsoft/qlib) | MIT | 50,332 | Quant research platform: data, models, backtests, experiment records | Infrastructure, not a strategy; this repository already uses its recorder through an adapter |
| `external/ai-trader` | [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) | No licence file at the pinned commit | 23,297 | Live benchmark of autonomous agents on US stocks, A-shares and crypto | Its paper reports clear limits of current agents and stresses risk management |
| `external/finsaber` | [waylonli/FINSABER](https://github.com/waylonli/FINSABER) | Apache-2.0 | 153 | Twenty-year, broad-universe backtests of published LLM strategies | Reported LLM advantages shrink or vanish over longer periods and more symbols |
| `external/stockbench` | [ChenYXxxx/stockbench](https://github.com/ChenYXxxx/stockbench) | Apache-2.0 | 188 | Multi-month stock trading benchmark dated after model training cut-offs | Most LLM agents struggle to beat equal-weight buy-and-hold |
| `external/live-trade-bench` | [ulab-uiuc/live-trade-bench](https://github.com/ulab-uiuc/live-trade-bench) | PolyForm Noncommercial 1.0.0 | 169 | Live streaming evaluation of 21 models over 50 days | General benchmark scores did not predict trading results; it omits costs and slippage |

Related evidence that is not a code repository: Nof1's real-money Alpha Arena, in
which four of six frontier models lost money in the first season, and Quantopian's
study of 888 algorithms, in which backtest Sharpe explained one to two percent of
live results.

## How this repository compares

| Question | The popular agent repos | This repository |
| --- | --- | --- |
| Is a language model in the trading loop? | Yes, it decides each trade | No. Models generate and audit research code |
| Main evidence offered | Short backtests, often inside the model's training period | A hash-checked register that includes failures |
| Out-of-sample test on a window fixed in advance | Rare | Yes: [results](edge-search-2026-11-17-results.md) |
| Outcome of that test | Mostly at or below buy-and-hold when others rerun them | No edge found |
| What it is good for | Learning agent orchestration | Learning how to test a claim without fooling yourself |

Nobody in this set has publicly shown a durable edge from a language model. The
projects differ mainly in whether they say so.

## Using a checkout

Treat the code as untrusted: read before running, use an isolated environment,
never give it brokerage credentials. Do not point one at a paid feed or a live
account from this repository; the [research policy](../../RESEARCH_POLICY.md)
authorises neither. To move a pin, update the submodule on a branch and record the
new commit and date here.

| Path | Pinned commit |
| --- | --- |
| `external/tradingagents` | `9fa1196fffbcfd9de5146d2dd2869df84ab99838` |
| `external/ai-hedge-fund` | `5c553b5222044f6a5527bd377b4104c50ee0ad28` |
| `external/rd-agent` | `484776c211e4fbbeef03e0ec00d6bbee7362a4f4` |
| `external/qlib` | `54355232463878d2eebb91fe0ee5fa7fa1f5976c` |
| `external/ai-trader` | `d03ff6c056b32ced735adf7c19ed8175adb1c8df` |
| `external/finsaber` | `e72c26206d08de2ffeab736b96878109f144040f` |
| `external/stockbench` | `ce8b2b3483590646ad3b650ac8221f43f76fd091` |
| `external/live-trade-bench` | `05fbfec8df76c2c309201e6b4b7c26aff7e6d0f9` |

## Sources

- [The Alpha Illusion, arXiv 2605.16895](https://arxiv.org/abs/2605.16895)
- [FINSABER, arXiv 2505.07078](https://arxiv.org/abs/2505.07078)
- [RD-Agent(Q), arXiv 2505.15155](https://arxiv.org/abs/2505.15155)
- [StockBench, arXiv 2510.02209](https://arxiv.org/abs/2510.02209)
- [LiveTradeBench, arXiv 2511.03628](https://arxiv.org/abs/2511.03628)
- [AI-Trader, arXiv 2512.10971](https://arxiv.org/abs/2512.10971)
- [Alpha Arena season one, Forklog](https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/)
- [Quantopian study summary, Quantpedia](https://quantpedia.com/quantopians-academic-paper-about-in-vs-out-of-sample-performance-of-trading-alg/)
