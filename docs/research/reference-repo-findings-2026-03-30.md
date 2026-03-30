# Reference Repository Findings — 2026-03-30

Exploration of three cloned reference repositories for actionable research leads.
All findings are read-only observations; no changes to main project code.

---

## Repo 1: machine-learning-for-trading (Stefan Jansen)

Location: `C:/Projects/quant-refs/machine-learning-for-trading/`

### Priority Notebooks for Immediate Study

| # | Path | Relevance | Target Family |
|---|------|-----------|---------------|
| 1 | `03_alternative_data/02_earnings_calls/` | Earnings call NLP pipeline using spaCy + sentiment scoring — direct hypothesis for Family 6 (Sentiment/Text) | 6 |
| 2 | `14_working_with_text_data/04_news_text_classification.ipynb` | Naive Bayes news classifier on BBC articles; applicable to financial headline routing | 6 |
| 3 | `14_working_with_text_data/05_sentiment_analysis_twitter.ipynb` | Twitter sentiment extraction pipeline — Family 7 (Social Sentiment) | 7 |
| 4 | `15_topic_modeling/06_lda_earnings_calls.ipynb` | LDA topic extraction from earnings transcripts — latent regime signals for Family 6 | 6 |
| 5 | `19_recurrent_neural_nets/07_sec_filings_return_prediction.ipynb` | LSTM return prediction from SEC filings — maps to Family 6 alternative data | 6 |
| 6 | `10_bayesian_machine_learning/03_bayesian_sharpe_ratio.ipynb` | Bayesian Sharpe ratio using PyMC3 — directly applicable to IS vs OOS comparison in our robustness gate | all |
| 7 | `10_bayesian_machine_learning/05_stochastic_volatility.ipynb` | Stochastic volatility (SV) model — regime classification input for VIX-based risk_on/risk_off signal | regime |
| 8 | `13_unsupervised_learning/04_hierarchical_risk_parity/01_hierarchical_risk_parity.ipynb` | HRP portfolio construction — direct Family 3 (Volatility/Risk Factor) candidate and alternative to our equal-weight combination | 3, combination |
| 9 | `20_autoencoders_for_conditional_risk_factors/05_conditional_autoencoder_for_asset_pricing_data.ipynb` | Conditional autoencoder for latent risk factor extraction — high-signal Family 4 (Statistical Factor) | 4 |
| 10 | `24_alpha_factor_library/03_101_formulaic_alphas.ipynb` | WorldQuant 101 formulaic alphas with holding periods 0.6–6.4 days — rapid Family 2–5 hypothesis generation | 2, 3, 4, 5 |

### CPCV Coverage Assessment

The book covers purged CV and combinatorial CV in `06_machine_learning_process/README.md` (section: "Purging, embargoing, and combinatorial CV") and in the notebook `06_machine_learning_process/04_cross_validation.py`. However, there is **no standalone CPCV worked example notebook**; the treatment is conceptual/reference-level rather than a code demonstration. Our `robustness.py` implementation cannot be validated against this repo — look to `mlfinlab` (also cloned at `C:/Projects/quant-refs/mlfinlab/`) for the actual CPCV implementation.

### ML Regime Classification (HMM)

No explicit HMM notebook found. The closest available tools are:
- `09_time_series_models/03_arch_garch_models.ipynb` — GARCH volatility regimes
- `10_bayesian_machine_learning/05_stochastic_volatility.ipynb` — Bayesian SV model as a regime proxy
- `10_bayesian_machine_learning/04_rolling_regression.ipynb` — rolling beta as a regime-sensitive factor

For HMM specifically, use `hmmlearn` or check `mlfinlab` which has a dedicated market microstructure module.

### Factor Analysis Hypotheses (Families 3–5)

The `04_alpha_factor_research/` chapter and `24_alpha_factor_library/` appendix are the richest source:
- `04_alpha_factor_research/03_kalman_filter_and_wavelets.ipynb` — Kalman filter for dynamic factor extraction (Family 4)
- `04_alpha_factor_research/02_how_to_use_talib.ipynb` + `24_alpha_factor_library/02_common_alpha_factors.ipynb` — momentum, volatility, and liquidity indicators (Families 3/5)
- `04_alpha_factor_research/06_performance_eval_alphalens.ipynb` — Alphalens factor IC/ICIR evaluation framework to adopt in our `/robustness` gate

---

## Repo 2: qlib (Microsoft)

Location: `C:/Projects/quant-refs/qlib/`

### Architecture Patterns Worth Adopting

**Pattern 1: Signal-as-first-class-object**

`qlib/contrib/strategy/signal_strategy.py` shows the `BaseSignalStrategy` class accepting a `Signal` object that can be constructed from a `(model, dataset)` tuple, a DataFrame, or a plain dict. This cleanly separates signal generation from execution — mirrors our Filters → Indicators → Signals → Rules decomposition but with stronger typing. Adoption path: wrap our Claude JSON output in a `Signal`-like dataclass that tracks source, confidence, and timestamp.

**Pattern 2: MLflow-backed Experiment Recorder**

`qlib/workflow/recorder.py` wraps MLflow with a `Recorder` class tracking status (SCHEDULED → RUNNING → FINISHED → FAILED). Every model run, prediction, and metric is persisted via `save_objects()` and `log_metrics()`. This directly maps to our research lifecycle: each stage gate (`/hypothesis`, `/backtest`, `/robustness`) could emit a recorder artifact instead of a markdown file. The DuckDB `strategy_changelog` table could be the backend. Key methods: `save_objects()`, `log_metrics()`, `load_object()`.

**Pattern 3: Rolling Task Generator for Walk-Forward Validation**

`qlib/contrib/rolling/base.py` and `qlib/workflow/task/gen.py` implement a `Rolling` class that wraps any model config YAML and generates a time-series of train/test splits via `RollingGen`. The `task_generator(tasks, generators)` function composes multiple generators (e.g., rolling window × loss function) into a Cartesian product of experiment tasks. This is exactly our CPCV outer loop — adopt `task_generator` as the scaffold for `robustness.py`'s combinatorial path expansion.

**Pattern 4: Meta-Learning for Regime-Adaptive Data Selection**

`qlib/contrib/meta/data_selection/model.py` implements `MetaModelDS` — a meta-learning model that learns *which historical periods* to weight heavily during training based on IC loss. The `TimeReweighter` class reweights training samples by time period. This is directly applicable to our regime-based parameter switching: instead of hard regime classification (risk_on/risk_off), MetaModelDS learns a continuous weight over historical periods, reducing the HARKing risk of discretionary regime labeling.

**Pattern 5: Online Model Rolling and Model Update**

`qlib/contrib/online/` and `qlib/workflow/online/` implement online serving with automatic model retraining on new data, recording predictions, and updating deployed model state. The `OnlineStrategy` class handles the equivalent of our `/promote` + live deployment loop. Key insight: qlib separates "offline rolling" (for backtesting) from "online serving" (for production), matching our Research Track vs Operations Track separation.

### RD-Agent Compatibility Assessment

RD-Agent (https://github.com/microsoft/RD-Agent) is a **separate repo** from qlib — not cloned. The qlib README documents it as an external LLM-driven quant factory that generates factor code, runs backtests via qlib, evaluates IC metrics, and iterates. The loop is: LLM proposes factor → qlib backtest → IC/Sharpe evaluated → LLM revises.

**Compatibility with our beads lifecycle:** Partial. RD-Agent's loop corresponds to our `/hypothesis` → `/backtest` → `/robustness` cycle, but RD-Agent lacks our governance gates (DSR, CPCV OOS/IS, kill-switch checks). It would generate candidate hypotheses but could not replace the full lifecycle. The practical adoption path is: use RD-Agent to auto-generate Family 2–7 hypothesis candidates, then funnel each through our lifecycle manually. Do NOT wire RD-Agent directly to `/promote`.

---

## Repo 3: pysystemtrade (Robert Carver)

Location: `C:/Projects/quant-refs/pysystemtrade/`

### FDM Formula — Correct Implementation Confirmed

The formula given in CLAUDE.md (`FDM = 1/sqrt(N × (1/N² × Σρ))`) is a simplified equal-weight approximation. Carver's implementation is the general weighted form:

```
FDM = 1 / sqrt(W × H × W^T)
```

where W is the weight vector and H is the full correlation matrix. Source: `sysquant/estimators/diversification_multipliers.py`, function `diversification_mult_single_period()`:

```python
def diversification_mult_single_period(corrmatrix, weights, dm_max=2.5):
    risk = weights.portfolio_stdev(corrmatrix)  # = sqrt(W × H × W^T)
    dm = min(1.0 / risk, dm_max)               # FDM = 1/sqrt(W × H × W^T)
    return dm
```

The function `diversification_multiplier_from_list()` builds a time series of FDMs by:
1. For each rolling correlation estimation period, computing a single-period FDM from the latest weights.
2. Reindexing to daily frequency via forward-fill.
3. Smoothing with EWMA (span=125 days by default) to prevent jumpy positions.
4. Capping at `dm_max=2.5`.

**Key finding:** The equal-weight simplification in CLAUDE.md (`1/sqrt(N × (1/N² × Σρ))`) is equivalent to `1/sqrt(mean(H))` which is correct only when all weights are 1/N. For our current 16-strategy portfolio with unequal allocation (70/30 Track A/B split), we should use the full W × H × W^T form. Implementation priority: medium — the equal-weight approximation is directionally correct.

**Two-level DM structure:** pysystemtrade uses *two* diversification multipliers in cascade:
- **FDM** (Forecast Diversification Multiplier): scales combined signals *within* each instrument/strategy, computed per-instrument from forecast correlation.
- **IDM** (Instrument Diversification Multiplier): scales the combined portfolio across instruments, computed from the full instrument return correlation matrix.

Our current architecture has no equivalent to FDM — we only have implicit IDM via equal weighting. Mapping to our framework: FDM ≈ within-strategy signal combination weight, IDM ≈ cross-strategy combination weight.

### Position Sizing Framework

`systems/positionsizing.py` implements target volatility scaling:

```
subsystem_position = combined_forecast × vol_scalar
vol_scalar = (target_risk / instrument_vol) / price
```

This is equivalent to our ATR-based sizing but is applied to the *combined capped forecast* (scaled -20 to +20), not the raw signal. The key advantage: position size automatically shrinks when volatility is high and expands when volatility is low, with the forecast providing direction and magnitude. Applicable to our Track A position sizing refinement.

`systems/portfolio.py` adds the IDM layer on top: `final_position = subsystem_position × instrument_weight × IDM`. The IDM is estimated via the same `diversification_mult_single_period()` function, time-varying and smoothed.

### Daily Governance Layer (Production)

`sysproduction/` implements a full daily operations loop directly applicable to our Operations Track:

- **`run_systems.py`** — runs overnight backtests to generate fresh optimal positions for each strategy. Maps to our `build_context.py`.
- **`syscontrol/run_process.py`** — `processToRun` class enforces process state machine (NO_RUN → RUNNING → STOP → FINISHED) with database-backed control. Prevents double execution. Maps to our `governance.toml` halt logic.
- **`sysproduction/reporting/`** — 20+ specialized reports including `risk_report.py`, `pandl_report.py`, `reconcile_report.py`, `strategies_report.py`. These are run daily via `run_reports.py`. The `reconcile_report.py` is particularly relevant — it compares expected vs actual positions and flags discrepancies.
- **`systems/risk_overlay.py`** — `get_risk_multiplier()` computes a continuous [0,1] portfolio risk scalar from four components: (1) normal volatility risk, (2) shocked-vol correlation risk, (3) sum-abs-risk (stress scenario), (4) leverage. Takes the *minimum* across all four. This is architecturally superior to our binary kill-switch approach — it *scales down* positions proportionally rather than triggering an all-or-nothing halt.

### Risk Overlay vs Our Kill Switches — Gap Analysis

| pysystemtrade | llm-quant | Recommendation |
|---|---|---|
| Continuous risk multiplier [0,1] | Binary halt (on/off) | Add analog risk scaling as a pre-cursor to full halt |
| 4 risk components, take minimum | 6 separate kill switches | Consider combining into a composite risk score |
| Smoothed EWMA position adjustment | Immediate halt + sells only | Add a "yellow zone" (multiplier 0.5) before red halt |
| Capped at dm_max=2.5 for FDM | No FDM, equal weighting | Implement FDM for strategy combination |

### Applicable Patterns for Track A/B Framework

1. **FDM for Track A:** Compute FDM across the 11 Track A strategies using rolling 6-month correlation of daily returns. Update monthly. Expected FDM for avg ρ=0.58 across 11 strategies ≈ 1.3–1.5 (matching our current rough estimate).

2. **IDM for Track B:** Separate IDM calculation using Track B's 5 higher-variance strategies. Apply IDM to the 30% capital allocation before sizing.

3. **Risk overlay for both tracks:** Replace the hard 15%/30% drawdown kill switches with a continuous risk multiplier that reduces positions by 50% at half the drawdown limit, then triggers full halt at the limit. Reduces whipsaw from brief drawdown breaches.

4. **Buffering:** `systems/buffering.py` implements a position buffer zone — only trade if the optimal position is outside a ±N% buffer around current position. This reduces unnecessary turnover at the margin. Applicable to our 2%/3% per-trade limits.

---

## Cross-Repo Synthesis

### Top 3 Immediate Actions

1. **Study HRP as strategy combination mechanism** (`machine-learning-for-trading/13_unsupervised_learning/04_hierarchical_risk_parity/01_hierarchical_risk_parity.ipynb`): HRP weights strategies by their hierarchical cluster structure rather than equal weight or Markowitz. For our correlated Family 1 cluster (10 credit strategies, avg ρ=0.58), HRP would naturally underweight them as a group relative to the independent SOXX-QQQ strategy. This directly addresses the ρ=0.58 problem identified in CLAUDE.md.

2. **Implement FDM using Carver's formula** (`pysystemtrade/sysquant/estimators/diversification_multipliers.py`): The `diversification_mult_single_period()` function is 15 lines of pure numpy. Port directly to `src/llm_quant/risk/` as `diversification.py`. Use it in the portfolio optimizer to replace naive equal weighting of the 16 active strategies.

3. **Use qlib's Rolling task generator as CPCV scaffold** (`qlib/workflow/task/gen.py`): `task_generator([config], [RollingGen(step=252, horizon=63)])` produces a list of walk-forward experiment configs. Adapt this pattern in `scripts/robustness.py` to replace the current manual CPCV loop. The key benefit: configs are serializable dicts, so each CPCV path is auditable and reproducible.

### Family Gap Priority (unchanged after review)

The repos confirm Families 2–7 are the right research targets. Specific notebook-to-family mapping:

| Family | Best Entry Point | Notebook |
|---|---|---|
| 2 (Microstructure) | No strong match; check mlfinlab | `mlfinlab/` — not yet reviewed |
| 3 (Volatility) | HRP + stochastic vol | `13_unsupervised_learning/04_hrp/`, `10_bayesian_ml/05_stochastic_volatility.ipynb` |
| 4 (Statistical Factor) | Conditional autoencoder | `20_autoencoders/05_conditional_autoencoder_for_asset_pricing_data.ipynb` |
| 5 (Momentum/Carry) | 101 Formulaic Alphas | `24_alpha_factor_library/03_101_formulaic_alphas.ipynb` |
| 6 (Alternative Data) | Earnings call LDA | `15_topic_modeling/06_lda_earnings_calls.ipynb`, `03_alternative_data/02_earnings_calls/` |
| 7 (Social/News Sentiment) | Twitter sentiment | `14_working_with_text_data/05_sentiment_analysis_twitter.ipynb` |
