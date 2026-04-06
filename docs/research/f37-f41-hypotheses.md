# F37-F41: Novel Mechanism Family Hypotheses (Batch 11)

Generated 2026-04-06. These five hypotheses target mechanisms that are
structurally different from the 20 passing families and 12 failed families.

## Design Constraints

1. Must avoid ratio momentum on cross-asset pairs (dominant signal type in F1-F22, F26)
2. Must avoid commodity/macro/inflation space (saturated, rho=0.75-0.85 cluster)
3. Must avoid credit-equity lead-lag (F1 saturated with 10 strategies)
4. Must use ETFs available on Yahoo Finance (no single stocks, no direct futures)
5. Short-horizon signals (5-20 day lookback) -- long lookback (252d) empirically fails
6. Target: Sharpe >= 0.80, MaxDD < 15%, DSR >= 0.95

## Why These Five

The existing 20 passing families overwhelmingly exploit two signal types:
**ratio momentum** (F1, F6, F8, F9, F11-F19, F21-F22, F26) and **absolute
momentum** (F3, F5). Only F2 (mean reversion), F7 (event-driven), and F13
(realized vol comparison) use fundamentally different signal construction.

These five hypotheses deliberately use signal constructions that have NO overlap
with ratio momentum or absolute momentum:

- F37: **Volume-price divergence** (volume confirms/denies price moves)
- F38: **Cross-asset volatility lead-lag** (vol transmission, not price)
- F39: **Equity-bond correlation regime change** (correlation dynamics, not levels)
- F40: **Metals-miners production leverage** (real-economy operating leverage)
- F41: **FX carry as equity risk signal** (currency risk premium channel)

---

## F37: Volume-Price Divergence Regime

**Mechanism**: When price rises on declining volume (or vice versa), the move
lacks conviction and is more likely to reverse. This is a classic Wyckoff/Dow
Theory principle grounded in information economics: volume represents the
intensity of informed trading. Price moves on thin volume reflect noise; price
moves on heavy volume reflect information. Applied systematically at the ETF
level, this detects when broad equity rallies are "hollow" (distribution phase)
versus supported by genuine accumulation. The signal exploits the lag between
price (which moves first on marginal trades) and volume (which confirms only
when participation broadens). This is mechanistically distinct from all
existing families because it uses volume data, which no current strategy
touches.

**Signal**: Compute a 10-day price-volume divergence score on SPY:

```
price_momentum = (close / close.shift(10)) - 1
volume_momentum = (volume.rolling(10).mean() / volume.rolling(40).mean()) - 1
divergence = price_momentum - volume_momentum
divergence_zscore = (divergence - divergence.rolling(60).mean()) / divergence.rolling(60).std()
```

When `divergence_zscore > 1.5` (price up, volume weak = hollow rally): reduce
equity, increase defensive. When `divergence_zscore < -1.5` (price down, volume
weak = exhausted selling): increase equity. Neutral otherwise.

Key design: 10-day price window, 10/40-day volume ratio (avoids calendar
effects in raw volume), 60-day z-score normalization.

**ETFs**:
- Signal source: SPY (volume + price)
- Long allocation: SPY 80%
- Defensive allocation: GLD 40%, SHY 40%, TLT 10%
- Neutral: SPY 50%, SHY 40%

**Expected Sharpe**: 0.80-1.10. Volume-price divergence has been validated in
academic literature (Blume, Easley, O'Hara 1994; Llorente et al. 2002) with
Sharpe ratios in the 0.6-1.0 range for daily frequency. The z-score
normalization should improve stability vs raw divergence.

**Orthogonality**: No existing family uses volume data. All 20 passing families
operate exclusively on close prices (and occasionally high/low for F34). Volume
is an independent information channel. Expected correlation to existing
portfolio: rho < 0.15 (volume dynamics have near-zero correlation with price
ratio momentum across different asset classes).

**Risk**: Primary failure mode is ETF volume reflecting passive flow (creation/
redemption) rather than informed trading. SPY is the most heavily traded ETF
with deep institutional participation, which mitigates this concern, but the
signal-to-noise ratio may be lower than for individual equities. Secondary
risk: volume data quality issues (stock splits, distribution dates cause volume
spikes that are not signal).

---

## F38: Cross-Asset Volatility Transmission

**Mechanism**: Realized volatility transmits across asset classes with a
measurable lag. When bond volatility (TLT realized vol) spikes before equity
volatility (SPY realized vol), it signals an incoming risk-off event that
has not yet been priced into equities. This is the volatility analogue of
credit-equity lead-lag (F1), but operates on a completely different data
channel. The economic rationale: fixed income markets are dominated by
institutional investors and central bank watchers who process macro information
faster than equity markets. The vol channel captures information flow speed
differences that price-level lead-lag may miss, particularly during regime
transitions when correlations are unstable.

**Signal**: Compute a vol-spread momentum score:

```
spy_vol = spy_close.pct_change().rolling(10).std() * sqrt(252)
tlt_vol = tlt_close.pct_change().rolling(10).std() * sqrt(252)
vol_spread = tlt_vol - spy_vol
vol_spread_momentum = vol_spread - vol_spread.shift(5)
vol_signal = vol_spread_momentum / vol_spread.rolling(60).std()
```

When `vol_signal > 1.0` (bond vol rising faster than equity vol = stress
incoming): shift to defensive. When `vol_signal < -1.0` (bond vol falling
faster = stress receding): shift to risk-on. The 5-day momentum on the spread
captures the DIRECTION of vol transmission, not the level -- avoiding the
stale-regime problem that killed F4 and F29.

**ETFs**:
- Signal source: SPY, TLT (realized vol computed from close prices)
- Risk-on: SPY 80%
- Defensive: GLD 40%, SHY 40%, IEF 10%
- Neutral: SPY 50%, GLD 20%, SHY 20%

**Expected Sharpe**: 0.85-1.15. The mechanism is well-documented (Diebold &
Yilmaz 2012, volatility spillover literature). The 5-day momentum on vol-spread
is the critical innovation vs F4 (which used VIX levels) and F29 (which used
wrong proxy). Using realized vol from close prices avoids the options-implied
data dependency.

**Orthogonality**: F13 (vol regime) compares SPY vs GLD realized vol levels.
F38 uses SPY vs TLT vol MOMENTUM (rate of change of vol spread). F4 used VIX
levels. F29 used bond vol levels. The key difference: F38 is a momentum signal
on the vol spread, not a level signal. The information content (direction of
vol transmission between bonds and equities) has no equivalent in the existing
families. Expected correlation to F13: rho < 0.25 (different assets, different
signal construction). Expected correlation to portfolio: rho < 0.20.

**Risk**: Primary failure mode is that SPY and TLT realized vol are too noisy
at 10-day windows, producing false signals. The 60-day normalization should
help, but short-lived vol spikes (e.g., options expiration) could pollute the
signal. Secondary risk: during crisis periods, all vol rises simultaneously
(no lead-lag), making the spread momentum uninformative precisely when it
matters most.

---

## F39: Equity-Bond Correlation Regime Change

**Mechanism**: The correlation between stocks and bonds is not constant -- it
shifts between positive (both move together, as in 2022 when rates drove
everything) and negative (flight-to-quality, as in 2008 and traditional risk-
off). The RATE OF CHANGE of this correlation is a powerful regime signal.
When stock-bond correlation is INCREASING (moving from negative toward
positive), the diversification value of bonds is eroding and portfolios become
fragile -- this is a "correlation crisis" regime. When correlation is
DECREASING (moving from positive toward negative), flight-to-quality dynamics
are strengthening -- bonds are becoming better hedges. This is mechanistically
different from F6 (rate momentum, which uses TLT price direction) and F14
(curve momentum, which uses TLT/SHY ratio). F39 uses the SECOND-ORDER
relationship between SPY and TLT, not their individual price dynamics.

**Signal**: Compute rolling correlation momentum:

```
spy_ret = spy_close.pct_change()
tlt_ret = tlt_close.pct_change()
corr_20 = spy_ret.rolling(20).corr(tlt_ret)
corr_60 = spy_ret.rolling(60).corr(tlt_ret)
corr_momentum = corr_20 - corr_60
corr_signal = sign(corr_momentum) * min(abs(corr_momentum) / 0.20, 1.0)
```

When `corr_momentum > 0.10` (correlation rising = diversification eroding):
risk-off with gold emphasis. When `corr_momentum < -0.10` (correlation falling
= flight-to-quality strengthening): risk-on. The 20d vs 60d comparison captures
the ACCELERATION of correlation shifts, avoiding the noise of raw rolling
correlation.

**ETFs**:
- Signal source: SPY, TLT (daily returns, rolling correlation)
- Decorrelating regime (corr falling): SPY 75%, TLT 15%
- Correlation crisis (corr rising): GLD 50%, SHY 30%, DBA 10%
- Neutral: SPY 45%, GLD 25%, SHY 20%

**Expected Sharpe**: 0.85-1.20. The stock-bond correlation regime is one of the
most studied macro relationships (Campbell, Sunderam, Viceira 2017). The
momentum-of-correlation construction is less explored than level-based
approaches, which is why it may retain more alpha. Similar to the mechanism
that produced the batch-3 near-miss (SPY-TLT-corr-sign-change, Sharpe=0.555)
but with a fundamentally different signal construction: momentum of correlation
instead of sign change. The near-miss failed because sign-change is binary and
rare; momentum is continuous and frequent.

**Orthogonality**: The existing F6 (rate momentum) uses TLT price direction.
F14 uses TLT/SHY ratio momentum. F15 uses TIP/TLT ratio momentum. All are
FIRST-ORDER price signals. F39 uses SECOND-ORDER interaction between SPY and
TLT returns -- the correlation is a fundamentally different mathematical
object. The near-miss M3 (SPY-TLT-corr-sign-change) was related but used
binary sign change, not momentum. Expected correlation to portfolio: rho < 0.15
(second-order signals have inherently low correlation with first-order signals).

**Risk**: Primary failure mode is that rolling correlation is extremely noisy
at 20-day windows, especially for daily returns. The 20d vs 60d comparison
helps, but may still produce whipsaw in calm markets where correlation
fluctuates randomly around zero. Secondary risk: the "correlation crisis"
regime (rising correlation) was dominant in 2022 -- if the strategy is
primarily profiting from that one episode, it will fail the regime-split
fraud detector.

---

## F40: Metals-Miners Operating Leverage

**Mechanism**: Gold miners (GDX) have approximately 2-3x operating leverage to
gold prices (GLD) because their costs are relatively fixed (labor, energy,
equipment) while revenue scales with gold price. When gold rises, miners
should rise faster. When this relationship BREAKS -- i.e., miners
underperform gold despite gold rising -- it signals either (a) rising input
costs (energy, labor) that signal broader inflation, or (b) equity market
stress pulling miners down alongside equities despite gold strength. In both
cases, the divergence carries information about the macro regime that pure
gold or pure equity signals miss. This is a LEAD-LAG signal in the same
family as F8 (SOXX-QQQ) but in a completely different asset class (precious
metals vs semiconductors) with a different economic mechanism (operating
leverage vs supply chain).

**Signal**: Compute the miners-metal divergence:

```
gold_ret_10d = gld_close.pct_change(10)
miners_ret_10d = gdx_close.pct_change(10)
expected_miners_ret = gold_ret_10d * 2.0  # approximate leverage ratio
divergence = miners_ret_10d - expected_miners_ret
div_zscore = (divergence - divergence.rolling(60).mean()) / divergence.rolling(60).std()
```

When `div_zscore > 1.0` (miners outperforming gold's leverage = strong risk
appetite, low input costs): favor equities. When `div_zscore < -1.0` (miners
lagging gold = input cost stress or equity contagion): favor gold + defensives.
The leverage ratio (2.0) is a stylized fact, not a fitted parameter --
empirical GDX/GLD beta ranges 1.5-3.0.

**ETFs**:
- Signal source: GDX (VanEck Gold Miners), GLD (Gold)
- Note: GDX is not in the current universe.toml -- would need to be added
  for fetching, but is liquid and available on Yahoo Finance
- Risk-on (miners outperforming): SPY 70%, GDX 15%
- Stress (miners underperforming): GLD 50%, SHY 30%, TIP 10%
- Neutral: SPY 40%, GLD 25%, SHY 25%

**Expected Sharpe**: 0.80-1.05. The gold-miner leverage relationship is one of
the most robust in commodity equities (Tufano 1998, Naylor et al. 2011). The
divergence signal has been shown to predict both gold returns and equity risk
appetite with Sharpe 0.6-0.9 in academic backtests. Using z-scored divergence
rather than raw difference should improve stability.

**Orthogonality**: F2 (GLD-SLV mean reversion) uses the gold-silver ratio.
F26 (dollar-gold regime) uses UUP/GLD ratio momentum. F40 uses the relationship
between gold and its equity derivatives (miners), which is a fundamentally
different economic channel -- operating leverage rather than monetary/inflation
dynamics. GDX correlates ~0.65 with GLD and ~0.45 with SPY, but the
DIVERGENCE signal (miners minus expected gold leverage) is decorrelated from
both. Expected correlation to existing portfolio: rho < 0.20.

**Risk**: Primary failure mode is that the 2.0x leverage ratio is not constant
-- it varies with gold price level, miner cost structures, and M&A activity.
A time-varying leverage estimate would fix this but adds a parameter to fit.
Secondary risk: GDX has higher idiosyncratic risk (Newmont, Barrick company-
specific events) that can produce false divergence signals. Tertiary risk:
GDX/GLD divergence was strongly positive during the 2020 gold rally and
strongly negative during the 2022 equity selloff -- the strategy may
overweight these two episodes.

---

## F41: FX Carry Risk Appetite Signal

**Mechanism**: High-carry currencies (AUD, commodity-linked) depreciate sharply
during risk-off events because carry trades unwind. Low-carry currencies (JPY,
CHF) appreciate as safe havens. The relative performance of carry vs safe-haven
currencies is a real-time measure of global risk appetite that is INDEPENDENT
of equity market signals -- FX markets trade 24 hours and incorporate
information from Asian and European sessions before US equity markets open.
This creates a genuine information lead. The signal: when AUD/JPY (the
canonical carry pair) is strengthening, global risk appetite is healthy. When
AUD/JPY is weakening, risk appetite is deteriorating. This is different from
F5 (overnight momentum, which uses SPY overnight gaps) and F17 (global yield
flow, which uses TLT/EFA ratio). F41 uses the FX market's risk appetite signal,
which reflects Asian/European macro conditions that US-centric signals miss.

**Signal**: Construct a synthetic FX carry signal from available pairs:

```
# AUD/JPY = AUDUSD / USDJPY (constructed from available pairs)
audjpy = audusd_close * usdjpy_close  # Note: AUDUSD=X * USDJPY=X gives AUD/JPY
carry_momentum_5d = audjpy.pct_change(5)
carry_momentum_20d = audjpy.pct_change(20)
carry_signal = carry_momentum_5d - carry_momentum_20d * 0.25  # fast vs slow
carry_zscore = (carry_signal - carry_signal.rolling(60).mean()) / carry_signal.rolling(60).std()
```

When `carry_zscore > 1.0` (carry strengthening = risk appetite expanding):
favor equities + EM. When `carry_zscore < -1.0` (carry unwinding = risk
appetite contracting): favor safe havens. The fast-slow differential (5d minus
scaled 20d) captures the ACCELERATION of carry flow, not the level.

**ETFs**:
- Signal source: AUDUSD=X, USDJPY=X (both in universe.toml, used to
  construct synthetic AUD/JPY)
- Risk-on (carry expanding): SPY 60%, EEM 15%, EFA 10%
- Risk-off (carry unwinding): GLD 40%, TLT 30%, SHY 20%
- Neutral: SPY 45%, GLD 20%, SHY 25%

**Expected Sharpe**: 0.85-1.10. FX carry as a risk signal is well-established
(Lustig, Roussanov, Verdelhan 2011; Menkhoff et al. 2012). AUD/JPY is the
single most studied carry pair and predicts equity risk appetite with
correlation 0.5-0.7. Using momentum-of-carry (acceleration) rather than carry
level should produce less crowded, more timely signals. The 24-hour FX market
provides genuine information lead over US equity hours.

**Orthogonality**: No existing family uses FX data as a signal source. All 20
passing families use US ETF prices, credit spreads, or volatility indices. FX
carry is driven by monetary policy differentials and global capital flows --
a completely separate information channel from US cross-asset ratios. Expected
correlation to existing portfolio: rho < 0.10 (FX carry dynamics have near-
zero correlation with US ETF ratio momentum in normal markets).

**Risk**: Primary failure mode is that AUD/JPY constructed from daily close
of AUDUSD=X and USDJPY=X loses the intraday information advantage. FX data
on Yahoo Finance uses 4PM ET snapshots, so the Asian/European session signal
is stale by the time US ETFs trade. The 5-day momentum window partially
addresses this but cannot fully capture the lead. Secondary risk: the FX
carry trade has experienced several "flash crash" unwinds (2015 CHF, 2019 JPY)
that produce extreme tail losses in the signal itself, potentially causing the
strategy to flip at the worst moment. Tertiary risk: correlation between FX
carry and equity risk appetite increases during crises (Brunnermeier, Nagel,
Pedersen 2009), reducing the diversification benefit precisely when it is most
needed.

---

## Testing Priority

| Family | Name | Expected Sharpe | Orthogonality (est. rho) | p(pass gates) | Priority |
|--------|------|----------------|-------------------------|---------------|----------|
| F41 | FX Carry Risk Appetite | 0.85-1.10 | < 0.10 | 0.40 | 1 |
| F39 | Equity-Bond Corr Regime | 0.85-1.20 | < 0.15 | 0.35 | 2 |
| F37 | Volume-Price Divergence | 0.80-1.10 | < 0.15 | 0.35 | 3 |
| F38 | Cross-Asset Vol Transmission | 0.85-1.15 | < 0.20 | 0.30 | 4 |
| F40 | Metals-Miners Op. Leverage | 0.80-1.05 | < 0.20 | 0.30 | 5 |

**Rationale for ordering**:

1. **F41** gets highest priority because FX data is a COMPLETELY unused
   information channel in the portfolio. If the signal works, it is almost
   guaranteed to be a new cluster representative (rho < 0.10). Both FX pairs
   are already in universe.toml.

2. **F39** is second because the correlation-momentum construction is genuinely
   novel (second-order), and the batch-3 near-miss (M3, Sharpe=0.555) suggests
   there IS a signal in SPY-TLT correlation dynamics -- it just needs a better
   signal construction than binary sign change.

3. **F37** is third because volume data is completely untapped in the portfolio.
   Lower priority than F41/F39 because SPY volume may be dominated by passive
   flow rather than informed trading.

4. **F38** is fourth because it rehabilitates the volatility channel (F4 failed,
   F29 failed) with a better construction (momentum of vol spread vs level).
   But two prior failures in the vol space lower the prior.

5. **F40** is last because it requires adding GDX to the universe (minor
   friction) and the precious metals space is partially covered by F2 (GLD-SLV)
   and F26 (UUP-GLD). The operating leverage mechanism IS genuinely different,
   but the overlap risk is highest.

## Data Requirements

| Family | New symbols needed | Already in universe? |
|--------|-------------------|---------------------|
| F37 | None | Yes (SPY volume data from OHLCV) |
| F38 | None | Yes (SPY, TLT close prices) |
| F39 | None | Yes (SPY, TLT close prices) |
| F40 | GDX | No -- add to universe.toml |
| F41 | None | Yes (AUDUSD=X, USDJPY=X) |

## Key Differences from Existing Families

- **No ratio momentum**: None of F37-F41 use the `ratio.pct_change(N)` pattern
  that dominates F1, F6, F8-F9, F11-F19, F21-F22, F26
- **No credit-equity**: None involve LQD, AGG, HYG, VCIT, or EMB
- **No commodity/macro**: None add to the saturated F11-F22 cluster
- **Three novel data channels**: volume (F37), FX carry (F41), and
  correlation dynamics (F39) -- none used by any existing family
- **Two rehabilitated channels**: volatility transmission (F38, improving on
  F4/F29) and precious metals leverage (F40, extending F2/F26)
- **Signal construction diversity**: z-scored divergence (F37, F40),
  vol-spread momentum (F38), correlation momentum (F39), carry acceleration
  (F41) -- all different from each other and from ratio momentum
