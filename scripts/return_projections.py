"""Portfolio return projections and capital requirements analysis."""

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent

slugs = [
    "lqd-spy-credit-lead",
    "agg-spy-credit-lead",
    "ief-qqq-rate-tech",
    "agg-qqq-credit-lead",
    "vcit-qqq-credit-lead",
    "lqd-qqq-credit-lead",
    "emb-spy-credit-lead",
    "tlt-qqq-rate-tech",
    "hyg-spy-5d-credit-lead",
    "hyg-qqq-credit-lead",
    "soxx-qqq-lead-lag",
    "agg-efa-credit-lead",
    "spy-overnight-momentum",
    "tlt-spy-rate-momentum",
]

base = ROOT / "data" / "strategies"
all_returns = {}

for slug in slugs:
    exp_dir = base / slug / "experiments"
    if not exp_dir.exists():
        continue
    exp_files = list(exp_dir.glob("*.yaml"))
    if not exp_files:
        continue
    with open(exp_files[0]) as f:
        exp = yaml.safe_load(f)
    daily_rets = exp.get("daily_returns", [])
    if daily_rets:
        all_returns[slug] = np.array(daily_rets)

min_len = min(len(r) for r in all_returns.values())
names = list(all_returns.keys())
R = np.array([all_returns[n][-min_len:] for n in names])
N, T = R.shape
n_years = T / 252


def portfolio_stats(weights, returns):
    port = (weights[:, None] * returns).sum(axis=0)
    cum = np.cumprod(1 + port)
    total_ret = cum[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(port)) - 1
    ann_vol = np.std(port) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    running_max = np.maximum.accumulate(cum)
    dd = abs(((cum - running_max) / running_max).min())
    downside = port[port < 0]
    ds_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1
    sortino = ann_ret / ds_vol
    return ann_ret, ann_vol, sharpe, dd, sortino


print("=" * 70)
print("THEORETICAL RETURN OPTIMIZATION")
print("=" * 70)
print(f"Strategies: {N}, Trading days: {T} ({n_years:.1f} years)")
print()

# 1. Equal Weight
ew_w = np.ones(N) / N
ew_ret, ew_vol, ew_sharpe, ew_dd, ew_sortino = portfolio_stats(ew_w, R)
print("--- 1. EQUAL WEIGHT (baseline) ---")
print(
    f"  Return: {ew_ret * 100:.1f}%   Vol: {ew_vol * 100:.1f}%   "
    f"Sharpe: {ew_sharpe:.2f}   MaxDD: {ew_dd * 100:.1f}%   Sortino: {ew_sortino:.2f}"
)
print()

# 2. Inverse Variance
vols = np.std(R, axis=1)
iv_w = (1 / vols**2) / (1 / vols**2).sum()
iv_ret, iv_vol, iv_sharpe, iv_dd, iv_sortino = portfolio_stats(iv_w, R)
print("--- 2. INVERSE VARIANCE (risk parity) ---")
print(
    f"  Return: {iv_ret * 100:.1f}%   Vol: {iv_vol * 100:.1f}%   "
    f"Sharpe: {iv_sharpe:.2f}   MaxDD: {iv_dd * 100:.1f}%   Sortino: {iv_sortino:.2f}"
)
top3 = sorted(zip(names, iv_w, strict=False), key=lambda x: -x[1])[:3]
for n, w in top3:
    print(f"    {n}: {w:.1%}")
print()

# 3. Max Sharpe (long-only tangency)
mu = R.mean(axis=1) * 252
Sigma = np.cov(R) * 252
try:
    Sigma_inv = np.linalg.inv(Sigma)
    raw_w = Sigma_inv @ mu
    tang_w = np.maximum(raw_w, 0)
    tang_w /= tang_w.sum()
    ms_ret, ms_vol, ms_sharpe, ms_dd, ms_sortino = portfolio_stats(tang_w, R)
    print("--- 3. MAX SHARPE (tangency, long-only) ---")
    print(
        f"  Return: {ms_ret * 100:.1f}%   Vol: {ms_vol * 100:.1f}%   "
        f"Sharpe: {ms_sharpe:.2f}   MaxDD: {ms_dd * 100:.1f}%   Sortino: {ms_sortino:.2f}"
    )
    top5 = sorted(zip(names, tang_w, strict=False), key=lambda x: -x[1])[:5]
    for n, w in top5:
        print(f"    {n}: {w:.1%}")
except Exception as e:
    print(f"  Max Sharpe failed: {e}")
    ms_ret, ms_vol, ms_sharpe, ms_dd = ew_ret, ew_vol, ew_sharpe, ew_dd
    tang_w = ew_w
print()

# 4. Kelly Criterion
best_ret = ms_ret
best_vol = ms_vol
full_kelly = best_ret / (best_vol**2)
half_kelly = full_kelly / 2

print("--- 4. KELLY CRITERION (optimal leverage) ---")
print(f"  Full Kelly: {full_kelly:.1f}x leverage (TOO AGGRESSIVE)")
print(f"  Half Kelly: {half_kelly:.1f}x leverage (standard practice)")
print(f"  Quarter Kelly: {full_kelly / 4:.1f}x leverage (conservative)")
print()

print("  Leveraged return projections:")
for lev, label in [
    (1.0, "No leverage"),
    (1.5, "1.5x margin"),
    (2.0, "2x margin"),
    (half_kelly, f"{half_kelly:.1f}x half Kelly"),
]:
    l_ret = best_ret * lev
    l_vol = best_vol * lev
    l_dd = min(ms_dd * lev, 0.99)
    print(
        f"    {label:<25}  Return: {l_ret * 100:>6.1f}%   "
        f"Vol: {l_vol * 100:>5.1f}%   MaxDD: ~{l_dd * 100:>5.1f}%"
    )
print()

# 5. Decorrelation future
corr = np.corrcoef(R)
avg_rho = (corr.sum() - N) / (N * (N - 1))
avg_sr_individual = np.mean(
    [R[i].mean() * np.sqrt(252) / np.std(R[i]) for i in range(N)]
)

print("--- 5. WHAT MORE STRATEGIES WOULD ADD ---")
print(f"  Current avg pairwise rho: {avg_rho:.3f}")
print(f"  Current avg individual Sharpe: {avg_sr_individual:.3f}")
print()

for extra, new_rho in [(5, 0.10), (10, 0.10), (20, 0.05)]:
    total = N + extra
    blended = (avg_rho * N * (N - 1) + new_rho * 2 * N * extra) / (total * (total - 1))
    proj_sr = avg_sr_individual * np.sqrt(total / (1 + (total - 1) * blended))
    proj_ret = proj_sr * best_vol
    print(
        f"  +{extra} uncorrelated strats (rho~{new_rho}): "
        f"SR -> {proj_sr:.2f}  return -> ~{proj_ret * 100:.1f}%"
    )
print()

# 6. Dollar amounts
print("=" * 70)
print("HOW MUCH CAPITAL DO YOU NEED?")
print("=" * 70)
print()

scenarios = [
    ("Conservative (1x, equal wt)", ew_ret, ew_dd),
    ("Optimized (1x, max Sharpe)", ms_ret, ms_dd),
    ("With 1.5x leverage", ms_ret * 1.5, min(ms_dd * 1.5, 0.99)),
]

for label, ret, dd in scenarios:
    print(f"  === {label} ===")
    print(f"  Annual return: {ret * 100:.1f}%   Worst drawdown: {dd * 100:.1f}%")
    print()
    for monthly in [500, 1000, 2000, 5000]:
        annual = monthly * 12
        capital = annual / ret if ret > 0 else float("inf")
        worst = capital * dd
        print(
            f"    ${monthly:>5,}/mo needs ${capital:>10,.0f}   "
            f"(worst loss: -${worst:>8,.0f})"
        )
    print()

print("=" * 70)
print("COMPOUND GROWTH TABLE")
print("=" * 70)
print()
for capital_start in [1000, 10000, 50000, 100000]:
    print(f"  Starting with ${capital_start:,}:")
    print(f"  {'':4} {'Conservative':>14} {'Optimized':>14} {'1.5x Lever':>14}")
    for yr in [1, 3, 5, 10, 20]:
        vals = []
        for _, ret, _ in scenarios:
            vals.append(capital_start * (1 + ret) ** yr)
        print(
            f"    Yr {yr:<3} ${vals[0]:>12,.0f}  ${vals[1]:>12,.0f}  ${vals[2]:>12,.0f}"
        )
    print()

print("=" * 70)
print("REALITY CHECK")
print("=" * 70)
print()
print("  Renaissance Medallion (best fund ever):  ~66% gross, ~39% net")
print("  Top quant hedge funds (D.E. Shaw, Two Sigma): 15-25% net")
print("  Good systematic retail trader:  8-15% net")
print("  Index fund (SPY buy-and-hold):  ~10% long-term avg")
print()
print(f"  Our backtest (equal weight):   {ew_ret * 100:.1f}%")
print(f"  Our backtest (optimized):      {ms_ret * 100:.1f}%")
print(f"  Our backtest (1.5x leverage):  {ms_ret * 150:.1f}%")
print()
print("  WHERE WE SIT: solidly in 'good systematic retail' range")
print("  The edge is NOT raw returns -- it is CONSISTENCY (low drawdown)")
print()
print("  HONEST ADJUSTMENT FOR LIVE TRADING:")
print("  - Backtest-to-live decay: typically 20-40%")
print(f"  - Realistic live return estimate: {ms_ret * 60:.1f}%-{ms_ret * 80:.1f}%")
print(
    f"  - Realistic live Sharpe estimate: {ms_sharpe * 0.6:.2f}-{ms_sharpe * 0.8:.2f}"
)
print()
print("  THE MATH TRUTH:")
print("  - You cannot compound to wealth at 8% on small capital")
print("  - $10k at 8% = $800/yr = $67/mo -- not life-changing")
print("  - $100k at 8% = $8k/yr = $667/mo -- noticeable")
print("  - $500k at 8% = $40k/yr -- meaningful supplemental income")
print("  - $1M+ at 8% = $80k/yr -- near full income replacement")
print("  - LEVERAGE helps but increases drawdown risk proportionally")
