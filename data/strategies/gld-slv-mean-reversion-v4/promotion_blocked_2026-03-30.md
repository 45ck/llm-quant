# GLD-SLV v4 Promotion Blocked — 2026-03-30

## Fraud Detector Results: FAIL

### Test 1: Shuffled Signal Test — FAIL
- Real Sharpe: 0.948
- Shuffled 95th percentile Sharpe: ~1.0+
- p-value: 0.274 (need < 0.05)
- Interpretation: Strategy does not significantly outperform random signal assignment

### Test 2: Mechanism Inversion Test — FAIL
- Original strategy SR: 0.948
- Inverted strategy SR: 1.075
- Genuine mean-reversion alpha should DEGRADE when inverted (lower SR)
- Inverted strategy outperforming = signal captures directional momentum or commodity beta, not pair spread mean-reversion

### Test 3: Permutation Test — PASS
- Returns not explained by simple permutation of dates

## Root Cause Hypothesis
The spread signal (z-score of GLD/SLV price ratio) likely correlates with:
- Gold/silver ratio momentum (directional commodity move)
- Macro risk-on/risk-off regime (both metals as safe haven)
- Not genuine mean-reversion of the pair spread

## Required Before Re-opening
1. Decompose signal into: (a) pure spread mean-reversion component, (b) commodity beta component
2. Remove directional beta: orthogonalize signal against GLD and SLV individual momentum
3. Re-run fraud detectors on beta-neutralized signal
4. Alternatively: reframe as a momentum/ratio strategy (different hypothesis, different lifecycle)

## Excellent Diversification Characteristics (Preserve)
- Avg portfolio ρ = 0.051 (lowest in portfolio)
- This asset pair IS worth keeping in the universe
- Just need a valid signal mechanism
