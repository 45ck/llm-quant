# Trading Decision Request — {{ date }}

## Current Portfolio State
- **NAV**: ${{ "%.2f"|format(nav) }}
- **Cash**: ${{ "%.2f"|format(cash) }} ({{ "%.1f"|format(cash_pct) }}%)
- **Gross Exposure**: {{ "%.1f"|format(gross_exposure_pct) }}%
- **Net Exposure**: {{ "%.1f"|format(net_exposure_pct) }}%
- **Positions**: {{ positions|length }}

{% if positions %}
### Current Positions
| Symbol | Shares | Avg Cost | Current | P&L % | Weight | Stop Loss |
|--------|--------|----------|---------|-------|--------|-----------|
{% for p in positions %}
| {{ p.symbol }} | {{ p.shares }} | ${{ "%.2f"|format(p.avg_cost) }} | ${{ "%.2f"|format(p.current_price) }} | {{ "%.1f"|format(p.pnl_pct) }}% | {{ "%.1f"|format(p.weight_pct) }}% | ${{ "%.2f"|format(p.stop_loss) }} |
{% endfor %}
{% endif %}

## Market Data (Top 30 ETFs, sorted by 20-day momentum)
| Symbol | Close | Chg% | SMA20 | SMA50 | RSI14 | MACD | ATR14 | Vol |
|--------|-------|------|-------|-------|-------|------|-------|-----|
{% for m in market_data %}
| {{ m.symbol }} | ${{ "%.2f"|format(m.close) }} | {{ "%.1f"|format(m.change_pct) }}% | ${{ "%.2f"|format(m.sma_20) }} | ${{ "%.2f"|format(m.sma_50) }} | {{ "%.1f"|format(m.rsi_14) }} | {{ "%.3f"|format(m.macd) }} | {{ "%.2f"|format(m.atr_14) }} | {{ m.volume }} |
{% endfor %}

## Macro Indicators
- **VIX**: {{ "%.2f"|format(vix) }} ({{ "%.1f"|format(vix_percentile_126d) }}th percentile of last 126 days)
- **VIX Regime**: {{ market_regime }} (thresholds: {{ "%.1f"|format(vix_regime_thresholds[0]) }} / {{ "%.1f"|format(vix_regime_thresholds[1]) }})
- **10Y-2Y Spread**: {{ "%.2f"|format(yield_spread) }} bps
- **SPY 50/200 SMA**: {{ spy_trend }}
{% if credit_spread_oas is not none %}
- **Credit OAS (BAMLC0A0CM)**: {{ "%.2f"|format(credit_spread_oas) }} bps (z-score: {{ "%.2f"|format(credit_spread_zscore) if credit_spread_zscore is not none else "N/A" }}){% if silent_stress %} ⚠️ **SILENT STRESS**: credit spread elevated while VIX is low — hidden risk building{% endif %}
{% else %}
- **Credit OAS**: N/A (FRED data not loaded)
{% endif %}

{% if governance is defined and governance %}
## Governance Status
- **Overall**: {{ governance.overall_severity | upper }}
- **Checks**: {{ governance.total_checks }} total, {{ governance.halts }} halt(s), {{ governance.warnings }} warning(s)
{% if governance.overall_severity == "halt" %}

**TRADING RESTRICTED**: Kill switch triggered. Only SELL/CLOSE actions permitted.
{% for h in governance.halt_details %}
- {{ h.detector }}: {{ h.message }}
{% endfor %}
{% elif governance.overall_severity == "warning" %}

**CAUTION**: Governance warnings active. Consider reducing position sizes.
{% for w in governance.warning_details %}
- {{ w.detector }}: {{ w.message }}
{% endfor %}
{% endif %}
{% endif %}

## Instructions
Analyze the data above and provide your trading decisions as JSON following the system prompt format. Consider:
1. Current market regime and any regime shifts (use adaptive VIX thresholds above, not fixed 20/25)
2. Sector rotation signals from momentum and RSI
3. Existing position management (stop-loss triggers, profit-taking)
4. New opportunities aligned with the regime
5. All hard constraints from your mandate
6. Governance status — respect halt restrictions, note warnings in analysis
7. **VIX percentile sizing rule**: when VIX percentile > 80, scale down target position sizes by 50% to reduce vol exposure
8. **Silent stress alert**: when silent_stress=True, treat as risk_off even if VIX appears benign — credit markets are leading equity stress indicators
