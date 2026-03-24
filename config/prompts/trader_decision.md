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
- **VIX**: {{ "%.2f"|format(vix) }}
- **10Y-2Y Spread**: {{ "%.2f"|format(yield_spread) }} bps
- **SPY 50/200 SMA**: {{ spy_trend }}

## Instructions
Analyze the data above and provide your trading decisions as JSON following the system prompt format. Consider:
1. Current market regime and any regime shifts
2. Sector rotation signals from momentum and RSI
3. Existing position management (stop-loss triggers, profit-taking)
4. New opportunities aligned with the regime
5. All hard constraints from your mandate
