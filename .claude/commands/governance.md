# /governance — Production Surveillance Scan

Run a full governance surveillance scan and display results. This checks all 7 detectors against the portfolio and market data.

## Run the Scan

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/run_surveillance.py
```

## Parse and Display Results

The script outputs JSON with:
- `overall_severity`: "ok", "warning", or "halt"
- `total_checks`: number of individual checks run
- `halts`: count of halt-level findings
- `warnings`: count of warning-level findings
- `checks`: array of individual check results

Display the results as a structured report:

### Governance Report — [date]

**Overall Status**: [OK / WARNING / HALT]

**Detectors**:
| Detector | Status | Finding |
|----------|--------|---------|
| ... | ... | ... |

**Kill Switches**: [all clear / triggered — list which ones]

### If HALT Status
Explain which kill switches or detectors triggered the halt, what the current values are vs thresholds, and what actions are needed to resolve. Trading is restricted to SELL/CLOSE only until the halt condition clears.

### If WARNING Status
List the warnings and their implications. Recommend caution — reduced position sizes, avoiding new entries in affected areas.

### Reference
- Full control matrix: `docs/governance/control-matrix.md`
- Model promotion policy: `docs/governance/model-promotion-policy.md`
- Governance config: `config/governance.toml`
