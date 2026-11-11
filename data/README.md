# Data boundary

[Repository map](../docs/repository-map.md) · [Research guide](../docs/contributing/research.md)

`data/strategies/` contains existing strategy lifecycle records. `data/arb/` contains
existing arbitrage research artifacts. These paths are preserved for compatibility;
the folder name does not grant permission to publish raw market data.

Put new downloaded histories, source captures, local databases, credentials and
private project inputs in ignored local storage, normally `research-local/`.
Review staged changes before pushing: existing legacy paths and ignore patterns
must not be assumed safe merely because they are under `data/`.

A contribution should disclose source permissions, observation and release dates,
vintage, units, calendar, currency, return basis, hashes and missing-value treatment.
Use permitted or synthetic fixtures for public tests. Publishing the code under MIT
does not relicense a vendor dataset. Never upload private client or account records.
