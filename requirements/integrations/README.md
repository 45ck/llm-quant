# Research integration profiles

Use a **separate virtual environment** for each optional backend, especially Qlib.
The files here are reviewed compatibility bounds, not a fully resolved dependency
lock and not a claim that every version within a range has been tested.
`../research-py313.lock` is the earlier exact executed core environment, not an
all-framework environment. Do not use optional installs to mutate that reference.

```sh
python -m venv .venv-integrations
# Activate the environment using the appropriate command for your platform.
python -m pip install -r requirements/integrations/arch.txt
python -m pip freeze --all > research-local/integration-environment.txt
python -m pip check
```

Create `research-local/` before redirecting output. Optional examples also need
NumPy, pandas, SciPy and scikit-learn from the core profile. A clean installation
and a successful native test are required before claiming runtime compatibility.
No install runs automatically at Python import or skill discovery time.

`backtesting.txt` is an **AGPL opt-in**. Other packages retain their upstream
licenses as recorded in the source catalogue; transitive dependencies have their
own terms. No dependency is relicensed by the repository's MIT code license.
