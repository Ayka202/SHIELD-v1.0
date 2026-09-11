# shield

Core SHIELD package. Public entry points are `split()`, `sweep()`, and `multi_split()`, exported from `shield/api.py` and documented there with a short comment next to each input's default value.

| Module | Role |
|---|---|
| `api.py` | Public entry points and configuration objects (`ShieldConfig`, `ShieldResult`) |
| `pipeline.py` | Block-coordinate optimization loop |
| `coarsening.py` | Hierarchy-first / internal coarsening into super-nodes |
| `stratification.py` | Quantile, moment, and sliced-Wasserstein stratification |
| `rounding.py` | Relaxation rounding and repair passes |
| `diagnostics.py` | Leakage, stability, and Pareto-frontier diagnostics |
| `solvers/` | LP/MILP, graph-cut, and OT/MMD backends (see `solvers/README.md`) |

See the root [README](../README.md) for installation and usage.
