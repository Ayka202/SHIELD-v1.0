# shield.solvers

Numerical backends used by the SHIELD pipeline.

| Module | Role |
|---|---|
| `gurobi_lp.py` | LP/MILP relaxation of the block-coordinate objective (requires a Gurobi license) |
| `graph_cut.py` | Graph-cut backend for the M-term (feature-manifold connectivity) |
| `distributional.py` | OT/MMD backend for the D-term (distributional matching) |

See the root [README](../../README.md) for installation and usage.
