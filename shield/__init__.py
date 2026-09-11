"""
SHIELD: Structure-aware Hybrid Integrated Evaluation by Leakage-aware Data splitting.

Public surface:
    shield.split(...)        - run a single SHIELD partition (Algorithm 1).
    shield.sweep(...)        - run a coefficient sweep (Algorithm 3).
    shield.ShieldConfig          - declarative configuration object.
    shield.ShieldResult          - bundle returned by split().
"""

from shield.api import (
    split,
    sweep,
    ShieldConfig,
    ShieldResult,
    ShieldSweepResult,
)
from shield import (
    coarsening,
    stratification,
    rounding,
    pipeline,
    diagnostics,
)
from shield.solvers import distributional, graph_cut, gurobi_lp

__all__ = [
    "split",
    "sweep",
    "ShieldConfig",
    "ShieldResult",
    "ShieldSweepResult",
    "coarsening",
    "stratification",
    "rounding",
    "pipeline",
    "diagnostics",
    "distributional",
    "graph_cut",
    "gurobi_lp",
]

__version__ = "0.1.0"
