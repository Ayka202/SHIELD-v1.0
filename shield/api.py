"""
User-facing API for SHIELD.

Two entry points:
    * shield.split(...)  - run a single SHIELD partition 
    * shield.sweep(...)  - run a (alpha, beta, gamma) coefficient sweep
                            with Pareto frontier + Hamming stability
                           

Both functions accept torch tensors or numpy arrays for every relational
object. Device placement (`device="cuda" | "cpu"`) and worker count
(`nodes=-1` for all-cores) are honoured globally.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from shield import diagnostics, pipeline, stratification as strat_mod
from shield.coarsening import (
    configure_threading,
    resolve_device,
)
from shield.solvers import distributional, graph_cut, gurobi_lp


# ---------------------------------------------------------------------------
# Type aliases / converters
# ---------------------------------------------------------------------------

ArrayLike = Union[torch.Tensor, np.ndarray, list]


def _to_tensor(
    x: Optional[ArrayLike],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device).to(dtype)
    try:
        import scipy.sparse as _sps
        if _sps.issparse(x):
            x = x.toarray()
    except ImportError:
        pass
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _to_long(x: Optional[ArrayLike], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        t = x.to(device)
        if t.is_floating_point() and not torch.all(t == t.floor()):
            raise ValueError(
                "class_labels / entity_of contains non-integer float values "
                "(e.g. 0.9 or 1.1). Refusing to truncate; cast to an integer "
                "dtype (torch.long / int64) before passing in."
            )
        return t.to(torch.long)
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating) and not np.all(arr == np.floor(arr)):
        raise ValueError(
            "class_labels / entity_of contains non-integer float values. "
            "Refusing to truncate; cast to an integer dtype before passing in."
        )
    return torch.as_tensor(arr, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Public configuration container
# ---------------------------------------------------------------------------

@dataclass
class ShieldConfig:
    # mandatory dimensions
    K: int = 3                             # number of folds
    r: Optional[ArrayLike] = None          # target fold proportions, shape (K,)
    # term mode (toggles are derived from which inputs are supplied, not stored here)
    d_mode: str = "match"                  # "match" | "shift"
    # composite-objective coefficients
    alpha: float = 1.0                     # weight on H (hierarchy) term
    beta: float = 1.0                      # weight on M (graph-cut) term
    gamma: float = 1.0                     # weight on D (distributional) term
    eta: float = 1.0                       # weight on the fold-balance penalty
    mu: float = 1.0                        # weight on the stratification term
    # -- under development, excluded from the manuscript; ships with SHIELD v1.1 --
    nu: float = 0.0                        # label-augmentation objective weight
    tau: float = 0.0                       # label-augmentation scale
    label_augmentation_justification: Optional[str] = None
    label_augmentation_override_target_derivation: bool = False
    # solver parameters
    max_iter: int = 25                     # outer block-coordinate iterations
    tolerance: float = 1e-3
    epsilon_OT: float = 5e-2               # Sinkhorn entropic regularizer
    sinkhorn_max_iter: int = 200
    seed: int = 0
    device: str = "cpu"                    # "cpu" | "cuda"
    nodes: int = -1                        # CPU worker count, -1 = all cores
    integer_z_final: bool = True           # allow last-resort MILP polish
    balance_eps: float = 0.05              # fold-balance tolerance
    output_flag: int = 0                   # Gurobi log verbosity
    enable_stage0: bool = True             # Monte-Carlo term calibration
    stage0_samples: int = 100
    # initialization / rounding controls
    init_backend: str = "auto"             # "auto" | "metis" | "kahip" | "spectral"
    rounding_mode: str = "pipage"          # "pipage" | "confidence_gap"
    enforce_class_delta_after_round: bool = True
    milp_last_resort_only: bool = True
    milp_size_limit: int = 10_000
    milp_time_limit_s: float = 120.0
    # diagnostics
    spearman_threshold: float = 0.05
    yhat_emb_knn_k: int = 15
    # stratification
    stratification_strategy: str = "none"  # "none" | "quantile" | "moment" | "sliced_wasserstein"
    n_bins: int = 10
    P_proj: int = 100
    Q_grid: int = 256
    omega: Optional[ArrayLike] = None
    class_delta: float = 0.05
    # coarsening
    coarsen: bool = False
    n_super: int = 0
    coarsen_max_iter: int = 25
    coarsen_balance_strength: float = 1.0
    # -- multi-split entity-perturbation strength; under development, excluded
    #    from the manuscript, ships with SHIELD v1.1 --
    entity_diversity_strength: float = 0.0
    # stopping criterion 1: Z-assignment Hamming stability
    z_hamming_eps: float = 0.01
    z_hamming_patience: int = 3
    # stopping criterion 2: per-term individual convergence
    tolerance_per_term: float = 1e-3
    patience_per_term: int = 3


@dataclass
class ShieldResult:
    z: torch.Tensor
    fold_assignment: torch.Tensor
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ShieldSweepResult:
    grid_points: List[Tuple[float, float, float]]
    results: List[ShieldResult]
    pareto_indices: List[int]
    recommended_index: int
    recommendation_info: Dict[str, float]
    stability: diagnostics.StabilityReport


@dataclass
class ShieldMultiSplitResult:
    splits: List[ShieldResult]
    n_splits: int
    seed_base: int
    entity_diversity_strength: float
    pairwise_hamming: "np.ndarray"
    pairwise_hamming_frac: "np.ndarray"
    mean_pairwise_hamming_frac: float
    entity_fold_matrix: Optional["np.ndarray"]
    entity_ids: Optional["np.ndarray"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_hierarchy_levels(
    levels_spec: Sequence[Dict[str, Any]],
) -> List[gurobi_lp.HierarchyLevel]:
    out: List[gurobi_lp.HierarchyLevel] = []
    for spec in levels_spec:
        sim = spec["similarity"]
        if isinstance(sim, tuple) and len(sim) == 3:
            rows, cols, vals = sim
            # Each element may itself be a torch tensor (including on CUDA).
            rows = (rows.detach().cpu().numpy() if isinstance(rows, torch.Tensor)
                    else np.asarray(rows, dtype=np.int64))
            cols = (cols.detach().cpu().numpy() if isinstance(cols, torch.Tensor)
                    else np.asarray(cols, dtype=np.int64))
            vals = (vals.detach().cpu().numpy() if isinstance(vals, torch.Tensor)
                    else np.asarray(vals, dtype=np.float64))
            rows = rows.astype(np.int64)
            cols = cols.astype(np.int64)
            vals = vals.astype(np.float64)
        else:
            # Dense matrix form: may be a torch tensor (including CUDA).
            if isinstance(sim, torch.Tensor):
                sim = sim.detach().cpu().numpy()
            arr = np.asarray(sim, dtype=np.float64)
            idx = np.nonzero(arr)
            rows = idx[0]
            cols = idx[1]
            vals = arr[idx]
        out.append(gurobi_lp.HierarchyLevel(
            name=str(spec["name"]),
            descendants=[list(d) for d in spec["descendants"]],
            similarity_rows=rows,
            similarity_cols=cols,
            similarity_vals=vals,
            lambda_level=float(spec.get("lambda", 1.0)),
        ))
    return out


# ---------------------------------------------------------------------------
# Public: shield.split
# ---------------------------------------------------------------------------

def split(
    *,
    n: int,                                # number of observations to partition
    config: Optional[ShieldConfig] = None, # optional ShieldConfig bundling the below
    weights: Optional[ArrayLike] = None,   # per-observation weight, shape (n,)
    K: int = 3,                            # number of folds
    r: Optional[ArrayLike] = None,         # target fold proportions, shape (K,)
    # H term: hierarchical containment
    hierarchy: Optional[Sequence[Dict[str, Any]]] = None,  # list of level specs, see build_hierarchy_levels
    # M term: graph-cut on the feature manifold
    affinity: Optional[ArrayLike] = None,       # symmetric affinity/similarity matrix, shape (n, n)
    affinity_provenance: Optional[str] = None,  # "disjoint-source" | "label-blind-handcrafted" | "self-supervised"
    # D term: distributional separation or matching
    d_mode: str = "match",                      # "match" (toward target_marginals) | "shift" (spread folds apart)
    cost: Optional[ArrayLike] = None,           # OT cost matrix, shape (n, m); match mode
    target_marginals: Optional[ArrayLike] = None,  # per-fold target measure, shape (K, m); match mode
    rho: Optional[ArrayLike] = None,            # per-fold weight for the match term, shape (K,)
    kernel_matrix: Optional[ArrayLike] = None,  # symmetric kernel, shape (n, n); shift mode
    rho_pair: Optional[ArrayLike] = None,       # per-fold-pair weight for the shift term, shape (K, K)
    # stratification / labels
    y_target: Optional[ArrayLike] = None,       # continuous target(s) to stratify on, shape (n,) or (n, q)
    class_labels: Optional[ArrayLike] = None,   # discrete class label per observation, shape (n,)
    stratification_strategy: str = "none",      # "none" | "quantile" | "moment" | "sliced_wasserstein"
    omega: Optional[ArrayLike] = None,          # moment-strategy weights, shape (4, q)
    n_bins: int = 10,                           # quantile-strategy bin count
    P_proj: int = 100,                          # sliced-Wasserstein projection count
    Q_grid: int = 256,                          # sliced-Wasserstein quantile grid resolution
    class_delta: float = 0.05,                  # per-class, per-fold mass tolerance
    # two-entity / cold-start splitting
    two_entity_mode: str = "warm",              # "warm" | "cold-left" | "cold-right" | "cold-both"
    pair_indices: Optional[ArrayLike] = None,       # (left, right) atom index pairs, shape (P, 2)
    left_entity_of: Optional[ArrayLike] = None,     # left entity id per atom
    right_entity_of: Optional[ArrayLike] = None,    # right entity id per atom
    pair_observation_targets: Optional[ArrayLike] = None,  # pair-level targets, forwarded for downstream use
    # coarsening: partition super-nodes instead of individual observations
    coarsen: bool = False,
    n_super: int = 0,                      # target super-node count
    coarsen_max_iter: int = 25,
    coarsen_balance_strength: float = 1.0,
    feature_matrix: Optional[ArrayLike] = None,  # observation features, shape (n, d); required if coarsen=True
    # composite-objective coefficients (Eq. 1)
    alpha: float = 1.0,                    # weight on H (hierarchy) term
    beta: float = 1.0,                     # weight on M (graph-cut) term
    gamma: float = 1.0,                    # weight on D (distributional) term
    eta: float = 1.0,                      # weight on the fold-balance penalty
    mu: float = 1.0,                       # weight on the stratification term
    # -- under development, excluded from the manuscript; ships with SHIELD v1.1 --
    nu: float = 0.0,                       # label-augmentation objective weight
    tau: float = 0.0,                      # label-augmentation scale
    label_augmentation_justification: Optional[str] = None,
    label_augmentation_override_target_derivation: bool = False,
    # solver controls
    max_iter: int = 25,                    # outer block-coordinate iterations
    tolerance: float = 1e-3,
    epsilon_OT: float = 5e-2,              # Sinkhorn entropic regularizer
    sinkhorn_max_iter: int = 200,
    seed: int = 0,
    device: str = "cpu",                   # "cpu" | "cuda"
    nodes: int = -1,                       # CPU worker count, -1 = all cores
    integer_z_final: bool = True,          # allow last-resort MILP polish
    milp_polish: Optional[bool] = None,    # explicit alias for integer_z_final
    balance_eps: float = 0.05,             # fold-balance tolerance
    output_flag: int = 0,                  # Gurobi log verbosity
    enable_stage0: bool = True,            # Monte-Carlo term calibration
    stage0_samples: int = 100,
    # initialization / rounding controls
    init_backend: str = "auto",            # "auto" | "metis" | "kahip" | "spectral"
    rounding_mode: str = "pipage",         # "pipage" | "confidence_gap"
    enforce_class_delta_after_round: bool = True,
    milp_last_resort_only: bool = True,
    milp_size_limit: int = 10_000,
    milp_time_limit_s: float = 120.0,
    # stopping criterion 1: fold-assignment Hamming stability
    z_hamming_eps: float = 0.01,
    z_hamming_patience: int = 3,
    # stopping criterion 2: per-term individual convergence
    tolerance_per_term: float = 1e-3,
    patience_per_term: int = 3,
    # -- multi-split entity-perturbation strength; under development, excluded
    #    from the manuscript, ships with SHIELD v1.1 --
    entity_diversity_strength: float = 0.0,
    # diagnostic inputs
    embedding: Optional[ArrayLike] = None,      # observation embedding used for diagnostics
    y_hat_emb: Optional[ArrayLike] = None,      # model-prediction embedding for the kNN diagnostic
    yhat_emb_knn_k: int = 15,
    spearman_threshold: float = 0.05,
    precomputed_normalizers: Optional[Dict[str, float]] = None,  # skip Stage-0 calibration with fixed values
) -> ShieldResult:
    # config values fill in any parameter left at its function-signature default
    if config is not None:
        _fd = ShieldConfig()
        if K == _fd.K:                           K = config.K
        if r is _fd.r:                           r = config.r
        if d_mode == _fd.d_mode:                 d_mode = config.d_mode
        if alpha == _fd.alpha:                   alpha = config.alpha
        if beta == _fd.beta:                     beta = config.beta
        if gamma == _fd.gamma:                   gamma = config.gamma
        if eta == _fd.eta:                       eta = config.eta
        if mu == _fd.mu:                         mu = config.mu
        if nu == _fd.nu:                         nu = config.nu
        if tau == _fd.tau:                       tau = config.tau
        if label_augmentation_justification is _fd.label_augmentation_justification:
            label_augmentation_justification = config.label_augmentation_justification
        if label_augmentation_override_target_derivation == _fd.label_augmentation_override_target_derivation:
            label_augmentation_override_target_derivation = config.label_augmentation_override_target_derivation
        if max_iter == _fd.max_iter:             max_iter = config.max_iter
        if tolerance == _fd.tolerance:           tolerance = config.tolerance
        if epsilon_OT == _fd.epsilon_OT:         epsilon_OT = config.epsilon_OT
        if sinkhorn_max_iter == _fd.sinkhorn_max_iter:
            sinkhorn_max_iter = config.sinkhorn_max_iter
        if seed == _fd.seed:                     seed = config.seed
        if device == _fd.device:                 device = config.device
        if nodes == _fd.nodes:                   nodes = config.nodes
        if integer_z_final == _fd.integer_z_final:
            integer_z_final = config.integer_z_final
        if balance_eps == _fd.balance_eps:       balance_eps = config.balance_eps
        if output_flag == _fd.output_flag:       output_flag = config.output_flag
        if enable_stage0 == _fd.enable_stage0:   enable_stage0 = config.enable_stage0
        if stage0_samples == _fd.stage0_samples: stage0_samples = config.stage0_samples
        if stratification_strategy == _fd.stratification_strategy:
            stratification_strategy = config.stratification_strategy
        if n_bins == _fd.n_bins:                 n_bins = config.n_bins
        if P_proj == _fd.P_proj:                 P_proj = config.P_proj
        if Q_grid == _fd.Q_grid:                 Q_grid = config.Q_grid
        if omega is _fd.omega:                   omega = config.omega
        if class_delta == _fd.class_delta:       class_delta = config.class_delta
        if coarsen == _fd.coarsen:               coarsen = config.coarsen
        if n_super == _fd.n_super:               n_super = config.n_super
        if coarsen_max_iter == _fd.coarsen_max_iter:
            coarsen_max_iter = config.coarsen_max_iter
        if coarsen_balance_strength == _fd.coarsen_balance_strength:
            coarsen_balance_strength = config.coarsen_balance_strength
        if init_backend == _fd.init_backend:
            init_backend = config.init_backend
        if rounding_mode == _fd.rounding_mode:
            rounding_mode = config.rounding_mode
        if enforce_class_delta_after_round == _fd.enforce_class_delta_after_round:
            enforce_class_delta_after_round = config.enforce_class_delta_after_round
        if milp_last_resort_only == _fd.milp_last_resort_only:
            milp_last_resort_only = config.milp_last_resort_only
        if milp_size_limit == _fd.milp_size_limit:
            milp_size_limit = config.milp_size_limit
        if milp_time_limit_s == _fd.milp_time_limit_s:
            milp_time_limit_s = config.milp_time_limit_s
        if spearman_threshold == _fd.spearman_threshold:
            spearman_threshold = config.spearman_threshold
        if yhat_emb_knn_k == _fd.yhat_emb_knn_k:
            yhat_emb_knn_k = config.yhat_emb_knn_k
        if entity_diversity_strength == _fd.entity_diversity_strength:
            entity_diversity_strength = config.entity_diversity_strength
        if z_hamming_eps == _fd.z_hamming_eps:
            z_hamming_eps = config.z_hamming_eps
        if z_hamming_patience == _fd.z_hamming_patience:
            z_hamming_patience = config.z_hamming_patience
        if tolerance_per_term == _fd.tolerance_per_term:
            tolerance_per_term = config.tolerance_per_term
        if patience_per_term == _fd.patience_per_term:
            patience_per_term = config.patience_per_term
    if milp_polish is not None:
        integer_z_final = bool(milp_polish)
    configure_threading(nodes)
    device_t = resolve_device(device)

    # ---- validation / safeguards ----------------------------------------
    use_H = hierarchy is not None
    use_M = affinity is not None
    use_D = cost is not None or kernel_matrix is not None
    print(
        f"[shield.api] Term flags derived from supplied inputs: "
        f"use_H={use_H} (hierarchy supplied={use_H}), "
        f"use_M={use_M} (affinity supplied={use_M}), "
        f"use_D={use_D} (cost/kernel supplied={use_D}). "
        f"api.py — flags are determined by which relational objects are passed, "
        f"not by ShieldConfig.use_H/use_M/use_D (those fields have been removed "
        f"as they were never read)."
    )
    if not (use_H or use_M or use_D):
        raise ValueError("At least one of (hierarchy, affinity, cost/kernel) is required.")

    # hierarchy input validation
    if use_H and hierarchy is not None:
        for spec_i, spec in enumerate(hierarchy):
            desc_list = spec.get("descendants", [])
            if spec.get("similarity") is None:
                raise ValueError(
                    f"[shield.api] Hierarchy level {spec_i} is missing the "
                    f"'similarity' key. api.py — each level needs a similarity "
                    f"matrix or (rows, cols, vals) COO tuple."
                )
            for g, d in enumerate(desc_list):
                for atom_idx in d:
                    if atom_idx < 0 or atom_idx >= n:
                        raise ValueError(
                            f"[shield.api] Hierarchy level {spec_i}, family {g}: "
                            f"descendant index {atom_idx} is out of bounds "
                            f"[0, n-1={n - 1}]. api.py — all descendant indices "
                            f"must reference valid atom positions in [0, n-1]."
                        )
            all_atoms: list = []
            for d in desc_list:
                all_atoms.extend(d)
            if len(all_atoms) != len(set(all_atoms)):
                raise ValueError(
                    f"[shield.api] Hierarchy level {spec_i} has overlapping "
                    f"families (non-disjoint descendants). api.py — each atom "
                    f"must appear in at most one family per level."
                )
            sim = spec.get("similarity")
            if not isinstance(sim, tuple):
                sim_arr = (sim.detach().cpu().numpy()
                           if isinstance(sim, torch.Tensor)
                           else np.asarray(sim, dtype=np.float64))
                if sim_arr.ndim == 2 and not np.allclose(sim_arr, sim_arr.T, atol=1e-6):
                    raise ValueError(
                        f"[shield.api] Hierarchy level {spec_i} similarity matrix "
                        f"is not symmetric (max|S-S^T|="
                        f"{float(np.abs(sim_arr - sim_arr.T).max()):.4e}). "
                        f"api.py — similarity must satisfy S[g,g']==S[g',g]."
                    )
        print(
            f"[shield.api] Hierarchy validated: {len(hierarchy)} level(s), n={n}. "
            f"api.py — checked descendant index bounds [0,{n-1}], disjoint "
            f"families within each level, and dense-matrix symmetry."
        )
    if use_M and affinity_provenance is None:
        raise ValueError(
            "M term active but affinity_provenance is None. Provide one of "
            "'disjoint-source', 'label-blind-handcrafted', or 'self-supervised' "
        )

    # label augmentation safeguards (under development, see notes above)
    if tau > 0.0 or nu > 0.0:
        if label_augmentation_justification is None:
            raise ValueError(
                "Label augmentation requested (tau>0 or nu>0) but "
                "label_augmentation_justification is None. Provide written justification"
            )
        if (not label_augmentation_override_target_derivation
                and y_target is not None and cost is not None):
            raise ValueError(
                "Label augmentation refused because cost may derive from the "
                "downstream prediction target. Pass "
                "label_augmentation_override_target_derivation=True to override"
            )

    if d_mode == "match" and use_D and cost is None:
        raise ValueError("D-match requires a cost matrix.")
    if d_mode == "match" and use_D and target_marginals is None:
        raise ValueError(
            "[shield.api] D-match requires target_marginals — a (K, m) array "
            "where each row k is the target measure for fold k and m is the "
            "number of support points. api.py — target_marginals missing while "
            "d_mode='match' is active."
        )
    if d_mode == "shift" and use_D and kernel_matrix is None:
        raise ValueError("D-shift requires a kernel_matrix.")

    if two_entity_mode != "warm":
        if pair_indices is None:
            raise ValueError(f"two_entity_mode='{two_entity_mode}' requires pair_indices.")
        if two_entity_mode in ("cold-left", "cold-both") and left_entity_of is None:
            raise ValueError("cold-left/both requires left_entity_of.")
        if two_entity_mode in ("cold-right", "cold-both") and right_entity_of is None:
            raise ValueError("cold-right/both requires right_entity_of.")

    if stratification_strategy != "none" and y_target is None:
        raise ValueError(
            f"stratification_strategy='{stratification_strategy}' requires y_target. "
            "Pass y_target or set stratification_strategy='none'."
        )

    # coarsening pre-flight checks
    if coarsen and feature_matrix is None:
        raise ValueError(
            "[shield.api] coarsen=True requires feature_matrix to be supplied. "
            "Coarsening uses balanced K-means to cluster n observations into "
            "n_super super-nodes in feature space, which requires a (n, d) "
            "feature tensor. Either provide feature_matrix as an (n, d) float "
            "tensor, or set coarsen=False to skip coarsening and solve the "
            "full n-scale problem directly."
        )
    if coarsen and two_entity_mode in ("cold-left", "cold-right", "cold-both"):
        raise ValueError(
            f"[shield.api] coarsen=True is incompatible with "
            f"two_entity_mode='{two_entity_mode}'. Cold-* constraints reference "
            "atom-level entity groups, but the LP operates on super-nodes after "
            "coarsening — atom indices are out of range for the super-node LP. "
            "Either set coarsen=False, or use two_entity_mode='warm'."
        )

    w_t = (_to_tensor(weights, device_t)
           if weights is not None
           else torch.ones(n, device=device_t, dtype=torch.float32))
    r_t = (_to_tensor(r, device_t)
           if r is not None
           else torch.full((K,), 1.0 / K, device=device_t, dtype=torch.float32))

    if r_t.shape[0] != K:
        raise ValueError(
            f"[shield.api] r has {r_t.shape[0]} element(s) but K={K}. "
            f"api.py — r must have exactly K target proportions, one per fold."
        )
    r_sum = float(r_t.sum().item())
    if abs(r_sum - 1.0) > 1e-3:
        raise ValueError(
            f"[shield.api] r sums to {r_sum:.6f} (must be 1.0 ± 1e-3). "
            f"api.py — normalize r before passing (e.g. r = r / r.sum())."
        )
    print(
        f"[shield.api] r validated: K={K}, sum={r_sum:.6f}, "
        f"values={r_t.tolist()}. api.py — r controls target fold-size "
        f"proportions passed to the LP and balance constraints."
    )

    affinity_t = _materialize_affinity(affinity, device_t) if use_M else None
    cost_t = _to_tensor(cost, device_t) if cost is not None else None
    target_marginals_t = _to_tensor(target_marginals, device_t)
    if target_marginals_t is not None and d_mode == "match":
        if target_marginals_t.dim() != 2 or target_marginals_t.shape[0] != K:
            raise ValueError(
                f"[shield.api] target_marginals must have shape (K, m) = ({K}, m), "
                f"got shape {list(target_marginals_t.shape)}. "
                f"api.py — each row k is the target measure for fold k."
            )
        if cost_t is not None and cost_t.shape[1] != target_marginals_t.shape[1]:
            raise ValueError(
                f"[shield.api] D-match support-dimension mismatch: cost has "
                f"m={cost_t.shape[1]} columns but target_marginals has "
                f"m={target_marginals_t.shape[1]} columns. api.py — cost (n, m) "
                f"and target_marginals (K, m) must share the same m (number of "
                f"target support points)."
            )
    rho_t = _to_tensor(rho, device_t) if rho is not None else None
    kernel_t = _to_tensor(kernel_matrix, device_t)
    if (use_D and d_mode == "shift" and kernel_t is not None
            and not kernel_t.is_sparse and n > 50_000):
        approx_gb = (float(n) ** 2 * 4.0) / (1024.0 ** 3)
        import warnings as _kwarn
        _kwarn.warn(
            f"[shield.api] D-shift kernel_matrix is dense and n={n}: the (n, n) "
            f"kernel needs ~{approx_gb:.1f} GB (float32). MMD is O(n^2) in memory "
            f"and is not rescued by coarsening. Reduce n, supply a sparse kernel, "
            f"or use d_mode='match' for large datasets.",
            UserWarning,
            stacklevel=2,
        )
    rho_pair_t = _to_tensor(rho_pair, device_t)
    y_target_t = _to_tensor(y_target, device_t) if y_target is not None else None
    if y_target_t is not None and y_target_t.dim() == 1:
        y_target_t = y_target_t.unsqueeze(1)
    # omega rows are central-moment orders 1..4, columns are output coordinates
    if stratification_strategy == "moment" and omega is not None:
        _omega_t = _to_tensor(omega, device_t)
        _q = int(y_target_t.shape[1]) if y_target_t is not None else None
        if (_omega_t.dim() != 2 or _omega_t.shape[0] != 4
                or (_q is not None and _omega_t.shape[1] != _q)):
            raise ValueError(
                f"[shield.api] moment stratification requires omega of shape "
                f"(4, q) = (4, {_q}); got {list(_omega_t.shape)}. api.py — rows "
                f"are central-moment orders 1..4 and columns are the q output "
                f"coordinates of y_target."
            )
    feature_t = _to_tensor(feature_matrix, device_t) if feature_matrix is not None else None
    class_t = _to_long(class_labels, device_t) if class_labels is not None else None
    pair_idx_t = _to_long(pair_indices, device_t) if pair_indices is not None else None
    if pair_idx_t is not None:
        if pair_idx_t.dim() != 2 or pair_idx_t.shape[1] != 2:
            raise ValueError(
                f"pair_indices must be a 2-D array of shape (P, 2) — "
                f"(left_atom_index, right_atom_index) per row — "
                f"but got shape {list(pair_idx_t.shape)}."
            )
    left_eid_t = _to_long(left_entity_of, device_t) if left_entity_of is not None else None
    right_eid_t = _to_long(right_entity_of, device_t) if right_entity_of is not None else None
    pot_t = (_to_tensor(pair_observation_targets, device_t)
             if pair_observation_targets is not None else None)
    if pot_t is not None and pot_t.dim() == 1:
        pot_t = pot_t.unsqueeze(1)

    if rho_t is None and (use_D and d_mode == "match"):
        rho_t = torch.ones(K, device=device_t, dtype=torch.float32)
    if rho_pair_t is None and (use_D and d_mode == "shift"):
        rho_pair_t = torch.ones((K, K), device=device_t, dtype=torch.float32)
        rho_pair_t.fill_diagonal_(0.0)

    # optional label augmentation, under development
    # effective_tau = nu * tau; nu=0 (default) disables it even if tau > 0
    import warnings as _aug_warnings
    if nu > 0.0 and tau == 0.0:
        _aug_warnings.warn(
            "nu > 0 but tau = 0; label augmentation is disabled. "
            "Set tau > 0 to activate the c̃ = c + effective_tau·c^(y) augmentation.",
            UserWarning,
            stacklevel=2,
        )
    effective_tau = nu * tau
    if effective_tau > 0.0 and y_target_t is not None:
        if cost_t is not None:
            y_cost = torch.cdist(y_target_t.to(torch.float32),
                                 y_target_t.to(torch.float32))
            cost_t = cost_t + effective_tau * y_cost[: cost_t.shape[0], : cost_t.shape[1]]
        if affinity_t is not None:
            if affinity_t.is_sparse:
                raise ValueError(
                    "Label augmentation requested (effective_tau > 0) but the "
                    "affinity matrix is sparse. SHIELD refuses to silently "
                    "drop the augmentation. Densify the affinity first via "
                    "`affinity = affinity.to_dense()` if you really want the "
                    "label-blended M term, or set tau=0/nu=0 to disable augmentation"
                )
            y_sim = torch.exp(-torch.cdist(y_target_t.to(torch.float32),
                                           y_target_t.to(torch.float32)) ** 2)
            affinity_t = affinity_t + effective_tau * y_sim

    hierarchy_levels = build_hierarchy_levels(hierarchy) if use_H else None

    # ---- build configs ---------------------------------------------------
    terms = pipeline.TermConfig(
        use_H=use_H, use_M=use_M, use_D=use_D,
        d_mode=d_mode,
        use_label_augmentation=(tau > 0.0 or nu > 0.0),
        tau=tau, alpha=alpha, beta=beta, gamma=gamma,
        eta=eta, mu=mu, nu=nu,
    )
    strat_cfg = pipeline.StratificationConfig(
        strategy=stratification_strategy,
        n_bins=n_bins,
        P_proj=P_proj,
        Q_grid=Q_grid,
        omega=_to_tensor(omega, device_t) if omega is not None else None,
        class_labels=class_t,
        class_delta=class_delta,
    )
    coarsen_cfg = pipeline.CoarseningConfig(
        enabled=coarsen,
        n_super=n_super,
        max_iter=coarsen_max_iter,
        balance_strength=coarsen_balance_strength,
    )
    two_cfg = pipeline.TwoEntityConfig(
        mode=two_entity_mode,
        pair_indices=pair_idx_t,
        left_entity_of=left_eid_t,
        right_entity_of=right_eid_t,
    )
    pcfg = pipeline.PipelineConfig(
        K=K,
        r=r_t,
        tolerance=tolerance,
        max_iter=max_iter,
        epsilon_OT=epsilon_OT,
        sinkhorn_max_iter=sinkhorn_max_iter,
        seed=seed,
        device=device,
        nodes=nodes,
        integer_z_final=integer_z_final,
        balance_eps=balance_eps,
        stage0_samples=stage0_samples,
        output_flag=output_flag,
        enable_stage0=enable_stage0,
        init_backend=init_backend,
        rounding_mode=rounding_mode,
        enforce_class_delta_after_round=enforce_class_delta_after_round,
        milp_last_resort_only=milp_last_resort_only,
        milp_size_limit=milp_size_limit,
        milp_time_limit_s=milp_time_limit_s,
        z_hamming_eps=float(z_hamming_eps),
        z_hamming_patience=int(z_hamming_patience),
        tolerance_per_term=float(tolerance_per_term),
        patience_per_term=int(patience_per_term),
        entity_diversity_strength=float(entity_diversity_strength),
    )

    embedding_t = (_to_tensor(embedding, device_t)
                   if embedding is not None else None)
    yhat_emb_t = (_to_tensor(y_hat_emb, device_t)
                  if y_hat_emb is not None else None)

    output = pipeline.run_shield(
        weights=w_t,
        n=n,
        affinity=affinity_t,
        cost=cost_t,
        target_marginals=target_marginals_t,
        rho=rho_t,
        kernel_matrix=kernel_t,
        rho_pair=rho_pair_t,
        hierarchy_levels=hierarchy_levels,
        y_target=y_target_t,
        two_entity=two_cfg,
        terms=terms,
        strat_cfg=strat_cfg,
        coarsen_cfg=coarsen_cfg,
        cfg=pcfg,
        feature_matrix=feature_t,
        pair_observation_targets=pot_t,
        embedding=embedding_t,
        y_hat_emb=yhat_emb_t,
        yhat_emb_knn_k=int(yhat_emb_knn_k),
        spearman_threshold=float(spearman_threshold),
        precomputed_normalizers=precomputed_normalizers,
    )

    output.metadata["affinity_provenance"] = affinity_provenance
    output.metadata["label_augmentation_justification"] = label_augmentation_justification
    return ShieldResult(
        z=output.z,
        fold_assignment=output.fold_assignment,
        diagnostics=output.diagnostics,
        metadata=output.metadata,
        history=output.history,
    )


# ---------------------------------------------------------------------------
# Public: shield.sweep
# ---------------------------------------------------------------------------

def sweep(
    *,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    gamma_grid: Sequence[float],
    metric_keys: Sequence[str] = ("M_normalized", "D_normalized", "H_normalized"),
    aggregation: str = "min_of_normalized",
    instability_threshold: float = 0.25,
    **shield_kwargs,
) -> ShieldSweepResult:
    """Run a coefficient sweep"""
    grid_points: List[Tuple[float, float, float]] = list(
        itertools.product(alpha_grid, beta_grid, gamma_grid)
    )
    results: List[ShieldResult] = []
    for (a, b, c) in grid_points:
        kw = dict(shield_kwargs)
        kw["alpha"] = float(a)
        kw["beta"] = float(b)
        kw["gamma"] = float(c)
        results.append(split(**kw))

    def _last_terms(res: ShieldResult) -> Dict[str, float]:
        return res.history[-1]["terms"] if res.history else {}

    present_keys = [k for k in metric_keys
                    if any(k in _last_terms(res) for res in results)]
    dropped = [k for k in metric_keys if k not in present_keys]
    if dropped:
        import warnings as _swarn
        _swarn.warn(
            f"[shield.sweep] metric key(s) {dropped} are not produced by any "
            f"sweep point (the corresponding term is inactive) and were dropped "
            f"from the Pareto frontier / recommendation. Pass metric_keys that "
            f"match the active terms to silence this.",
            UserWarning, stacklevel=2,
        )
    if not present_keys:
        raise ValueError(
            "[shield.sweep] none of the requested metric_keys were produced by "
            "the sweep, so no frontier can be built. Check that metric_keys "
            "match the active terms (e.g. H_normalized requires a hierarchy)."
        )
    eff_keys = tuple(present_keys)

    diag_rows: List[Dict[str, float]] = []
    for res in results:
        lt = _last_terms(res)
        diag_rows.append({k: float(lt.get(k, 0.0)) for k in eff_keys})

    matrix = [[row[k] for k in eff_keys] for row in diag_rows]
    pareto = diagnostics.pareto_frontier(matrix)
    rec_idx, rec_info = diagnostics.recommend_coefficients(
        grid_points, diag_rows, eff_keys, aggregation=aggregation,
    )
    stability = diagnostics.assignment_stability(
        grid_points,
        [r.z for r in results],
        instability_threshold=instability_threshold,
    )
    return ShieldSweepResult(
        grid_points=grid_points,
        results=results,
        pareto_indices=pareto,
        recommended_index=rec_idx,
        recommendation_info=rec_info,
        stability=stability,
    )


# ---------------------------------------------------------------------------
# Public: shield.multi_split
# ---------------------------------------------------------------------------

# Under development, excluded from the manuscript; ships with SHIELD v1.1.
def multi_split(
    *,
    # multi-split controls
    n_splits: int,
    seed_base: int = 0,
    entity_diversity_strength: float = 0.1,
    n: int,
    # objective / term inputs, identical to shield.split
    config: Optional[ShieldConfig] = None,
    weights: Optional[ArrayLike] = None,
    K: int = 3,
    r: Optional[ArrayLike] = None,
    # H term
    hierarchy: Optional[Sequence[Dict[str, Any]]] = None,
    # M term
    affinity: Optional[ArrayLike] = None,
    affinity_provenance: Optional[str] = None,
    # D term
    d_mode: str = "match",
    cost: Optional[ArrayLike] = None,
    target_marginals: Optional[ArrayLike] = None,
    rho: Optional[ArrayLike] = None,
    kernel_matrix: Optional[ArrayLike] = None,
    rho_pair: Optional[ArrayLike] = None,
    # stratification / labels
    y_target: Optional[ArrayLike] = None,
    class_labels: Optional[ArrayLike] = None,
    stratification_strategy: str = "none",
    omega: Optional[ArrayLike] = None,
    n_bins: int = 10,
    P_proj: int = 100,
    Q_grid: int = 256,
    class_delta: float = 0.05,
    # two-entity / cold-* mode
    two_entity_mode: str = "warm",
    pair_indices: Optional[ArrayLike] = None,
    left_entity_of: Optional[ArrayLike] = None,
    right_entity_of: Optional[ArrayLike] = None,
    pair_observation_targets: Optional[ArrayLike] = None,
    # coarsening
    coarsen: bool = False,
    n_super: int = 0,
    coarsen_max_iter: int = 25,
    coarsen_balance_strength: float = 1.0,
    feature_matrix: Optional[ArrayLike] = None,
    # objective coefficients
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    eta: float = 1.0,
    mu: float = 1.0,
    nu: float = 0.0,
    tau: float = 0.0,
    label_augmentation_justification: Optional[str] = None,
    label_augmentation_override_target_derivation: bool = False,
    # solver controls
    max_iter: int = 25,
    tolerance: float = 1e-3,
    epsilon_OT: float = 5e-2,
    sinkhorn_max_iter: int = 200,
    device: str = "cpu",
    nodes: int = -1,
    integer_z_final: bool = True,
    milp_polish: Optional[bool] = None,
    balance_eps: float = 0.05,
    output_flag: int = 0,
    enable_stage0: bool = True,
    stage0_samples: int = 100,
    init_backend: str = "auto",
    rounding_mode: str = "pipage",
    enforce_class_delta_after_round: bool = True,
    milp_last_resort_only: bool = True,
    milp_size_limit: int = 10_000,
    milp_time_limit_s: float = 120.0,
    # stopping criterion 1: Z-assignment Hamming stability
    z_hamming_eps: float = 0.01,
    z_hamming_patience: int = 3,
    # stopping criterion 2: per-term individual convergence
    tolerance_per_term: float = 1e-3,
    patience_per_term: int = 3,
    # diagnostic inputs
    embedding: Optional[ArrayLike] = None,
    y_hat_emb: Optional[ArrayLike] = None,
    yhat_emb_knn_k: int = 15,
    spearman_threshold: float = 0.05,
    precomputed_normalizers: Optional[Dict[str, float]] = None,
) -> ShieldMultiSplitResult:
    """Generate ``n_splits`` independent data partitions for cross-validation"""
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
    if entity_diversity_strength < 0.0:
        raise ValueError(
            f"entity_diversity_strength must be >= 0.0, got {entity_diversity_strength}."
        )

    # kwargs shared unchanged across every split call
    _shared: Dict[str, Any] = dict(
        n=n,
        config=config,
        weights=weights,
        K=K,
        r=r,
        hierarchy=hierarchy,
        affinity=affinity,
        affinity_provenance=affinity_provenance,
        d_mode=d_mode,
        cost=cost,
        target_marginals=target_marginals,
        rho=rho,
        kernel_matrix=kernel_matrix,
        rho_pair=rho_pair,
        y_target=y_target,
        class_labels=class_labels,
        stratification_strategy=stratification_strategy,
        omega=omega,
        n_bins=n_bins,
        P_proj=P_proj,
        Q_grid=Q_grid,
        class_delta=class_delta,
        two_entity_mode=two_entity_mode,
        pair_indices=pair_indices,
        left_entity_of=left_entity_of,
        right_entity_of=right_entity_of,
        pair_observation_targets=pair_observation_targets,
        coarsen=coarsen,
        n_super=n_super,
        coarsen_max_iter=coarsen_max_iter,
        coarsen_balance_strength=coarsen_balance_strength,
        feature_matrix=feature_matrix,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        eta=eta,
        mu=mu,
        nu=nu,
        tau=tau,
        label_augmentation_justification=label_augmentation_justification,
        label_augmentation_override_target_derivation=label_augmentation_override_target_derivation,
        max_iter=max_iter,
        tolerance=tolerance,
        epsilon_OT=epsilon_OT,
        sinkhorn_max_iter=sinkhorn_max_iter,
        device=device,
        nodes=nodes,
        integer_z_final=integer_z_final,
        milp_polish=milp_polish,
        balance_eps=balance_eps,
        output_flag=output_flag,
        enable_stage0=enable_stage0,
        stage0_samples=stage0_samples,
        init_backend=init_backend,
        rounding_mode=rounding_mode,
        enforce_class_delta_after_round=enforce_class_delta_after_round,
        milp_last_resort_only=milp_last_resort_only,
        milp_size_limit=milp_size_limit,
        milp_time_limit_s=milp_time_limit_s,
        z_hamming_eps=z_hamming_eps,
        z_hamming_patience=z_hamming_patience,
        tolerance_per_term=tolerance_per_term,
        patience_per_term=patience_per_term,
        entity_diversity_strength=entity_diversity_strength,
        embedding=embedding,
        y_hat_emb=y_hat_emb,
        yhat_emb_knn_k=yhat_emb_knn_k,
        spearman_threshold=spearman_threshold,
        precomputed_normalizers=precomputed_normalizers,
    )

    splits: List[ShieldResult] = []
    for i in range(n_splits):
        print(
            f"[shield.multi_split] Running split {i + 1}/{n_splits}  "
            f"(seed={seed_base + i}, entity_diversity_strength={entity_diversity_strength})"
        )
        res = split(seed=seed_base + i, **_shared)
        splits.append(res)

    # ---- pairwise Hamming distances -----------------------------------------
    n_obs = int(splits[0].fold_assignment.shape[0])
    H = np.zeros((n_splits, n_splits), dtype=np.int64)
    for a in range(n_splits):
        fa = splits[a].fold_assignment.cpu()
        for b in range(a + 1, n_splits):
            fb = splits[b].fold_assignment.cpu()
            d = int((fa != fb).sum().item())
            H[a, b] = H[b, a] = d
    H_frac = H.astype(np.float64) / max(n_obs, 1)
    n_pairs = n_splits * (n_splits - 1)
    mean_h = float(H_frac[np.triu_indices(n_splits, k=1)].mean()) if n_pairs > 0 else 0.0

    # ---- entity-fold matrix (cold-* modes only) -----------------------------
    efm: Optional[np.ndarray] = None
    eids_out: Optional[np.ndarray] = None
    if left_entity_of is not None:
        left_arr = np.asarray(left_entity_of)
        unique_eids = np.sort(np.unique(left_arr[left_arr >= 0]))
        efm = np.full((len(unique_eids), n_splits), -1, dtype=np.int64)
        for si, res in enumerate(splits):
            fa_np = res.fold_assignment.cpu().numpy()
            for ei, eid in enumerate(unique_eids):
                mask = left_arr == eid
                if mask.any():
                    counts = np.bincount(fa_np[mask], minlength=K)
                    efm[ei, si] = int(counts.argmax())
        eids_out = unique_eids

    print(
        f"[shield.multi_split] Done. {n_splits} splits produced. "
        f"Mean pairwise Hamming fraction = {mean_h:.3f} "
        f"({'good diversity' if mean_h > 0.05 else 'low diversity — consider increasing entity_diversity_strength'})."
    )

    return ShieldMultiSplitResult(
        splits=splits,
        n_splits=n_splits,
        seed_base=seed_base,
        entity_diversity_strength=float(entity_diversity_strength),
        pairwise_hamming=H,
        pairwise_hamming_frac=H_frac,
        mean_pairwise_hamming_frac=mean_h,
        entity_fold_matrix=efm,
        entity_ids=eids_out,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _materialize_affinity(
    affinity: ArrayLike,
    device: torch.device,
) -> torch.Tensor:
    """Convert a user-supplied affinity into a torch tensor (dense or sparse)"""
    if isinstance(affinity, torch.Tensor):
        return affinity.to(device)
    try:
        import scipy.sparse as _sps
        if _sps.issparse(affinity):
            affinity = affinity.toarray()
    except ImportError:
        pass
    arr = np.asarray(affinity)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)
