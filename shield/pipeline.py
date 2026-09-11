"""
SHIELD pipeline 

    1) hierarchy LP in (y, w);
    2) D-term update (Sinkhorn for match, MMD for shift);
    3) z LP-relax update with all term gradients merged into the linear
       coefficient handed to Gurobi;
    4) rounding + entity repair + greedy leakage-aware swap repair;
    5) deterministic (y, w) recomputation from the rounded z;
    6) cold-* hard constraint verification + diagnostics.

The user interacts through `shield.api.split` / `shield.api.sweep`; this
module exposes the implementation primitive `run_shield`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from shield import coarsening, diagnostics, rounding, stratification
from shield.solvers import distributional, graph_cut, gurobi_lp


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------

@dataclass
class TermConfig:
    """Active-term toggles + normalized coefficients.

    The coefficients are *user-supplied* multipliers applied to Stage-0
    normalized terms. Defaults follow (nu=0, tau=0).
    """
    use_H: bool = False
    use_M: bool = False
    use_D: bool = False
    d_mode: str = "match"               # 'match' | 'shift'
    # backs label augmentation (under development, ships with SHIELD v1.1)
    use_label_augmentation: bool = False
    tau: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    mu: float = 1.0
    nu: float = 0.0


@dataclass
class TwoEntityConfig:
    """Cold-* hard-constraint specification"""
    mode: str = "warm"                   # 'warm' | 'cold-left' | 'cold-right' | 'cold-both'
    pair_indices: Optional[torch.Tensor] = None     # (P, 2) atom-index pairs
    left_entity_of: Optional[torch.Tensor] = None   # (n,)
    right_entity_of: Optional[torch.Tensor] = None  # (n,)


@dataclass
class StratificationConfig:
    """Stratification strategy + parameters"""
    strategy: str = "none"                # 'none' | 'quantile' | 'moment' | 'sliced_wasserstein'
    n_bins: int = 10
    P_proj: int = 100
    Q_grid: int = 256
    omega: Optional[torch.Tensor] = None   # moment weights (4, q)
    class_labels: Optional[torch.Tensor] = None      # (n,) for classification stratification
    class_delta: float = 0.05


@dataclass
class CoarseningConfig:
    """Stage-2 coarsening parameters"""
    enabled: bool = False
    n_super: int = 0                       # if 0 with enabled=True, set to max(8K, n/10)
    max_iter: int = 25
    balance_strength: float = 1.0


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration (mirrors `shield.split` signature)"""
    K: int = 3
    r: Optional[torch.Tensor] = None       # (K,) target proportions; defaults to uniform
    tolerance: float = 1e-4
    max_iter: int = 25
    epsilon_OT: float = 5e-2
    sinkhorn_max_iter: int = 200
    seed: int = 0
    device: str = "cpu"
    nodes: int = -1
    integer_z_final: bool = True
    balance_eps: float = 0.05
    stage0_samples: int = 100
    output_flag: int = 0
    enable_stage0: bool = True
    # Initializer back-end: 'auto' tries METIS, then KaHIP, then spectral.
    init_backend: str = "auto"
    # Rounding strategy: 'pipage' or 'confidence_gap'
    rounding_mode: str = "pipage"
    # Verify class-stratification tolerance after rounding.
    enforce_class_delta_after_round: bool = True
    # MILP polish is opt-in last resort: invoked only on user request and
    # residual repair failure.
    milp_last_resort_only: bool = True
    milp_size_limit: int = 10_000
    milp_time_limit_s: float = 120.0
    # ---- Stopping criterion 1: Z-assignment Hamming stability -------------------
    # Stop when the fraction of atoms changing fold between consecutive rounded
    # assignments is below z_hamming_eps for z_hamming_patience iterations.
    z_hamming_eps: float = 0.01          # fraction of atoms; 0.01 = 1%
    z_hamming_patience: int = 3          # consecutive stable iterations required
    # ---- Stopping criterion 2: per-term individual convergence -----------------
    # Stop when every active normalized term changes by less than
    # tolerance_per_term for patience_per_term consecutive iterations.
    tolerance_per_term: float = 1e-3
    patience_per_term: int = 3
    # ---- Case-specific internal thresholds (not exposed in public API yet) ----------
    # H-only case: LP y-solution stability (max |Δy_{g,k}| threshold).
    h_y_eps: float = 1e-4
    # D-match only: Sinkhorn outer-stability (max inner iters from warm-start).
    sinkhorn_conv_threshold: int = 5
    # ---- Multi-split diversity ---------------------------------------------------
    # Strength of seed-dependent entity-level perturbation in the LP objective;
    # 0.0 = off. Steers each seed toward a different valid fold assignment.
    # backs multi_split (under development, ships with SHIELD v1.1)
    entity_diversity_strength: float = 0.0


@dataclass
class ShieldOutput:
    """Bundle returned by `run_shield`"""
    z: torch.Tensor                                # (n, K) one-hot final assignment
    fold_assignment: torch.Tensor                  # (n,) int64 fold ids
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _default_proportions(K: int, device) -> torch.Tensor:
    return torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)


def _maybe_normalize(value: float, normalizer: float) -> float:
    if normalizer is None or normalizer == 0.0:
        return value
    return value / normalizer


def _torch_to_np(z: torch.Tensor) -> np.ndarray:
    return z.detach().cpu().to(torch.float64).numpy()


def _build_class_groups(
    class_labels: Optional[torch.Tensor],
) -> Tuple[Optional[Dict[int, list]], Optional[List[torch.Tensor]]]:
    """Convert (n,) class labels into (dict for Gurobi, list[Tensor] for swap)"""
    if class_labels is None:
        return None, None
    cls_dict: Dict[int, list] = {}
    cls_list: List[torch.Tensor] = []
    for c in torch.unique(class_labels).tolist():
        idx = torch.nonzero(class_labels == c, as_tuple=False).flatten()
        cls_dict[int(c)] = idx.detach().cpu().tolist()
        cls_list.append(idx)
    return cls_dict, cls_list


def _build_cold_groups(
    mode: str,
    entity_of: Optional[torch.Tensor],
) -> Optional[Dict[Any, list]]:
    if mode == "warm" or entity_of is None:
        return None
    eids = torch.unique(entity_of)
    out: Dict[Any, list] = {}
    for e in eids.tolist():
        if e < 0:
            continue
        idx = torch.nonzero(entity_of == e, as_tuple=False).flatten()
        out[int(e)] = idx.detach().cpu().tolist()
    return out


def _build_compound_cold_groups(
    left_entity_of: torch.Tensor,
    right_entity_of: torch.Tensor,
) -> Dict[int, list]:
    """Build compound (left, right) entity groups for cold-both mode"""
    n = int(left_entity_of.shape[0])
    left_cpu  = left_entity_of.detach().cpu().tolist()
    right_cpu = right_entity_of.detach().cpu().tolist()

    # Collect observation indices per unique (left, right) pair.
    pair_to_obs: Dict[tuple, list] = {}
    for i in range(n):
        l, r = int(left_cpu[i]), int(right_cpu[i])
        if l < 0 or r < 0:
            continue
        key = (l, r)
        if key not in pair_to_obs:
            pair_to_obs[key] = []
        pair_to_obs[key].append(i)

    # Assign deterministic integer IDs to each unique pair.
    return {gid: pair_to_obs[key]
            for gid, key in enumerate(sorted(pair_to_obs.keys()))}


def _recompute_y_w_from_rounded(
    z: torch.Tensor,
    levels: Sequence[gurobi_lp.HierarchyLevel],
) -> "gurobi_lp.HierarchySolution":
    K = int(z.shape[1])
    y_out: Dict = {}
    w_out: Dict = {}
    z_int = (z > 0.5).to(torch.float32)
    obj_val = 0.0
    for level in levels:
        # per-(g, k) "all descendants in same fold k" indicator
        per_node_y = []
        for g, desc in enumerate(level.descendants):
            if len(desc) == 0:
                yg = torch.zeros(K, device=z.device, dtype=torch.float32)
            else:
                desc_t = torch.as_tensor(desc, dtype=torch.long, device=z.device)
                yg = z_int.index_select(0, desc_t).min(dim=0).values
            per_node_y.append(yg)
            for k in range(K):
                y_out[(level.name, g, k)] = float(yg[k].item())
        if not per_node_y:
            continue
        y_mat = torch.stack(per_node_y, dim=0)             # (n_nodes, K)
        rows = torch.as_tensor(level.similarity_rows, dtype=torch.long, device=z.device)
        cols = torch.as_tensor(level.similarity_cols, dtype=torch.long, device=z.device)
        vals = torch.as_tensor(level.similarity_vals, dtype=torch.float32, device=z.device)
        if rows.numel() == 0:
            continue
        w_mat = y_mat[rows] * y_mat[cols]                  # (n_edges, K)
        for e in range(rows.shape[0]):
            for k in range(K):
                w_out[(level.name, int(rows[e].item()), int(cols[e].item()), k)] = \
                    float(w_mat[e, k].item())
        # contribution to the H objective
        sum_w_k = w_mat.sum(dim=1)                          # (n_edges,)
        obj_val += float(level.lambda_level) * float((vals * (1.0 - sum_w_k)).sum().item())
    return gurobi_lp.HierarchySolution(y=y_out, w_aux=w_out, objective=obj_val)


# ---------------------------------------------------------------------------
# Post-rounding cold-entity repair (majority vote per entity)
# ---------------------------------------------------------------------------

def _post_round_entity_repair(
    z: torch.Tensor,
    weights: torch.Tensor,
    cold_left_groups: Optional[Dict[Any, list]],
    cold_right_groups: Optional[Dict[Any, list]],
) -> Tuple[torch.Tensor, int]:
    """Force every cold-* entity into a single fold via weighted majority vote"""
    if not cold_left_groups and not cold_right_groups:
        return z, 0
    device = z.device
    K = int(z.shape[1])
    n = int(z.shape[0])
    w = weights.to(device).to(torch.float32)
    z_new = z.clone()
    moved = 0
    fold_of = z_new.argmax(dim=1)

    for groups in (cold_left_groups, cold_right_groups):
        if not groups:
            continue
        for _eid, obs_list in groups.items():
            if not obs_list:
                continue
            idx = torch.as_tensor(obs_list, dtype=torch.long, device=device)
            if idx.numel() <= 1:
                continue
            # weighted-mode fold for this entity
            tally = torch.zeros(K, device=device, dtype=torch.float32)
            tally.scatter_add_(0, fold_of[idx], w[idx])
            target_k = int(tally.argmax().item())
            wrong = fold_of[idx] != target_k
            if torch.any(wrong):
                moved += int(wrong.sum().item())
                z_new[idx] = 0.0
                z_new[idx, target_k] = 1.0
                fold_of[idx] = target_k
    return z_new, moved


# ---------------------------------------------------------------------------
# Stage-0 calibration helper
# ---------------------------------------------------------------------------

def _random_feasible_z(
    n: int,
    K: int,
    r: torch.Tensor,
    weights: torch.Tensor,
    *,
    device,
    seed: int,
) -> torch.Tensor:
    return graph_cut.random_balanced_init(
        n=n, K=K, r=r, weights=weights, device=device, seed=seed,
    )


def _evaluate_terms_for_z(
    z: torch.Tensor,
    *,
    weights: torch.Tensor,
    affinity: Optional[torch.Tensor],
    cost: Optional[torch.Tensor],
    target_marginals: Optional[torch.Tensor],
    rho: Optional[torch.Tensor],
    kernel_matrix: Optional[torch.Tensor],
    rho_pair: Optional[torch.Tensor],
    hierarchy_levels: Optional[Sequence[gurobi_lp.HierarchyLevel]],
    terms: TermConfig,
    epsilon_OT: float,
    sinkhorn_max_iter: int,
    strat_cfg: StratificationConfig,
    y_target: Optional[torch.Tensor],
    bins: Optional[stratification.QuantileBins],
    r: torch.Tensor,
    verbose: bool = False,
) -> Dict[str, float]:
    """Compute raw term values for a fixed z (used by Stage-0 calibration)"""
    out: Dict[str, float] = {}
    if terms.use_M and affinity is not None:
        out["M"] = graph_cut.cut_energy(z, affinity)
    if terms.use_D:
        if terms.d_mode == "match":
            res = distributional.evaluate_d_term(
                z, weights, mode="match",
                C=cost, target_marginals=target_marginals,
                rho=rho, epsilon_OT=epsilon_OT,
                sinkhorn_max_iter=sinkhorn_max_iter,
                verbose=verbose,
            )
            out["D"] = float((rho.to(res.per_fold_cost.device)
                              * res.per_fold_cost).sum().item())
        else:
            res = distributional.evaluate_d_term(
                z, weights, mode="shift",
                K_mat=kernel_matrix, rho_pair=rho_pair,
                verbose=verbose,
            )
            out["D"] = res.total
    if terms.use_H and hierarchy_levels is not None:
        # H term value is independent of (y, w) once z is fixed; we evaluate
        # using y_{gk}=prod_{i in D(g)} z_{ik} and w=y*y' directly.
        out["H"] = _evaluate_h_term_from_z(z, hierarchy_levels)
    bv = diagnostics.balance_violation(z, weights, r)
    out["bal"] = float((bv * bv).sum().item())
    if strat_cfg.strategy != "none":
        rep = stratification.evaluate_stratification(
            z, weights,
            strategy=strat_cfg.strategy,
            y=y_target,
            bins=bins,
            r=r,
            omega=strat_cfg.omega,
            P=strat_cfg.P_proj,
            Q=strat_cfg.Q_grid,
        )
        out["strat"] = rep.total
    return out


def _compute_h_gradient_z(
    h_sol: "gurobi_lp.HierarchySolution",
    levels: Sequence[gurobi_lp.HierarchyLevel],
    n: int,
    K: int,
    device,
) -> torch.Tensor:
    """Subgradient of L_H w.r.t. z from the current H-LP solution (y*, w*)

    grad_H[i, k] += lambda_l * sim_total_g * (1 - 2 * y*_{g,k}) for atom i
    in family g at level l; zero at y*=0.5.
    """
    grad = torch.zeros((n, K), device=device, dtype=torch.float32)
    for level in levels:
        rows_np = np.asarray(level.similarity_rows, dtype=np.int64)
        cols_np = np.asarray(level.similarity_cols, dtype=np.int64)
        vals_np = np.asarray(level.similarity_vals, dtype=np.float64)
        keep = vals_np > 0.0
        rows_np = rows_np[keep]
        cols_np = cols_np[keep]
        vals_np = vals_np[keep]
        n_nodes = len(level.descendants)
        # per-node total similarity weight 
        sim_total = np.zeros(n_nodes, dtype=np.float64)
        for ei in range(len(rows_np)):
            sim_total[rows_np[ei]] += vals_np[ei]
            sim_total[cols_np[ei]] += vals_np[ei]
        lambda_l = float(level.lambda_level)
        for g, desc in enumerate(level.descendants):
            if len(desc) == 0 or sim_total[g] == 0.0:
                continue
            desc_t = torch.as_tensor(desc, dtype=torch.long, device=device)
            for k in range(K):
                y_gk = float(h_sol.y.get((level.name, g, k), 0.0))
                coeff = lambda_l * sim_total[g] * (1.0 - 2.0 * y_gk)
                if abs(coeff) > 1e-15:
                    grad[desc_t, k] = grad[desc_t, k] + float(coeff)
    return grad


def _evaluate_h_term_from_z(
    z: torch.Tensor,
    levels: Sequence[gurobi_lp.HierarchyLevel],
) -> float:
    """L_H using y_{g,k} = min_{i in D(g)} z_{i,k}

    Exact for one-hot z; matches the LP containment constraint
    y_{g,k} <= z_{i,k} for fractional z, so the surrogate stays continuous
    """
    total = 0.0
    K = z.shape[1]
    for level in levels:
        y_vals: List[torch.Tensor] = []
        for desc in level.descendants:
            if len(desc) == 0:
                y_vals.append(torch.zeros(K, device=z.device, dtype=torch.float32))
                continue
            desc_t = torch.as_tensor(desc, dtype=torch.long, device=z.device)
            sub = z.index_select(0, desc_t)
            # y_{g,k} = min_{i in D(g)} z_{i,k}
            y_g = sub.min(dim=0).values.to(torch.float32)
            y_vals.append(y_g)
        y_mat = (torch.stack(y_vals, dim=0)
                 if y_vals else torch.zeros((0, K), device=z.device))
        rows = torch.as_tensor(level.similarity_rows, dtype=torch.long, device=z.device)
        cols = torch.as_tensor(level.similarity_cols, dtype=torch.long, device=z.device)
        vals = torch.as_tensor(level.similarity_vals, dtype=torch.float32, device=z.device)
        if rows.numel() == 0:
            continue
        prod = (y_mat[rows] * y_mat[cols]).sum(dim=1)
        total += float(level.lambda_level) * float((vals * (1.0 - prod)).sum().item())
    return total


# ---------------------------------------------------------------------------
# Per-iteration progress logger
# ---------------------------------------------------------------------------

_LOG_W = 80  # terminal width for separator lines


def _log_iteration(
    it: int,
    term_values: Dict[str, float],
    composite_obj: float,
    prev_obj: float,
    alignment: Dict[str, float],
    z_round: torch.Tensor,
    weights: torch.Tensor,
    r: torch.Tensor,
    tol: float,
    wall_s: float,
) -> None:
    W = float(weights.sum().item())
    fold_mass = diagnostics.fold_sizes(z_round, weights)
    actual_pcts = [f"{float(v):.1f}" for v in (fold_mass / W * 100).tolist()]
    target_pcts = [f"{float(v):.1f}" for v in (r.to(z_round.device) * 100).tolist()]
    bal_viol = diagnostics.balance_violation(z_round, weights, r)
    max_viol_pct = float(bal_viol.abs().max().item()) / W * 100

    delta = abs(prev_obj - composite_obj)
    delta_str = "---" if math.isinf(delta) else f"{delta:.4g}"

    header = f" Iteration {it}  ({wall_s:.2f}s) "
    top = "━" * 3 + header + "━" * max(0, _LOG_W - 3 - len(header))
    bot = "─" * _LOG_W
    L = 24  # label column width

    lines = [top]
    lines.append(
        f"  {'Composite objective':<{L}}: {composite_obj:>10.4f}  (Δ={delta_str})"
        f"   target → minimize; converges when Δ < {tol:.2g}"
    )
    lines.append(
        f"  {'Balance violation':<{L}}: {max_viol_pct:>8.3f}%"
        f"                        target → 0%  (each fold reaches its proportional size r_k)"
    )
    lines.append(
        f"  {'Fold weights (actual)':<{L}}: [{', '.join(actual_pcts)}]%"
        f"   target → [{', '.join(target_pcts)}]%"
    )

    if "H_normalized" in term_values:
        lines.append(
            f"  {'H — hierarchy':<{L}}: norm={term_values['H_normalized']:>8.4f}"
            f"  raw={term_values['H_raw']:>10.3g}"
            f"   target → 0  (all families assigned to a single fold)"
        )
    if "M_normalized" in term_values:
        lines.append(
            f"  {'M — affinity cut':<{L}}: norm={term_values['M_normalized']:>8.4f}"
            f"  raw={term_values['M_raw']:>10.3g}"
            f"   target → 0  (no edges in the affinity graph cut across folds)"
        )
    if "D_normalized" in term_values:
        lines.append(
            f"  {'D — distribution':<{L}}: norm={term_values['D_normalized']:>8.4f}"
            f"  raw={term_values['D_raw']:>10.3g}"
            f"   target → 0  (each fold's feature distribution matches the target)"
        )
    if "strat_normalized" in term_values:
        lines.append(
            f"  {'Strat — quantile':<{L}}: norm={term_values['strat_normalized']:>8.4f}"
            f"  raw={term_values['strat_raw']:>10.3g}"
            f"   target → 0  (outcome quantiles balanced evenly across folds)"
        )

    if alignment:
        pairs = "  ".join(
            f"{k.replace('/', '↔')}={v:+.2f}" for k, v in alignment.items()
        )
        lines.append(f"  {'Gradient alignment':<{L}}: {pairs}")
        lines.append(
            f"  {'':<{L}}  "
            f"(positive = terms agree on split direction; negative = terms conflict)"
        )

    lines.append(bot)
    print("\n".join(lines))


def _log_gradient_analysis(
    grads_for_alignment: Dict[str, torch.Tensor],
    normalizers: Dict[str, float],
    term_coeffs: Dict[str, float],
) -> None:
    if not grads_for_alignment:
        return
    report = diagnostics.effective_gradient_magnitudes(
        grads_for_alignment, normalizers, term_coeffs
    )
    eff_norms  = report["effective_norms"]
    cosines    = report["pairwise_cosines"]
    dominant   = report["dominant_term"]

    norm_str = "  ".join(f"{k}={v:.3g}" for k, v in eff_norms.items())
    print(f"  [grad magnitudes] {norm_str}   dominant={dominant}")

    conflicts = {k: v for k, v in cosines.items() if v < 0}
    if conflicts:
        conf_str = "  ".join(f"{k}={v:+.3f}" for k, v in conflicts.items())
        print(f"  [gradient CONFLICT (<0)] {conf_str}")
    else:
        cos_str = "  ".join(f"{k}={v:+.3f}" for k, v in cosines.items())
        print(f"  [gradient cosines (all >=0: aligned)] {cos_str}")


# ---------------------------------------------------------------------------
# H-aware initialization (used when M is inactive)
# ---------------------------------------------------------------------------

def _hierarchy_group_pack_init(
    hierarchy_levels: Sequence[gurobi_lp.HierarchyLevel],
    K: int,
    r: torch.Tensor,
    weights: torch.Tensor,
    device,
    seed: int,
) -> torch.Tensor:
    n = int(weights.shape[0])
    w = weights.to(device).to(torch.float32)
    W = float(w.sum().item())
    r_dev = r.to(device).to(torch.float32)
    targets = r_dev * W                                   # (K,) target mass per fold

    z = torch.zeros((n, K), device=device, dtype=torch.float32)
    fold_load = torch.zeros(K, device=device, dtype=torch.float32)

    level0 = hierarchy_levels[0]
    groups = level0.descendants                           # list[list[int]]

    # Compute weighted size of every group
    group_sizes = []
    for desc in groups:
        if len(desc) == 0:
            group_sizes.append(0.0)
        else:
            idx = torch.as_tensor(desc, dtype=torch.long, device=device)
            group_sizes.append(float(w[idx].sum().item()))

    order = sorted(range(len(groups)), key=lambda i: -group_sizes[i])

    assigned_atoms: set = set()
    for gi in order:
        desc = groups[gi]
        if len(desc) == 0:
            continue
        # Assign to the fold most under-filled relative to its target.
        slack = targets - fold_load                       # positive = under-filled
        assign_k = int(slack.argmax().item())
        idx = torch.as_tensor(desc, dtype=torch.long, device=device)
        z[idx, assign_k] = 1.0
        fold_load[assign_k] += group_sizes[gi]
        assigned_atoms.update(desc)

    # Collect uncovered atoms (atoms in no H group, or in empty groups).
    all_atoms = set(range(n))
    uncovered = sorted(all_atoms - assigned_atoms)
    if uncovered:
        unc_t = torch.as_tensor(uncovered, dtype=torch.long, device=device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(len(uncovered), generator=g)
        for pos, ui in enumerate(perm.tolist()):
            atom_idx = uncovered[ui]
            slack = targets - fold_load
            k = int(slack.argmax().item())
            z[atom_idx, k] = 1.0
            fold_load[k] += float(w[atom_idx].item())

    unset = (z.sum(dim=1) == 0).nonzero(as_tuple=False).flatten()
    if unset.numel() > 0:
        z[unset, 0] = 1.0

    return z


# ---------------------------------------------------------------------------
# H-group post-rounding repair (H-solo only)
# ---------------------------------------------------------------------------

def _post_round_h_group_repair(
    z: torch.Tensor,
    hierarchy_levels: Sequence[gurobi_lp.HierarchyLevel],
    h_sol_pre_round: gurobi_lp.HierarchySolution,
    weights: torch.Tensor,
    r: torch.Tensor,
    balance_eps: float,
) -> Tuple[torch.Tensor, int]:
    """Majority-vote repair for H-groups after rounding (H-solo mode only)"""
    device = z.device
    K = int(z.shape[1])
    w = weights.to(device).to(torch.float32)
    W = float(w.sum().item())
    r_dev = r.to(device).to(torch.float32)
    targets = r_dev * W                                   # (K,) target mass
    cap = targets + balance_eps * W                       # per-fold upper capacity

    z_new = z.clone()
    fold_of = z_new.argmax(dim=1)
    total_moved = 0

    level = hierarchy_levels[0]
    level_name = level.name

    committed: List[Tuple[int, int, int, list]] = []
    for g, desc in enumerate(level.descendants):
        if len(desc) == 0:
            continue
        for k in range(K):
            y_val = float(h_sol_pre_round.y.get((level_name, g, k), 0.0))
            if y_val >= 0.5:
                idx = torch.as_tensor(desc, dtype=torch.long, device=device)
                n_escaped = int((fold_of[idx] != k).sum().item())
                if n_escaped > 0:
                    committed.append((k, n_escaped, g, desc))
                break  # each group is committed to at most one fold

    committed.sort(key=lambda t: -t[1])

    fold_load = (w.unsqueeze(1) * z_new).sum(dim=0)      # (K,) current load

    for committed_k, _, _, desc in committed:
        idx = torch.as_tensor(desc, dtype=torch.long, device=device)
        escaped_mask = fold_of[idx] != committed_k
        if not escaped_mask.any():
            continue
        escaped_idx = idx[escaped_mask]
        # Check balance: would adding escaped atoms to committed_k overflow?
        mass_to_move = float(w[escaped_idx].sum().item())
        if float(fold_load[committed_k].item()) + mass_to_move > float(cap[committed_k].item()):
            # Skip: repairing this group would violate the balance cap.
            continue
        # Repair: move escaped atoms into the committed fold.
        for src_k in range(K):
            if src_k == committed_k:
                continue
            in_src = (fold_of[escaped_idx] == src_k)
            if not in_src.any():
                continue
            moving = escaped_idx[in_src]
            z_new[moving, src_k] = 0.0
            z_new[moving, committed_k] = 1.0
            fold_load[committed_k] += float(w[moving].sum().item())
            fold_load[src_k] -= float(w[moving].sum().item())
            total_moved += int(moving.shape[0])
        fold_of = z_new.argmax(dim=1)

    return z_new, total_moved


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_shield(
    *,
    weights: torch.Tensor,
    n: int,
    affinity: Optional[torch.Tensor] = None,
    cost: Optional[torch.Tensor] = None,
    target_marginals: Optional[torch.Tensor] = None,
    rho: Optional[torch.Tensor] = None,
    kernel_matrix: Optional[torch.Tensor] = None,
    rho_pair: Optional[torch.Tensor] = None,
    hierarchy_levels: Optional[Sequence[gurobi_lp.HierarchyLevel]] = None,
    y_target: Optional[torch.Tensor] = None,
    two_entity: Optional[TwoEntityConfig] = None,
    terms: Optional[TermConfig] = None,
    strat_cfg: Optional[StratificationConfig] = None,
    coarsen_cfg: Optional[CoarseningConfig] = None,
    cfg: Optional[PipelineConfig] = None,
    feature_matrix: Optional[torch.Tensor] = None,
    pair_observation_targets: Optional[torch.Tensor] = None,
    # ---- Diagnostic inputs ------------------
    embedding: Optional[torch.Tensor] = None,        # (n, d) for kNN y_hat_emb
    y_hat_emb: Optional[torch.Tensor] = None,        # explicit predictions
    yhat_emb_knn_k: int = 15,
    spearman_threshold: float = 0.05,
    precomputed_normalizers: Optional[Dict[str, float]] = None,
) -> ShieldOutput:

    if cfg is None:
        cfg = PipelineConfig()
    if terms is None:
        terms = TermConfig()
    if strat_cfg is None:
        strat_cfg = StratificationConfig()
    if coarsen_cfg is None:
        coarsen_cfg = CoarseningConfig()
    if two_entity is None:
        two_entity = TwoEntityConfig()

    threads = coarsening.configure_threading(cfg.nodes)
    device = coarsening.resolve_device(cfg.device)
    weights = weights.to(device).to(torch.float32)
    r = cfg.r.to(device).to(torch.float32) if cfg.r is not None else _default_proportions(cfg.K, device)
    K = cfg.K

    metadata: Dict[str, Any] = {
        "device": str(device),
        "threads": threads,
        "K": K,
        "n": int(n),
        "stage0_samples": cfg.stage0_samples,
        "d_mode": terms.d_mode,
        "tau": terms.tau,
        "nu": terms.nu,
        "two_entity_mode": two_entity.mode,
        "stratification_strategy": strat_cfg.strategy,
        "coarsening_enabled": coarsen_cfg.enabled,
    }

    # ---- Stage-2 coarsening (optional) ------------------------------------
    # Save atom-level copies before any projection so Stage 6 diagnostics
    # always operate on the original n-scale data.
    y_target_atom: Optional[torch.Tensor] = y_target
    class_labels_atom: Optional[torch.Tensor] = strat_cfg.class_labels
    # class_labels_for_lp may be projected to super-node level below.
    class_labels_for_lp: Optional[torch.Tensor] = strat_cfg.class_labels

    coarsening_result = None
    super_to_atoms = None
    if coarsen_cfg.enabled and feature_matrix is not None:
        n_super = coarsen_cfg.n_super
        if n_super <= 0:
            # Cap auto formula at 45000 super-nodes to keep the K-means
            # E-step (O(n * n_super * d)) tractable at large n.
            n_super = min(45_000, n // 10)
            print(
                f"[shield.coarsening] n_super_default = min(45000, n//10) "
                f"= {n_super} chosen because coarsen_cfg.n_super<=0 "
                f"(n={n}, K={K}). Set coarsen_cfg.n_super explicitly if you "
                f"need a different value; keep n_super << n for efficiency."
            )
        coarsening_result = coarsening.coarsen_balanced_kmeans(
            feature_matrix.to(device),
            n_clusters=n_super,
            weights=weights,
            max_iter=coarsen_cfg.max_iter,
            balance_strength=coarsen_cfg.balance_strength,
            seed=cfg.seed,
            device=device,
        )
        super_to_atoms = coarsening_result.labels
        # operate downstream at the super-node level
        atom_weights = weights
        weights = coarsening_result.super_weights.clone()
        n_eff = int(coarsening_result.centroids.shape[0])
    else:
        if coarsen_cfg.enabled and feature_matrix is None:
            # secondary guard (primary check is in api.split())
            raise ValueError(
                "[shield.pipeline] coarsen=True requires feature_matrix. "
                "Pass feature_matrix=(n, d) float tensor or set coarsen=False."
            )
        atom_weights = weights
        n_eff = n

    # ---- Project relational matrices + y_target onto super-node domain ----
    if super_to_atoms is not None and n_eff < n:
        # Always project y_target when coarsening is active so stratification,
        # quantile bins, and Stage-0 calibration see (n_eff, q) targets.
        if y_target is not None:
            agg = coarsening.aggregate_super_nodes(
                labels=super_to_atoms,
                weights=atom_weights,
                targets=y_target,
                n_clusters=n_eff,
            )
            y_target = agg["targets"]

        if class_labels_atom is not None:
            agg_cls = coarsening.aggregate_super_nodes(
                labels=super_to_atoms,
                weights=atom_weights,
                classes=class_labels_atom,
                n_clusters=n_eff,
            )
            class_labels_for_lp = agg_cls["classes"]

        active_relational = (
            (terms.use_H and hierarchy_levels is not None) or
            (terms.use_M and affinity is not None) or
            (terms.use_D and (cost is not None or kernel_matrix is not None))
        )
        if active_relational:
            print(
                "[shield.coarsening] Projecting H/M/D relational matrices onto "
                f"{n_eff} super-nodes (Galerkin coarsening). Note: this is an "
                "approximation — term objectives at super-node level differ from "
                "the exact atom-level objectives."
            )
            if cost is not None:
                agg = coarsening.aggregate_super_nodes(
                    labels=super_to_atoms,
                    weights=atom_weights,
                    features=cost,
                    n_clusters=n_eff,
                )
                cost = agg["features"]
            # Memory-safe Galerkin projection P^T M P; sparse affinities stay
            # sparse (O(nnz)), avoiding a dense (n, n_eff) one-hot
            if affinity is not None:
                affinity = coarsening.galerkin_project(
                    affinity, super_to_atoms, n_eff,
                )
            if kernel_matrix is not None:
                kernel_matrix = coarsening.galerkin_project(
                    kernel_matrix, super_to_atoms, n_eff,
                )
            if hierarchy_levels is not None:

                super_to_atoms_np = super_to_atoms.cpu().numpy()
                new_hierarchy_levels = []
                for level in hierarchy_levels:
                    new_desc = []
                    for desc in level.descendants:
                        if len(desc) == 0:
                            new_desc.append([])
                            continue
                        super_ids = list(set(int(super_to_atoms_np[i]) for i in desc))
                        new_desc.append(super_ids)
                    new_hierarchy_levels.append(gurobi_lp.HierarchyLevel(
                        name=level.name,
                        descendants=new_desc,
                        similarity_rows=level.similarity_rows,
                        similarity_cols=level.similarity_cols,
                        similarity_vals=level.similarity_vals,
                        lambda_level=level.lambda_level,
                    ))
                hierarchy_levels = new_hierarchy_levels

    bins = (stratification.build_quantile_bins(y_target, B=strat_cfg.n_bins)
            if (strat_cfg.strategy == "quantile" and y_target is not None) else None)

    # ---- Stage-0 calibration ---------------------------------------------
    normalizers: Dict[str, float] = {}
    if precomputed_normalizers is not None:
        normalizers = dict(precomputed_normalizers)
        metadata["stage0_normalizers"] = {k: float(v) for k, v in normalizers.items()}
        metadata["stage0_source"] = "precomputed"
    elif cfg.enable_stage0:
        def _sample_fn(seed):
            z_rand = _random_feasible_z(
                n_eff, K, r, weights,
                device=device, seed=seed,
            )
            return _evaluate_terms_for_z(
                z_rand,
                weights=weights,
                affinity=affinity,
                cost=cost,
                target_marginals=target_marginals,
                rho=rho,
                kernel_matrix=kernel_matrix,
                rho_pair=rho_pair,
                hierarchy_levels=hierarchy_levels,
                terms=terms,
                epsilon_OT=cfg.epsilon_OT,
                sinkhorn_max_iter=cfg.sinkhorn_max_iter,
                strat_cfg=strat_cfg,
                y_target=y_target,
                bins=bins,
                r=r,
            )
        normalizers = diagnostics.stage0_calibrate(
            _sample_fn,
            n_samples=cfg.stage0_samples,
            seed=cfg.seed,
            n_jobs=cfg.nodes,
        )
        metadata["stage0_normalizers"] = {k: float(v) for k, v in normalizers.items()}
        metadata["stage0_source"] = "mc"

    _precomp_keys = (set(precomputed_normalizers.keys())
                     if precomputed_normalizers is not None else set())

    # Stratification normalizer: 
    #   - quantile: worst-case bound -> strat_normalized in [0, 1]
    #   - moment: range bounds (odd orders) / pooled-moment squared (even)
    #   - sliced_wasserstein: covariance-trace reference K * tr(Cov_w(y))
    if (strat_cfg.strategy == "quantile" and bins is not None
            and "strat" not in _precomp_keys):
        normalizers["strat"] = stratification.quantile_worst_case_scale(
            bins, weights, r,
        )
        metadata.setdefault("stage0_normalizers", {})["strat"] = float(
            normalizers["strat"]
        )
    elif (strat_cfg.strategy == "moment" and y_target is not None
          and "strat" not in _precomp_keys):
        normalizers["strat"] = stratification.moment_worst_case_scale(
            y_target, weights, r, omega=strat_cfg.omega,
        )
        metadata.setdefault("stage0_normalizers", {})["strat"] = float(
            normalizers["strat"]
        )
    elif (strat_cfg.strategy == "sliced_wasserstein" and y_target is not None
          and "strat" not in _precomp_keys):
        normalizers["strat"] = stratification.sw_covariance_trace_scale(
            y_target, weights, r,
        )
        metadata.setdefault("stage0_normalizers", {})["strat"] = float(
            normalizers["strat"]
        )

    # D-shift normalizer
    if (terms.use_D and terms.d_mode == "shift" and kernel_matrix is not None
            and "D" not in _precomp_keys):
        if rho_pair is None:
            rho_pair_eff = torch.ones((K, K), device=device, dtype=torch.float32)
            rho_pair_eff.fill_diagonal_(0.0)
        else:
            rho_pair_eff = rho_pair
        normalizers["D"] = distributional.mmd_reference_scale(
            kernel_matrix, rho_pair_eff,
        )
        metadata.setdefault("stage0_normalizers", {})["D"] = float(
            normalizers["D"]
        )

    _active_norm_keys = []
    if terms.use_H:
        _active_norm_keys.append("H")
    if terms.use_M:
        _active_norm_keys.append("M")
    if terms.use_D:
        _active_norm_keys.append("D")
    if strat_cfg.strategy != "none":
        _active_norm_keys.append("strat")
    _missing_norm = [k for k in _active_norm_keys if k not in normalizers]
    if _missing_norm:
        metadata["unnormalized_terms"] = _missing_norm
        print(
            f"[shield.pipeline] WARNING — Stage-0 normalizers absent for active "
            f"term(s) {_missing_norm}; they default to scale 1.0 (raw magnitude). "
            "pipeline.py — set enable_stage0=True or pass precomputed_normalizers "
            "so coefficients act on comparably-scaled terms."
        )

    # ---- Mode flags for H-aware behavior --------------------------------
    # _h_no_m : H active, M inactive (H-only or H+D). Selects group-packing
    #           init and the 0.90 pin-IN threshold (no affinity graph exists).
    # _h_solo : H active, M and D both inactive. Activates post-rounding
    #           H-group repair and the strict pin-OUT constraint.
    _h_no_m = terms.use_H and hierarchy_levels is not None and not terms.use_M
    _h_solo  = _h_no_m and not terms.use_D

    # ---- Stage-3 initial assignment --------------------------------------
    init_backend_used = "random"
    if _h_no_m:
        z_curr = _hierarchy_group_pack_init(
            hierarchy_levels, K, r, weights, device, seed=cfg.seed,
        )
        init_backend_used = "h_group_pack"
    elif terms.use_M and affinity is not None:
        # METIS -> KaHIP -> spectral cascade 
        z_curr, init_backend_used = graph_cut.balanced_partition_init(
            affinity, K=K, r=r, weights=weights,
            backend=cfg.init_backend, device=device, seed=cfg.seed,
        )
    else:
        z_curr = _random_feasible_z(n_eff, K, r, weights, device=device, seed=cfg.seed)
    metadata["init_backend"] = init_backend_used
    metadata["h_mode"] = {"h_no_m": _h_no_m, "h_solo": _h_solo}

    # ---- two-entity hard constraints --------------------------------------
    cold_left_groups = None
    cold_right_groups = None
    if two_entity.mode == "cold-both":
        # Compound (left, right) entities avoid the fold-collapse that
        # separate left/right constraints cause on a connected co-occurrence
        # graph. 
        if (two_entity.left_entity_of is not None
                and two_entity.right_entity_of is not None):
            cold_left_groups = _build_compound_cold_groups(
                two_entity.left_entity_of, two_entity.right_entity_of,
            )
    elif two_entity.mode == "cold-left" and two_entity.left_entity_of is not None:
        cold_left_groups = _build_cold_groups(two_entity.mode, two_entity.left_entity_of)
    elif two_entity.mode == "cold-right" and two_entity.right_entity_of is not None:
        cold_right_groups = _build_cold_groups(two_entity.mode, two_entity.right_entity_of)
    class_groups_gurobi, class_groups_swap = _build_class_groups(class_labels_for_lp)

    history: List[Dict[str, Any]] = []
    prev_obj = math.inf                  # kept for iteration logging only
    h_sol: Optional[gurobi_lp.HierarchySolution] = None
    z_linear_coeff = torch.zeros((n_eff, K), device=device, dtype=torch.float32)
    # per-fold (f, g) Sinkhorn dual potentials carried across iterations.
    sinkhorn_warm: Optional[list] = None

    # ---- Stopping-criterion state ----------------------------------------
    # Criterion 1: Z-Assignment Hamming Stability
    z_prev_round: Optional[torch.Tensor] = None
    hamming_stable_count: int = 0
    # Criterion 2: Per-Term Individual Convergence
    prev_term_values: Dict[str, float] = {}
    per_term_stable_count: int = 0
    # Criterion 3 (case-specific): H-LP y-solution stability (H-only)
    h_sol_prev: Optional[gurobi_lp.HierarchySolution] = None
    h_y_stable_count: int = 0
    # Criterion 3 (case-specific): Sinkhorn outer-stability (D-match only)
    sinkhorn_stable_count: int = 0

    # ---- Algorithm 2 alternation loop ------------------------------------
    for it in range(cfg.max_iter):
        t0 = time.time()
        z_linear_coeff = torch.zeros((n_eff, K), device=device, dtype=torch.float32)
        grads_for_alignment: Dict[str, torch.Tensor] = {}
        term_values: Dict[str, float] = {}

        h_sol = None
        if terms.use_H and hierarchy_levels is not None:
            try:
                h_sol = gurobi_lp.solve_hierarchy_lp(
                    z=_torch_to_np(z_curr),
                    levels=hierarchy_levels,
                    K=K,
                    output_flag=cfg.output_flag,
                )
                h_norm = max(1e-12, normalizers.get("H", 1.0))
                h_value = h_sol.objective / h_norm
                term_values["H_normalized"] = h_value
                term_values["H_raw"] = h_sol.objective
                grad_h = _compute_h_gradient_z(h_sol, hierarchy_levels, n_eff, K, device)
                z_linear_coeff += terms.alpha * grad_h / h_norm
                grads_for_alignment["H"] = grad_h
            except RuntimeError as e:
                metadata.setdefault("warnings", []).append(f"H LP error: {e}")

        # ---- M block: spectral gradient at current z --------------------
        if terms.use_M and affinity is not None:
            grad_m = graph_cut.m_gradient_z(z_curr, affinity)
            m_value = graph_cut.cut_energy(z_curr, affinity)
            m_norm = m_value / max(1e-12, normalizers.get("M", 1.0))
            term_values["M_normalized"] = m_norm
            term_values["M_raw"] = m_value
            z_linear_coeff += terms.beta * grad_m / max(1e-12, normalizers.get("M", 1.0))
            grads_for_alignment["M"] = grad_m

        # ---- D block: Sinkhorn or MMD ----------------------------------
        d_eval = None
        if terms.use_D:
            d_eval = distributional.evaluate_d_term(
                z_curr, weights,
                mode=terms.d_mode,
                C=cost, target_marginals=target_marginals, rho=rho,
                epsilon_OT=cfg.epsilon_OT,
                K_mat=kernel_matrix, rho_pair=rho_pair,
                sinkhorn_max_iter=cfg.sinkhorn_max_iter,
                warm_start=sinkhorn_warm,
                verbose=True,
            )
            if terms.d_mode == "match" and hasattr(d_eval, "plans"):
                sinkhorn_warm = [(p.f, p.g) if p is not None else None
                                 for p in d_eval.plans]

            if terms.d_mode == "match":
                d_value = float((rho.to(d_eval.per_fold_cost.device)
                                 * d_eval.per_fold_cost).sum().item())
            else:
                d_value = d_eval.total
            d_norm = d_value / max(1e-12, normalizers.get("D", 1.0))
            term_values["D_normalized"] = d_norm
            term_values["D_raw"] = d_value
            z_linear_coeff += terms.gamma * d_eval.gradient_z \
                               / max(1e-12, normalizers.get("D", 1.0))
            grads_for_alignment["D"] = d_eval.gradient_z

        # ---- stratification gradient ------------------------------------
        strat_eval: Optional[stratification.StratEvaluation] = None
        if strat_cfg.strategy != "none":
            strat_eval = stratification.evaluate_stratification(
                z_curr, weights,
                strategy=strat_cfg.strategy,
                y=y_target,
                bins=bins,
                r=r,
                omega=strat_cfg.omega,
                P=strat_cfg.P_proj,
                Q=strat_cfg.Q_grid,
                seed=cfg.seed + it,
            )
            term_values["strat_normalized"] = (
                strat_eval.total / max(1e-12, normalizers.get("strat", 1.0))
            )
            term_values["strat_raw"] = strat_eval.total
            z_linear_coeff += terms.mu * strat_eval.gradient_z \
                               / max(1e-12, normalizers.get("strat", 1.0))
            grads_for_alignment["strat"] = strat_eval.gradient_z



        # ---- diversity perturbation (multi_split / CV) -----------------------
        # Seed-dependent noise steers the solver toward different valid
        # assignments per seed. cfg.seed (not cfg.seed+it) keeps the push
        # consistent across iterations within a single split.
        if cfg.entity_diversity_strength > 0.0:
            _div_rng = np.random.default_rng(cfg.seed)
            if cold_left_groups or cold_right_groups:
                # cold-* mode: share one noise vector per entity so all atoms of
                # the same entity are steered toward the same fold.
                for _div_groups in (cold_left_groups, cold_right_groups):
                    if not _div_groups:
                        continue
                    for _eid, _obs_list in _div_groups.items():
                        _noise = _div_rng.standard_normal(K).astype(np.float32)
                        _noise *= float(cfg.entity_diversity_strength)
                        _noise_t = torch.as_tensor(_noise, device=device)
                        _idx_t = torch.as_tensor(_obs_list, dtype=torch.long, device=device)
                        z_linear_coeff[_idx_t] += _noise_t
            else:
                # warm mode: no entity groups, so draw an independent noise
                # vector per observation.
                _noise = _div_rng.standard_normal((n_eff, K)).astype(np.float32)
                _noise *= float(cfg.entity_diversity_strength)
                z_linear_coeff += torch.as_tensor(_noise, device=device)



        # ---- effective gradient magnitude analysis (diagnostic) --------
        # Builds coeff map from active terms and logs norms + conflict cosines.
        _term_coeffs_diag: Dict[str, float] = {}
        if terms.use_H:   _term_coeffs_diag["H"]    = terms.alpha
        if terms.use_M:   _term_coeffs_diag["M"]    = terms.beta
        if terms.use_D:   _term_coeffs_diag["D"]    = terms.gamma
        if strat_cfg.strategy != "none": _term_coeffs_diag["strat"] = terms.mu
        _log_gradient_analysis(grads_for_alignment, normalizers, _term_coeffs_diag)

        hard = gurobi_lp.HardConstraints(
            cold_left_groups=cold_left_groups,
            cold_right_groups=cold_right_groups,
            class_groups=class_groups_gurobi,
            class_weights=_torch_to_np(weights),
            class_delta=strat_cfg.class_delta,
            balance_eps=cfg.balance_eps,
        )
        
        _pin_in_frac  = 0.90 if _h_no_m else 0.70
        
        _pin_out_strict = _h_solo

        _lp_ok = True   
        try:
            sol = gurobi_lp.solve_assignment_lp(
                n=n_eff, K=K,
                weights=_torch_to_np(weights),
                r=_torch_to_np(r),
                z_linear_coeff=_torch_to_np(z_linear_coeff),
                eta_balance=terms.eta / max(1e-12, normalizers.get("bal", 1.0)),
                hard=hard,
                hierarchy_levels=(hierarchy_levels
                                  if (terms.use_H and h_sol is not None)
                                  else None),
                y_fixed=(h_sol.y
                         if (terms.use_H and h_sol is not None)
                         else None),
                pin_in_frac=_pin_in_frac,
                pin_out_strict=_pin_out_strict,
                integer=False,
                output_flag=cfg.output_flag,
            )
            Z = gurobi_lp.to_tensor(sol.z, device)
        except RuntimeError as e:
            _lp_ok = False
            warn_msg = f"Assignment LP fallback (iteration {it}): {e}"
            metadata.setdefault("warnings", []).append(warn_msg)
            print(
                f"[shield.pipeline] WARNING — {warn_msg}. "
                f"pipeline.py — assignment LP failed; z is unchanged for this "
                f"iteration. Check Gurobi license / constraint feasibility."
            )
            Z = z_curr

        # ---- Rounding + entity/leakage repair pipeline ---------------
        z_round = rounding.round_and_repair(
            Z,
            weights=weights, r=r,
            z_cost=z_linear_coeff,
            class_groups=class_groups_swap,
            class_delta=strat_cfg.class_delta,
            cold_left_groups=cold_left_groups,
            cold_right_groups=cold_right_groups,
            seed=cfg.seed + it,
            mode=cfg.rounding_mode,
        )

        # post-rounding entity repair via weighted majority vote
        z_round, _moved = _post_round_entity_repair(
            z_round, weights,
            cold_left_groups=cold_left_groups,
            cold_right_groups=cold_right_groups,
        )
        if _moved:
            metadata.setdefault("entity_repair_moves", []).append(
                {"iteration": it, "atoms_moved": int(_moved)}
            )


        if _h_solo and h_sol is not None and hierarchy_levels is not None:
            z_round, _h_moved = _post_round_h_group_repair(
                z_round, hierarchy_levels, h_sol,
                weights=weights, r=r,
                balance_eps=cfg.balance_eps,
            )
            if _h_moved:
                metadata.setdefault("h_group_repair_moves", []).append(
                    {"iteration": it, "atoms_moved": int(_h_moved)}
                )

        # Leakage-aware swap repair against the M (cut) energy: flips the
        # worst-violating cross-fold pairs without breaking cold-*/class
        # constraints. Only runs when M is active.
        if terms.use_M and affinity is not None:
            
            h_pinned_mask = None
            if terms.use_H and h_sol is not None and hierarchy_levels is not None:
                h_pinned_mask = torch.zeros(n_eff, dtype=torch.bool, device=device)
                fold_now = z_round.argmax(dim=1)
                for level in hierarchy_levels:
                    for g, desc in enumerate(level.descendants):
                        if len(desc) <= 1:
                            continue
                        committed_k = None
                        for k in range(K):
                            if float(h_sol.y.get((level.name, g, k), 0.0)) >= 0.5:
                                committed_k = k
                                break
                        if committed_k is None:
                            continue
                        idx = torch.as_tensor(desc, dtype=torch.long, device=device)
                        in_fold = fold_now[idx] == committed_k
                        h_pinned_mask[idx[in_fold]] = True
            z_round = rounding.leakage_aware_swap_repair(
                z_round, affinity,
                weights=weights, r=r,
                class_groups=class_groups_swap,
                class_delta=strat_cfg.class_delta,
                cold_left_groups=cold_left_groups,
                cold_right_groups=cold_right_groups,
                pinned_mask=h_pinned_mask,
            )

        # Verify class-stratification residual after all repairs
        if (cfg.enforce_class_delta_after_round
                and class_groups_swap is not None
                and class_labels_for_lp is not None):
            _violation = _class_strat_max_violation(
                z_round, weights, r, class_labels_for_lp
            )
            if _violation is not None and _violation > strat_cfg.class_delta:
                metadata.setdefault("warnings", []).append(
                    f"Iteration {it}: class-stratification residual "
                    f"{_violation:.4f} exceeds class_delta={strat_cfg.class_delta:.4f}. "
                    "Repair budget exhausted; final assignment may not meet hard "
                    "stratification target."
                )

        # ---- H-group integrity diagnostic (post-rounding) ---------------
        # Fraction of hierarchy groups fully committed to a single fold.
        if terms.use_H and hierarchy_levels is not None:
            _integrity = diagnostics.hierarchy_group_integrity_report(
                z_round, hierarchy_levels
            )
            _agg = _integrity.get("_aggregate", {})
            _frac_c  = _agg.get("frac_committed", float("nan"))
            _frac_cw = _agg.get("frac_committed_wt", float("nan"))
            _n_split = _agg.get("n_split", 0)
            _n_total = _agg.get("n_groups", 0)
            print(
                f"  [H-group integrity] committed={_frac_c*100:.1f}%  "
                f"weighted={_frac_cw*100:.1f}%  "
                f"split={_n_split}/{_n_total} groups"
            )
            for _lname, _lstats in _integrity.items():
                if _lname == "_aggregate":
                    continue
                print(
                    f"    level '{_lname}': "
                    f"{_lstats['n_committed']}/{_lstats['n_groups']} committed  "
                    f"(mean max_y={_lstats['mean_max_y']:.3f})"
                )

        if terms.use_H and hierarchy_levels is not None:
            h_sol = _recompute_y_w_from_rounded(z_round, hierarchy_levels)

        # ---- track gradient alignment + objective ----------------------
        alignment = diagnostics.gradient_alignment(grads_for_alignment)
        # D-shift contributes with a sign flip in the composite objective.
        _d_contrib = term_values.get("D_normalized", 0.0)
        if terms.use_D and terms.d_mode == "shift":
            _d_contrib = _d_contrib * (-1.0)
        composite_obj = (
            terms.alpha * term_values.get("H_normalized", 0.0)
            + terms.beta  * term_values.get("M_normalized", 0.0)
            + terms.gamma * _d_contrib
            + terms.mu    * term_values.get("strat_normalized", 0.0)
        )
        wall_s = time.time() - t0
        history.append({
            "iteration": it,
            "terms": term_values,
            "alignment": alignment,
            "composite_objective": composite_obj,
            "wall_clock_s": wall_s,
        })

        _log_iteration(
            it=it,
            term_values=term_values,
            composite_obj=composite_obj,
            prev_obj=prev_obj,
            alignment=alignment,
            z_round=z_round,
            weights=weights,
            r=r,
            tol=cfg.tolerance,
            wall_s=wall_s,
        )

        z_curr = z_round
        prev_obj = composite_obj  # update for logging only

        # ------------------------------------------------------------------
        # Stopping criteria (any that fires ends the loop)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Criterion 1: Z-assignment Hamming stability
        # ------------------------------------------------------------------
        if not _lp_ok:
            hamming_stable_count = 0
            metadata.setdefault("warnings", []).append(
                f"Iteration {it}: assignment LP did not solve; Hamming-stability "
                "convergence suppressed this iteration (possible infeasibility)."
            )
        elif z_prev_round is not None:
            hamming_frac = float(
                (z_round.argmax(dim=1) != z_prev_round.argmax(dim=1)).sum().item()
            ) / max(n_eff, 1)
            if hamming_frac < cfg.z_hamming_eps:
                hamming_stable_count += 1
            else:
                hamming_stable_count = 0

            if hamming_stable_count >= cfg.z_hamming_patience:
                print(
                    f"[shield.pipeline] Converged (Hamming stability) at iteration {it}: "
                    f"assignment Hamming fraction {hamming_frac:.4f} < "
                    f"z_hamming_eps={cfg.z_hamming_eps:.4g} for "
                    f"{cfg.z_hamming_patience} consecutive iterations. "
                    f"Completed {it + 1} of {cfg.max_iter} requested iterations."
                )
                metadata["converged"] = {
                    "criterion": "hamming_stability",
                    "iteration": it,
                    "hamming_frac": hamming_frac,
                    "consecutive_count": hamming_stable_count,
                }
                break
        z_prev_round = z_round.clone()

        # ------------------------------------------------------------------
        # Criterion 2: per-term individual convergence
        # ------------------------------------------------------------------
        active_term_keys = []
        if terms.use_H:
            active_term_keys.append("H_normalized")
        if terms.use_M:
            active_term_keys.append("M_normalized")
        if terms.use_D:
            active_term_keys.append("D_normalized")
        if strat_cfg.strategy != "none":
            active_term_keys.append("strat_normalized")

        strat_tol_effective = cfg.tolerance_per_term
        if (strat_cfg.strategy == "sliced_wasserstein"
                and strat_eval is not None
                and "sliced_wasserstein" in strat_eval.report):
            mc_se = strat_eval.report["sliced_wasserstein"].monte_carlo_se
            strat_tol_effective = max(cfg.tolerance_per_term, 2.0 * mc_se)

        if prev_term_values and active_term_keys:
            all_terms_converged = True
            for key in active_term_keys:
                if key not in term_values or key not in prev_term_values:
                    all_terms_converged = False
                    break
                tol_k = (strat_tol_effective
                         if key == "strat_normalized"
                         else cfg.tolerance_per_term)
                delta_k = abs(term_values[key] - prev_term_values[key])
                if delta_k >= tol_k * max(1.0, abs(prev_term_values[key])):
                    all_terms_converged = False
                    break
            if all_terms_converged:
                per_term_stable_count += 1
            else:
                per_term_stable_count = 0

            if per_term_stable_count >= cfg.patience_per_term:
                print(
                    f"[shield.pipeline] Converged (per-term) at iteration {it}: "
                    f"all active normalized terms stable for "
                    f"{cfg.patience_per_term} consecutive iterations "
                    f"(tolerance_per_term={cfg.tolerance_per_term:.2g}). "
                    f"Completed {it + 1} of {cfg.max_iter} requested iterations."
                )
                metadata["converged"] = {
                    "criterion": "per_term_convergence",
                    "iteration": it,
                    "consecutive_count": per_term_stable_count,
                    "active_terms": active_term_keys,
                }
                break
        else:
            per_term_stable_count = 0
        prev_term_values = dict(term_values)

        # ------------------------------------------------------------------
        # Criterion 3: case-specific stopping rules (OR-combined with 1 & 2)
        # ------------------------------------------------------------------

        # H-only: stop when H-LP y-variables stop changing.
        _h_only = (terms.use_H and not terms.use_M and not terms.use_D
                   and strat_cfg.strategy == "none")
        if _h_only and h_sol is not None and hierarchy_levels is not None:
            if h_sol_prev is not None:
                _all_keys = set(h_sol.y.keys()) | set(h_sol_prev.y.keys())
                _max_y_delta = max(
                    (abs(h_sol.y.get(k, 0.0) - h_sol_prev.y.get(k, 0.0))
                     for k in _all_keys),
                    default=0.0,
                )
                if _max_y_delta < cfg.h_y_eps:
                    h_y_stable_count += 1
                else:
                    h_y_stable_count = 0
                if h_y_stable_count >= cfg.patience_per_term:
                    print(
                        f"[shield.pipeline] Converged (H-LP y-stability, H-only) "
                        f"at iteration {it}: max |Δy_{{g,k}}| = {_max_y_delta:.4e} "
                        f"< h_y_eps={cfg.h_y_eps:.2g} for {cfg.patience_per_term} "
                        f"consecutive iterations."
                    )
                    metadata["converged"] = {
                        "criterion": "h_y_stability",
                        "iteration": it,
                        "max_y_delta": _max_y_delta,
                    }
                    break
            # Save the recomputed y (post-round, consistent with z_round).
            h_sol_prev = gurobi_lp.HierarchySolution(
                y=dict(h_sol.y),
                w_aux=dict(h_sol.w_aux),
                objective=h_sol.objective,
            )

        # D-match only: stop when Sinkhorn converges from warm-start in few
        # inner iterations (the transport plan has stabilized).
        _d_match_only = (terms.use_D and terms.d_mode == "match"
                         and not terms.use_H and not terms.use_M
                         and strat_cfg.strategy == "none")
        if _d_match_only and d_eval is not None and hasattr(d_eval, "plans"):
            _valid_plans = [p for p in d_eval.plans
                            if p.stop_reason != "empty_marginal"]
            if _valid_plans:
                _max_inner = max(p.n_iter for p in _valid_plans)
                if _max_inner <= cfg.sinkhorn_conv_threshold:
                    sinkhorn_stable_count += 1
                else:
                    sinkhorn_stable_count = 0
                if sinkhorn_stable_count >= cfg.patience_per_term:
                    print(
                        f"[shield.pipeline] Converged (Sinkhorn outer-stability, "
                        f"D-match only) at iteration {it}: Sinkhorn converged in "
                        f"≤{cfg.sinkhorn_conv_threshold} inner iters from warm-start "
                        f"for {cfg.patience_per_term} consecutive outer iterations."
                    )
                    metadata["converged"] = {
                        "criterion": "sinkhorn_outer_stability",
                        "iteration": it,
                        "max_inner_iters": _max_inner,
                    }
                    break

    # ---- Optional final MILP polish (last-resort gating) -----------------
    # Invoked only when the user requests an integer-clean answer AND the
    # repair budget left a measurable residual.
    def _needs_milp(z_now: torch.Tensor) -> Tuple[bool, str]:
        # 1) class-stratification residual above tolerance?
        # Use class_labels_for_lp (projected to super-nodes when coarsening active).
        if (cfg.enforce_class_delta_after_round
                and class_groups_swap is not None
                and class_labels_for_lp is not None):
            v = _class_strat_max_violation(z_now, weights, r, class_labels_for_lp)
            if v is not None and v > strat_cfg.class_delta:
                return True, f"class-strat residual {v:.4f} > {strat_cfg.class_delta:.4f}"
        # 2) balance residual above eps?
        bal_residual = diagnostics.balance_violation(z_now, weights, r).abs().max().item()
        bal_thresh = cfg.balance_eps * float(weights.sum().item())
        if bal_residual > bal_thresh:
            return True, f"balance residual {bal_residual:.4f} > {bal_thresh:.4f}"
        # 3) cold-* hardness violated (any entity straddling folds)?
        for label, groups in (("cold-left", cold_left_groups),
                              ("cold-right", cold_right_groups)):
            if not groups:
                continue
            fold_of = z_now.argmax(dim=1)
            for eid, obs_list in groups.items():
                if not obs_list:
                    continue
                idx = torch.as_tensor(obs_list, dtype=torch.long, device=device)
                if int(torch.unique(fold_of[idx]).numel()) > 1:
                    return True, f"{label} entity {eid} straddles folds after repair"
        return False, ""

    if cfg.integer_z_final:
        if cfg.milp_last_resort_only:
            need_milp, reason = _needs_milp(z_curr)
        else:
            need_milp, reason = True, "user requested unconditional MILP polish"

        if not need_milp:
            metadata["milp_polish"] = {"invoked": False,
                                       "reason": "repairs sufficed"}
        elif n_eff > cfg.milp_size_limit:
            print(
                f"[shield.pipeline] milp_size_limit = {cfg.milp_size_limit} "
                f"has been reached in pipeline.py - Final MIP polish is "
                f"skipped (n_eff={n_eff}, trigger='{reason}')."
            )
            metadata.setdefault("warnings", []).append(
                f"integer_z_final=True but n_eff={n_eff} > "
                f"milp_size_limit={cfg.milp_size_limit}; MILP polish skipped."
            )
            metadata["milp_polish"] = {"invoked": False,
                                       "reason": "size_limit",
                                       "size_limit": cfg.milp_size_limit}
        else:
            # Final H-LP recompute 
            milp_h_sol = h_sol if (terms.use_H and hierarchy_levels is not None) else None
            try:
                sol = gurobi_lp.solve_full_milp(
                    n=n_eff, K=K,
                    weights=_torch_to_np(weights),
                    r=_torch_to_np(r),
                    z_linear_coeff=_torch_to_np(z_linear_coeff),
                    eta_balance=terms.eta / max(1e-12, normalizers.get("bal", 1.0)),
                    hard=gurobi_lp.HardConstraints(
                        cold_left_groups=cold_left_groups,
                        cold_right_groups=cold_right_groups,
                        class_groups=class_groups_gurobi,
                        class_weights=_torch_to_np(weights),
                        class_delta=strat_cfg.class_delta,
                        balance_eps=cfg.balance_eps,
                    ),
                    hierarchy_levels=hierarchy_levels if milp_h_sol is not None else None,
                    y_fixed=milp_h_sol.y if milp_h_sol is not None else None,
                    pin_in_frac=_pin_in_frac,
                    pin_out_strict=_pin_out_strict,
                    time_limit=cfg.milp_time_limit_s,
                    output_flag=cfg.output_flag,
                    initial_z=_torch_to_np(z_curr),
                    seed=cfg.seed,
                )
                # detect Gurobi TIME_LIMIT termination and report it.
                try:
                    from gurobipy import GRB as _GRB
                    if int(sol.status) == int(_GRB.TIME_LIMIT):
                        print(
                            f"[shield.pipeline] milp_time_limit_s = "
                            f"{cfg.milp_time_limit_s} has been reached in "
                            "pipeline.py - returning best incumbent rather "
                            "than proven optimum."
                        )
                        metadata.setdefault("warnings", []).append(
                            f"MILP polish hit time_limit={cfg.milp_time_limit_s}s; "
                            "returned incumbent may be suboptimal."
                        )
                except Exception:                                          
                    pass
                z_curr = gurobi_lp.to_tensor(sol.z, device)
                metadata["milp_polish"] = {"invoked": True,
                                           "trigger": reason,
                                           "status": int(sol.status)}
            except RuntimeError as e:
                metadata.setdefault("warnings", []).append(
                    f"Final MILP polish skipped: {e}"
                )
                metadata["milp_polish"] = {"invoked": False,
                                           "reason": f"error: {e}"}

    # ---- Stage-5 unpacking back to atoms --------------------------------
    if super_to_atoms is not None:
        z_atom = coarsening.lift_assignment_to_atoms(z_curr, super_to_atoms)
        labels = z_atom.argmax(dim=1)
        z_final = torch.zeros((n, K), device=device, dtype=torch.float32)
        z_final.scatter_(1, labels.unsqueeze(1), 1.0)
        weights_final = atom_weights
    else:
        z_final = z_curr
        weights_final = atom_weights

    # ---- Stage-6 diagnostics --------------------------------------------
    diag_bundle: Dict[str, Any] = {
        "fold_sizes": diagnostics.fold_sizes(z_final, weights_final).detach().cpu().numpy(),
        "balance_residuals": diagnostics.balance_violation(z_final, weights_final, r)
                                       .detach().cpu().numpy(),
        "stage0_normalizers": normalizers,
        "history": history,
    }


    if y_target_atom is not None:
        diag_bundle["I_fold_y"] = diagnostics.mutual_info_continuous(z_final, y_target_atom)
        diag_bundle["anderson_darling"] = stratification.anderson_darling_per_fold(
            z_final, y_target_atom, weights=weights_final
        ).detach().cpu().numpy()
        diag_bundle["kolmogorov_smirnov"] = stratification.kolmogorov_smirnov_per_fold(
            z_final, y_target_atom, weights=weights_final
        ).detach().cpu().numpy()

        # per-fold vs pooled Spearman rank-correlation deviation.
        if y_target_atom.shape[-1] > 1:
            spearman_rep = diagnostics.spearman_correlation_deviation(
                z_final, y_target_atom, weights=weights_final,
                threshold=float(spearman_threshold),
            )
            diag_bundle["spearman"] = {
                "max_abs_deviation_per_fold":
                    spearman_rep.max_abs_deviation.detach().cpu().numpy(),
                "flagged_folds": list(spearman_rep.flagged_folds),
                "threshold": spearman_rep.threshold,
                "pooled": spearman_rep.pooled.detach().cpu().numpy(),
                "per_fold": spearman_rep.per_fold.detach().cpu().numpy(),
            }
        else:
            print(
                f"[shield.pipeline] Spearman rank-correlation diagnostic skipped: "
                f"y_target has q={y_target_atom.shape[-1]} output dimension "
                f"(shape {list(y_target_atom.shape)}). pipeline.py — Spearman requires "
                f"q>=2 (pairwise rank-correlations are undefined for a single "
                f"output coordinate)."
            )

    yhat_emb_t: Optional[torch.Tensor] = None
    if y_hat_emb is not None:
        yhat_emb_t = y_hat_emb.to(device)
    elif embedding is not None:
        target_for_knn = None
        if class_labels_atom is not None:
            target_for_knn = class_labels_atom.to(device)
        elif y_target_atom is not None:
            target_for_knn = y_target_atom
        if target_for_knn is not None:
            yhat_emb_t = diagnostics.knn_embedding_predict(
                embedding.to(device),
                target_for_knn,
                k=int(yhat_emb_knn_k),
            )
    if yhat_emb_t is not None:
        diag_bundle["I_fold_yhat_emb"] = diagnostics.mi_fold_yhat_emb(
            z_final, yhat_emb_t
        )

    if class_labels_atom is not None:
        diag_bundle["I_fold_class"] = diagnostics.mutual_info_discrete(
            z_final, class_labels_atom
        )

    
    if (two_entity.pair_indices is not None
            and two_entity.mode != "warm"):
        rep = diagnostics.cold_discard_report(
            two_entity.pair_indices,
            z_final,
            left_entity_of=two_entity.left_entity_of,
            right_entity_of=two_entity.right_entity_of,
        )
        diag_bundle["discard"] = {
            "total_pairs": rep.total_pairs,
            "discarded": rep.discarded,
            "discard_fraction": rep.discard_fraction,
            "retained_left_deg_summary": _deg_summary(rep.retained_degree_left),
            "discarded_left_deg_summary": _deg_summary(rep.discarded_degree_left),
            "retained_right_deg_summary": _deg_summary(rep.retained_degree_right),
            "discarded_right_deg_summary": _deg_summary(rep.discarded_degree_right),
        }

    # ---- pair-level target diagnostics (pair_observation_targets) --------
    # Warm mode permits cross-fold pairs, so same_fold isn't a discard signal
    # there:
    #   * "discarded" counts apply only under cold-* modes.
    #   * fold-wise pair stats in warm mode use the left entity's fold.
    #   * I(fold; y_pair) uses every pair in warm mode, retained-only in cold.
    if pair_observation_targets is not None and two_entity.pair_indices is not None:
        pot = pair_observation_targets.to(device)
        if pot.dim() == 1:
            pot = pot.unsqueeze(1)
        left_idx  = two_entity.pair_indices[:, 0].to(device).to(torch.long)
        right_idx = two_entity.pair_indices[:, 1].to(device).to(torch.long)
        fold_assign = z_final.argmax(dim=1)
        fold_left  = fold_assign[left_idx]
        fold_right = fold_assign[right_idx]
        same_fold = fold_left == fold_right

        is_cold = two_entity.mode in ("cold-left", "cold-right", "cold-both")
        # In warm mode, every pair counts; in cold modes, only same-fold
        # pairs are observable as the soft side of the discard accounting.
        retained_mask = same_fold if is_cold else torch.ones_like(same_fold)
        # Per-pair fold attribution: cold modes -> shared fold; warm mode ->
        # left-entity fold (downstream model is trained on the left side).
        per_pair_fold = fold_left

        # (1) fold-wise pair target statistics over the retained pairs
        pair_fold_stats: Dict[int, Any] = {}
        for k in range(K):
            m = (per_pair_fold == k) & retained_mask
            if m.any():
                n_m = int(m.sum().item())
                pair_fold_stats[k] = {
                    "n_retained": n_m,
                    "mean_y_pair": pot[m].mean(dim=0).cpu().tolist(),
                    "std_y_pair": (pot[m].std(dim=0).cpu().tolist()
                                   if n_m > 1 else [0.0] * pot.shape[1]),
                }
        diag_bundle["pair_fold_stats"] = pair_fold_stats
        diag_bundle["pair_diagnostics_mode"] = (
            "cold" if is_cold else "warm"
        )

        # (2) I(fold; y_pair). In warm mode every pair is included.
        n_ret = int(retained_mask.sum().item())
        if n_ret > 0:
            z_pairs = torch.zeros(n_ret, K, device=device, dtype=torch.float32)
            z_pairs.scatter_(1, per_pair_fold[retained_mask].unsqueeze(1), 1.0)
            diag_bundle["I_fold_y_pair"] = diagnostics.mutual_info_continuous(
                z_pairs, pot[retained_mask]
            )

        # (3) weighted discard fraction is only meaningful under cold-* modes.
        if is_cold:
            pair_magnitude = pot.abs().mean(dim=1)
            total_w   = float(pair_magnitude.sum().item())
            discard_w = float(pair_magnitude[~retained_mask].sum().item())
            diag_bundle["weighted_discard_fraction"] = discard_w / max(total_w, 1e-12)
        else:
            # Warm mode: cross-fold pairs are legitimate, not discarded.
            diag_bundle["weighted_discard_fraction"] = 0.0
            diag_bundle["cross_fold_pair_fraction"] = float(
                (~same_fold).float().mean().item()
            )

    # gradient alignment summary across iterations
    if history:
        avg_align: Dict[str, float] = {}
        for h in history:
            for k, v in h["alignment"].items():
                avg_align[k] = avg_align.get(k, 0.0) + v
        for k in avg_align:
            avg_align[k] /= len(history)
        diag_bundle["mean_gradient_alignment"] = avg_align

    return ShieldOutput(
        z=z_final,
        fold_assignment=z_final.argmax(dim=1),
        diagnostics=diag_bundle,
        metadata=metadata,
        history=history,
    )


def _class_strat_max_violation(
    z: torch.Tensor,
    weights: torch.Tensor,
    r: torch.Tensor,
    class_labels: torch.Tensor,
) -> Optional[float]:
    device = z.device
    w = weights.to(device).to(torch.float32)
    c = class_labels.to(device).to(torch.long)
    K = int(z.shape[1])
    classes = torch.unique(c)
    worst = 0.0
    r_dev = r.to(device).to(torch.float32)
    for cl in classes.tolist():
        if cl < 0:
            continue
        mask = c == cl
        W_c = float(w[mask].sum().item())
        if W_c <= 0:
            continue
        for k in range(K):
            v = float(((w * z[:, k])[mask].sum() - r_dev[k] * W_c).abs().item()) / W_c
            if v > worst:
                worst = v
    return worst if worst > 0 else None


def _deg_summary(deg: torch.Tensor) -> Dict[str, float]:
    if deg.numel() == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(deg.numel()),
        "mean": float(deg.float().mean().item()),
        "min": float(deg.float().min().item()),
        "max": float(deg.float().max().item()),
    }
