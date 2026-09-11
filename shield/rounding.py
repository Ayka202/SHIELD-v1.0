"""
Rounding from a relaxed assignment Z in [0,1]^{n x K} to a feasible one-hot z.

Two rounding back-ends are exposed (selected via `round_and_repair(..., mode=)`):

    'pipage'         -- Dependent-sampling pipage analog (default): 
                        row-wise categorical draw against Z's
                        relaxed marginals, then a vectorized balance-repair
                        sweep that re-enforces column marginals exactly. 
    'confidence_gap' -- Deterministic margin-driven swap rounding. Assigns
                        each row to its argmax column, then re-routes the
                        least-confident (smallest top-1 margin) rows into
                        under-full folds. 

Both back-ends are followed by:
    1. `_post_round_entity_repair`   (pipeline.py): cold-* atoms.
    2. `greedy_swap_repair`          (here): class-stratification residuals.
    3. `leakage_aware_swap_repair`   (here): M-cut leakage residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Pipage rounding 
# ---------------------------------------------------------------------------

def pipage_round(
    Z: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-9,
    seed: int = 0,
) -> torch.Tensor:
    """Pipage-style dependent-sampling rounding of Z in [0,1]^{n,K} to one-hot.

    1) Row-wise categorical draw against Z's row distribution (single
       multinomial call across all n rows) — preserves row marginals
       exactly, column marginals only in expectation.
    2) Vectorized argsort-driven swap sweep that drives the realized column
       residual to within numerical tolerance.

    `r` (target column proportions) is required.
    """
    if r is None:
        raise ValueError(
            "pipage_round requires the target proportions `r`. "
            "Pass an explicit r (length K, sums to 1)."
        )
    Z = Z.to(torch.float32).clamp(0.0, 1.0)
    n, K = Z.shape
    device = Z.device
    if weights is None:
        weights = torch.ones(n, device=device, dtype=torch.float32)
    else:
        weights = weights.to(device).to(torch.float32)
    W = weights.sum()
    target = r.to(device).to(torch.float32) * W

    # ---- step 1: dependent rounding via direct categorical sampling -------
    if device.type == "cuda":
        g = torch.Generator(device="cuda").manual_seed(seed)
        u = torch.rand(n, generator=g, device="cuda")
        Zc = Z.clamp_min(eps)
        Zc = Zc / Zc.sum(dim=1, keepdim=True)
        cum = torch.cumsum(Zc, dim=1)
        # searchsorted along row axis for each i: pick smallest k with cum[i,k] >= u[i]
        labels = torch.searchsorted(
            cum, u.unsqueeze(1)
        ).squeeze(1).clamp_max(K - 1).to(device)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        labels = torch.multinomial(
            Z.cpu().clamp_min(eps), num_samples=1, generator=g,
        ).squeeze(1).to(device)
    z = torch.zeros_like(Z)
    z.scatter_(1, labels.unsqueeze(1), 1.0)

    # ---- step 2: balance repair via sorted-swap sweep ---------------------
    achieved = (weights.unsqueeze(1) * z).sum(dim=0)                # (K,)
    for _ in range(2 * K):
        deficit = target - achieved                                  # (K,)
        if deficit.abs().max() < eps * W:
            break
        # pick over-full fold k_over and under-full fold k_under
        k_over = int(deficit.argmin().item())
        k_under = int(deficit.argmax().item())
        if k_over == k_under:
            break
        # candidates in k_over ranked by preference to flip to k_under
        in_over = (z[:, k_over] > 0.5)
        if not torch.any(in_over):
            break
        cand_idx = in_over.nonzero(as_tuple=False).flatten()
        score = Z[cand_idx, k_under] - Z[cand_idx, k_over]
        order = torch.argsort(score, descending=True)
        ranked = cand_idx[order]
        cum = torch.cumsum(weights[ranked], dim=0)
        gap = float(deficit[k_under].item())
        if gap <= 0:
            break
        cutoff = torch.searchsorted(cum, torch.tensor(gap, device=device))
        cutoff = int(min(cutoff.item() + 1, len(ranked)))
        to_flip = ranked[:cutoff]
        z[to_flip, k_over] = 0.0
        z[to_flip, k_under] = 1.0
        achieved = (weights.unsqueeze(1) * z).sum(dim=0)
    return z


# ---------------------------------------------------------------------------
# Greedy local swap repair
# ---------------------------------------------------------------------------

@dataclass
class SwapConfig:
    max_passes: int = 5
    swap_chunk: int = 16384


def greedy_swap_repair(
    z: torch.Tensor,
    weights: torch.Tensor,
    *,
    r: torch.Tensor,
    z_cost: torch.Tensor,
    class_groups: Optional[Sequence[torch.Tensor]] = None,
    class_delta: float = 0.05,
    config: Optional[SwapConfig] = None,
    cold_left_groups: Optional[dict] = None,
    cold_right_groups: Optional[dict] = None,
) -> torch.Tensor:
    """Greedy local swap improvement against z_cost while preserving hard rules.

    Iterates in passes:
      - Compute per-fold residuals against (r, class targets).
      - For each violating (k, c) bucket, propose swapping a sample i in fold
        k with a sample j in another fold k' that fixes the residual and
        strictly decreases z_cost contribution.
      - Skip any swap that would break a cold-left/right entity assignment.
    """
    if config is None:
        config = SwapConfig()
    device = z.device
    n, K = z.shape
    weights = weights.to(device).to(torch.float32)
    z = z.clone()
    z_cost = z_cost.to(device).to(torch.float32)

    # build a per-observation cold-group lookup (entity id) for fast checks
    entity_of_left = torch.full((n,), -1, dtype=torch.long, device=device)
    if cold_left_groups:
        for eid, obs_list in cold_left_groups.items():
            entity_of_left[torch.as_tensor(obs_list, device=device, dtype=torch.long)] = int(eid)
    entity_of_right = torch.full((n,), -1, dtype=torch.long, device=device)
    if cold_right_groups:
        for eid, obs_list in cold_right_groups.items():
            entity_of_right[torch.as_tensor(obs_list, device=device, dtype=torch.long)] = int(eid)

    # any atom that participates in a cold-* group is fully managed by the
    # Gurobi LP; the greedy swap pass is forbidden from moving it on its
    # own (that would split an entity across folds and break hardness).
    cold_mask = torch.zeros(n, dtype=torch.bool, device=device)
    if cold_left_groups:
        cold_mask = cold_mask | (entity_of_left >= 0)
    if cold_right_groups:
        cold_mask = cold_mask | (entity_of_right >= 0)

    class_membership: Optional[torch.Tensor] = None
    n_classes = 0
    if class_groups is not None and len(class_groups) > 0:
        n_classes = len(class_groups)
        class_membership = torch.full((n,), -1, dtype=torch.long, device=device)
        for c_idx, idx in enumerate(class_groups):
            class_membership[idx.to(device).to(torch.long)] = c_idx
        class_total_w = torch.zeros(n_classes, device=device, dtype=torch.float32)
        for c_idx, idx in enumerate(class_groups):
            class_total_w[c_idx] = weights[idx.to(device).to(torch.long)].sum()
        class_target_const = (r.to(device).to(torch.float32).unsqueeze(0)
                              * class_total_w.unsqueeze(1))

        def _class_residual_now() -> torch.Tensor:
            valid = class_membership >= 0
            cidx = class_membership.clamp_min(0)
            tally = torch.zeros((n_classes, K), device=device, dtype=torch.float32)
            for k in range(K):
                contribution = torch.where(valid, weights * z[:, k],
                                           torch.zeros_like(weights))
                tally[:, k].scatter_add_(0, cidx, contribution)
            return tally - class_target_const

    for _ in range(config.max_passes):
        improved = False
        achieved = (weights.unsqueeze(1) * z).sum(dim=0)
        target = r.to(device).to(torch.float32) * weights.sum()
        residual = achieved - target

        # class-stratification residuals, refreshed each pass
        class_residual = (_class_residual_now()
                          if class_membership is not None else None)

        for k_over in range(K):
            for k_under in range(K):
                if k_over == k_under:
                    continue
                # do nothing if both fold-balance and class-balance are within tol
                fold_ok = (residual[k_over] <= 0 or residual[k_under] >= 0)
                class_violation = False
                if class_residual is not None:
                    cv = (class_residual[:, k_over] > class_delta * class_total_w) | \
                         (class_residual[:, k_under] < -class_delta * class_total_w)
                    class_violation = bool(cv.any().item())
                if fold_ok and not class_violation:
                    continue
                # candidates: in fold k_over now AND not under cold-constraint
                cand_mask = (z[:, k_over] > 0.5) & (~cold_mask)
                cand_in = cand_mask.nonzero(as_tuple=False).flatten()
                if cand_in.numel() == 0:
                    continue
                deltas = z_cost[cand_in, k_under] - z_cost[cand_in, k_over]
                # class-violation repair accepts cost-neutral swaps too;
                # pure balance repair requires a strict cost decrease
                accept_thresh = 1e-9 if class_violation else -1e-9
                neg_mask = deltas < accept_thresh
                if not torch.any(neg_mask):
                    continue
                cand = cand_in[neg_mask]
                # apply at most `swap_chunk` of the best (most-negative) swaps
                # but only enough to close the residual gap
                order = torch.argsort(deltas[neg_mask])
                cand = cand[order]
                cum_w = torch.cumsum(weights[cand], dim=0)
                gap = float(residual[k_under].abs().item())
                if gap <= 0:
                    gap = float((class_total_w.max() * class_delta).item()) \
                        if class_residual is not None else 0.0
                cutoff_pos = torch.searchsorted(cum_w, torch.tensor(gap, device=device)).item() + 1
                cutoff_pos = min(int(cutoff_pos), cand.numel(), config.swap_chunk)
                to_flip = cand[:cutoff_pos]
                if to_flip.numel() == 0:
                    continue
                z[to_flip, k_over] = 0.0
                z[to_flip, k_under] = 1.0
                improved = True
                achieved = (weights.unsqueeze(1) * z).sum(dim=0)
                residual = achieved - target
                if class_membership is not None:
                    class_residual = _class_residual_now()

        if not improved:
            break
    return z


# ---------------------------------------------------------------------------
# End-to-end rounding entry point
# ---------------------------------------------------------------------------

def confidence_gap_round(
    Z: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    if r is None:
        raise ValueError(
            "confidence_gap_round requires the target proportions `r`."
        )
    Z = Z.to(torch.float32).clamp(0.0, 1.0)
    n, K = Z.shape
    device = Z.device
    if weights is None:
        weights = torch.ones(n, device=device, dtype=torch.float32)
    else:
        weights = weights.to(device).to(torch.float32)
    W = weights.sum()
    target = r.to(device).to(torch.float32) * W

    # 1) deterministic argmax assignment.
    labels = Z.argmax(dim=1)
    z = torch.zeros_like(Z)
    z.scatter_(1, labels.unsqueeze(1), 1.0)
    achieved = (weights.unsqueeze(1) * z).sum(dim=0)

    # 2) margin-driven re-balancing.
    for _ in range(4 * K):
        deficit = target - achieved
        if deficit.abs().max() < eps * W:
            break
        k_over = int(deficit.argmin().item())
        k_under = int(deficit.argmax().item())
        if k_over == k_under:
            break
        in_over = z[:, k_over] > 0.5
        if not torch.any(in_over):
            break
        cand_idx = in_over.nonzero(as_tuple=False).flatten()
        # margin = how much we lose by flipping i from k_over to k_under
        # (smaller margin first = least confident in current assignment).
        margin = Z[cand_idx, k_over] - Z[cand_idx, k_under]
        order = torch.argsort(margin)            # ascending
        ranked = cand_idx[order]
        cum_w = torch.cumsum(weights[ranked], dim=0)
        gap = float(deficit[k_under].item())
        cutoff = int(torch.searchsorted(cum_w, torch.tensor(gap, device=device)).item()) + 1
        cutoff = min(cutoff, ranked.numel())
        to_flip = ranked[:cutoff]
        z[to_flip, k_over] = 0.0
        z[to_flip, k_under] = 1.0
        achieved = (weights.unsqueeze(1) * z).sum(dim=0)
    return z


def leakage_aware_swap_repair(
    z: torch.Tensor,
    affinity: torch.Tensor,
    *,
    weights: torch.Tensor,
    r: torch.Tensor,
    class_groups: Optional[Sequence[torch.Tensor]] = None,
    class_delta: float = 0.05,
    cold_left_groups: Optional[dict] = None,
    cold_right_groups: Optional[dict] = None,
    pinned_mask: Optional[torch.Tensor] = None,
    max_iter: int = 3,
    swap_chunk: int = 8192,
) -> torch.Tensor:
    """Greedy leakage-aware swap pass against the M-cut energy.

    For each over-full fold, identifies the atoms with the highest weighted
    edge mass crossing into other folds and migrates them. Skips cold-*
    entity atoms and any atom in `pinned_mask` (e.g. a committed H-group).
    A swap is vetoed if it would push a class's weighted mass outside the
    band [r_k W_c - delta W_c, r_k W_c + delta W_c] on the source or target
    fold, checked conservatively against the entire candidate class mass.
    """
    device = z.device
    n, K = z.shape
    z_new = z.clone()
    w = weights.to(device).to(torch.float32)
    r_t = r.to(device).to(torch.float32)
    target = r_t * w.sum()

    cold_mask = torch.zeros(n, dtype=torch.bool, device=device)
    for groups in (cold_left_groups, cold_right_groups):
        if not groups:
            continue
        for _eid, obs in groups.items():
            if obs:
                cold_mask[torch.as_tensor(obs, dtype=torch.long, device=device)] = True
    # atoms locked by committed H-groups are also immovable here
    if pinned_mask is not None:
        cold_mask = cold_mask | pinned_mask.to(device).to(torch.bool)

    # class membership + per-class total weight for the stratification-band
    # veto; class_membership[i] = class index or -1
    class_membership: Optional[torch.Tensor] = None
    n_classes = 0
    class_total_w: Optional[torch.Tensor] = None
    class_target: Optional[torch.Tensor] = None
    if class_groups is not None and len(class_groups) > 0:
        n_classes = len(class_groups)
        class_membership = torch.full((n,), -1, dtype=torch.long, device=device)
        class_total_w = torch.zeros(n_classes, device=device, dtype=torch.float32)
        for c_idx, idx in enumerate(class_groups):
            ii = idx.to(device).to(torch.long)
            class_membership[ii] = c_idx
            class_total_w[c_idx] = w[ii].sum()
        # band center per (class, fold): r_k * W_c
        class_target = r_t.unsqueeze(0) * class_total_w.unsqueeze(1)   # (C, K)

    def _class_tally(zc: torch.Tensor) -> Optional[torch.Tensor]:
        if class_membership is None:
            return None
        valid = class_membership >= 0
        cidx = class_membership.clamp_min(0)
        tally = torch.zeros((n_classes, K), device=device, dtype=torch.float32)
        for k in range(K):
            contrib = torch.where(valid, w * zc[:, k], torch.zeros_like(w))
            tally[:, k].scatter_add_(0, cidx, contrib)
        return tally

    is_sparse = affinity.is_sparse
    if is_sparse:
        A = affinity.coalesce()
        a_rows = A.indices()[0]
        a_cols = A.indices()[1]
        a_vals = A.values().to(torch.float32)
    # cross_mass[i, k'] = sum_{j: fold(j)=k'} A_ij  -> contribution of i to fold k'.
    for _ in range(max_iter):
        fold_of = z_new.argmax(dim=1)
        if is_sparse:
            # Build a sparse one-hot for fold-of, then A @ one-hot.
            oh = torch.zeros((n, K), device=device, dtype=torch.float32)
            oh.scatter_(1, fold_of.unsqueeze(1), 1.0)
            cross_mass = torch.sparse.mm(A, oh)   # (n, K)
        else:
            oh = torch.zeros((n, K), device=device, dtype=torch.float32)
            oh.scatter_(1, fold_of.unsqueeze(1), 1.0)
            cross_mass = affinity.to(torch.float32) @ oh

        self_mass = cross_mass.gather(1, fold_of.unsqueeze(1)).squeeze(1)
        max_other_mass, max_other_k = cross_mass.scatter(
            1, fold_of.unsqueeze(1), float("-inf")
        ).max(dim=1)
        leak_score = max_other_mass - self_mass          # high = good swap candidate

        achieved = (w.unsqueeze(1) * z_new).sum(dim=0)
        deficit = target - achieved
        improved = False
        for k_over in range(K):
            if deficit[k_over] >= 0:
                continue
            cand = ((fold_of == k_over) & (~cold_mask) & (leak_score > 0)).nonzero(as_tuple=False).flatten()
            if cand.numel() == 0:
                continue
            cand_k_target = max_other_k[cand]
            free_cap = deficit > 0
            mask = free_cap[cand_k_target]
            cand = cand[mask]
            cand_k_target = cand_k_target[mask]
            if cand.numel() == 0:
                continue
            order = torch.argsort(leak_score[cand], descending=True)
            cand = cand[order][:swap_chunk]
            cand_k_target = cand_k_target[order][:swap_chunk]

            # class-stratification band veto (conservative, worst-case mass)
            if class_membership is not None:
                tally = _class_tally(z_new)                          # (C, K)
                lower = class_target - class_delta * class_total_w.unsqueeze(1)
                upper = class_target + class_delta * class_total_w.unsqueeze(1)
                cand_c = class_membership[cand]                      # (m,)
                has_class = cand_c >= 0
                cc = cand_c.clamp_min(0)
                wc = w[cand]
                # worst-case mass leaving k_over per class
                out_mass = torch.zeros(n_classes, device=device, dtype=torch.float32)
                out_mass.scatter_add_(0, cc[has_class], wc[has_class])
                # worst-case mass arriving per (class, target fold)
                in_mass = torch.zeros(n_classes * K, device=device, dtype=torch.float32)
                flat_ct = (cc * K + cand_k_target)[has_class]
                in_mass.scatter_add_(0, flat_ct, wc[has_class])
                in_mass = in_mass.view(n_classes, K)
                src_ok = (tally[:, k_over] - out_mass) >= lower[:, k_over]   # (C,)
                tgt_ok = (tally + in_mass) <= upper                          # (C, K)
                atom_ok = (~has_class) | (
                    src_ok[cc] & tgt_ok[cc, cand_k_target]
                )
                cand = cand[atom_ok]
                cand_k_target = cand_k_target[atom_ok]
                if cand.numel() == 0:
                    continue

            # apply the swaps
            z_new[cand, k_over] = 0.0
            z_new[cand, :] = z_new[cand, :].scatter_(
                1, cand_k_target.unsqueeze(1), 1.0
            )
            improved = True
            fold_of = z_new.argmax(dim=1)
            achieved = (w.unsqueeze(1) * z_new).sum(dim=0)
            deficit = target - achieved
        if not improved:
            break
    return z_new


def round_and_repair(
    Z: torch.Tensor,
    *,
    weights: torch.Tensor,
    r: torch.Tensor,
    z_cost: torch.Tensor,
    class_groups: Optional[Sequence[torch.Tensor]] = None,
    class_delta: float = 0.05,
    cold_left_groups: Optional[dict] = None,
    cold_right_groups: Optional[dict] = None,
    seed: int = 0,
    mode: str = "pipage",
) -> torch.Tensor:
    """Run rounding (`mode`) then class-aware greedy swap repair.

    `mode` is one of:
        'pipage'         -- dependent-sampling pipage analog (default).
        'confidence_gap' -- deterministic margin-driven swap rounding.
    """
    mode_l = str(mode).lower()
    if mode_l == "pipage":
        z_int = pipage_round(Z, weights=weights, r=r, seed=seed)
    elif mode_l in ("confidence_gap", "deterministic"):
        z_int = confidence_gap_round(Z, weights=weights, r=r)
    else:
        raise ValueError(
            f"Unknown rounding mode '{mode}'. Choose 'pipage' or 'confidence_gap'."
        )
    z_rep = greedy_swap_repair(
        z_int,
        weights=weights,
        r=r,
        z_cost=z_cost,
        class_groups=class_groups,
        class_delta=class_delta,
        cold_left_groups=cold_left_groups,
        cold_right_groups=cold_right_groups,
    )
    return z_rep
