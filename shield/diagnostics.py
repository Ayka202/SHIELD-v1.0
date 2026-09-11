"""
SHIELD diagnostic primitives 

Covers:
    * fold-size and stratification residuals;
    * cold-discard fraction and entity-degree distributions;
    * mutual information I(fold; y), I(fold; y_hat_emb);
    * embedding-based prediction `y_hat_emb` via a vectorized weighted k-NN
      regressor / classifier on the user-supplied embedding;
    * per-fold Spearman rank-correlation matrices on multi-output y, with
      max-deviation threshold flagging;
    * pre- and post-Stage-0 normalized term values;
    * per-iteration gradient inner products (alignment diagnostic);
    * Hamming distance between assignments (sweep stability);
    * Pareto frontier construction and recommendation.
    * hierarchy group integrity (fraction of H-groups committed to a single fold);
    * effective gradient magnitudes and pairwise conflict analysis (H/M/D balance).

"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import joblib as _joblib
    _HAS_JOBLIB = True
except ImportError:
    _joblib = None  # type: ignore
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# Fold sizes and balance violation
# ---------------------------------------------------------------------------

def fold_sizes(
    z: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(z.device).to(torch.float32)
    return (weights.unsqueeze(1) * z).sum(dim=0)


def balance_violation(
    z: torch.Tensor,
    weights: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    s = fold_sizes(z, weights)
    W = float(weights.sum().item())
    return s - r.to(z.device).to(torch.float32) * W


# ---------------------------------------------------------------------------
# Cold-* discard accounting 
# ---------------------------------------------------------------------------

@dataclass
class DiscardReport:
    total_pairs: int
    discarded: int
    discard_fraction: float
    retained_degree_left: torch.Tensor       # (E_L,)
    discarded_degree_left: torch.Tensor
    retained_degree_right: torch.Tensor      # (E_R,)
    discarded_degree_right: torch.Tensor


def cold_discard_report(
    pair_indices: torch.Tensor,            # (P, 2) - (left atom, right atom)
    z: torch.Tensor,
    *,
    left_entity_of: Optional[torch.Tensor] = None,   # (n,) entity id per left atom
    right_entity_of: Optional[torch.Tensor] = None,  # (n,) entity id per right atom
) -> DiscardReport:
    """Compute discard counts and entity-degree distributions under cold-* """
    device = z.device
    fold_of = z.argmax(dim=1)
    left = pair_indices[:, 0].to(device).to(torch.long)
    right = pair_indices[:, 1].to(device).to(torch.long)
    same_fold = (fold_of[left] == fold_of[right])
    discarded_mask = ~same_fold

    def _degrees(entity_ids):
        if entity_ids.numel() == 0:
            return torch.zeros(0, device=device, dtype=torch.long)
        uniq, inv = torch.unique(entity_ids, return_inverse=True)
        deg = torch.zeros(uniq.numel(), device=device, dtype=torch.long)
        deg.scatter_add_(0, inv, torch.ones_like(inv))
        return deg

    _empty = torch.zeros(0, device=device, dtype=torch.long)
    if left_entity_of is not None:
        le = left_entity_of.to(device).to(torch.long)[left]
        retained_left_deg = _degrees(le[same_fold])
        discarded_left_deg = _degrees(le[discarded_mask])
    else:
        retained_left_deg = _empty
        discarded_left_deg = _empty
    if right_entity_of is not None:
        re = right_entity_of.to(device).to(torch.long)[right]
        retained_right_deg = _degrees(re[same_fold])
        discarded_right_deg = _degrees(re[discarded_mask])
    else:
        retained_right_deg = _empty
        discarded_right_deg = _empty
    total = int(pair_indices.shape[0])
    n_dis = int(discarded_mask.sum().item())
    return DiscardReport(
        total_pairs=total,
        discarded=n_dis,
        discard_fraction=(n_dis / max(1, total)),
        retained_degree_left=retained_left_deg,
        discarded_degree_left=discarded_left_deg,
        retained_degree_right=retained_right_deg,
        discarded_degree_right=discarded_right_deg,
    )


# ---------------------------------------------------------------------------
# Mutual information diagnostics 
# ---------------------------------------------------------------------------

def _entropy_from_counts(counts: torch.Tensor) -> torch.Tensor:
    p = counts / counts.sum().clamp_min(1e-12)
    nz = p[p > 0]
    return -(nz * torch.log(nz)).sum()


def mutual_info_discrete(
    z: torch.Tensor,
    y_class: torch.Tensor,
) -> float:
    """Empirical I(fold; y_class) for categorical y, vectorized """
    device = z.device
    fold = z.argmax(dim=1).to(device).to(torch.long)
    y_class = y_class.to(device).to(torch.long)
    valid = y_class >= 0
    if not bool(valid.all()):
        fold = fold[valid]
        y_class = y_class[valid]
    if y_class.numel() == 0:
        return 0.0
    K = int(z.shape[1])
    C = int(y_class.max().item() + 1)
    joint = torch.zeros((K, C), device=device, dtype=torch.float32)
    flat = fold * C + y_class
    joint.view(-1).scatter_add_(0, flat,
                                torch.ones_like(flat, dtype=torch.float32))
    p_xy = joint / joint.sum().clamp_min(1e-12)
    p_x = p_xy.sum(dim=1, keepdim=True)
    p_y = p_xy.sum(dim=0, keepdim=True)
    mask = p_xy > 0
    mi = (p_xy[mask] * (p_xy[mask].log() - (p_x * p_y).expand_as(p_xy)[mask].log())).sum()
    return float(mi.item())


def mutual_info_continuous(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    bins: int = 32,
) -> float:
    """Empirical I(fold; y) for a multi-dim continuous y using per-coord binning """
    if y.dim() == 1:
        y = y.unsqueeze(1)
    n, q = y.shape
    mi_total = 0.0
    for j in range(q):
        edges = torch.linspace(
            y[:, j].min().item() - 1e-6,
            y[:, j].max().item() + 1e-6,
            bins + 1,
            device=y.device,
        )
        bj = torch.bucketize(y[:, j].contiguous(), edges[1:-1])
        mi_total += mutual_info_discrete(z, bj)
    return mi_total / max(1, q)


# ---------------------------------------------------------------------------
# Embedding-based prediction y_hat_emb 
# ---------------------------------------------------------------------------

def knn_embedding_predict(
    embedding: torch.Tensor,
    y: torch.Tensor,
    *,
    k: int = 15,
    chunk: int = 4096,
    mode: str = "auto",
) -> torch.Tensor:
    """Predict y from `embedding` via weighted k-NN, fully vectorized """
    device = embedding.device
    n = int(embedding.shape[0])
    if y.shape[0] != n:
        raise ValueError(
            f"embedding has {n} rows but y has {y.shape[0]}; lengths must match."
        )
    if mode == "auto":
        is_class = not torch.is_floating_point(y)
    elif mode == "classification":
        is_class = True
    elif mode == "regression":
        is_class = False
    else:
        raise ValueError(
            f"Unknown knn_embedding_predict mode '{mode}'. "
            "Use 'auto', 'regression', or 'classification'."
        )

    k_eff = max(1, min(int(k), n - 1))
    emb = embedding.to(torch.float32)
    if is_class:
        y_long = y.to(torch.long).to(device)
        C = int(y_long.max().item()) + 1
        out = torch.empty(n, dtype=torch.long, device=device)
    else:
        y_float = y.to(torch.float32).to(device)
        if y_float.dim() == 1:
            y_float = y_float.unsqueeze(1)
        q = int(y_float.shape[1])
        out = torch.empty((n, q), dtype=torch.float32, device=device)

    emb_sq = (emb * emb).sum(dim=1)               # (n,)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = emb[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        cross = xb @ emb.t()
        d2 = (x_sq + emb_sq.unsqueeze(0) - 2.0 * cross).clamp_min_(0.0)
        d2[torch.arange(end - start, device=device),
           torch.arange(start, end, device=device)] = float("inf")
        topd, topi = torch.topk(d2, k_eff, dim=1, largest=False, sorted=False)
        w = 1.0 / (topd + 1e-12)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-30)
        if is_class:
            b = end - start
            tally = torch.zeros((b, C), device=device, dtype=torch.float32)
            tally.scatter_add_(1, y_long[topi], w)
            out[start:end] = tally.argmax(dim=1)
        else:
            neigh = y_float[topi]                  # (b, k_eff, q)
            out[start:end] = (w.unsqueeze(2) * neigh).sum(dim=1)
    return out


def mi_fold_yhat_emb(
    z: torch.Tensor,
    y_hat_emb: torch.Tensor,
    *,
    bins: int = 32,
) -> float:
    if not torch.is_floating_point(y_hat_emb):
        return mutual_info_discrete(z, y_hat_emb.to(torch.long))
    return mutual_info_continuous(z, y_hat_emb, bins=bins)


# ---------------------------------------------------------------------------
# Per-fold Spearman rank-correlation diagnostic
# ---------------------------------------------------------------------------

def _rank_columns(x: torch.Tensor) -> torch.Tensor:
    n, q = x.shape
    order = torch.argsort(x, dim=0)
    rank = torch.empty_like(order, dtype=torch.float32)
    rows = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
    rank.scatter_(0, order, rows.unsqueeze(1).expand(-1, q))

    x_sorted = torch.gather(x, 0, order)
    for j in range(q):
        col = x_sorted[:, j]
        diffs = torch.ones(n, dtype=torch.bool, device=x.device)
        diffs[1:] = col[1:] != col[:-1]
        group_id = torch.cumsum(diffs.to(torch.long), dim=0) - 1     # (n,)
        n_groups = int(group_id.max().item()) + 1
        sums = torch.zeros(n_groups, device=x.device, dtype=torch.float32)
        counts = torch.zeros(n_groups, device=x.device, dtype=torch.float32)
        sums.scatter_add_(0, group_id, rows)
        counts.scatter_add_(0, group_id, torch.ones_like(rows))
        avg_rank = sums / counts.clamp_min(1.0)
        rank_sorted = avg_rank[group_id]
        rank[:, j] = torch.empty(n, device=x.device, dtype=torch.float32) \
                          .scatter_(0, order[:, j], rank_sorted)
    return rank


def _corr_matrix(x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    n, q = x.shape
    if w is None:
        w = torch.ones(n, device=x.device, dtype=torch.float32)
    w = w.to(torch.float32)
    W = w.sum().clamp_min(1e-30)
    mean = (w.unsqueeze(1) * x).sum(dim=0) / W                        # (q,)
    xc = x - mean.unsqueeze(0)
    cov = (xc.t() * w.unsqueeze(0)) @ xc / W                          # (q, q)
    std = cov.diagonal().clamp_min(1e-30).sqrt()
    return cov / (std.unsqueeze(0) * std.unsqueeze(1))


@dataclass
class SpearmanReport:
    pooled: torch.Tensor                       # (q, q)
    per_fold: torch.Tensor                     # (K, q, q)
    max_abs_deviation: torch.Tensor            # (K,) sup_{i,j} |ρ^k_ij - ρ^pool_ij|
    flagged_folds: List[int]                   # folds whose deviation > threshold
    threshold: float


def spearman_correlation_deviation(
    z: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    threshold: float = 0.05,
) -> SpearmanReport:
    if y.dim() == 1:
        y = y.unsqueeze(1)
    n, q = y.shape
    if z.shape[0] != n:
        raise ValueError(
            f"z has {z.shape[0]} rows but y has {n}; lengths must match."
        )
    device = z.device
    y = y.to(torch.float32).to(device)
    if weights is None:
        w = torch.ones(n, device=device, dtype=torch.float32)
    else:
        w = weights.to(torch.float32).to(device)

    ranks_pooled = _rank_columns(y)                                    # (n, q)
    pooled = _corr_matrix(ranks_pooled, w)                             # (q, q)

    K = int(z.shape[1])
    per_fold = torch.zeros((K, q, q), device=device, dtype=torch.float32)
    max_dev = torch.zeros(K, device=device, dtype=torch.float32)
    flagged: List[int] = []
    for k in range(K):
        mask = z[:, k] > 0.5
        if int(mask.sum().item()) < 3:
            per_fold[k] = float("nan")
            max_dev[k] = float("nan")
            continue
        wk = w[mask]
        rk = ranks_pooled[mask]   # project pooled ranks onto fold-k subset
        rho_k = _corr_matrix(rk, wk)
        per_fold[k] = rho_k
        dev = (rho_k - pooled).abs().max()
        max_dev[k] = dev
        if float(dev.item()) > float(threshold):
            flagged.append(k)
            warnings.warn(
                f"Spearman rank-correlation deviation in fold {k} exceeds "
                f"threshold ({float(dev.item()):.4f} > {threshold:.4f}). "
                "Coordinate-pair dependence structure differs from the pooled distribution"
                UserWarning,
                stacklevel=2,
            )

    return SpearmanReport(
        pooled=pooled.detach(),
        per_fold=per_fold.detach(),
        max_abs_deviation=max_dev.detach(),
        flagged_folds=flagged,
        threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Hierarchy group integrity diagnostic
# ---------------------------------------------------------------------------

def hierarchy_group_integrity_report(
    z: torch.Tensor,
    hierarchy_levels: Sequence,
) -> Dict[str, Dict]:
    device = z.device
    results: Dict[str, Dict] = {}

    total_committed = total_split = 0
    total_committed_wt = total_wt = 0.0

    for level in hierarchy_levels:
        lev_committed = lev_split = 0
        lev_committed_wt = lev_total_wt = 0.0
        sum_max_y = 0.0
        n_nonempty = 0

        for g, desc in enumerate(level.descendants):
            if len(desc) == 0:
                continue
            desc_t = torch.as_tensor(desc, dtype=torch.long, device=device)
            y_g = z.index_select(0, desc_t).min(dim=0).values  # (K,)
            max_y = float(y_g.max().item())
            group_size = float(len(desc))

            sum_max_y += max_y
            n_nonempty += 1
            lev_total_wt += group_size
            if max_y >= 0.5:
                lev_committed += 1
                lev_committed_wt += group_size
            else:
                lev_split += 1

        n_groups = lev_committed + lev_split
        results[level.name] = {
            "n_groups": n_groups,
            "n_committed": lev_committed,
            "n_split": lev_split,
            "frac_committed": lev_committed / max(n_groups, 1),
            "frac_committed_wt": lev_committed_wt / max(lev_total_wt, 1.0),
            "mean_max_y": sum_max_y / max(n_nonempty, 1),
        }
        total_committed += lev_committed
        total_split += lev_split
        total_committed_wt += lev_committed_wt
        total_wt += lev_total_wt

    total_groups = total_committed + total_split
    results["_aggregate"] = {
        "n_groups": total_groups,
        "n_committed": total_committed,
        "n_split": total_split,
        "frac_committed": total_committed / max(total_groups, 1),
        "frac_committed_wt": total_committed_wt / max(total_wt, 1.0),
    }
    return results


# ---------------------------------------------------------------------------
# Effective gradient magnitudes and pairwise conflict analysis
# ---------------------------------------------------------------------------

def effective_gradient_magnitudes(
    grads_dict: Dict[str, torch.Tensor],
    normalizers: Dict[str, float],
    coefficients: Dict[str, float],
) -> Dict[str, object]:
    effective: Dict[str, torch.Tensor] = {}
    eff_norms: Dict[str, float] = {}
    raw_norms: Dict[str, float] = {}

    for name, grad in grads_dict.items():
        coeff    = float(coefficients.get(name, 1.0))
        norm_val = max(float(normalizers.get(name, 1.0)), 1e-12)
        raw_norm = float(grad.norm().item())
        eff      = coeff * grad / norm_val
        effective[name]   = eff
        eff_norms[name]   = float(eff.norm().item())
        raw_norms[name]   = raw_norm

    names = list(effective.keys())
    cosines: Dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            ga = effective[na].reshape(-1)
            gb = effective[nb].reshape(-1)
            denom = float(ga.norm().item()) * float(gb.norm().item())
            cos   = float((ga * gb).sum().item()) / denom if denom > 1e-30 else 0.0
            cosines[f"{na}↔{nb}"] = cos

    dominant = max(eff_norms, key=lambda k: eff_norms[k]) if eff_norms else ""
    return {
        "effective_norms": eff_norms,
        "raw_norms": raw_norms,
        "pairwise_cosines": cosines,
        "dominant_term": dominant,
    }


# ---------------------------------------------------------------------------
# Gradient alignment 
# ---------------------------------------------------------------------------

def gradient_alignment(
    grads: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    out = {}
    names = list(grads.keys())
    flat = {n: g.reshape(-1) for n, g in grads.items()}
    norms = {n: float(v.norm().item()) + 1e-12 for n, v in flat.items()}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            dot = float((flat[na] * flat[nb]).sum().item())
            out[f"{na}/{nb}"] = dot / (norms[na] * norms[nb])
    return out


# ---------------------------------------------------------------------------
# Hamming distance for sweep stability
# ---------------------------------------------------------------------------

def hamming_distance(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> int:
    return int((z_a.argmax(dim=1) != z_b.argmax(dim=1)).sum().item())


# ---------------------------------------------------------------------------
# Pareto frontier helper 
# ---------------------------------------------------------------------------

def pareto_frontier(
    points: Sequence[Sequence[float]],
) -> List[int]:
    P = np.asarray(points, dtype=np.float64)
    n = P.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        better_or_equal = (P <= P[i]).all(axis=1)
        strictly_better = (P < P[i]).any(axis=1)
        dominated = better_or_equal & strictly_better
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return [int(i) for i in np.where(keep)[0]]


def recommend_coefficients(
    grid_points: Sequence[Sequence[float]],
    diagnostics: Sequence[Dict[str, float]],
    metric_keys: Sequence[str],
    *,
    aggregation: str = "min_of_normalized",
) -> Tuple[int, Dict[str, float]]:
    arr = np.array([[float(d[k]) for k in metric_keys] for d in diagnostics],
                   dtype=np.float64)
    mx = arr.max(axis=0)
    mx[mx == 0] = 1.0
    norm = arr / mx

    if aggregation == "min_of_normalized":
        score = norm.max(axis=1)
        var = norm.var(axis=1)
        gp = np.array(list(grid_points), dtype=np.float64)
        if gp.ndim == 2 and gp.shape[0] == len(diagnostics) and gp.shape[1] > 0:
            gp_max = gp.max(axis=0)
            gp_max[gp_max == 0] = 1.0
            gp_norm = gp / gp_max
            equal_w = np.ones(gp.shape[1]) / np.sqrt(gp.shape[1])
            coeff_dist = np.linalg.norm(gp_norm - equal_w, axis=1)
        else:
            coeff_dist = np.zeros(len(diagnostics))
        order = np.lexsort((coeff_dist, var, score))
        pick = int(order[0])
        return pick, {
            "aggregation": aggregation,
            "score": float(score[pick]),
            "variance_tie_break": float(var[pick]),
            "coeff_l2_dist": float(coeff_dist[pick]),
        }

    if aggregation == "min_mean":
        mean_score = norm.mean(axis=1)
        max_score = norm.max(axis=1)
        var = norm.var(axis=1)
        order = np.lexsort((var, max_score, mean_score))
        pick = int(order[0])
        return pick, {
            "aggregation": aggregation,
            "mean_normalized": float(mean_score[pick]),
            "max_normalized": float(max_score[pick]),
            "variance_tie_break": float(var[pick]),
        }

    if aggregation == "pareto_closest":
        pareto_idx_list = pareto_frontier(norm.tolist())
        if not pareto_idx_list:
            pareto_idx_list = list(range(norm.shape[0]))
        pareto_idx = np.asarray(pareto_idx_list, dtype=np.int64)
        dist = np.linalg.norm(norm[pareto_idx], axis=1)
        pick = int(pareto_idx[int(np.argmin(dist))])
        return pick, {
            "aggregation": aggregation,
            "n_pareto": int(pareto_idx.size),
            "dist_from_origin": float(dist.min()),
        }

    raise ValueError(
        f"Unknown aggregation '{aggregation}'. "
        "Choose: 'min_of_normalized', 'min_mean', or 'pareto_closest'."
    )


# ---------------------------------------------------------------------------
# Stability report
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    grid_points: List[Tuple[float, ...]]
    assignments: List[torch.Tensor]
    pairwise_hamming: np.ndarray         # (G, G)
    flagged_unstable: List[Tuple[int, int]]


def assignment_stability(
    grid_points: Sequence[Tuple[float, ...]],
    assignments: Sequence[torch.Tensor],
    *,
    instability_threshold: float = 0.25,
) -> StabilityReport:
    G = len(grid_points)
    H = np.zeros((G, G), dtype=np.float64)
    n = assignments[0].shape[0]
    for i in range(G):
        for j in range(i + 1, G):
            H[i, j] = H[j, i] = hamming_distance(assignments[i], assignments[j]) / max(1, n)
    flagged = [(i, j) for i in range(G) for j in range(i + 1, G)
               if H[i, j] > instability_threshold]
    return StabilityReport(
        grid_points=list(grid_points),
        assignments=list(assignments),
        pairwise_hamming=H,
        flagged_unstable=flagged,
    )


# ---------------------------------------------------------------------------
# Stage-0 calibration (Monte Carlo expectation of term values)
# ---------------------------------------------------------------------------

def stage0_calibrate(
    sample_eval_fn,
    *,
    n_samples: int = 100,
    seed: int = 0,
    n_jobs: int = 1,
) -> Dict[str, float]:
    if n_samples < 100:
        print(
            f"[shield.diagnostics] stage0_samples = {n_samples} is below the "
            "lower bound of 100 - normalizers will "
            "have > 10% Monte-Carlo standard error."
        )

    rng = np.random.default_rng(seed)
    all_seeds = [int(rng.integers(low=0, high=2**31 - 1)) for _ in range(n_samples)]

    use_parallel = (n_jobs != 1) and _HAS_JOBLIB and (n_samples > 1)

    if use_parallel:
        print(
            f"[shield.diagnostics] Stage-0: running {n_samples} MC samples "
            f"in parallel (n_jobs={n_jobs}, backend=threading)."
        )
        results = _joblib.Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            _joblib.delayed(sample_eval_fn)(s) for s in all_seeds
        )
    else:
        if n_jobs != 1 and not _HAS_JOBLIB:
            print(
                "[shield.diagnostics] Stage-0: joblib not installed — "
                "falling back to serial evaluation."
            )
        results = [sample_eval_fn(s) for s in all_seeds]

    accumulator: Dict[str, float] = {}
    for d in results:
        for k, v in d.items():
            accumulator[k] = accumulator.get(k, 0.0) + abs(float(v))
    count = len(results)
    return {k: v / max(1, count) for k, v in accumulator.items()}
