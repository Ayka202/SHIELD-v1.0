from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# 1. Cost matrix construction (representation-agnostic)
# ---------------------------------------------------------------------------

def pairwise_cost(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    metric: str = "sqeuclidean",
    chunk: int = 4096,
) -> torch.Tensor:
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    n, m = X.shape[0], Y.shape[0]
    out = torch.empty((n, m), device=X.device, dtype=torch.float32)
    if metric == "cosine":
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-12)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out[start:end] = 1.0 - (Xn[start:end] @ Yn.t())
        return out

    y_sq = (Y * Y).sum(dim=1)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)
        cross = xb @ Y.t()
        d2 = (x_sq + y_sq.unsqueeze(0) - 2.0 * cross).clamp_min_(0.0)
        if metric == "euclidean":
            out[start:end] = d2.sqrt()
        else:
            out[start:end] = d2
    return out


# ---------------------------------------------------------------------------
# 2. Entropic OT (D-match) - log-domain Sinkhorn
# ---------------------------------------------------------------------------

@dataclass
class SinkhornResult:
    P: torch.Tensor          # (n, m) transport plan in primal space
    f: torch.Tensor          # (n,) source dual potential
    g: torch.Tensor          # (m,) target dual potential
    cost: float              # <C, P> (raw OT cost, no entropy term)
    entropy: float           # sum P log P
    n_iter: int
    stop_reason: str         # 'max_iter' | 'dual_tol' | 'cost_change' | 'dual_tol+cost_change' | 'empty_marginal'


def sinkhorn_log(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float = 5e-2,
    max_iter: int = 500,
    tol: float = 1e-3,
    cost_rel_tol: float = 1e-4,
    n_verify: int = 3,
    warm_start: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> SinkhornResult:
    a = a.to(torch.float32).to(C.device)
    b = b.to(torch.float32).to(C.device)
    # numerical guards against zero mass
    a = a.clamp_min(1e-30)
    b = b.clamp_min(1e-30)
    a = a / a.sum()
    b = b / b.sum()
    log_a = torch.log(a)
    log_b = torch.log(b)
    n, m = C.shape
    if warm_start is not None:
        f, g = warm_start
        f = f.to(C.device)
        g = g.to(C.device)
    else:
        f = torch.zeros(n, device=C.device, dtype=torch.float32)
        g = torch.zeros(m, device=C.device, dtype=torch.float32)

    inv_eps = 1.0 / max(epsilon, 1e-12)
    converged_iter = max_iter
    stop_reason = "max_iter"

    pending_stop = False
    verify_start = -1
    pending_reason = ""
    prev_cost_proxy = float("inf")

    for it in range(max_iter):
        # f update
        M = (g.unsqueeze(0) - C) * inv_eps
        lse_j = torch.logsumexp(M, dim=1)
        f_new = epsilon * (log_a - lse_j)
        # g update with new f
        M = (f_new.unsqueeze(1) - C) * inv_eps
        lse_i = torch.logsumexp(M, dim=0)
        g_new = epsilon * (log_b - lse_i)

        delta = max((f_new - f).abs().max().item(), (g_new - g).abs().max().item())
        f, g = f_new, g_new

        # cost proxy
        cost_proxy = float((a * f).sum().item() + (b * g).sum().item())
        if it > 0:
            cost_change_rel = abs(cost_proxy - prev_cost_proxy) / (
                abs(cost_proxy) + 1e-10
            )
            cost_fire = cost_change_rel < cost_rel_tol
        else:
            cost_fire = False
        prev_cost_proxy = cost_proxy

        dual_fire = delta < tol

        if dual_fire or cost_fire:
            if not pending_stop:
                pending_stop = True
                verify_start = it
                reasons: list = []
                if dual_fire:
                    reasons.append("dual_tol")
                if cost_fire:
                    reasons.append("cost_change")
                pending_reason = "+".join(reasons)
            elif (it - verify_start) >= n_verify:
                stop_reason = pending_reason
                converged_iter = it + 1
                break
        else:
            pending_stop = False
            verify_start = -1
            pending_reason = ""

    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) * inv_eps
    P = torch.exp(log_P)
    P = P / P.sum().clamp_min(1e-30)
    cost = float((P * C).sum().item())
    safe_log_P = torch.where(P > 0, torch.log(P.clamp_min(1e-300)), torch.zeros_like(P))
    H = float((P * safe_log_P).sum().item())
    return SinkhornResult(
        P=P, f=f, g=g, cost=cost, entropy=H,
        n_iter=converged_iter, stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# 3. D-match wrapper across folds
# ---------------------------------------------------------------------------

@dataclass
class MatchEvaluation:
    total: float                      # sum_k rho_k (<C, Pi^k> + eps * H(Pi^k))
    per_fold_cost: torch.Tensor       # (K,) raw OT cost
    per_fold_entropy: torch.Tensor    # (K,) entropy
    plans: list                       # list[SinkhornResult]
    gradient_z: torch.Tensor          # (n, K) linear-in-z gradient for next LP


def _normalize_marginal(
    z_col: torch.Tensor,
    weights: torch.Tensor,
    floor: float = 1e-30,
) -> torch.Tensor:
    w = (z_col * weights).clamp_min(0.0)
    total = w.sum().clamp_min(floor)
    return w / total


def d_match_evaluate(
    z: torch.Tensor,
    weights: torch.Tensor,
    C: torch.Tensor,
    target_marginals: torch.Tensor,
    rho: torch.Tensor,
    *,
    epsilon_OT: float = 5e-2,
    max_iter: int = 200,
    tol: float = 1e-3,
    cost_rel_tol: float = 1e-4,
    n_verify: int = 5,
    warm_start: Optional[list] = None,
    verbose: bool = False,
) -> MatchEvaluation:
    device = z.device
    n, K = z.shape
    per_cost = torch.zeros(K, device=device)
    per_ent = torch.zeros(K, device=device)
    plans = []
    grad = torch.zeros((n, K), device=device, dtype=torch.float32)
    weights = weights.to(torch.float32).to(device)

    # loop over K folds only; sinkhorn_log is vectorized over (n, m)
    for k in range(K):
        marginal = _normalize_marginal(z[:, k], weights)
        if marginal.sum() <= 0:
            per_cost[k] = 0.0
            per_ent[k] = 0.0
            plans.append(SinkhornResult(
                P=torch.zeros_like(C), f=torch.zeros(n, device=device),
                g=torch.zeros(C.shape[1], device=device),
                cost=0.0, entropy=0.0, n_iter=0, stop_reason="empty_marginal",
            ))
            continue
        ws_k = None
        if warm_start is not None and k < len(warm_start) and warm_start[k] is not None:
            ws_k = warm_start[k]
        res = sinkhorn_log(
            C=C,
            a=marginal,
            b=target_marginals[k],
            epsilon=epsilon_OT,
            max_iter=max_iter,
            tol=tol,
            cost_rel_tol=cost_rel_tol,
            n_verify=n_verify,
            warm_start=ws_k,
        )
        per_cost[k] = res.cost
        per_ent[k] = res.entropy
        plans.append(res)
        W_k = (z[:, k] * weights).sum().clamp_min(1e-30)
        f = res.f
        mean_f = (f * marginal).sum()
        grad[:, k] = rho[k] * (f - mean_f) * weights / W_k

    rho_dev = rho.to(device)
    total = float((rho_dev * per_cost).sum().item()
                  + epsilon_OT * (rho_dev * per_ent).sum().item())

    if verbose:
        saved = sum(max_iter - p.n_iter for p in plans if p.stop_reason != "empty_marginal")
        lines = [
            f"[shield.distributional] D-match Sinkhorn summary  "
            f"(K={K}  max_iter={max_iter}  tol={tol:.0e}  cost_rel_tol={cost_rel_tol:.0e}  n_verify={n_verify}):"
        ]
        for k, p in enumerate(plans):
            if p.stop_reason == "empty_marginal":
                lines.append(f"  Fold {k}: skipped — empty marginal")
            else:
                tag = (
                    f"early stop [{p.stop_reason}]"
                    if p.stop_reason != "max_iter"
                    else "max_iter reached (did not converge)"
                )
                lines.append(f"  Fold {k}: {p.n_iter:>4d} iters — {tag}")
        if saved > 0:
            lines.append(
                f"  → {saved} iteration(s) saved vs max_iter cap across all folds"
            )
        print("\n".join(lines))

    return MatchEvaluation(
        total=total,
        per_fold_cost=per_cost.detach(),
        per_fold_entropy=per_ent.detach(),
        plans=plans,
        gradient_z=grad.detach(),
    )


# ---------------------------------------------------------------------------
# 4. D-shift mode - MMD^2 between fold pairs
# ---------------------------------------------------------------------------

def build_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
    chunk: int = 4096,
) -> torch.Tensor:
    if kernel == "linear":
        return X.to(torch.float32) @ Y.to(torch.float32).t()

    d2 = pairwise_cost(X, Y, metric="sqeuclidean", chunk=chunk)
    if bandwidth is None:
        med = d2.flatten().median().clamp_min(1e-12).item()
        bandwidth_eff = (med ** 0.5)
    else:
        bandwidth_eff = float(bandwidth)
    if kernel == "rbf":
        return torch.exp(-d2 / (2.0 * (bandwidth_eff ** 2) + 1e-12))
    if kernel == "laplace":
        return torch.exp(-d2.sqrt() / (bandwidth_eff + 1e-12))
    raise ValueError(f"Unknown kernel '{kernel}'")


@dataclass
class ShiftEvaluation:
    total: float                            # -sum_{k!=k'} rho_kk' MMD^2_kk'
    per_pair_mmd2: torch.Tensor             # (K, K) symmetric, zero diag
    gradient_z: torch.Tensor                # (n, K) linearization


def _fold_marginal_weight(
    z: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    wz = z * weights.unsqueeze(1)
    W = wz.sum(dim=0).clamp_min(1e-30)
    return wz / W.unsqueeze(0)


def d_shift_evaluate(
    z: torch.Tensor,
    weights: torch.Tensor,
    K_mat: torch.Tensor,
    rho_pair: torch.Tensor,
) -> ShiftEvaluation:
    device = z.device
    n, K = z.shape
    weights = weights.to(torch.float32).to(device)
    u = _fold_marginal_weight(z, weights)  # (n, K)
    Ku = K_mat @ u                          # (n, K)
    diag = (u * Ku).sum(dim=0)              # (K,)
    cross = u.t() @ Ku                      # (K, K)
    mmd2 = diag.unsqueeze(1) + diag.unsqueeze(0) - 2.0 * cross
    mmd2.fill_diagonal_(0.0)
    mmd2.clamp_min_(0.0)
    rho_pair = rho_pair.to(device)
    total = -float((rho_pair * mmd2).sum().item())

    grad = torch.zeros((n, K), device=device, dtype=torch.float32)
    W = (z * weights.unsqueeze(1)).sum(dim=0).clamp_min(1e-30)
    for k in range(K):
        acc = torch.zeros(n, device=device, dtype=torch.float32)
        for kp in range(K):
            if kp == k:
                continue
            acc = acc + rho_pair[k, kp] * (Ku[:, k] - Ku[:, kp])
        # negation because we minimize -MMD^2
        grad[:, k] = -2.0 * (weights / W[k]) * acc
    return ShiftEvaluation(
        total=total,
        per_pair_mmd2=mmd2.detach(),
        gradient_z=grad.detach(),
    )


# ---------------------------------------------------------------------------
# 4b. Analytical reference scales for D-shift normalization
# ---------------------------------------------------------------------------
def _densify_kernel(K_mat: torch.Tensor) -> torch.Tensor:
    if K_mat.is_sparse:
        return K_mat.to_dense()
    return K_mat


def mmd_worst_case_scale(
    K_mat: torch.Tensor,
    rho_pair: torch.Tensor,
) -> float:
    K_dense = _densify_kernel(K_mat)
    max_diag = float(K_dense.diag().max().item())
    min_K    = float(K_dense.min().item())
    mmd2_max = max(2.0 * max_diag - 2.0 * min_K, 0.0)
    rho_sum  = float(rho_pair.to(K_dense.device).sum().item())
    return rho_sum * mmd2_max


def mmd_average_pairwise_scale(
    K_mat: torch.Tensor,
    rho_pair: torch.Tensor,
) -> float:
    K_dense   = _densify_kernel(K_mat)
    mean_diag = float(K_dense.diag().mean().item())
    mean_K    = float(K_dense.mean().item())
    mmd2_typ  = max(2.0 * mean_diag - 2.0 * mean_K, 0.0)
    rho_sum   = float(rho_pair.to(K_dense.device).sum().item())
    return rho_sum * mmd2_typ


def mmd_reference_scale(
    K_mat: torch.Tensor,
    rho_pair: torch.Tensor,
) -> float:
    worst        = mmd_worst_case_scale(K_mat, rho_pair)
    K_dense      = _densify_kernel(K_mat)
    kernel_scale = max(float(K_dense.abs().max().item()), 1e-30)
    rho_sum      = float(rho_pair.to(K_dense.device).sum().item())
    threshold    = 1e-6 * rho_sum * kernel_scale
    if worst > threshold:
        return worst
    typical = mmd_average_pairwise_scale(K_mat, rho_pair)
    return max(typical, threshold)


# ---------------------------------------------------------------------------
# 5. unified D-term API used by the pipeline
# ---------------------------------------------------------------------------

def evaluate_d_term(
    z: torch.Tensor,
    weights: torch.Tensor,
    *,
    mode: str,
    C: Optional[torch.Tensor] = None,
    target_marginals: Optional[torch.Tensor] = None,
    rho: Optional[torch.Tensor] = None,
    epsilon_OT: float = 5e-2,
    K_mat: Optional[torch.Tensor] = None,
    rho_pair: Optional[torch.Tensor] = None,
    sinkhorn_max_iter: int = 200,
    sinkhorn_tol: float = 1e-3,
    sinkhorn_cost_rel_tol: float = 1e-4,
    sinkhorn_n_verify: int = 3,
    warm_start: Optional[list] = None,
    verbose: bool = False,
):
    if mode == "match":
        if C is None or target_marginals is None or rho is None:
            raise ValueError("D-match requires C, target_marginals, rho.")
        return d_match_evaluate(
            z=z, weights=weights, C=C,
            target_marginals=target_marginals, rho=rho,
            epsilon_OT=epsilon_OT,
            max_iter=sinkhorn_max_iter,
            tol=sinkhorn_tol,
            cost_rel_tol=sinkhorn_cost_rel_tol,
            n_verify=sinkhorn_n_verify,
            warm_start=warm_start,
            verbose=verbose,
        )
    if mode == "shift":
        if K_mat is None or rho_pair is None:
            raise ValueError("D-shift requires K_mat, rho_pair.")
        return d_shift_evaluate(
            z=z, weights=weights,
            K_mat=K_mat, rho_pair=rho_pair,
        )
    raise ValueError(f"Unknown D mode: '{mode}'")
