"""
Multi-output stratification utilities

Three strategies are supported and vectorized in torch:

    1. Quantile-binned
       per-coordinate binning + class-style residuals.
    2. Higher-moment matching up to fourth order
       Reports the empirical fourth-moment standard error so users can flag
       coordinates whose tails make the criterion unstable.
    3. Sliced Wasserstein (recommended default for q >= 2)

For each strategy the module also exposes a linear-in-z gradient that the
assignment LP (`shield.solvers.gurobi_lp.solve_assignment_lp`) can fold into
its objective at every alternation step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fold_marginal_mass(
    z: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (weights.unsqueeze(1) * z).sum(dim=0)


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return num / den.clamp_min(eps)


@dataclass
class QuantileBins:
    edges: torch.Tensor              # (q, B+1) bin edges per output coord
    bin_index: torch.Tensor          # (n, q) int64 bin assignment


def build_quantile_bins(
    y: torch.Tensor,
    B: int = 10,
) -> QuantileBins:
    n, q = y.shape
    # quantile edges per coordinate
    qs = torch.linspace(0.0, 1.0, B + 1, device=y.device, dtype=torch.float32)
    edges = torch.quantile(y.to(torch.float32), qs, dim=0).t()      # (q, B+1)
    bins = torch.empty((n, q), dtype=torch.long, device=y.device)
    for j in range(q):
        bins[:, j] = torch.bucketize(y[:, j].contiguous(), edges[j, 1:-1])
    return QuantileBins(edges=edges, bin_index=bins.clamp_max(B - 1))


def quantile_residuals(
    z: torch.Tensor,
    weights: torch.Tensor,
    bins: QuantileBins,
    r: torch.Tensor,
) -> torch.Tensor:
    n, q = bins.bin_index.shape
    K = z.shape[1]
    device = z.device
    B = int(bins.edges.shape[1] - 1)
    r = r.to(device).to(torch.float32)
    weights = weights.to(device).to(torch.float32)

    total = torch.zeros((), device=device, dtype=torch.float32)
    # per coordinate: (K, B) tally of weights vs. target r_k * W_{j,b}
    for j in range(q):
        bj = bins.bin_index[:, j]
        # weights per bin (target = sum_i w_i 1[bin_j(i)=b])
        target_per_bin = torch.zeros(B, device=device, dtype=torch.float32)
        target_per_bin.scatter_add_(0, bj, weights)
        # fold-wise weight per bin: combine (K,B) via flat index
        flat = z.t().reshape(K, n)                  # (K, n)
        tally = torch.zeros((K, B), device=device, dtype=torch.float32)
        idx_flat = bj.unsqueeze(0).expand(K, -1)    # (K, n)
        w_z = (weights.unsqueeze(0) * flat)         # (K, n)
        tally.scatter_add_(1, idx_flat, w_z)
        target_kb = r.unsqueeze(1) * target_per_bin.unsqueeze(0)  # (K, B)
        diff = tally - target_kb
        total = total + (diff * diff).sum()
    return total


def quantile_worst_case_scale(
    bins: QuantileBins,
    weights: torch.Tensor,
    r: torch.Tensor,
) -> float:
    n, q = bins.bin_index.shape
    B = int(bins.edges.shape[1] - 1)
    device = weights.device
    w = weights.to(device).to(torch.float32)
    total_W_b_sq = torch.zeros((), device=device, dtype=torch.float32)
    for j in range(q):
        bj = bins.bin_index[:, j]
        W_b = torch.zeros(B, device=device, dtype=torch.float32)
        W_b.scatter_add_(0, bj, w)
        total_W_b_sq = total_W_b_sq + (W_b * W_b).sum()
    r_dev = r.to(device).to(torch.float32)
    coeff = 1.0 - 2.0 * float(r_dev.min().item()) + float((r_dev * r_dev).sum().item())
    return float(total_W_b_sq.item()) * coeff


def quantile_gradient_z(
    z: torch.Tensor,
    weights: torch.Tensor,
    bins: QuantileBins,
    r: torch.Tensor,
) -> torch.Tensor:
    n, q = bins.bin_index.shape
    K = z.shape[1]
    device = z.device
    B = int(bins.edges.shape[1] - 1)
    r = r.to(device).to(torch.float32)
    weights = weights.to(device).to(torch.float32)

    grad = torch.zeros((n, K), device=device, dtype=torch.float32)
    for j in range(q):
        bj = bins.bin_index[:, j]
        target_per_bin = torch.zeros(B, device=device, dtype=torch.float32)
        target_per_bin.scatter_add_(0, bj, weights)
        flat = z.t().reshape(K, n)
        tally = torch.zeros((K, B), device=device, dtype=torch.float32)
        idx_flat = bj.unsqueeze(0).expand(K, -1)
        w_z = (weights.unsqueeze(0) * flat)
        tally.scatter_add_(1, idx_flat, w_z)
        target_kb = r.unsqueeze(1) * target_per_bin.unsqueeze(0)  # (K, B)
        # diff per (k, b)
        diff = tally - target_kb                                  # (K, B)
        # gather d/dz_{i,k} contribution: 2 w_i * diff[k, bin_j(i)]
        gathered = diff[:, bj]                                    # (K, n)
        grad = grad + 2.0 * weights.unsqueeze(1) * gathered.t()
    return grad



@dataclass
class MomentReport:
    moments_per_fold: torch.Tensor              # (4, K, q)
    moments_target: torch.Tensor                # (4, q)
    residual: float
    fourth_moment_std_error: torch.Tensor       # (q,) for diagnostic flagging


def _weighted_moments(
    y: torch.Tensor,                              # (n, q)
    w: torch.Tensor,                              # (n,)
    z_col: Optional[torch.Tensor] = None,         # (n,) in [0,1] or None for pooled
) -> torch.Tensor:
    if z_col is None:
        eff_w = w
    else:
        eff_w = w * z_col
    total = eff_w.sum().clamp_min(1e-12)
    mean = (eff_w.unsqueeze(1) * y).sum(dim=0) / total           # (q,)
    centered = y - mean.unsqueeze(0)                               # (n, q)
    out = torch.empty((4, y.shape[1]), device=y.device, dtype=torch.float32)
    out[0] = mean
    for idx, m in enumerate((2, 3, 4), start=1):
        out[idx] = (eff_w.unsqueeze(1) * centered.pow(m)).sum(dim=0) / total
    return out


def moment_residual(
    z: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    omega: Optional[torch.Tensor] = None,
) -> MomentReport:
    K = z.shape[1]
    q = y.shape[1]
    device = z.device
    y = y.to(torch.float32).to(device)
    weights = weights.to(torch.float32).to(device)
    if omega is None:
        omega = torch.ones((4, q), device=device, dtype=torch.float32)
    else:
        omega = omega.to(device).to(torch.float32)

    target = _weighted_moments(y, weights, None)                  # (4, q)
    per_fold = torch.zeros((4, K, q), device=device, dtype=torch.float32)
    _mass_floor = 1e-12 * float(weights.sum().item())
    for k in range(K):
        if float((weights * z[:, k]).sum().item()) <= _mass_floor:
            per_fold[:, k] = target
            continue
        per_fold[:, k] = _weighted_moments(y, weights, z[:, k])
    diff = per_fold - target.unsqueeze(1)                          # (4, K, q)
    residual = float((omega.unsqueeze(1) * diff * diff).sum().item())

    # empirical 4th-moment standard error per coord (diagnostic)
    centered = y - target[0].unsqueeze(0)
    fourth = centered.pow(4) * weights.unsqueeze(1)
    mean4 = fourth.mean(dim=0)
    var4 = (fourth - mean4.unsqueeze(0)).pow(2).mean(dim=0)
    se4 = (var4 / max(1.0, y.shape[0])).sqrt()
    return MomentReport(
        moments_per_fold=per_fold,
        moments_target=target,
        residual=residual,
        fourth_moment_std_error=se4,
    )


def moment_worst_case_scale(
    y: torch.Tensor,
    weights: torch.Tensor,
    r: torch.Tensor,
    omega: Optional[torch.Tensor] = None,
) -> float:
    device = weights.device
    y = y.to(torch.float32).to(device)
    w = weights.to(torch.float32).to(device)
    n, q = y.shape
    K = int(r.shape[0])
    if omega is None:
        omega = torch.ones((4, q), device=device, dtype=torch.float32)
    else:
        omega = omega.to(device).to(torch.float32)

    y_min = y.min(dim=0).values
    y_max = y.max(dim=0).values
    R = (y_max - y_min).clamp_min(0.0)                                # (q,)

    # Pooled moments for the even-order relative scales.
    target = _weighted_moments(y, w, None)                            # (4, q)

    S_m1 = R.pow(2)
    eps_m2 = (1e-6 * R.pow(2)).clamp_min(1e-30)
    S_m2 = torch.maximum(target[1].abs(), eps_m2).pow(2)
    S_m3 = R.pow(6) / 64.0
    eps_m4 = (1e-6 * R.pow(4)).clamp_min(1e-30)
    S_m4 = torch.maximum(target[3].abs(), eps_m4).pow(2)

    per_coord = (omega[0] * S_m1 + omega[1] * S_m2
                 + omega[2] * S_m3 + omega[3] * S_m4)
    return float(K) * float(per_coord.sum().item())


def moment_gradient_z(
    z: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    omega: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    K = z.shape[1]
    q = y.shape[1]
    n = z.shape[0]
    device = z.device
    y = y.to(torch.float32).to(device)
    weights = weights.to(torch.float32).to(device)
    if omega is None:
        omega = torch.ones((4, q), device=device, dtype=torch.float32)
    else:
        omega = omega.to(device).to(torch.float32)

    target = _weighted_moments(y, weights, None)                    # (4, q)
    grad = torch.zeros((n, K), device=device, dtype=torch.float32)

    # relative mass floor below which a fold is treated as empty
    _mass_floor = 1e-12 * float(weights.sum().item())

    for k in range(K):
        a_k = weights * z[:, k]                                     # (n,)
        if float(a_k.sum().item()) <= _mass_floor:
            continue
        S_k = a_k.sum().clamp_min(1e-30)
        mu_k = (a_k.unsqueeze(1) * y).sum(dim=0) / S_k              # (q,)
        centered_k = y - mu_k.unsqueeze(0)                           # (n, q)

        # central moments M^{(m)} at the current z, m = 1..4.
        M_k = torch.empty((4, q), device=device, dtype=torch.float32)
        M_k[0] = mu_k                                                # raw mean
        for m_idx, m_order in enumerate((2, 3, 4), start=1):
            M_k[m_idx] = (a_k.unsqueeze(1) * centered_k.pow(m_order)).sum(dim=0) / S_k
        diff = M_k - target                                          # (4, q)

        # m = 1 (raw): ∂M^{(1)}/∂z_{l,k} = w_l (y_l - μ_k) / S_k
        g_m1 = weights.unsqueeze(1) * centered_k / S_k                # (n, q)
        grad[:, k] = grad[:, k] + (2.0 * omega[0] * diff[0] * g_m1).sum(dim=1)

        # m = 2, 3, 4 (central). M_prev_central = M^{(m-1)}; for m=2 use 0.
        for m_idx, m_order in enumerate((2, 3, 4), start=1):
            if m_order == 2:
                M_prev_c = torch.zeros(q, device=device, dtype=torch.float32)
            else:
                M_prev_c = M_k[m_idx - 1]                            # (q,)
            d_M = (centered_k.pow(m_order)
                   - float(m_order) * centered_k * M_prev_c.unsqueeze(0)
                   - M_k[m_idx].unsqueeze(0)) / S_k                  # (n, q)
            g_m = weights.unsqueeze(1) * d_M                         # (n, q)
            grad[:, k] = grad[:, k] + (2.0 * omega[m_idx] * diff[m_idx] * g_m).sum(dim=1)
    return grad



def _draw_projections(P: int, q: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.randn(P, q, generator=g).to(device)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return theta.to(torch.float32)


@dataclass
class SlicedWassersteinReport:
    sw2_per_fold: torch.Tensor                # (K,)
    sw2_total: float
    P: int
    monte_carlo_se: float                     # SE of the SW^2 estimator across projections


def _weighted_quantile_via_sort(
    proj: torch.Tensor,                          # (n,) projected values
    w: torch.Tensor,                              # (n,) weights (must be >=0)
    levels: torch.Tensor,                         # (Q,) in [0, 1]
) -> torch.Tensor:
    order = torch.argsort(proj)
    s_proj = proj[order]
    s_w = w[order]
    cw = torch.cumsum(s_w, dim=0)
    cw = cw / cw[-1].clamp_min(1e-30)
    idx = torch.searchsorted(cw, levels.clamp(0.0, 1.0)).clamp_max(len(s_proj) - 1)
    return s_proj[idx]


def _trapezoidal_integral_unit(values: torch.Tensor) -> torch.Tensor:
    Q = int(values.shape[0])
    if Q < 2:
        return values.mean()
    return (0.5 * (values[0] + values[-1]) + values[1:-1].sum()) / float(Q - 1)


def sliced_wasserstein_stratification(
    z: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    *,
    P: int = 100,
    Q: int = 256,
    seed: int = 0,
) -> SlicedWassersteinReport:
    device = z.device
    n, q = y.shape
    K = z.shape[1]
    y = y.to(torch.float32).to(device)
    weights = weights.to(torch.float32).to(device)
    theta = _draw_projections(P, q, device, seed=seed)
    levels = torch.linspace(0.0, 1.0, Q, device=device, dtype=torch.float32)

    # running E[X], E[X^2] of the per-projection SW^2 sample
    mean_per_fold = torch.zeros(K, device=device, dtype=torch.float32)
    sqmean_per_fold = torch.zeros(K, device=device, dtype=torch.float32)

    inv_P = 1.0 / float(max(1, P))
    for p in range(P):
        proj = (y * theta[p]).sum(dim=1)                              # (n,)
        q_star = _weighted_quantile_via_sort(proj, weights, levels)   # (Q,)
        for k in range(K):
            wk = weights * z[:, k]
            wk_sum = wk.sum().clamp_min(1e-30)  # fold-k total mass
            q_k = _weighted_quantile_via_sort(proj, wk, levels)
            diff_sq = (q_k - q_star).pow(2)                           # (Q,)
            sq = _trapezoidal_integral_unit(diff_sq)                  # scalar
            mean_per_fold[k]   = mean_per_fold[k]   + sq * inv_P
            sqmean_per_fold[k] = sqmean_per_fold[k] + sq.pow(2) * inv_P

    # sample variance 
    sample_var = (sqmean_per_fold - mean_per_fold.pow(2)).clamp_min_(0.0)
    # SE of the mean estimator: sqrt(Var / P), averaged across folds
    se_per_fold = (sample_var / float(max(1, P))).sqrt()
    se = float(se_per_fold.mean().item())

    return SlicedWassersteinReport(
        sw2_per_fold=mean_per_fold.detach(),
        sw2_total=float(mean_per_fold.sum().item()),
        P=P,
        monte_carlo_se=se,
    )


def sw_gradient_z(
    z: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    *,
    P: int = 64,
    Q: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    device = z.device
    n, q = y.shape
    K = z.shape[1]
    y = y.to(torch.float32).to(device)
    weights = weights.to(torch.float32).to(device)
    theta = _draw_projections(P, q, device, seed=seed)
    levels = torch.linspace(0.0, 1.0, Q, device=device, dtype=torch.float32)
    grad = torch.zeros((n, K), device=device, dtype=torch.float32)

    _mass_floor = 1e-12 * float(weights.sum().item())

    inv_P = 1.0 / float(max(1, P))
    for p in range(P):
        proj = (y * theta[p]).sum(dim=1)
        q_star = _weighted_quantile_via_sort(proj, weights, levels)
        order = torch.argsort(proj)
        s_proj = proj[order]
        # per-fold cumulative-weight level -> per-fold target quantile
        for k in range(K):
            wk = weights * z[:, k]
            if float(wk.sum().item()) <= _mass_floor:
                continue
            wk_sorted = wk[order]
            cw_k = torch.cumsum(wk_sorted, dim=0)
            cw_k = cw_k / cw_k[-1].clamp_min(1e-30)
            target_at_obs_k = q_star[
                torch.searchsorted(levels, cw_k).clamp_max(Q - 1)
            ]
            per_obs_sq_k = (s_proj - target_at_obs_k).pow(2)
            # Un-permute back to original observation order before scattering
            # into the (n, K) gradient column.
            col = torch.empty(n, device=device, dtype=torch.float32)
            col[order] = per_obs_sq_k
            grad[:, k] = grad[:, k] + col * inv_P
    return grad


def sw_covariance_trace_scale(
    y: torch.Tensor,
    weights: torch.Tensor,
    r: torch.Tensor,
) -> float:
    device = weights.device
    y = y.to(torch.float32).to(device)
    w = weights.to(torch.float32).to(device).clamp_min(0.0)
    W = w.sum().clamp_min(1e-30)
    mean_y = (w.unsqueeze(1) * y).sum(dim=0) / W                      # (q,)
    centered = y - mean_y.unsqueeze(0)
    var_y = (w.unsqueeze(1) * centered.pow(2)).sum(dim=0) / W          # (q,)
    K = int(r.shape[0])
    return float(K) * float(var_y.sum().item())


def sliced_wasserstein_and_gradient(
    z: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    *,
    P: int = 100,
    Q: int = 256,
    seed: int = 0,
) -> Tuple[SlicedWassersteinReport, torch.Tensor]:
    device = z.device
    n, q = y.shape
    K = z.shape[1]
    y = y.to(torch.float32).to(device)
    weights = weights.to(torch.float32).to(device)
    theta = _draw_projections(P, q, device, seed=seed)
    levels = torch.linspace(0.0, 1.0, Q, device=device, dtype=torch.float32)

    mean_per_fold = torch.zeros(K, device=device, dtype=torch.float32)
    sqmean_per_fold = torch.zeros(K, device=device, dtype=torch.float32)
    grad = torch.zeros((n, K), device=device, dtype=torch.float32)
    inv_P = 1.0 / float(max(1, P))
    _mass_floor = 1e-12 * float(weights.sum().item())

    for p in range(P):
        proj = (y * theta[p]).sum(dim=1)                              # (n,)
        q_star = _weighted_quantile_via_sort(proj, weights, levels)   # (Q,)
        order = torch.argsort(proj)
        s_proj = proj[order]
        for k in range(K):
            wk = weights * z[:, k]
            if float(wk.sum().item()) <= _mass_floor:
                continue
            q_k = _weighted_quantile_via_sort(proj, wk, levels)
            diff_sq = (q_k - q_star).pow(2)
            sq = _trapezoidal_integral_unit(diff_sq)
            mean_per_fold[k] = mean_per_fold[k] + sq * inv_P
            sqmean_per_fold[k] = sqmean_per_fold[k] + sq.pow(2) * inv_P
            # gradient: per-observation squared deviation at fold-k quantile level
            wk_sorted = wk[order]
            cw_k = torch.cumsum(wk_sorted, dim=0)
            cw_k = cw_k / cw_k[-1].clamp_min(1e-30)
            target_at_obs_k = q_star[
                torch.searchsorted(levels, cw_k).clamp_max(Q - 1)
            ]
            per_obs_sq_k = (s_proj - target_at_obs_k).pow(2)
            col = torch.empty(n, device=device, dtype=torch.float32)
            col[order] = per_obs_sq_k
            grad[:, k] = grad[:, k] + col * inv_P

    sample_var = (sqmean_per_fold - mean_per_fold.pow(2)).clamp_min_(0.0)
    se_per_fold = (sample_var / float(max(1, P))).sqrt()
    se = float(se_per_fold.mean().item())
    report = SlicedWassersteinReport(
        sw2_per_fold=mean_per_fold.detach(),
        sw2_total=float(mean_per_fold.sum().item()),
        P=P,
        monte_carlo_se=se,
    )
    print(
        f"[shield.stratification] sliced_wasserstein_and_gradient: computed "
        f"SW2 total and gradient in one pass over P={P} projections."
    )
    return report, grad



def _weighted_cdf_at(
    sorted_values: torch.Tensor,            # (n,) ascending
    cum_weights: torch.Tensor,              # (n,) ascending, normalized
    query: torch.Tensor,                    # (m,) values
) -> torch.Tensor:
    idx = torch.searchsorted(sorted_values, query, right=True).clamp_min(1) - 1
    return cum_weights[idx]


def anderson_darling_per_fold(
    z: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = z.device
    K = z.shape[1]
    n, q = y.shape
    y = y.to(torch.float32).to(device)
    if weights is None:
        weights = torch.ones(n, device=device, dtype=torch.float32)
    else:
        weights = weights.to(torch.float32).to(device)

    out = torch.zeros((K, q), device=device, dtype=torch.float32)
    for j in range(q):
        order_pool = torch.argsort(y[:, j])
        sorted_pool_j = y[order_pool, j]
        w_pool_sorted = weights[order_pool]
        cw_pool = torch.cumsum(w_pool_sorted, dim=0)
        cw_pool = cw_pool / cw_pool[-1].clamp_min(1e-30)
        for k in range(K):
            mask = z[:, k] > 0.5
            nk = int(mask.sum().item())
            if nk < 2:
                continue
            yk = y[mask, j]
            wk = weights[mask]
            order_k = torch.argsort(yk)
            sorted_k = yk[order_k]
            w_k_sorted = wk[order_k]
            cw_k = torch.cumsum(w_k_sorted, dim=0)
            cw_k = cw_k / cw_k[-1].clamp_min(1e-30)
            # u_i = F_pool(x_i), the pooled CDF at fold-k point x_i
            u = _weighted_cdf_at(sorted_pool_j, cw_pool, sorted_k)
            u = u.clamp(1e-7, 1 - 1e-7)
            sq_diff = (cw_k - u).pow(2)
            denom = (u * (1.0 - u)).clamp_min(1e-30)
            W_eff = float(wk.sum().item())
            out[k, j] = float(
                (w_k_sorted * sq_diff / denom).sum().item()
            ) / max(W_eff, 1e-30)
    return out


def kolmogorov_smirnov_per_fold(
    z: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = z.device
    K = z.shape[1]
    n, q = y.shape
    y = y.to(torch.float32).to(device)
    if weights is None:
        weights = torch.ones(n, device=device, dtype=torch.float32)
    else:
        weights = weights.to(torch.float32).to(device)
    out = torch.zeros((K, q), device=device, dtype=torch.float32)

    for j in range(q):
        order_pool = torch.argsort(y[:, j])
        sorted_pool_j = y[order_pool, j]
        w_pool_sorted = weights[order_pool]
        cw_pool = torch.cumsum(w_pool_sorted, dim=0)
        cw_pool = cw_pool / cw_pool[-1].clamp_min(1e-30)
        for k in range(K):
            mask = z[:, k] > 0.5
            nk = int(mask.sum().item())
            if nk < 2:
                continue
            yk = y[mask, j]
            wk = weights[mask]
            order_k = torch.argsort(yk)
            sorted_k = yk[order_k]
            w_k_sorted = wk[order_k]
            cw_k = torch.cumsum(w_k_sorted, dim=0)
            cw_k = cw_k / cw_k[-1].clamp_min(1e-30)
            pool_at_k = _weighted_cdf_at(sorted_pool_j, cw_pool, sorted_k)
            out[k, j] = (cw_k - pool_at_k).abs().max()
    return out


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

@dataclass
class StratEvaluation:
    total: float
    gradient_z: torch.Tensor
    report: dict


def evaluate_stratification(
    z: torch.Tensor,
    weights: torch.Tensor,
    *,
    strategy: str,
    y: Optional[torch.Tensor] = None,
    bins: Optional[QuantileBins] = None,
    r: Optional[torch.Tensor] = None,
    omega: Optional[torch.Tensor] = None,
    P: int = 100,
    Q: int = 256,
    seed: int = 0,
) -> StratEvaluation:
    report: dict = {}
    if strategy == "quantile":
        if bins is None or r is None:
            raise ValueError("quantile stratification requires bins and r")
        total = float(quantile_residuals(z, weights, bins, r).item())
        grad = quantile_gradient_z(z, weights, bins, r)
        report["bins"] = bins
        return StratEvaluation(total=total, gradient_z=grad, report=report)
    if strategy == "moment":
        if y is None:
            raise ValueError("moment stratification requires y")
        rep = moment_residual(z, weights, y, omega=omega)
        grad = moment_gradient_z(z, weights, y, omega=omega)
        report["moments"] = rep
        return StratEvaluation(total=rep.residual, gradient_z=grad, report=report)
    if strategy == "sliced_wasserstein":
        if y is None:
            raise ValueError("sliced_wasserstein stratification requires y")
        rep, grad = sliced_wasserstein_and_gradient(z, weights, y, P=P, Q=Q, seed=seed)
        report["sliced_wasserstein"] = rep
        return StratEvaluation(total=rep.sw2_total, gradient_z=grad, report=report)
    if strategy == "none":
        return StratEvaluation(
            total=0.0,
            gradient_z=torch.zeros_like(z),
            report={},
        )
    raise ValueError(f"Unknown stratification strategy '{strategy}'")
