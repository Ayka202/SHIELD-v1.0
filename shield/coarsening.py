"""
Coarsening for SHIELD

Two responsibilities for +1M-scale data:
    1) cluster n raw observations into n' super-nodes using **balanced batch
       K-means** in pure PyTorch (vectorized over data dimension);
    2) sparsify a dense affinity / distance source into a symmetric k-NN graph
       on either the raw observations or the super-nodes.

All heavy operations:
    * run on torch.Tensor objects placed on a user-selected device
      (`device="cpu"` or `device="cuda"`);
    * are mini-batched along the n axis to bound peak memory at O(B * n);

The torch worker-thread count can be controlled with `nodes` (-1 -> use all
physical CPU cores; honoured globally via torch.set_num_threads).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# device / parallelism helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' requested but torch.cuda.is_available() is False."
        )
    return torch.device(device)


def configure_threading(nodes: int) -> int:
    """
    nodes = -1  -> use all available logical CPUs
    nodes =  0  -> single threaded
    nodes >  0  -> use exactly that many threads
    """
    if nodes == -1:
        n = os.cpu_count() or 1
    elif nodes <= 0:
        n = 1
    else:
        n = int(nodes)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except RuntimeError:
        # interop threads can only be set once per process; ignore if locked.
        pass
    return torch.get_num_threads()


# ---------------------------------------------------------------------------
# pairwise distance helpers 
# ---------------------------------------------------------------------------

def pairwise_sqdist_chunked(
    X: torch.Tensor,
    Y: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:

    n = X.shape[0]
    m = Y.shape[0]
    out = torch.empty((n, m), dtype=torch.float32, device=X.device)
    y_sq = (Y * Y).sum(dim=1)  # (m,)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)         # (b, 1)
        cross = xb @ Y.t()                                # (b, m)
        out[start:end] = (x_sq + y_sq.unsqueeze(0) - 2.0 * cross).clamp_min_(0.0)
    return out


def argmin_chunked(
    X: torch.Tensor,
    centroids: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:

    n = X.shape[0]
    out = torch.empty(n, dtype=torch.long, device=X.device)
    c_sq = (centroids * centroids).sum(dim=1)             # (K,)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xb = X[start:end]
        x_sq = (xb * xb).sum(dim=1, keepdim=True)         # (b, 1)
        cross = xb @ centroids.t()                        # (b, K)
        d = x_sq + c_sq.unsqueeze(0) - 2.0 * cross
        out[start:end] = d.argmin(dim=1)
    return out


# ---------------------------------------------------------------------------
# Balanced (batch) K-means
# ---------------------------------------------------------------------------

@dataclass
class CoarseningResult:
    labels: torch.Tensor               # (n,)
    centroids: torch.Tensor            # (n_clusters, d) 
    super_weights: torch.Tensor        # (n_clusters,) 
    inertia: float


def coarsen_balanced_kmeans(
    X: torch.Tensor,
    n_clusters: int,
    weights: Optional[torch.Tensor] = None,
    *,
    max_iter: int = 25,
    tol: float = 1e-4,
    chunk: int = 8192,
    balance_strength: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoarseningResult:

    if X.dtype != torch.float32:
        X = X.to(torch.float32)
    device_t = resolve_device(device)
    X = X.to(device_t)
    n, d = X.shape
    if n_clusters >= n:
        # No coarsening: every point is its own super-node.
        labels = torch.arange(n, device=device_t, dtype=torch.long)
        w = (weights.to(device_t) if weights is not None
             else torch.ones(n, device=device_t, dtype=torch.float32))
        return CoarseningResult(
            labels=labels,
            centroids=X.clone(),
            super_weights=w.clone(),
            inertia=0.0,
        )

    if weights is None:
        w = torch.ones(n, device=device_t, dtype=torch.float32)
    else:
        w = weights.to(device_t).to(torch.float32)

    g = torch.Generator(device="cpu").manual_seed(seed)
    first_idx = int(torch.randint(0, n, (1,), generator=g).item())
    centroids = X[first_idx:first_idx + 1].clone()                          # (1, d)
    for _ in range(n_clusters - 1):
        d2 = pairwise_sqdist_chunked(X, centroids, chunk=chunk).min(dim=1).values
        probs = (d2 * w).clamp_min_(1e-12)
        probs = probs / probs.sum()
        idx = int(torch.multinomial(probs.cpu(), 1, generator=g).item())
        centroids = torch.cat([centroids, X[idx:idx + 1]], dim=0)

    target = w.sum() / float(n_clusters)
    prev_inertia = math.inf
    labels = torch.zeros(n, device=device_t, dtype=torch.long)
    occupancy = torch.zeros(n_clusters, device=device_t, dtype=torch.float32)

    c_sq_base = (centroids * centroids).sum(dim=1)
    for it in range(max_iter):
        new_labels = torch.empty(n, dtype=torch.long, device=device_t)
        chunk_inertia = torch.zeros((), device=device_t, dtype=torch.float32)
        new_occupancy = torch.zeros(n_clusters, device=device_t, dtype=torch.float32)
        if it > 0 and balance_strength > 0:
            penalty = balance_strength * (occupancy - target) / (target + 1e-9)
        else:
            penalty = torch.zeros(n_clusters, device=device_t, dtype=torch.float32)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xb = X[start:end]
            x_sq = (xb * xb).sum(dim=1, keepdim=True)
            cross = xb @ centroids.t()
            cost = (x_sq + c_sq_base.unsqueeze(0) - 2.0 * cross) + penalty.unsqueeze(0)
            cost.clamp_min_(0.0)
            lab = cost.argmin(dim=1)
            picked = (x_sq + c_sq_base.unsqueeze(0) - 2.0 * cross).clamp_min_(0.0)
            picked = picked.gather(1, lab.unsqueeze(1)).squeeze(1)
            wb = w[start:end]
            chunk_inertia = chunk_inertia + (picked * wb).sum()
            new_labels[start:end] = lab
            new_occupancy.scatter_add_(0, lab, wb)

        labels = new_labels
        occupancy = new_occupancy

        new_centroids = torch.zeros_like(centroids)
        ws = w.unsqueeze(1) * X                                                   # (n, d)
        new_centroids.index_add_(0, labels, ws)
        denom = occupancy.clamp_min(1e-12).unsqueeze(1)
        new_centroids = new_centroids / denom
        empty = (occupancy < 1e-9).nonzero(as_tuple=False).flatten()
        if empty.numel() > 0:
            d2 = pairwise_sqdist_chunked(X, new_centroids, chunk=chunk).min(dim=1).values
            order = torch.argsort(d2, descending=True)
            new_centroids[empty] = X[order[:empty.numel()]]
        centroids = new_centroids
        c_sq_base = (centroids * centroids).sum(dim=1)

        inertia = float(chunk_inertia.item())
        if abs(prev_inertia - inertia) <= tol * max(1.0, prev_inertia):
            prev_inertia = inertia
            break
        prev_inertia = inertia

    super_w = torch.zeros(n_clusters, device=device_t, dtype=torch.float32)
    super_w.scatter_add_(0, labels, w)

    return CoarseningResult(
        labels=labels,
        centroids=centroids,
        super_weights=super_w,
        inertia=float(prev_inertia),
    )


# ---------------------------------------------------------------------------
# Symmetric k-NN sparsification
# ---------------------------------------------------------------------------

def knn_sparsify(
    X: torch.Tensor,
    k: int,
    *,
    chunk: int = 4096,
    mode: str = "gaussian",
    sigma: Optional[float] = None,
    symmetrize: str = "max",
    device: str | torch.device = "cpu",
) -> torch.sparse.Tensor:

    device_t = resolve_device(device)
    X = X.to(device_t).to(torch.float32)
    n = X.shape[0]
    k_eff = min(k, n - 1)
    rows_buf = torch.empty(n * k_eff, dtype=torch.long, device=device_t)
    cols_buf = torch.empty(n * k_eff, dtype=torch.long, device=device_t)
    vals_buf = torch.empty(n * k_eff, dtype=torch.float32, device=device_t)
    write = 0
    median_pool: list[torch.Tensor] = []
    N_SIGMA_CHUNKS = 8
    n_chunks_total = math.ceil(n / chunk)
    sigma_sample_stride = max(1, n_chunks_total // N_SIGMA_CHUNKS)
    if sigma is None and n_chunks_total > N_SIGMA_CHUNKS:
        print(
            f"[shield.coarsening] N_SIGMA_CHUNKS = {N_SIGMA_CHUNKS} has been "
            f"reached in coarsening.py - global Gaussian-kernel sigma is "
            f"estimated from {N_SIGMA_CHUNKS} of {n_chunks_total} chunks "
            "(set `sigma` explicitly to skip this heuristic)."
        )

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d2 = pairwise_sqdist_chunked(X[start:end], X, chunk=chunk)
        idx_local = torch.arange(start, end, device=device_t)
        d2[torch.arange(end - start, device=device_t), idx_local] = float("inf")
        topd, topi = torch.topk(d2, k_eff, dim=1, largest=False, sorted=False)
        chunk_idx = start // chunk
        if sigma is None and chunk_idx % sigma_sample_stride == 0:
            median_pool.append(topd.reshape(-1))
        nrows = end - start
        rows_local = torch.arange(start, end, device=device_t).unsqueeze(1).expand(-1, k_eff).reshape(-1)
        cols_local = topi.reshape(-1)
        dist_local = topd.reshape(-1)
        rows_buf[write:write + nrows * k_eff] = rows_local
        cols_buf[write:write + nrows * k_eff] = cols_local
        vals_buf[write:write + nrows * k_eff] = dist_local
        write += nrows * k_eff

    rows_buf = rows_buf[:write]
    cols_buf = cols_buf[:write]
    dist_vals = vals_buf[:write]

    if mode == "binary":
        weights = torch.ones_like(dist_vals)
    elif mode == "inverse":
        weights = 1.0 / (1.0 + dist_vals)
    else:  
        if sigma is None:
            med = torch.cat(median_pool).median().item()
            sigma_eff = math.sqrt(max(med, 1e-12))
        else:
            sigma_eff = float(sigma)
        weights = torch.exp(-dist_vals / (sigma_eff ** 2 + 1e-12))

    indices = torch.stack([rows_buf, cols_buf], dim=0)
    A = torch.sparse_coo_tensor(
        indices, weights, size=(n, n), device=device_t
    ).coalesce()
    A_t = torch.sparse_coo_tensor(
        torch.stack([cols_buf, rows_buf], dim=0), weights, size=(n, n),
        device=device_t,
    ).coalesce()
    if symmetrize == "max":
        merged_idx = torch.cat([A.indices(), A_t.indices()], dim=1)
        merged_val = torch.cat([A.values(), A_t.values()], dim=0)
        keys = merged_idx[0].to(torch.long) * n + merged_idx[1].to(torch.long)
        uniq_keys, inverse = torch.unique(keys, return_inverse=True)
        out_vals = torch.full(
            (uniq_keys.numel(),), float("-inf"),
            dtype=merged_val.dtype, device=device_t,
        )
        out_vals.scatter_reduce_(0, inverse, merged_val, reduce="amax", include_self=True)
        out_rows = (uniq_keys // n)
        out_cols = (uniq_keys % n)
        A_sym = torch.sparse_coo_tensor(
            torch.stack([out_rows, out_cols], dim=0), out_vals,
            size=(n, n), device=device_t,
        ).coalesce()
    else:
        A_sym = (A + A_t).coalesce()
        A_sym = torch.sparse_coo_tensor(
            A_sym.indices(),
            A_sym.values() * 0.5,
            size=(n, n),
            device=device_t,
        ).coalesce()

    return A_sym


# ---------------------------------------------------------------------------
# Aggregation of attributes onto super-nodes
# ---------------------------------------------------------------------------

def aggregate_super_nodes(
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    features: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    classes: Optional[torch.Tensor] = None,
    n_clusters: Optional[int] = None,
) -> dict:

    device_t = labels.device
    if n_clusters is None:
        n_clusters = int(labels.max().item()) + 1

    out: dict = {}
    super_w = torch.zeros(n_clusters, device=device_t, dtype=torch.float32)
    super_w.scatter_add_(0, labels, weights.to(torch.float32))
    out["super_weights"] = super_w
    denom = super_w.clamp_min(1e-12).unsqueeze(1)

    if features is not None:
        feats = features.to(torch.float32) * weights.to(torch.float32).unsqueeze(1)
        agg = torch.zeros(n_clusters, feats.shape[1], device=device_t, dtype=torch.float32)
        agg.index_add_(0, labels, feats)
        out["features"] = agg / denom

    if targets is not None:
        targs = targets.to(torch.float32) * weights.to(torch.float32).unsqueeze(1)
        agg = torch.zeros(n_clusters, targs.shape[1], device=device_t, dtype=torch.float32)
        agg.index_add_(0, labels, targs)
        out["targets"] = agg / denom

    if classes is not None:
        C = int(classes.max().item()) + 1
        hist = torch.zeros((n_clusters, C), device=device_t, dtype=torch.float32)
        flat = labels * C + classes.to(torch.long)
        hist.view(-1).index_add_(
            0, flat, weights.to(torch.float32)
        )
        out["classes"] = hist.argmax(dim=1)
        out["class_hist"] = hist

    return out


def lift_assignment_to_atoms(
    super_z: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return super_z[labels]


def galerkin_project(
    M: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:

    device = M.device
    labels = labels.to(device).to(torch.long)
    if M.is_sparse:
        A = M.coalesce()
        idx = A.indices()
        rows_s = labels[idx[0]]
        cols_s = labels[idx[1]]
        new_idx = torch.stack([rows_s, cols_s], dim=0)
        return torch.sparse_coo_tensor(
            new_idx, A.values(), size=(n_clusters, n_clusters), device=device,
        ).coalesce()
    Md = M.to(torch.float32)
    n = Md.shape[0]
    M1 = torch.zeros((n_clusters, n), device=device, dtype=torch.float32)
    M1.index_add_(0, labels, Md)
    M1_T = M1.t().contiguous()                              # (n, n_clusters)
    out_T = torch.zeros((n_clusters, n_clusters), device=device, dtype=torch.float32)
    out_T.index_add_(0, labels, M1_T)
    return out_T.t().contiguous()                          # (n_clusters, n_clusters)
