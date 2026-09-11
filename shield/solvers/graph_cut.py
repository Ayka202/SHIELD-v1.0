from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except Exception:                                         
    _HAS_SCIPY = False

# Optional balanced graph partitioners. Either back-end
# is enough; pure-torch spectral + balanced K-means is the always-available
# fallback so SHIELD still runs without these third-party libraries.
try:
    import pymetis as _pymetis                              # type: ignore
    _HAS_METIS = True
except Exception:                                         
    try:
        import metis as _pymetis                            # type: ignore
        _HAS_METIS = True
    except Exception:                                     
        _pymetis = None                                     # type: ignore
        _HAS_METIS = False

try:
    import kahip as _kahip                                  # type: ignore
    _HAS_KAHIP = True
except Exception:                                         
    _kahip = None                                           # type: ignore
    _HAS_KAHIP = False


from shield.coarsening import (
    pairwise_sqdist_chunked,
    argmin_chunked,
    resolve_device,
)


# ---------------------------------------------------------------------------
# Affinity / Laplacian construction
# ---------------------------------------------------------------------------

def _sparse_row_sum(A: torch.Tensor) -> torch.Tensor:
    A = A.coalesce()
    deg = torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    deg.scatter_add_(0, A.indices()[0], A.values())
    return deg


def laplacian_from_affinity(
    A: torch.Tensor,
    normalized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.is_sparse:
        A = A.coalesce()
        deg = _sparse_row_sum(A)
        n = A.shape[0]
        idx_diag = torch.arange(n, device=A.device)

        D = torch.sparse_coo_tensor(
            torch.stack([idx_diag, idx_diag]), deg, (n, n), device=A.device,
        ).coalesce()
        if not normalized:
            L = (D + (-1.0) * A).coalesce()
            return L, deg

        inv_sqrt = deg.clamp_min(1e-12).rsqrt()
        vals = A.values() * inv_sqrt[A.indices()[0]] * inv_sqrt[A.indices()[1]]
        A_norm = torch.sparse_coo_tensor(A.indices(), vals, A.shape, device=A.device).coalesce()
        I_idx = torch.stack([idx_diag, idx_diag])
        I = torch.sparse_coo_tensor(I_idx, torch.ones(n, device=A.device, dtype=A.dtype),
                                    (n, n), device=A.device).coalesce()
        L = (I + (-1.0) * A_norm).coalesce()
        return L, deg

    # dense path
    deg = A.sum(dim=1)
    if not normalized:
        return torch.diag(deg) - A, deg
    inv_sqrt = deg.clamp_min(1e-12).rsqrt()
    A_norm = (A * inv_sqrt.unsqueeze(1)) * inv_sqrt.unsqueeze(0)
    L = torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - A_norm
    return L, deg


# ---------------------------------------------------------------------------
# Cut-energy evaluation
# ---------------------------------------------------------------------------

def cut_energy(
    z: torch.Tensor,
    A: torch.Tensor,
) -> float:
    z = z.to(torch.float32)
    if A.is_sparse:
        A = A.coalesce()
        deg = _sparse_row_sum(A)
        deg_part = (deg.unsqueeze(1) * (z * z)).sum()
        Az = torch.sparse.mm(A, z)
        a_part = (z * Az).sum()
        return float((deg_part - a_part).item())
    L = torch.diag(A.sum(dim=1)) - A
    return float((z * (L @ z)).sum().item())


def m_gradient_z(
    z: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    if A.is_sparse:
        A = A.coalesce()
        deg = _sparse_row_sum(A)
        Az = torch.sparse.mm(A, z)
        Lz = deg.unsqueeze(1) * z - Az
        return 2.0 * Lz
    L = torch.diag(A.sum(dim=1)) - A
    return 2.0 * (L @ z)


# ---------------------------------------------------------------------------
# Spectral embedding + balanced rounding
# ---------------------------------------------------------------------------

def spectral_embedding(
    A: torch.Tensor,
    K: int,
    *,
    normalized: bool = True,
    device: str | torch.device = "cpu",
    use_scipy: Optional[bool] = None,
) -> torch.Tensor:
    device_t = resolve_device(device)
    L, _ = laplacian_from_affinity(A.to(device_t), normalized=normalized)

    if use_scipy is None:
        n = L.shape[0]
        use_scipy = (L.is_sparse and n > 4096 and _HAS_SCIPY)

    if use_scipy:
        if not _HAS_SCIPY:
            raise RuntimeError("scipy not available for sparse eigendecomposition")
        L_cpu = L.cpu().coalesce()
        rows = L_cpu.indices()[0].numpy()
        cols = L_cpu.indices()[1].numpy()
        vals = L_cpu.values().numpy()
        n = L_cpu.shape[0]
        L_sp = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        vals_eig, vecs = spla.eigsh(L_sp, k=min(K + 1, n - 1), which="SM")
        order = vals_eig.argsort()
        vecs = vecs[:, order][:, 1:K + 1]
        emb = torch.from_numpy(vecs).to(torch.float32).to(device_t)
        return emb

    # dense path
    if L.is_sparse:
        L_dense = L.to_dense()
    else:
        L_dense = L
    vals_eig, vecs = torch.linalg.eigh(L_dense)
    emb = vecs[:, 1:K + 1]
    return emb.to(torch.float32)


def balanced_kmeans_assign(
    emb: torch.Tensor,
    K: int,
    r: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    max_iter: int = 100,
    seed: int = 0,
    chunk: int = 4096,
) -> torch.Tensor:
    device = emb.device
    n = emb.shape[0]
    if weights is None:
        w = torch.ones(n, device=device, dtype=torch.float32)
    else:
        w = weights.to(device).to(torch.float32)
    W = w.sum()
    target = r.to(device).to(torch.float32) * W

    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:K].to(device)
    centroids = emb[init_idx].clone()

    labels = torch.zeros(n, dtype=torch.long, device=device)
    occupancy = torch.zeros(K, device=device, dtype=torch.float32)
    for it in range(max_iter):
        c_sq = (centroids * centroids).sum(dim=1)
        if it > 0:
            penalty = (occupancy - target) / (target + 1e-9)
        else:
            penalty = torch.zeros(K, device=device, dtype=torch.float32)

        new_labels = torch.empty(n, dtype=torch.long, device=device)
        new_occupancy = torch.zeros(K, device=device, dtype=torch.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xb = emb[start:end]
            x_sq = (xb * xb).sum(dim=1, keepdim=True)
            cross = xb @ centroids.t()
            cost = (x_sq + c_sq.unsqueeze(0) - 2.0 * cross).clamp_min_(0.0)
            cost = cost + penalty.unsqueeze(0)
            lab = cost.argmin(dim=1)
            new_labels[start:end] = lab
            new_occupancy.scatter_add_(0, lab, w[start:end])

        if torch.equal(new_labels, labels):
            labels = new_labels
            occupancy = new_occupancy
            break
        labels = new_labels
        occupancy = new_occupancy

        new_centroids = torch.zeros_like(centroids)
        ws = w.unsqueeze(1) * emb
        new_centroids.index_add_(0, labels, ws)
        denom = occupancy.clamp_min(1e-12).unsqueeze(1)
        centroids = new_centroids / denom

    return labels


def spectral_balanced_init(
    A: torch.Tensor,
    K: int,
    r: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    device: str | torch.device = "cpu",
    normalized: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    device_t = resolve_device(device)
    emb = spectral_embedding(A, K=K, normalized=normalized, device=device_t)
    labels = balanced_kmeans_assign(emb, K=K, r=r, weights=weights, seed=seed)
    n = labels.shape[0]
    z = torch.zeros((n, K), device=device_t, dtype=torch.float32)
    z.scatter_(1, labels.unsqueeze(1), 1.0)
    return z


# ---------------------------------------------------------------------------
# METIS / KaHIP back-ends
# ---------------------------------------------------------------------------

def _affinity_to_csr(A: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if A.is_sparse:
        A = A.coalesce()
        rows = A.indices()[0].detach().cpu().numpy()
        cols = A.indices()[1].detach().cpu().numpy()
        vals = A.values().detach().cpu().to(torch.float64).numpy()
        n = int(A.shape[0])
    else:
        A_dense = A.detach().cpu().to(torch.float64).numpy()
        n = int(A_dense.shape[0])
        idx = np.nonzero(A_dense)
        rows, cols = idx[0], idx[1]
        vals = A_dense[idx]
    keep = rows != cols
    rows = rows[keep].astype(np.int64)
    cols = cols[keep].astype(np.int64)
    vals = vals[keep].astype(np.float64)
    if vals.size:
        scale = 10_000.0 / max(float(vals.max()), 1e-30)
        wq = np.clip(np.round(vals * scale).astype(np.int64), 1, None)
    else:
        wq = np.zeros(0, dtype=np.int64)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    wq = wq[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, cols, wq, n


def _proportions_to_tpwgts(r: torch.Tensor, K: int) -> list:
    r_np = r.detach().cpu().to(torch.float64).numpy().reshape(-1)
    if r_np.shape[0] != K:
        raise ValueError(f"r must have length K={K}, got {r_np.shape[0]}.")
    s = r_np.sum()
    if s <= 0:
        raise ValueError("Sum of target proportions r must be positive.")
    return (r_np / s).tolist()


def metis_balanced_init(
    A: torch.Tensor,
    K: int,
    r: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    device: str | torch.device = "cpu",
    seed: int = 0,
    imbalance: float = 1.05,
) -> torch.Tensor:
    if not _HAS_METIS:
        raise RuntimeError(
            "METIS back-end requested but `pymetis` / `metis` is not installed. "
            "Install via `pip install pymetis` (Linux/macOS) or fall back to "
            "spectral_balanced_init."
        )
    device_t = resolve_device(device)
    indptr, cols, wq, n = _affinity_to_csr(A)
    if weights is None:
        vw = np.ones(n, dtype=np.int64)
    else:
        w_np = weights.detach().cpu().to(torch.float64).numpy().reshape(-1)
        scale = 1000.0 / max(float(w_np.max()), 1e-30)
        vw = np.clip(np.round(w_np * scale).astype(np.int64), 1, None)
    tpwgts = _proportions_to_tpwgts(r, K)

    try:
        _, membership = _pymetis.part_graph( 
            K,
            xadj=indptr.astype(np.int32).tolist(),
            adjncy=cols.astype(np.int32).tolist(),
            eweights=wq.astype(np.int32).tolist(),
            vweights=vw.astype(np.int32).tolist(),
            tpwgts=tpwgts,
            options={"seed": int(seed), "ufactor": int(max(1, round((imbalance - 1.0) * 1000)))},
        )
    except (TypeError, AttributeError):
        _, membership = _pymetis.part_graph(  
            K,
            xadj=indptr.astype(np.int32).tolist(),
            adjncy=cols.astype(np.int32).tolist(),
            eweights=wq.astype(np.int32).tolist(),
            vweights=vw.astype(np.int32).tolist(),
        )

    labels = torch.as_tensor(membership, dtype=torch.long, device=device_t)
    z = torch.zeros((n, K), device=device_t, dtype=torch.float32)
    z.scatter_(1, labels.unsqueeze(1), 1.0)
    return z


def kahip_balanced_init(
    A: torch.Tensor,
    K: int,
    r: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    device: str | torch.device = "cpu",
    seed: int = 0,
    imbalance: float = 0.05,
    mode: str = "FAST",
) -> torch.Tensor:
    if not _HAS_KAHIP:
        raise RuntimeError(
            "KaHIP back-end requested but `kahip` is not installed. "
            "Install via `pip install kahip` or fall back to spectral_balanced_init."
        )
    device_t = resolve_device(device)
    indptr, cols, wq, n = _affinity_to_csr(A)
    if weights is None:
        vw = np.ones(n, dtype=np.int64)
    else:
        w_np = weights.detach().cpu().to(torch.float64).numpy().reshape(-1)
        scale = 1000.0 / max(float(w_np.max()), 1e-30)
        vw = np.clip(np.round(w_np * scale).astype(np.int64), 1, None)


    mode_int = {"FAST": 0, "ECO": 1, "STRONG": 2}.get(str(mode).upper(), 0)
    try:
        _, membership = _kahip.kaffpa(
            vw.astype(np.int32).tolist(),
            indptr.astype(np.int32).tolist(),
            cols.astype(np.int32).tolist(),
            wq.astype(np.int32).tolist(),
            K,
            float(imbalance),
            False,        # suppress_output
            int(seed),
            mode_int,
        )
    except Exception as exc:
        raise RuntimeError(f"KaHIP partitioning failed: {exc}") from exc

    labels = torch.as_tensor(np.asarray(membership), dtype=torch.long, device=device_t)

    if weights is None:
        w_repair = torch.ones(n, device=device_t, dtype=torch.float32)
    else:
        w_repair = weights.to(device_t).to(torch.float32)
    W_repair = w_repair.sum()
    target_repair = r.to(device_t).to(torch.float32) * W_repair
    r_vals = r.detach().cpu().to(torch.float64).numpy()
    if float(r_vals.max() - r_vals.min()) > 1e-6:
        fold_mass = torch.zeros(K, device=device_t, dtype=torch.float32)
        fold_mass.scatter_add_(0, labels, w_repair)
        n_repair_iters = max(10 * K, 100)
        moved_total = 0
        for _ in range(n_repair_iters):
            deficit = target_repair - fold_mass
            if deficit.abs().max() <= 1e-9 * float(W_repair.item()):
                break
            k_over = int(deficit.argmin().item())
            k_under = int(deficit.argmax().item())
            if deficit[k_under] <= 0 or deficit[k_over] >= 0:
                break
            in_over = (labels == k_over).nonzero(as_tuple=False).flatten()
            if in_over.numel() == 0:
                break
            order = torch.argsort(w_repair[in_over])
            budget = float((-deficit[k_over]).item())
            moved_any = False
            for idx in order.tolist():
                atom = int(in_over[idx].item())
                w_atom = float(w_repair[atom].item())
                if w_atom <= budget + 1e-9:
                    labels[atom] = k_under
                    fold_mass[k_over] -= w_atom
                    fold_mass[k_under] += w_atom
                    moved_any = True
                    moved_total += 1
                    break
            if not moved_any:
                break
        print(
            f"[shield.graph_cut] KaHIP balance-repair: {moved_total} atom(s) moved "
            f"over up to {n_repair_iters} iterations for non-uniform r "
            f"(r_range={float(r_vals.max() - r_vals.min()):.4f}). "
            f"graph_cut.py — KaHIP ignores tpwgts; greedy repair pushes fold "
            f"masses toward the target proportions before the LP step."
        )

    z = torch.zeros((n, K), device=device_t, dtype=torch.float32)
    z.scatter_(1, labels.unsqueeze(1), 1.0)
    return z


def balanced_partition_init(
    A: torch.Tensor,
    K: int,
    r: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    backend: str = "auto",
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Tuple[torch.Tensor, str]:
    backend = str(backend).lower()
    if backend not in {"auto", "metis", "kahip", "spectral"}:
        raise ValueError(
            f"Unknown initializer backend '{backend}'. "
            "Choose 'auto', 'metis', 'kahip', or 'spectral'."
        )

    order: list = []
    if backend == "auto":
        order = ["metis", "kahip", "spectral"]
    else:
        order = [backend]

    last_exc: Optional[Exception] = None
    for b in order:
        try:
            if b == "metis":
                z = metis_balanced_init(A, K=K, r=r, weights=weights,
                                        device=device, seed=seed)
                print(f"[shield.graph_cut] Stage-3 init: using METIS partitioner.")
                return z, "metis"
            if b == "kahip":
                z = kahip_balanced_init(A, K=K, r=r, weights=weights,
                                        device=device, seed=seed)
                print(f"[shield.graph_cut] Stage-3 init: using KaHIP partitioner.")
                return z, "kahip"
            if b == "spectral":
                z = spectral_balanced_init(A, K=K, r=r, weights=weights,
                                           device=device, seed=seed)
                print(f"[shield.graph_cut] Stage-3 init: using spectral partitioner (METIS/KaHIP unavailable).")
                return z, "spectral"
        except RuntimeError as exc:
            last_exc = exc
            if backend == "auto":
                _fb_msg = (
                    f"[shield.graph_cut] Stage-3 init back-end '{b}' unavailable "
                    f"({exc}); falling back to the next option."
                )
                print(_fb_msg)
                warnings.warn(
                    f"Stage-3 init back-end '{b}' unavailable ({exc}); "
                    "falling back to the next option.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            raise

    raise RuntimeError(
        f"No Stage-3 init back-end succeeded; last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Fallback: random feasible assignment with balance repair
# ---------------------------------------------------------------------------

def random_balanced_init(
    n: int,
    K: int,
    r: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    device_t = resolve_device(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if weights is None:
        w = torch.ones(n, device=device_t, dtype=torch.float32)
    else:
        w = weights.to(device_t).to(torch.float32)
    W = w.sum()
    target = r.to(device_t).to(torch.float32) * W
    perm = torch.randperm(n, generator=g, device="cpu").to(device_t)
    cum_target = torch.cumsum(target, dim=0)
    cum_weight = torch.cumsum(w[perm], dim=0)
    fold_of_position = torch.searchsorted(cum_target, cum_weight - 1e-12).clamp_max(K - 1)
    labels = torch.empty(n, dtype=torch.long, device=device_t)
    labels[perm] = fold_of_position

    # greedy rebalance to tighten the heavy-weight tail 
    fold_mass = torch.zeros(K, device=device_t, dtype=torch.float32)
    fold_mass.scatter_add_(0, labels, w)
    for _ in range(max(10 * K, 50)):
        deficit = target - fold_mass
        if deficit.abs().max() <= 1e-9 * W:
            break
        k_over = int(deficit.argmin().item())     # most over-full
        k_under = int(deficit.argmax().item())    # most under-full
        if deficit[k_under] <= 0 or deficit[k_over] >= 0:
            break
        in_over = (labels == k_over).nonzero(as_tuple=False).flatten()
        if in_over.numel() == 0:
            break
        # smallest-weight atom that, once moved, doesn't push k_over below target
        order = torch.argsort(w[in_over])        
        moved_any = False
        budget = float((-deficit[k_over]).item())
        for idx in order.tolist():
            atom = int(in_over[idx].item())
            w_atom = float(w[atom].item())
            if w_atom <= budget + 1e-9 or budget < 1e-12:
                labels[atom] = k_under
                fold_mass[k_over] -= w_atom
                fold_mass[k_under] += w_atom
                moved_any = True
                break
        if not moved_any:
            break

    z = torch.zeros((n, K), device=device_t, dtype=torch.float32)
    z.scatter_(1, labels.unsqueeze(1), 1.0)
    return z
