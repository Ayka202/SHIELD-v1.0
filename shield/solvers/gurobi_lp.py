from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception:                                                 
    _HAS_GUROBI = False


def _require_gurobi() -> None:
    if not _HAS_GUROBI:
        raise RuntimeError(
            "Gurobi (gurobipy) is required for the LP/MILP backend. "
            "Install gurobipy and a Gurobi license before calling this solver."
        )


# ---------------------------------------------------------------------------
# Hierarchy data structures
# ---------------------------------------------------------------------------

@dataclass
class HierarchyLevel:
    name: str
    descendants: list                
    similarity_rows: np.ndarray      
    similarity_cols: np.ndarray      
    similarity_vals: np.ndarray      
    lambda_level: float = 1.0


@dataclass
class HierarchySolution:
    y: dict                          # {(level_name, g, k): float}
    w_aux: dict                      # {(level_name, g, g', k): float}
    objective: float


# ---------------------------------------------------------------------------
# Hierarchy block LP 
# ---------------------------------------------------------------------------

def solve_hierarchy_lp(
    z: np.ndarray,
    levels: Sequence[HierarchyLevel],
    K: int,
    *,
    time_limit: Optional[float] = None,
    output_flag: int = 0,
    relax_y: bool = True,
) -> HierarchySolution:
    _require_gurobi()
    model = gp.Model("shield_hierarchy_lp")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    y_vars_per_level: dict = {}
    w_vars_per_level: dict = {}
    # objective accumulated in the matrix API (no mixing gp.LinExpr with MLinExpr)
    obj_const = 0.0       
    obj_parts: list = []  

    for level in levels:
        n_nodes = len(level.descendants)
        if n_nodes == 0:
            continue
        vtype = GRB.CONTINUOUS if relax_y else GRB.BINARY

        # vectorized y_{g,k}
        y_var = model.addMVar(
            shape=(n_nodes, K), lb=0.0, ub=1.0, vtype=vtype,
            name=f"y_{level.name}",
        )
        y_vars_per_level[level.name] = y_var

        min_z = np.zeros((n_nodes, K), dtype=np.float64)
        sum_z = np.zeros((n_nodes, K), dtype=np.float64)
        size_dg = np.zeros(n_nodes, dtype=np.float64)
        for g, desc in enumerate(level.descendants):
            if len(desc) == 0:
                min_z[g] = 0.0
                sum_z[g] = 0.0
                size_dg[g] = 0.0
                continue
            desc_arr = np.asarray(desc, dtype=np.int64)
            z_slice = z[desc_arr]                                    # (|D(g)|, K)
            min_z[g] = z_slice.min(axis=0)
            sum_z[g] = z_slice.sum(axis=0)
            size_dg[g] = float(desc_arr.size)

        # Vectorized containment constraints (one per (g, k))
        model.addConstr(y_var <= min_z, name=f"cont_le_{level.name}")
        rhs_lo = sum_z - size_dg.reshape(-1, 1) + 1.0
        model.addConstr(y_var >= rhs_lo, name=f"cont_ge_{level.name}")

        # ----- w_{e,k} aux variables + linearization --------------------
        rows = np.asarray(level.similarity_rows, dtype=np.int64)
        cols = np.asarray(level.similarity_cols, dtype=np.int64)
        vals = np.asarray(level.similarity_vals, dtype=np.float64)
        keep = vals > 0.0
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        n_edges = rows.shape[0]
        if n_edges == 0:
            w_vars_per_level[level.name] = None
            continue
        w_var = model.addMVar(
            shape=(n_edges, K), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
            name=f"w_{level.name}",
        )
        w_vars_per_level[level.name] = w_var

        for k in range(K):
            y_rows = y_var[rows.tolist(), k]    
            y_cols = y_var[cols.tolist(), k]
            wk = w_var[:, k]
            model.addConstr(wk <= y_rows, name=f"w_le_y_a_{level.name}_{k}")
            model.addConstr(wk <= y_cols, name=f"w_le_y_b_{level.name}_{k}")
            model.addConstr(wk >= y_rows + y_cols - 1.0,
                            name=f"w_ge_{level.name}_{k}")

        # objective contribution
        obj_const += float(level.lambda_level * vals.sum())
        coefs = -float(level.lambda_level) * vals      # (n_edges,)
        obj_parts.append((coefs @ w_var).sum())

    if obj_parts:
        obj_mlin = obj_parts[0]
        for part in obj_parts[1:]:
            obj_mlin = obj_mlin + part
        model.setObjective(obj_mlin, GRB.MINIMIZE)
    else:
        model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi H-LP failed with status {model.Status}")

    y_out: dict = {}
    w_out: dict = {}
    for level in levels:
        y_var = y_vars_per_level.get(level.name)
        if y_var is None:
            continue
        yv = np.asarray(y_var.X)
        for g in range(yv.shape[0]):
            for k in range(K):
                y_out[(level.name, g, k)] = float(yv[g, k])
        w_var = w_vars_per_level.get(level.name)
        if w_var is None:
            continue
        wv = np.asarray(w_var.X)
        rows = np.asarray(level.similarity_rows, dtype=np.int64)
        cols = np.asarray(level.similarity_cols, dtype=np.int64)
        vals = np.asarray(level.similarity_vals, dtype=np.float64)
        keep = vals > 0.0
        rows, cols = rows[keep], cols[keep]
        for e in range(wv.shape[0]):
            for k in range(K):
                w_out[(level.name, int(rows[e]), int(cols[e]), k)] = float(wv[e, k])
    return HierarchySolution(y=y_out, w_aux=w_out,
                             objective=float(model.ObjVal) + obj_const)


# ---------------------------------------------------------------------------
# Assignment block LP / MILP
# ---------------------------------------------------------------------------

@dataclass
class HardConstraints:
    cold_left_groups: Optional[dict] = None      # {entity_id: [obs idx, ...]}
    cold_right_groups: Optional[dict] = None
    class_groups: Optional[dict] = None          # {class_id: [obs idx, ...]}
    class_weights: Optional[np.ndarray] = None
    class_delta: float = 0.05
    balance_eps: float = 0.05


@dataclass
class AssignmentSolution:
    z: np.ndarray
    objective: float
    status: int


def solve_assignment_lp(
    *,
    n: int,
    K: int,
    weights: np.ndarray,
    r: np.ndarray,
    z_linear_coeff: np.ndarray,
    eta_balance: float,
    hard: HardConstraints,
    hierarchy_levels: Optional[Sequence[HierarchyLevel]] = None,
    y_fixed: Optional[dict] = None,
    pin_in_frac: float = 0.90,
    pin_out_strict: bool = False,
    integer: bool = False,
    time_limit: Optional[float] = None,
    output_flag: int = 0,
    initial_z: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> AssignmentSolution:
    _require_gurobi()
    model = gp.Model("shield_assignment_lp")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if eta_balance > 0:
        model.Params.NumericFocus = 2
    if seed is not None and integer:
        model.Params.Seed = int(seed) % 2_000_000_000

    vtype = GRB.BINARY if integer else GRB.CONTINUOUS
    z = model.addMVar(
        shape=(n, K), lb=0.0, ub=1.0, vtype=vtype, name="z",
    )
    if initial_z is not None and integer:
        z.Start = initial_z.astype(np.float64)

    # ---- one-hot rows ----------------------------------------------------
    model.addConstr(z.sum(axis=1) == np.ones(n, dtype=np.float64),
                    name="row_onehot")

    # ---- hard balance equality ------------------------------------------
    W = float(weights.sum())
    rW = r.astype(np.float64) * W
    eps_arr = np.ones(K, dtype=np.float64) * float(hard.balance_eps * W)
    col_sums = weights.astype(np.float64) @ z               
    model.addConstr(col_sums <= rW + eps_arr, name="bal_hi")
    model.addConstr(col_sums >= rW - eps_arr, name="bal_lo")

    # ---- cold-left / cold-right -----------------------------------------
    if hard.cold_left_groups:
        _add_cold_constraints(model, z, K, hard.cold_left_groups, prefix="cold_left",
                              integer=integer)
    if hard.cold_right_groups:
        _add_cold_constraints(model, z, K, hard.cold_right_groups, prefix="cold_right",
                              integer=integer)

    # ---- classification stratification ----------------------------------
    if hard.class_groups is not None:
        cw = (hard.class_weights if hard.class_weights is not None
              else np.ones(n, dtype=np.float64))
        for c, obs_list in hard.class_groups.items():
            if len(obs_list) == 0:
                continue
            obs_arr = np.asarray(obs_list, dtype=np.int64)
            W_c = float(cw[obs_arr].sum())
            wc = cw[obs_arr].astype(np.float64)
            col_sums_c = wc @ z[obs_arr.tolist(), :]
            model.addConstr(col_sums_c <= r.astype(np.float64) * W_c
                            + hard.class_delta * W_c,
                            name=f"strat_hi_{c}")
            model.addConstr(col_sums_c >= r.astype(np.float64) * W_c
                            - hard.class_delta * W_c,
                            name=f"strat_lo_{c}")

    # hierarchy containment using y_fixed 
    if hierarchy_levels is not None and y_fixed is not None:
        for level in hierarchy_levels:
            for g, desc in enumerate(level.descendants):
                if len(desc) == 0:
                    continue
                obs_arr = np.asarray(desc, dtype=np.int64)
                obs_list = obs_arr.tolist()
                desc_size = float(obs_arr.shape[0])
                group_committed = any(
                    float(y_fixed.get((level.name, g, kk), 0.0)) >= 0.5
                    for kk in range(K)
                )
                for k in range(K):
                    y_val = float(y_fixed.get((level.name, g, k), 0.0))
                    sub = z[obs_list, k]                    # MVar of length |D(g)|
                    sum_expr = sub.sum()
                    if y_val >= 0.5:
                        model.addConstr(sum_expr >= pin_in_frac * desc_size,
                                        name=f"pin_in_{level.name}_{g}_{k}")
                    elif desc_size > 1:
                        if pin_out_strict and group_committed:
                            model.addConstr(sum_expr <= 0.10 * desc_size,
                                            name=f"pin_out_{level.name}_{g}_{k}")
                        else:
                            model.addConstr(sum_expr <= desc_size - 1.0,
                                            name=f"pin_out_{level.name}_{g}_{k}")

    # objective 
    coeff = z_linear_coeff.astype(np.float64)              # (n, K)
    obj_linear = (coeff * z).sum()                         # scalar MLinExpr

    if eta_balance > 0:
        s_bal = model.addMVar(shape=(K,), lb=-GRB.INFINITY, ub=GRB.INFINITY,
                              vtype=GRB.CONTINUOUS, name="s_bal")
        model.addConstr(s_bal == col_sums - rW, name="s_bal_def")
        model.setObjective(obj_linear + eta_balance * (s_bal @ s_bal),
                           GRB.MINIMIZE)
    else:
        model.setObjective(obj_linear, GRB.MINIMIZE)

    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi assignment LP failed with status {model.Status}")

    z_out = np.asarray(z.X, dtype=np.float64)
    return AssignmentSolution(z=z_out, objective=float(model.ObjVal),
                              status=int(model.Status))


def solve_full_milp(
    *,
    n: int,
    K: int,
    weights: np.ndarray,
    r: np.ndarray,
    z_linear_coeff: np.ndarray,
    eta_balance: float,
    hard: HardConstraints,
    hierarchy_levels: Optional[Sequence[HierarchyLevel]] = None,
    y_fixed: Optional[dict] = None,
    pin_in_frac: float = 0.90,
    pin_out_strict: bool = False,
    time_limit: Optional[float] = 300.0,
    output_flag: int = 0,
    initial_z: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> AssignmentSolution:
    return solve_assignment_lp(
        n=n, K=K,
        weights=weights, r=r,
        z_linear_coeff=z_linear_coeff,
        eta_balance=eta_balance,
        hard=hard,
        hierarchy_levels=hierarchy_levels,
        y_fixed=y_fixed,
        pin_in_frac=pin_in_frac,
        pin_out_strict=pin_out_strict,
        integer=True,
        time_limit=time_limit,
        output_flag=output_flag,
        initial_z=initial_z,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_cold_constraints(
    model: "gp.Model",
    z: "gp.MVar",
    K: int,
    cold_groups: dict,
    prefix: str,
    *,
    integer: bool = False,
) -> None:
    etype = GRB.BINARY if integer else GRB.CONTINUOUS
    print(
        f"[shield.gurobi_lp] _add_cold_constraints: {len(cold_groups)} "
        f"'{prefix}' group(s) declared as {'BINARY' if integer else 'CONTINUOUS'} "
        f"(integer={integer}). gurobi_lp.py — LP mode uses CONTINUOUS "
        f"indicators; post-rounding entity repair enforces the hard constraint. "
        f"MILP mode uses BINARY for an exact solver guarantee."
    )
    for entity_id, obs_list in cold_groups.items():
        if len(obs_list) == 0:
            continue
        obs_arr = np.asarray(obs_list, dtype=np.int64)
        obs_idx = obs_arr.tolist()
        e = model.addMVar(
            shape=(K,), lb=0.0, ub=1.0, vtype=etype,
            name=f"{prefix}_e_{entity_id}",
        )
        model.addConstr(e.sum() == 1.0, name=f"{prefix}_onehot_{entity_id}")
        for k in range(K):
            sub = z[obs_idx, k]
            n_obs = len(obs_idx)
            model.addConstr(sub - e[k] * np.ones(n_obs)
                            == np.zeros(n_obs),
                            name=f"{prefix}_pin_{entity_id}_{k}")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().to(torch.float64).numpy()


def to_tensor(arr: np.ndarray, device: str | torch.device,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype).to(device)
