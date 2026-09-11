<h1 align="center">SHIELD v1.0</h1>
<h3 align="center">Structure-aware Hybrid Integrated Evaluation by Leakage-aware Data splitting</h3>

<p align="center">
  <img src="abstract.png" alt="SHIELD graphical abstract" width="100%">
</p>

## Abstract

The reliability of machine-learning models for scientific discovery depends on their ability to generalize to genuinely unseen data rather than interpolate between structurally similar examples. Conventional train-test splits frequently place closely related samples on both sides of the partition, leading to information leakage and overly optimistic estimates of predictive performance. Existing splitting methods typically address only a single source of leakage, limiting their ability to construct realistic out-of-distribution benchmarks across diverse scientific domains. SHIELD is a general framework that jointly optimizes hierarchical containment, feature-space connectivity, and distributional constraints to generate leakage-aware dataset partitions while remaining applicable across heterogeneous data modalities. Across eight benchmarks spanning materials science, additive manufacturing, structural biology, toxicology, and quantum chemistry, SHIELD consistently produces more structurally distinct train-test partitions than random, clustering-based, and domain-specific splitting methods, yielding more stringent and realistic assessments of model generalization while resolving overlapping paired entities without discarding data.

## Repository structure

```
shield/                          core SHIELD package — see shield/README.md
├── api.py                       public entry points: split()
├── pipeline.py                  block-coordinate optimization loop
├── coarsening.py                hierarchy-first / internal coarsening
├── stratification.py            quantile, moment, and sliced-Wasserstein stratification
├── rounding.py                  relaxation rounding and repair passes
├── diagnostics.py               leakage, stability, and Pareto-frontier diagnostics
└── solvers/                     LP/MILP (Gurobi), graph-cut, and OT/MMD backends

benchmarks/                      one .py script per benchmark (HIV, LP_PDBBind,
                                  Lipophilicity, glass, lpbf, qm8 x2, qm9, tox21) plus
                                  the raw datasets they read — see benchmarks/README.md
                                  for the script list and dataset install links (the
                                  datasets themselves are not included in this repository)

splits/                          reference splits used for comparison against SHIELD
                                  — see splits/README.md

outputs/                         SHIELD and baseline splits produced by the benchmark
                                  scripts
```

Each benchmark script is self-contained and imports `shield` and reads from `benchmarks/` and `splits/` using paths relative to the repository root, so the folder layout above should be kept intact when running the code.

## Installation

SHIELD is built on PyTorch and uses [Gurobi](https://www.gurobi.com/) as its LP/MILP backend; a Gurobi license is required to run `shield.split`.

```bash
pip install -r requirements.txt
```

The benchmark scripts additionally depend on RDKit, pymatgen, scikit-learn, and a few other scientific-computing packages used for featurization and visualization - see `requirements.txt` for the full list.

## Usage

```python
import shield

result = shield.split(
    n=n,                                  # number of observations
    K=3, r=[0.7, 0.15, 0.15],             # folds and target proportions
    hierarchy=hierarchy_levels,           # H term: keep related groups intact
    affinity=affinity_matrix,             # M term: feature-manifold graph cut
    affinity_provenance="disjoint-source",
    cost=cost_matrix, target_marginals=target_marginals,  # D term (match mode)
    alpha=1.0, beta=1.0, gamma=1.0,
)

fold_assignment = result.fold_assignment
```

Every input SHIELD accepts is documented with a short comment next to its default value in `shield/api.py`. See any of the benchmark scripts in `benchmarks/` for a complete worked example, from raw data to a saved split.

### Pre-computed splits

Producing a split from scratch (featurization, hierarchy/affinity construction, and the SHIELD solve) can take a while for the larger benchmarks. The `outputs/` folder is where we publish the SHIELD and baseline splits produced by each script, so you can use them directly without rerunning any code. We are populating the repository with more benchamrks over time, if a split you need is not yet there, you may request sooner as below.

### Datasets

The raw benchmark datasets are not included in this repository due to licensing and redistribution restrictions on the original sources. Download each dataset separately and place it in `benchmarks/` under the filename each script expects; see [benchmarks/README.md](benchmarks/README.md) for the full list, expected filenames, and download links.

## Requesting a split

Only the splits reported in the manuscript are included so far. If you would like SHIELD for a dataset that isn't in this repository yet, please reach out to **ayka89500@hbku.edu.qa** — we would be happy to generate the split and share it with you.

## Citation

This repository accompanies a manuscript currently under peer review. A full citation will be added here once it is published; in the meantime, please cite the repository directly.

## License

SHIELD is released under the **Academic and Non-Commercial Research License (ANCRL) v1.0** — see [LICENSE](LICENSE). Commercial use requires prior written permission from the copyright holder.
