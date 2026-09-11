# Benchmarks

One `.py` script per benchmark. Each script loads the raw dataset, builds the SHIELD inputs (hierarchy, affinity, cost/kernel), runs SHIELD and the baseline splitters, and writes diagnostics and figures to `outputs/`. Run a script from the repository root, e.g. `python benchmarks/tox21.py`.

| Benchmark | Script | Expected dataset file | Source |
|---|---|---|---|
| HIV | `hiv.py` | `HIV.csv` | [Deepchem Loader](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function) |
| LP-PDBBind | `pdbbind.py` | `LP_PDBBind.csv` | [LP-PDBBind Repo](https://github.com/THGLab/LP-PDBBind/tree/master) |
| Lipophilicity | `lipophilicity.py` | `Lipophilicity.csv` | [Deepchem Loader](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function) |
| Additive Manufacturing (LPBF) | `lpbf.py` | `lpbf.csv` | [Github Repo](https://github.com/GermanOmar/data_LPBF) |
| Matbench glass | `matbench_glass.py` | `glass.json` | [MatBench Website](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_glass/) |
| QM8 | `qm8_full.py`, `qm8_coarsened.py` | `qm8.csv` | [Deepchem Loader](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function) |
| QM9 | `qm9.py` | `qm9.csv` | [Deepchem Loader](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function) |
| Tox21 | `tox21.py` | `tox21.csv` | [Deepchem Loader](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function) |

## Datasets

The dataset files above are **not included** in this repository due to licensing and redistribution restrictions on the original sources. Download each one from the link in the table and place it directly in this folder (`benchmarks/`) under the exact filename shown, before running the corresponding script.

`qm8_full.py` and `qm8_coarsened.py` both read the same `qm8.csv`; the latter additionally exercises SHIELD's external-coarsening path.
