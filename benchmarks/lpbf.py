# ## 1. Setup, imports, configuration, output directories

# In[1]:


# Cell 1.1 - imports
import os, sys, json, time, math, hashlib, pickle, warnings, importlib, platform, re
import itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance, pearsonr, spearmanr

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

_HERE = Path(os.getcwd()).resolve()
_REPO_ROOT = _HERE if (_HERE / 'shield').is_dir() else _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from shield import split as shield_split
from shield import sweep as shield_sweep
from shield.api import build_hierarchy_levels
from shield.pipeline import _evaluate_h_term_from_z
from shield.solvers.graph_cut import cut_energy
from shield import diagnostics as shield_diag

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from shield.solvers.distributional import build_kernel
import torch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA as _PCA
from sklearn.manifold import TSNE as _TSNE
from scipy.stats import wasserstein_distance, gaussian_kde
import umap as _umap
import pandas as _pd
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors

print('SHIELD version:', __import__('shield').__version__)
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, 'PyTorch:', torch.__version__)


# In[57]:


# Cell 1.2 - global configurations
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'lpbf.csv')
DATASET_ENC  = 'latin-1'
TARGET_COL   = 'RD (%)'
MATERIAL_COL = 'Material'
MACHINE_COL  = 'Printer Model'
DROP_COLS    = ['DOI']

# Size control
MAX_ROWS    = None
RANDOM_SEED = 42

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.10
STRAT_TOL   = 0.10


# Stratification 
STRAT_STRATEGY = 'none'

# Affinity
KNN_AFFINITY_K = 25

# Hierarchy lambdas
LAMBDA_MAIN = [1.0, 0.5]
LAMBDA_VARIANTS_SI = {
    'plan_default':                 [1.0, 0.4],
    'inverse_plan_default':         [0.4, 1.0],
    'material_identity_only':       [1.0, 0.0],
    'solidification_only':          [0.0, 1.0],
    'balanced':                     [1.0, 1.0],
}

# SHIELD coefficients 
ALPHA, BETA, GAMMA, ETA, MU = 1.0, 1.0, 1.0, 1.0, 1.0
NU, TAU = 0.0, 0.0
DIST_MODE = 'shift'
SHIELD_SINKHORN_ITER = 50

# Two-entity mode
TWO_ENTITY_MODES = ['warm', 'cold-left', 'cold-right', 'cold-both']

# Sweep H/M/D coefficients
SWEEP_GRID = {
    'alpha': [0.2, 0.35, 0.5, 0.75, 1.0],
    'beta':  [0.2, 0.35, 0.5, 0.75, 1.0],
    'gamma': [0.2, 0.35, 0.5, 0.75, 1.0],
}
HAMMING_STABILITY_THRESHOLD = 0.10  

# kNN purity / alpha-shape
KNN_K = [5, 10, 20]
ALPHA_SHAPE_PARAMS = [0.01, 0.03, 0.05, 0.1, 0.15]  

# Solver / computational
SHIELD_MAX_ITER     = 10
SHIELD_STAGE0_SAMPLES  = 100        
SHIELD_INIT_BACKEND = 'auto'
SHIELD_ROUNDING     = 'pipage'
SHIELD_MILP_SIZE    = 10_000
SHIELD_DEVICE       = 'cpu'
N_JOBS              = -1

# ML evaluation
ML_RUN = True


# In[3]:


# Cell 1.3 - output directories and figure-saving utility
OUT_DIR   = _HERE / 'outputs'
OUT_MAIN  = OUT_DIR / 'main_text'
OUT_SI    = OUT_DIR / 'supplementary'
OUT_INTER = OUT_DIR / 'intermediate'
OUT_META  = OUT_DIR / 'metadata'

for d in [OUT_MAIN, OUT_SI, OUT_INTER, OUT_META]:
    for sub in ['figures', 'tables', 'data']:
        (d / sub).mkdir(parents=True, exist_ok=True)
for d in ['featurization', 'splits', 'shield_runs']:
    (OUT_INTER / d).mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
mpl.rcParams.update({
    'figure.dpi':      300,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'Liberation Sans', 'DejaVu Sans'],
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'pdf.fonttype':    42,
    'svg.fonttype':    'none',
})

def save_fig(fig, name, where='main', subdir='figures', formats=('png','pdf')):
    base = OUT_MAIN if where == 'main' else OUT_SI
    out  = base / subdir
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f'{name}.{ext}')
    return [str(out / f'{name}.{ext}') for ext in formats]

def save_table(df, name, where='main', subdir='tables'):
    base = OUT_MAIN if where == 'main' else OUT_SI
    out  = base / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{name}.csv'
    df.to_csv(path, index=False)
    return str(path)

def save_array(arr, name, where='main', subdir='data'):
    base = OUT_MAIN if where == 'main' else OUT_SI
    out  = base / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{name}.npy'
    np.save(path, arr)
    return str(path)

print('Output tree:')
for p in sorted(OUT_DIR.glob('*')):
    print(f'  {p.relative_to(_HERE)}')


# In[4]:


# Cell 1.4 - environment / optional-dependency probe
_OPT = {}
def _try(name, alias=None):
    try:
        mod = importlib.import_module(name)
        _OPT[alias or name] = mod
        return getattr(mod, '__version__', 'present')
    except Exception:
        _OPT[alias or name] = None
        return None

_versions = {
    'xgboost':     _try('xgboost'),
    'umap':        _try('umap', 'umap'),
    'alphashape':  _try('alphashape'),
    'shapely':     _try('shapely'),
    'gpytorch':    _try('gpytorch'),
    'gurobipy':    _try('gurobipy'),
}
for k, v in _versions.items():
    print(f'  {k:14s}  {v if v else "MISSING; required to get the same exact results"}')

with open(OUT_META / 'environment.json', 'w') as fh:
    json.dump({'python': sys.version, 'platform': platform.platform(),
               'versions': _versions,
               'shield_version': __import__('shield').__version__}, fh, indent=2)

if _OPT['gurobipy'] is None:
    raise RuntimeError('gurobipy + a Gurobi license are required for SHIELD LP/MILP.')


# ## 2. Data loading and cleaning

# In[5]:


# Cell 2.1 - load LPBF CSV with the latin-1 encoding
print(f'Loading from {DATASET_PATH} with encoding={DATASET_ENC}')
df_raw = pd.read_csv(DATASET_PATH, encoding=DATASET_ENC)
print(f'Raw shape: {df_raw.shape}')
print(f'Raw columns: {df_raw.columns.tolist()}')

to_drop = [c for c in DROP_COLS if c in df_raw.columns]
df = df_raw.drop(columns=to_drop)
print(f'Dropped {to_drop}')

df = df.rename(columns={c: 'D50_um' for c in df.columns if c.startswith('D50')})
print(f'Renamed D50 column. Final columns: {df.columns.tolist()}')

print(f'\nMissing values: {df.isna().sum().sum()} (should be 0)')


# In[6]:


# Cell 2.2 - canonicalize printer-model strings (whitespace + dash variants)
def _canonical_machine(s):
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(' - ', '-')
    s = re.sub(r'([A-Za-z]+)[\s-]+([0-9])', r'\1-\2', s)
    return s

df[MACHINE_COL] = df[MACHINE_COL].astype(str).map(_canonical_machine)

print(f'Unique alloys:    {df[MATERIAL_COL].nunique()}')
print(f'  {df[MATERIAL_COL].value_counts().to_dict()}')
print(f'Unique machines after canonicalization: {df[MACHINE_COL].nunique()}')
print('  Top 15 machines (highest counts):')
for m, c in df[MACHINE_COL].value_counts().head(15).items():
    print(f'    {m:35s}  {c}')


# In[7]:


# Cell 2.3 - subsample if MAX_ROWS is set, capture working n
if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = (df.groupby([MATERIAL_COL, MACHINE_COL], group_keys=False)
            .apply(lambda g: g.sample(n=max(1, int(round(MAX_ROWS * len(g) / len(df_raw)))),
                                      random_state=RANDOM_SEED, replace=False))
            .reset_index(drop=True))
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df):,} rows preserving alloy x machine coverage')
df = df.reset_index(drop=True)
n = len(df)
y_RD = df[TARGET_COL].values.astype(np.float32)
print(f'Working n={n}, RD(%) range=[{y_RD.min():.2f}, {y_RD.max():.2f}], '
      f'mean={y_RD.mean():.2f}')


# ## 3. Domain-knowledge tables - alloy thermophysical properties and nominal compositions

# In[8]:


# Cell 3.1 - Add alloy thermophysical properties
ALLOY_THERMO = {
    '316L':     (8000., 15.0,  1723., 500., 6.0e-3, -0.40e-3,    16e-6),
    'AlSi10Mg': (2670., 146.0, 868.,  910., 1.3e-3, -0.35e-3,    21e-6),
    '18Ni300':  (8100., 20.0,  1686., 450., 5.0e-3, -0.40e-3,    10e-6),
    'IN718':    (8190., 11.4,  1609., 435., 7.0e-3, -0.38e-3,    13e-6),
    'Ti6Al4V':  (4430., 6.7,   1933., 560., 3.25e-3,-0.26e-3,    8.6e-6),
    'CuCrZr':   (8890., 320.0, 1348., 390., 3.5e-3, -0.20e-3,    18e-6),
}
THERMO_COLS = ['rho_kg_m3', 'k_W_mK', 'Tm_K', 'Cp_J_kgK',
               'mu_melt_Pas', 'dsigma_dT_NmK', 'alpha_thermexp_1K']
thermo_df = (pd.DataFrame.from_dict(ALLOY_THERMO, orient='index', columns=THERMO_COLS)
               .rename_axis(MATERIAL_COL).reset_index())
save_table(thermo_df, 'alloy_thermophysical_properties', where='si')
thermo_df


# In[9]:


# Cell 3.2 - Add nominal alloy compositions
ALLOY_COMPOSITION = {
    '316L':     {'Fe': 0.67, 'Cr': 0.17, 'Ni': 0.12, 'Mo': 0.025, 'Mn': 0.015,
                 'Si': 0.0075, 'C': 0.0003, 'P': 0.00045, 'S': 0.0003, 'N': 0.001},
    'AlSi10Mg': {'Al': 0.895, 'Si': 0.10, 'Mg': 0.004, 'Fe': 0.001},
    '18Ni300':  {'Fe': 0.675, 'Ni': 0.18, 'Co': 0.090, 'Mo': 0.05, 'Ti': 0.007,
                 'Al': 0.001, 'C': 0.0003},
    'IN718':    {'Ni': 0.53, 'Cr': 0.19, 'Fe': 0.18, 'Nb': 0.05, 'Mo': 0.03,
                 'Ti': 0.009, 'Al': 0.005, 'C': 0.0004, 'Co': 0.001},
    'Ti6Al4V':  {'Ti': 0.895, 'Al': 0.06, 'V': 0.04, 'Fe': 0.003, 'O': 0.002},
    'CuCrZr':   {'Cu': 0.986, 'Cr': 0.010, 'Zr': 0.0015, 'Fe': 0.0005},
}
ALL_ELEMENTS = sorted(set(el for cmp in ALLOY_COMPOSITION.values() for el in cmp))
print(f'Element universe: {ALL_ELEMENTS}')

comp_rows = []
for alloy, cmp in ALLOY_COMPOSITION.items():
    row = {'Material': alloy}
    for el in ALL_ELEMENTS:
        row[f'comp_{el}'] = cmp.get(el, 0.0)
    comp_rows.append(row)
comp_df = pd.DataFrame(comp_rows)
save_table(comp_df, 'alloy_composition_mass_fraction', where='si')
comp_df


# ## 4. Feature engineering: VED / LED / AED, normalized energy, Peclet, thermophysical merge, composition

# In[10]:


# Cell 4.1 - process-physics derived features (VED, LED, AED, alpha, Pe)
P = df['Laser Power (W)'].values.astype(np.float32)
v = df['Scan Speed (mm/s)'].values.astype(np.float32)
h = df['Hatch space (mm)'].values.astype(np.float32)
t = df['Layer thickness (mm)'].values.astype(np.float32)
spot = df['Spot size (mm)'].values.astype(np.float32)
D50 = df['D50_um'].values.astype(np.float32)

df['VED']  = P / np.maximum(v * h * t, 1e-9)
df['LED']  = P / np.maximum(v, 1e-9)
df['AED']  = P / np.maximum(v * h, 1e-9)
df['v_h_ratio']   = v / np.maximum(h, 1e-9)
df['spot_layer']  = spot / np.maximum(t, 1e-9)

for col in THERMO_COLS:
    df[col] = df[MATERIAL_COL].map(lambda a: ALLOY_THERMO[a][THERMO_COLS.index(col)])

df['alpha_thermdiff_m2s'] = df['k_W_mK'] / (df['rho_kg_m3'] * df['Cp_J_kgK'])

v_si = v * 1e-3
h_si = h * 1e-3
df['E_star']  = P / np.maximum(v_si * h_si * np.sqrt(df['alpha_thermdiff_m2s']), 1e-30)

spot_si = spot * 1e-3
df['Pe']     = v_si * spot_si / np.maximum(df['alpha_thermdiff_m2s'], 1e-30)

df['Marangoni_proxy'] = (np.abs(df['dsigma_dT_NmK']) * np.clip(df['Tm_K'] - 300, 0, None)
                          * spot_si /
                          np.maximum(df['mu_melt_Pas'] * df['alpha_thermdiff_m2s'], 1e-30))

DERIVED_COLS = ['VED', 'LED', 'AED', 'v_h_ratio', 'spot_layer', 'E_star',
                'Pe', 'Marangoni_proxy', 'alpha_thermdiff_m2s']
print('Process-physics + thermophysical derived features:')
print(df[DERIVED_COLS].describe().T[['mean', 'std', 'min', 'max']].round(4))


# In[11]:


# Cell 4.2 - elemental composition columns (one per element observed)
COMP_COLS = []
for el in ALL_ELEMENTS:
    col = f'comp_{el}'
    df[col] = df[MATERIAL_COL].map(lambda a: ALLOY_COMPOSITION[a].get(el, 0.0))
    COMP_COLS.append(col)

print(f'Composition columns added: {len(COMP_COLS)}')
print(df[COMP_COLS].describe().T[['mean', 'std']].round(4).head(-1))


# In[12]:


# Cell 4.3 - assemble the joint feature matrix and z-score it
PROCESS_COLS = ['Laser Power (W)', 'Scan Speed (mm/s)', 'Hatch space (mm)',
                'Layer thickness (mm)', 'Spot size (mm)', 'D50_um']
THERMO_COLS_F = THERMO_COLS + ['alpha_thermdiff_m2s']
FEATURE_COLS  = PROCESS_COLS + DERIVED_COLS + THERMO_COLS_F + COMP_COLS

X = df[FEATURE_COLS].values.astype(np.float32)
log_cols = ['VED', 'LED', 'AED', 'E_star', 'Pe', 'Marangoni_proxy']
for c in log_cols:
    j = FEATURE_COLS.index(c)
    X[:, j] = np.log10(np.maximum(X[:, j], 1e-30))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
print(f'Feature matrix: shape={X_scaled.shape}; '
      f'{len(PROCESS_COLS)} process + {len(DERIVED_COLS)} derived + '
      f'{len(THERMO_COLS_F)} thermophysical + {len(COMP_COLS)} composition')


# ## 5. Affinity, kernel matrix, hierarchy

# In[13]:


# Cell 5.1 - cosine-similarity affinity over process+derived feature subspace (M1)
def _knn_sparsify(A, k):
    n_ = A.shape[0]; kk = min(k, n_ - 1)
    knn_idx = np.argpartition(-A, kth=kk, axis=1)[:, :kk]
    out = np.zeros_like(A)
    rr = np.repeat(np.arange(n_), kk); cc = knn_idx.reshape(-1)
    out[rr, cc] = A[rr, cc]
    return np.maximum(out, out.T)

_M1_COLS   = PROCESS_COLS + [c for c in DERIVED_COLS if c != 'alpha_thermdiff_m2s']
_m1_idx    = [FEATURE_COLS.index(c) for c in _M1_COLS]
X_affinity = X_scaled[:, _m1_idx]

A_full = cosine_similarity(X_affinity).astype(np.float32)
np.fill_diagonal(A_full, 0.0)
AM     = _knn_sparsify(A_full, KNN_AFFINITY_K)
print(f'Affinity matrix (M1 - process+derived, {len(_M1_COLS)} features): '
      f'shape={A_full.shape}; k-NN density={(AM > 0).sum() / AM.size:.4f}')

np.save(OUT_INTER / 'featurization' / 'affinity_full.npy', A_full)
np.save(OUT_INTER / 'featurization' / 'affinity_knn.npy', AM)


# In[14]:


# Cell 5.2 - kernel matrix for D-shift (RBF over the scaled feature space)
X_t = torch.from_numpy(X_scaled)
K_mat = build_kernel(X_t, X_t, kernel='rbf').numpy()
np.save(OUT_INTER / 'featurization' / 'kernel_matrix.npy', K_mat)
print(f'Kernel matrix: shape={K_mat.shape}, mean={K_mat.mean():.4f}, '
      f'min={K_mat.min():.4f}, max={K_mat.max():.4f}, '
      f'symmetric? {np.allclose(K_mat, K_mat.T)}')


# ## 6. Hierarchy and (alloy, machine) entity identifiers

# In[15]:


# Cell 6.1 - alloy-family + solidification-class classification
ALLOY_FAMILY = {
    '316L':     'stainless_steel',
    '18Ni300':  'maraging_steel',
    'AlSi10Mg': 'Al_alloy',
    'IN718':    'Ni_superalloy',
    'Ti6Al4V':  'Ti_alloy',
    'CuCrZr':   'Cu_alloy',
}
del ALLOY_FAMILY



SOLIDIFICATION_CLASS = {
    '316L':     'FCC',
    '18Ni300':  'BCT',
    'AlSi10Mg': 'FCC',
    'IN718':    'FCC',
    'Ti6Al4V':  'HCP_BCC',
    'CuCrZr':   'FCC',
}
df['_solidification']    = df[MATERIAL_COL].map(SOLIDIFICATION_CLASS)
print('Alloy family x solidification class:')
print(df[[MATERIAL_COL, '_solidification']].drop_duplicates().to_string(index=False))


# In[16]:


# Cell 6.2 - hierarchy builder
def build_hierarchy(df_local, lambda_per_level):
    """Three-level alloy hierarchy.

    lambda_per_level = [lambda_identity, lambda_solidification].
    Deepest -> shallowest (per SHIELD convention).
    """
    def _grp(series):
        uniq = list(dict.fromkeys(series))
        idx  = {u: i for i, u in enumerate(uniq)}
        desc = [[] for _ in uniq]
        for i, v in enumerate(series): desc[idx[v]].append(i)
        return uniq, desc
    L3_names, L3_desc = _grp(df_local[MATERIAL_COL])
    L1_names, L1_desc = _grp(df_local['_solidification'])



    comp_arr = np.array([[ALLOY_COMPOSITION[a].get(el, 0.0) for el in ALL_ELEMENTS]
                         for a in L3_names], dtype=np.float64)
    sim3 = cosine_similarity(comp_arr)
    rs, cs, vs = [], [], []
    for i in range(len(L3_names)):
        for j in range(i + 1, len(L3_names)):
            if sim3[i, j] > 0:
                rs += [i, j]; cs += [j, i]; vs += [sim3[i, j], sim3[i, j]]
    r3 = np.array(rs, np.int64); c3 = np.array(cs, np.int64); v3 = np.array(vs)

    r1, c1, v1 = [], [], []
    for i in range(len(L1_names)):
        for j in range(i + 1, len(L1_names)):
            r1 += [i, j]; c1 += [j, i]; v1 += [0.5, 0.5]
    r1 = np.array(r1, np.int64); c1 = np.array(c1, np.int64); v1 = np.array(v1)

    return [
        {'name': 'alloy_identity',  'descendants': L3_desc, 'similarity': (r3, c3, v3),
         'lambda': lambda_per_level[0]},

        {'name': 'solidification',  'descendants': L1_desc, 'similarity': (r1, c1, v1),
         'lambda': lambda_per_level[1]},
    ]

hierarchy_main = build_hierarchy(df, LAMBDA_MAIN)
for L in hierarchy_main:
    print(f"  {L['name']:18s}: {len(L['descendants'])} nodes, "
          f"{len(L['similarity'][2])} edges, lambda={L['lambda']}")
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)


# In[17]:


# Cell 6.3 - entity IDs for cold-* modes
alloy_entity_of   = pd.factorize(df[MATERIAL_COL])[0].astype(np.int64)
machine_entity_of = pd.factorize(df[MACHINE_COL])[0].astype(np.int64)

pair_indices = np.column_stack([np.arange(n), np.arange(n)])

print(f'Alloy entities:   {len(np.unique(alloy_entity_of))}  '
      f'(mean rows / entity = {n / len(np.unique(alloy_entity_of)):.1f})')
print(f'Machine entities: {len(np.unique(machine_entity_of))}  '
      f'(mean rows / entity = {n / len(np.unique(machine_entity_of)):.1f})')


# ## 7. Baseline splitters

# In[18]:


# Cell 7.1 - implement baselines
def split_random(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[perm[:cut]] = 0
    return f

def split_stratified(seed):
    y_bins = pd.qcut(y_RD, q=5, labels=False, duplicates='drop')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=FOLD_RATIOS[1], random_state=seed)
    train_idx, _ = next(sss.split(np.zeros(n), y_bins))
    f = np.ones(n, dtype=int); f[train_idx] = 0
    return f

def split_pc1(seed):
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(X_scaled)[:, 0]
    cutoff = np.percentile(pc1, FOLD_RATIOS[0] * 100)
    f = np.where(pc1 <= cutoff, 0, 1).astype(int)
    return f


def split_kmeans(seed, n_clusters=20):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(X_scaled)
    return _greedy_pack(km.labels_, n_clusters)

def split_agglomerative(seed, n_clusters=18, linkage='ward'):
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X_scaled)
    return _greedy_pack(ac.labels_, n_clusters)

def _greedy_pack(labels, n_clusters):
    grp_to_idx = defaultdict(list)
    for i, c in enumerate(labels): grp_to_idx[c].append(i)
    order = sorted(grp_to_idx.values(), key=len, reverse=True)
    f = np.ones(n, dtype=int)
    target_train = int(round(FOLD_RATIOS[0] * n)); cum = 0
    for g in order:
        if cum + len(g) <= target_train:
            for j in g: f[j] = 0
            cum += len(g)
        else: break
    return f


BASELINE_SPLITTERS = {
    'RANDOM':       lambda seed=RANDOM_SEED: split_random(seed),
    'STRATIFIED':   lambda seed=RANDOM_SEED: split_stratified(seed),
    'KMEANS':       lambda seed=RANDOM_SEED: split_kmeans(seed),
    'AGGLOMERATIVE':lambda seed=RANDOM_SEED: split_agglomerative(seed),
    'PC1_SCAFFOLD': lambda seed=RANDOM_SEED: split_pc1(seed),
}


# In[19]:


# Cell 7.2 - run baselines
baseline_results = {}
for name, fn in BASELINE_SPLITTERS.items():
    f = fn()
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(
        OUT_INTER / 'splits' / f'{name}.csv', index=False)
    print(f'  {name:22s}  sizes={sizes.round(3).tolist()}')


# ## 8. SHIELD (warm + cold-alloy + cold-machine + cold-both + ablations for warm mode)

# In[20]:


# Cell 8.1 - SHIELD config grid
SHIELD_CONFIGS = {}

for mode in TWO_ENTITY_MODES:
    SHIELD_CONFIGS[f'HMD-SHIELD_{mode}'] = dict(
        alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, mode=mode)

# Warm-mode single-term runs (keep only one term)
for term, param in [('H', 'alpha'), ('M', 'beta'), ('D', 'gamma')]:
    cfg = dict(alpha=0.0, beta=0.0, gamma=0.0, eta=0.0, mu=0.0, mode='warm')
    cfg[param] = ALPHA if param == 'alpha' else (BETA if param == 'beta' else GAMMA)
    SHIELD_CONFIGS[f'{term}-SHIELD_warm'] = cfg



print('SHIELD config grid:')
for k, c in SHIELD_CONFIGS.items():
    print(f'  {k:30s}  {c}')


# In[21]:


# Cell 8.2 - run all SHIELD configurations
shield_results = {}
shield_diagnostics = {}
t_all = time.time()
CACHED_NORMALIZERS = None


Y_TARGET = y_RD.reshape(-1, 1).astype(np.float32)

for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    mode = cfg['mode']
    kwargs = dict(
        n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main if cfg['alpha'] > 0 else None,
        affinity=AM if cfg['beta'] > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        kernel_matrix=K_mat if cfg['gamma'] > 0 else None,
        y_target=Y_TARGET,
        stratification_strategy=STRAT_STRATEGY if cfg['mu'] > 0 else 'none',
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        two_entity_mode=mode,
        pair_indices=pair_indices if mode != 'warm' else None,
        left_entity_of=alloy_entity_of   if mode in ('cold-left',  'cold-both') else None,
        right_entity_of=machine_entity_of if mode in ('cold-right', 'cold-both') else None,
        pair_observation_targets=y_RD.reshape(-1, 1) if mode != 'warm' else None,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'], mu=cfg['mu'], nu=NU, tau=TAU,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        stage0_samples=SHIELD_STAGE0_SAMPLES,
        precomputed_normalizers=CACHED_NORMALIZERS,
        sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        res = shield_split(**kwargs)
        if CACHED_NORMALIZERS is None:                   
            CACHED_NORMALIZERS = res.metadata.get('stage0_normalizers')
        fold = res.fold_assignment.cpu().numpy().astype(int)
        shield_results[cfg_name] = fold
        shield_diagnostics[cfg_name] = {
            'history': res.history,
            'diagnostics': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                            for k, v in res.diagnostics.items() if k != 'history'},
            'metadata': res.metadata,
        }
        (OUT_INTER / 'shield_runs' / cfg_name).mkdir(exist_ok=True)
        pd.DataFrame({'index': np.arange(n), 'fold': fold}).to_csv(
            OUT_INTER / 'shield_runs' / cfg_name / 'split.csv', index=False)
        print(f'  {cfg_name:30s}  wall={time.time()-t0:.1f}s  '
              f'sizes={(np.bincount(fold, minlength=N_FOLDS)/n).round(3).tolist()}')
    except Exception as exc:
        print(f'  {cfg_name:30s}  FAILED: {exc}')

print(f'Total SHIELD wall time: {time.time()-t_all:.1f}s')


# In[22]:


# Cell 8.3 - Distribution shifts across SHIELD splits: showing D term impact

d_term_configs = [cfg for cfg in SHIELD_CONFIGS.keys() if 'HMD-SHIELD_warm' in cfg or 'D-SHIELD_warm' in cfg]
d_term_configs = ['HMD-SHIELD_warm', 'H-SHIELD_warm', 'M-SHIELD_warm', 'D-SHIELD_warm']
d_term_configs = [c for c in d_term_configs if c in shield_results]

if 'embeds' not in locals() or 'PCA' not in embeds:
    emb2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_scaled)
else:
    emb2d = embeds['PCA']

try:
    from sklearn.manifold import TSNE
    print('Computing t-SNE embedding...')
    tsne_emb = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=1000).fit_transform(X_scaled)
    print('t-SNE done')
except Exception as e:
    print(f't-SNE failed: {e}')
    tsne_emb = None

fig, axes = plt.subplots(5, len(d_term_configs), figsize=(5 * len(d_term_configs), 16))
if len(d_term_configs) == 1:
    axes = axes.reshape(-1, 1)

for col, cfg_name in enumerate(d_term_configs):
    fold = shield_results[cfg_name]

    # Row 0: 2D PCA scatter plot
    ax = axes[0, col]
    ax.scatter(emb2d[fold == 0, 0], emb2d[fold == 0, 1], s=12, alpha=0.5, 
               c='#4477AA', label='train', edgecolor='none')
    ax.scatter(emb2d[fold == 1, 0], emb2d[fold == 1, 1], s=12, alpha=0.5, 
               c='#CC6677', label='test', edgecolor='none')

    try:
        from scipy.stats import gaussian_kde
        xy_train = emb2d[fold == 0].T
        xy_test = emb2d[fold == 1].T
        if len(xy_train[0]) > 10 and len(xy_test[0]) > 10:
            z_train = gaussian_kde(xy_train)(emb2d.T)
            z_test = gaussian_kde(xy_test)(emb2d.T)
            levels_train = np.percentile(z_train[fold == 0], [68, 95])
            levels_test = np.percentile(z_test[fold == 1], [68, 95])
            xx, yy = np.meshgrid(np.linspace(emb2d[:, 0].min(), emb2d[:, 0].max(), 50),
                                 np.linspace(emb2d[:, 1].min(), emb2d[:, 1].max(), 50))
            zz_train = gaussian_kde(xy_train)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            zz_test = gaussian_kde(xy_test)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz_train, levels=levels_train, colors='#4477AA', alpha=0.4, linewidths=1)
            ax.contour(xx, yy, zz_test, levels=levels_test, colors='#CC6677', alpha=0.4, linewidths=1)
    except:
        pass

    ax.set_title(f'{cfg_name}\n(train: {np.sum(fold==0)}, test: {np.sum(fold==1)})', fontsize=9)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1: PC1 distribution (SAME BIN WIDTHS)
    ax = axes[1, col]
    pc1_min, pc1_max = emb2d[:, 0].min(), emb2d[:, 0].max()
    bins_pc1 = np.linspace(pc1_min, pc1_max, 30)
    ax.hist(emb2d[fold == 0, 0], bins=bins_pc1, alpha=0.6, label='train', color='#4477AA', density=True)
    ax.hist(emb2d[fold == 1, 0], bins=bins_pc1, alpha=0.6, label='test', color='#CC6677', density=True)

    try:
        from scipy.stats import gaussian_kde
        kde_train = gaussian_kde(emb2d[fold == 0, 0])
        kde_test = gaussian_kde(emb2d[fold == 1, 0])
        x_range = np.linspace(pc1_min, pc1_max, 100)
        ax.plot(x_range, kde_train(x_range), color='#4477AA', linewidth=2, alpha=0.7)
        ax.plot(x_range, kde_test(x_range), color='#CC6677', linewidth=2, alpha=0.7)
    except:
        pass

    ax.set_xlabel('PC1'); ax.set_ylabel('Density')
    ax.set_title('PC1 distribution', fontsize=8)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Row 2: PC2 distribution (SAME BIN WIDTHS)
    ax = axes[2, col]
    pc2_min, pc2_max = emb2d[:, 1].min(), emb2d[:, 1].max()
    bins_pc2 = np.linspace(pc2_min, pc2_max, 30)
    ax.hist(emb2d[fold == 0, 1], bins=bins_pc2, alpha=0.6, label='train', color='#4477AA', density=True)
    ax.hist(emb2d[fold == 1, 1], bins=bins_pc2, alpha=0.6, label='test', color='#CC6677', density=True)

    try:
        from scipy.stats import gaussian_kde
        kde_train = gaussian_kde(emb2d[fold == 0, 1])
        kde_test = gaussian_kde(emb2d[fold == 1, 1])
        x_range = np.linspace(pc2_min, pc2_max, 100)
        ax.plot(x_range, kde_train(x_range), color='#4477AA', linewidth=2, alpha=0.7)
        ax.plot(x_range, kde_test(x_range), color='#CC6677', linewidth=2, alpha=0.7)
    except:
        pass

    ax.set_xlabel('PC2'); ax.set_ylabel('Density')
    ax.set_title('PC2 distribution', fontsize=8)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Row 3: Wasserstein distance (PCA space)
    ax = axes[3, col]
    try:
        from scipy.stats import wasserstein_distance
        w_pc1 = wasserstein_distance(emb2d[fold == 0, 0], emb2d[fold == 1, 0])
        w_pc2 = wasserstein_distance(emb2d[fold == 0, 1], emb2d[fold == 1, 1])
        ax.bar(['PC1', 'PC2'], [w_pc1, w_pc2], 
               color=['#4477AA', '#CC6677'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Distributional Shift Magnitude (PCA)', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate([w_pc1, w_pc2]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=8)
    except:
        ax.text(0.5, 0.5, 'Wasserstein unavailable', ha='center', va='center', transform=ax.transAxes)

    # Row 4: t-SNE 2D scatter plot
    ax = axes[4, col]
    if tsne_emb is not None:
        ax.scatter(tsne_emb[fold == 0, 0], tsne_emb[fold == 0, 1], s=12, alpha=0.5, 
                   c='#4477AA', label='train', edgecolor='none')
        ax.scatter(tsne_emb[fold == 1, 0], tsne_emb[fold == 1, 1], s=12, alpha=0.5, 
                   c='#CC6677', label='test', edgecolor='none')
        ax.set_title('t-SNE (2D) embedding', fontsize=8)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 't-SNE unavailable', ha='center', va='center', transform=ax.transAxes)

fig.suptitle('D-Term Impact Analysis)', fontsize=12, y=0.995)
plt.tight_layout()
save_fig(fig, 'distribution_shift_d_term_analysis', where='main')
plt.show()

# Summary statistics
print('='*90)
print('Distribution Shift Summary (D-Term Analysis)')
print('='*90)
for cfg_name in d_term_configs:
    fold = shield_results[cfg_name]
    from scipy.stats import wasserstein_distance
    w_pc1 = wasserstein_distance(emb2d[fold == 0, 0], emb2d[fold == 1, 0])
    w_pc2 = wasserstein_distance(emb2d[fold == 0, 1], emb2d[fold == 1, 1])
    if tsne_emb is not None:
        w_tsne1 = wasserstein_distance(tsne_emb[fold == 0, 0], tsne_emb[fold == 1, 0])
        w_tsne2 = wasserstein_distance(tsne_emb[fold == 0, 1], tsne_emb[fold == 1, 1])
    mean_sep_pc1 = np.abs(emb2d[fold == 0, 0].mean() - emb2d[fold == 1, 0].mean())
    mean_sep_pc2 = np.abs(emb2d[fold == 0, 1].mean() - emb2d[fold == 1, 1].mean())
    print(f'\n{cfg_name}:')
    print(f'  Wasserstein (PC1):    {w_pc1:.6f}  |  Mean separation: {mean_sep_pc1:.6f}')
    print(f'  Wasserstein (PC2):    {w_pc2:.6f}  |  Mean separation: {mean_sep_pc2:.6f}')
    if tsne_emb is not None:
        print(f'  Wasserstein (t-SNE1): {w_tsne1:.6f}')
        print(f'  Wasserstein (t-SNE2): {w_tsne2:.6f}')
    print(f'  Train size: {np.sum(fold==0):4d}  |  Test size: {np.sum(fold==1):4d}')


# ## 9. Leakage diagnostics L(pi), channels, entity-straddle, VED preservation, per-machine R^2

# In[23]:


# Cell 9.1 - cross-fold similarity L(pi)

A_pos = np.clip(A_full, 0.0, 1.0)

kappa = np.ones(A_pos.shape[0])

kappa_matrix = kappa.reshape(-1, 1) * kappa

TOTAL_SIM = float((A_pos * kappa_matrix).sum())

def cross_fold_similarity(fold, A=A_pos, kappa=kappa_matrix):
    """
    Calculate cross-fold similarity leak L(Ï) with cardinality weighting.

    Args:
        fold: array of fold assignments Ï(x) for each element
        A: similarity matrix (clipped to [0,1])
        kappa: cardinality weighting matrix Îº(x) * Îº(x')
    """
    F = fold.reshape(-1, 1)

    cross = (F != F.T).astype(np.float32)

    raw = float((A * cross * kappa).sum())

    scaled = raw / max(TOTAL_SIM, 1e-9)

    return raw, scaled

lpi_rows = []
for name, fold in baseline_results.items():
    raw, scaled = cross_fold_similarity(fold)
    lpi_rows.append({'method': 'baseline', 'config': name,
                     'L_pi': raw, 'scaled_L_pi': scaled})

for name, fold in shield_results.items():
    raw, scaled = cross_fold_similarity(fold)
    lpi_rows.append({'method': 'SHIELD', 'config': name,
                     'L_pi': raw, 'scaled_L_pi': scaled})

lpi_df = pd.DataFrame(lpi_rows).sort_values('scaled_L_pi')
save_table(lpi_df, 'cross_fold_similarity', where='main')
lpi_df


# In[24]:


# Cell 9.2 - L(pi) bar chart
fig, ax = plt.subplots(figsize=(12, 4))
agg = lpi_df.set_index('config')['scaled_L_pi'].sort_values()
colors = ['#888888' if c in baseline_results else '#4477AA' for c in agg.index]
ax.bar(range(len(agg)), agg.values, color=colors, edgecolor='black')
ax.axhline(agg.loc['RANDOM'] if 'RANDOM' in agg.index else 1.0,
           color='red', ls='--', lw=1, label='RANDOM baseline')
ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg.index, rotation=45, ha='right')
ax.set_ylabel('scaled L(pi)')
ax.set_title('Cross-fold feature-space similarity (lower = stronger OOD)')
ax.legend()
plt.tight_layout()
save_fig(fig, 'L_pi_bar_chart', where='main')
plt.show()


# In[25]:


# Cell 9.3 - per-channel L_H, L_M, L_W
hier_levels_obj = build_hierarchy_levels(hierarchy_main)
AM_tensor = torch.from_numpy(AM).to(torch.float32)

def per_channel_leakage(fold):
    z = torch.zeros((n, N_FOLDS), dtype=torch.float32)
    z[np.arange(n), fold] = 1.0
    L_H = float(_evaluate_h_term_from_z(z, hier_levels_obj))
    L_M = float(cut_energy(z, AM_tensor))
    tr = y_RD[fold == 0]; te = y_RD[fold == 1]
    L_W = float(wasserstein_distance(tr, te)) if len(tr) and len(te) else float('nan')
    return L_H, L_M, L_W

ch_rows = []
for name, fold in baseline_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'baseline', 'config': name,
                    'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
for name, fold in shield_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'SHIELD', 'config': name,
                    'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
channel_df = pd.DataFrame(ch_rows)
save_table(channel_df, 'channel_decomposition', where='main')
channel_df


# In[26]:


# Cell 9.4 - entity-straddle audit (cold-discard accounting for the (alloy, machine) setting)
def entity_straddle(fold, entity_of):
    n_split = 0
    for eid, grp in pd.Series(fold).groupby(entity_of):
        if grp.nunique() > 1:
            n_split += 1
    return n_split, int(np.unique(entity_of).size)

cold_audit = []
for name, fold in {**baseline_results, **shield_results}.items():
    nA, totA = entity_straddle(fold, alloy_entity_of)
    nM, totM = entity_straddle(fold, machine_entity_of)
    cold_audit.append({
        'config': name,
        'alloys_straddled':       nA, 'alloys_total':   totA,
        'machines_straddled':     nM, 'machines_total': totM,
        'fraction_alloys_straddled':   nA / max(totA, 1),
        'fraction_machines_straddled': nM / max(totM, 1),
    })
cold_df = pd.DataFrame(cold_audit)
save_table(cold_df, 'entity_straddle_audit', where='main')
cold_df


# In[27]:


# Cell 9.5 - per-channel radar (main-text exhibit)
configs_for_radar = list(channel_df['config'])
ch_arr = channel_df[['L_H', 'L_M', 'L_W']].values
ch_norm = ch_arr / np.maximum(ch_arr.max(axis=0, keepdims=True), 1e-12)
angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
for cfg, row in zip(configs_for_radar, ch_norm):
    values = row.tolist() + [row[0]]
    color = '#888888' if cfg in baseline_results else '#4477AA'
    lw = 2.5 if 'cold-left' in cfg and 'SHIELD' in cfg else 0.8
    ax.plot(angles, values, label=cfg, color=color, lw=lw,
            alpha=1.0 if 'cold-left' in cfg else 0.5)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(['L_H', 'L_M', 'L_W'])
ax.set_title('Per-channel leakage (normalized per axis)')
ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1.1), fontsize=6)
plt.tight_layout()
save_fig(fig, 'channel_leakage_radar', where='main')
plt.show()


# In[28]:


# Cell 9.6 - kNN purity in joint scaled feature space
def knn_purity(fold, X, k):
    test_idx = np.where(fold == 1)[0]
    if not len(test_idx): return float('nan')

    Xt = X[test_idx].astype(np.float32)
    X_full = X.astype(np.float32)

    d = (Xt * Xt).sum(1, keepdims=True) + (X_full * X_full).sum(1)[None, :] - 2 * Xt @ X_full.T

    nn = np.argpartition(d, kth=min(k+1, X_full.shape[0]-1), axis=1)[:, :k+1]

    for i in range(nn.shape[0]):
        nn[i] = nn[i][np.argsort(d[i, nn[i]])]

    nn = nn[:, 1:k+1]

    return float(np.mean([np.mean(fold[row] == 0) for row in nn]))

purity_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    for k in KNN_K:
        purity_rows.append({'config': name, 'k': k,
                            'knn_purity': knn_purity(fold, X_scaled, k)})
purity_df = pd.DataFrame(purity_rows)
save_table(purity_df, 'knn_purity', where='main')

display(purity_df)

fig, axes = plt.subplots(1, len(KNN_K), figsize=(5 * len(KNN_K), 4))
if len(KNN_K) == 1: axes = [axes]
for ax, k in zip(axes, KNN_K):
    sub = purity_df[purity_df['k'] == k].set_index('config')['knn_purity'].sort_values()
    colors = ['#888888' if c in baseline_results else '#4477AA' for c in sub.index]
    ax.bar(range(len(sub)), sub.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub.index, rotation=45, ha='right')
    ax.axhline(FOLD_RATIOS[0], color='red', ls='--', lw=1, label='random baseline')
    ax.set_title(f'kNN purity (k={k})')
    ax.set_ylim(0, 1)
    ax.legend()
plt.tight_layout()
save_fig(fig, 'knn_purity_panels', where='main')
plt.show()


# In[29]:


# Cell 9.7 - alpha-shape IoU in 2-D PCA with improved visualization

emb2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_scaled)
save_array(emb2d, 'pca_2d_embedding', where='main', subdir='data')

def alpha_iou(fold, alpha):
    if _OPT['alphashape'] is None or _OPT['shapely'] is None: 
        return float('nan')

    import alphashape
    from shapely.geometry import Polygon

    tr_coords = emb2d[fold == 0]
    te_coords = emb2d[fold == 1]

    if len(tr_coords) < 4 or len(te_coords) < 4: 
        return float('nan')

    try:
        st = alphashape.alphashape(tr_coords, alpha)
        ss = alphashape.alphashape(te_coords, alpha)

        if not isinstance(st, Polygon) or not isinstance(ss, Polygon):
            return float('nan')

        if st.is_empty or ss.is_empty: 
            return float('nan')

        inter = st.intersection(ss).area
        union = st.union(ss).area
        return float(inter / union) if union > 0 else float('nan')

    except Exception:
        return float('nan')

iou_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    for a in ALPHA_SHAPE_PARAMS:
        iou_rows.append({'config': name, 'alpha': a, 'iou': alpha_iou(fold, a)})

iou_df = pd.DataFrame(iou_rows)
save_table(iou_df, 'alpha_shape_iou', where='main')
display(iou_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: Heatmap (unchanged)
pivot_iou = iou_df.pivot(index='config', columns='alpha', values='iou')
im = axes[0].imshow(pivot_iou.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
axes[0].set_xticks(range(len(pivot_iou.columns)))
axes[0].set_xticklabels([f'{a:.3f}' for a in pivot_iou.columns], rotation=45, ha='right')
axes[0].set_yticks(range(len(pivot_iou.index)))
axes[0].set_yticklabels(pivot_iou.index, fontsize=9)
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Config')
axes[0].set_title('Alpha-Shape IoU (Red=High Leakage, Green=Low Leakage)')
plt.colorbar(im, ax=axes[0])

# RIGHT: Line plot — all baselines + all SHIELD configs
baseline_names = list(baseline_results.keys())
shield_names   = list(shield_results.keys())

baseline_colors = plt.cm.tab10(np.linspace(0, 0.9, len(baseline_names)))
shield_colors   = plt.cm.cool(np.linspace(0.1, 0.9, len(shield_names)))

for i, name in enumerate(baseline_names):
    data = iou_df[iou_df['config'] == name].sort_values('alpha')
    axes[1].plot(data['alpha'], data['iou'], marker='o', linewidth=2, alpha=0.85,
                 color=baseline_colors[i], label=f'baseline: {name}')

for i, name in enumerate(shield_names):
    data = iou_df[iou_df['config'] == name].sort_values('alpha')
    axes[1].plot(data['alpha'], data['iou'], marker='s', linewidth=1.5, alpha=0.75,
                 color=shield_colors[i], linestyle='--', label=f'SHIELD: {name}')

# SHIELD average on top
shield_avg = iou_df[iou_df['config'].isin(shield_names)].groupby('alpha')['iou'].mean().sort_index()
axes[1].plot(shield_avg.index, shield_avg.values, marker='D', linewidth=2.5,
             color='black', linestyle='-', label='SHIELD AVG', zorder=5)

axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('IoU')
axes[1].set_title('IoU Trend by Alpha — All Baselines & All SHIELD Configs')
axes[1].legend(fontsize=8, loc='best', ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, 'alpha_shape_iou_analysis', where='main')
plt.show()


# In[30]:


# Cell 9.8 - Train Potential Leakage (TPL)
def tpl(X, fold, train_label=0, test_label=1, m=64, n_bootstrap=100):
    """
    Compute Train Potential Leakage (TPL) using density-based Mann-Whitney U statistic.

    TPL = 2*P(train_potential > test_potential) - 1

    where potential is a Gaussian-kernel density estimate using only train points.

    Args:
        X: feature array (n, d)
        fold: label array (n,), 0=train, 1=test
        train_label, test_label: fold labels
        m: number of neighbors to use in potential calculation
        n_bootstrap: number of bootstrap replicates for CI

    Returns:
        tpl_value, tpl_ci_lower, tpl_ci_upper, (phi_train, phi_test, h) diagnostics
    """
    try:
        import hnswlib
        use_hnsw = True
    except ImportError:
        use_hnsw = False

    train_idx = np.where(fold == train_label)[0]
    test_idx = np.where(fold == test_label)[0]

    if len(test_idx) == 0 or len(train_idx) < 2:
        return float('nan'), float('nan'), float('nan'), (np.array([]), np.array([]), float('nan'))

    d = X.shape[1]
    Xtr = X[train_idx].astype('float32')

    if use_hnsw:
        idx = hnswlib.Index(space='l2', dim=d)
        idx.init_index(max_elements=len(train_idx), ef_construction=200, M=32)
        idx.add_items(Xtr, np.arange(len(train_idx)))
        idx.set_ef(max(m + 16, 64))

        dist2_nn, _ = idx.knn_query(Xtr, k=2)
        h = np.median(np.sqrt(dist2_nn[:, 1])) + 1e-9

        def potential(Q, drop_self):
            dist2, nn = idx.knn_query(Q.astype('float32'), k=m + (1 if drop_self else 0))
            if drop_self:
                dist2, nn = dist2[:, 1:], nn[:, 1:]
            w = np.exp(-dist2 / (2 * h * h))
            return w.mean(axis=1)

        phi_test = potential(X[test_idx], drop_self=False)
        phi_train = potential(Xtr, drop_self=True)
    else:

        nn_train = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X[train_idx])
        dist_nn, _ = nn_train.kneighbors(X[train_idx])
        h = np.median(dist_nn[:, 1]) + 1e-9

        def potential(Q, drop_self):
            nn_obj = NearestNeighbors(n_neighbors=m + (1 if drop_self else 0), metric='euclidean').fit(X[train_idx])
            dist, _ = nn_obj.kneighbors(Q)
            if drop_self:
                dist = dist[:, 1:]
            w = np.exp(-dist**2 / (2 * h * h))
            return w.mean(axis=1)

        phi_test = potential(X[test_idx], drop_self=False)
        phi_train = potential(X[train_idx], drop_self=True)

    U = mannwhitneyu(phi_train, phi_test, alternative='greater').statistic
    prob = U / (len(phi_train) * len(phi_test))
    tpl_value = 2 * prob - 1

    # Bootstrap CI (resample test potentials)
    boot_tpl = []
    for _ in range(n_bootstrap):
        phi_test_boot = np.random.choice(phi_test, size=len(phi_test), replace=True)
        U_boot = mannwhitneyu(phi_train, phi_test_boot, alternative='greater').statistic
        prob_boot = U_boot / (len(phi_train) * len(phi_test_boot))
        boot_tpl.append(2 * prob_boot - 1)
    boot_tpl = np.array(boot_tpl)
    tpl_ci_lower = np.percentile(boot_tpl, 2.5)
    tpl_ci_upper = np.percentile(boot_tpl, 97.5)

    return tpl_value, tpl_ci_lower, tpl_ci_upper, (phi_train, phi_test, h)

tpl_rows = []

for name, fold in {**baseline_results, **shield_results}.items():
    tpl_val, tpl_lo, tpl_hi, (phi_tr, phi_te, h) = tpl(X_scaled, fold)
    tpl_rows.append({
        'config': name,
        'tpl': tpl_val,
        'tpl_ci_lower': tpl_lo,
        'tpl_ci_upper': tpl_hi,
        'bandwidth_h': h,
        'mean_phi_train': phi_tr.mean(),
        'mean_phi_test': phi_te.mean(),
        'n_test_pts': len(phi_te)
    })

tpl_df = pd.DataFrame(tpl_rows)
save_table(tpl_df, 'train_potential_leakage', where='main')

print('Train Potential Leakage (TPL) Summary:')
print('  TPL  +1 : test in low-train-density regions (good OOD separation)')
print('  TPL   0 : test in same train-density field (random split leakage)')
print('  TPL  -1 : test buried in high train density (severe leakage)\n')

# Single bar chart
fig, ax = plt.subplots(figsize=(13, 5))
sub = tpl_df.set_index('config')['tpl'].sort_values()
colors = ['#2ecc71' if x > 0.1 else '#f39c12' if x > -0.1 else '#e74c3c' for x in sub.values]

err_lower = sub.values - tpl_df.set_index('config').loc[sub.index, 'tpl_ci_lower'].values
err_upper = tpl_df.set_index('config').loc[sub.index, 'tpl_ci_upper'].values - sub.values

bars = ax.bar(range(len(sub)), sub.values, yerr=np.array([err_lower, err_upper]),
              color=colors, edgecolor='black', linewidth=1.2, capsize=5, alpha=0.8)
ax.set_xticks(range(len(sub)))
ax.set_xticklabels(sub.index, rotation=45, ha='right', fontsize=9)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.axhline(y=0.1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Good separation threshold')
ax.axhline(y=-0.1, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Sever leakage threshold')
ax.set_ylabel('TPL  [mean Â± 95% CI]')
ax.set_title('Train Potential Leakage (higher = better OOD; density-based; error bar = bootstrap CI)')
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='best', fontsize=9)

plt.tight_layout()
save_fig(fig, 'train_potential_leakage', where='main')
plt.show()

print('Train Potential Leakage (TPL):')
print(tpl_df[['config', 'tpl', 'tpl_ci_lower', 'tpl_ci_upper', 'bandwidth_h', 
              'mean_phi_train', 'mean_phi_test']].to_string(index=False))
print()

tpl_df.head(10)


# ## 10. Convergence and gradient-alignment diagnostics

# In[31]:


# Cell 10.1 - gradient alignment cosine warm vs cold-both
focus_cfgs = ['HMD-SHIELD_warm', 'HMD-SHIELD_cold-both']
fig, axes = plt.subplots(1, len(focus_cfgs), figsize=(6 * len(focus_cfgs), 4), sharey=True)
for ax, cfg in zip(axes, focus_cfgs):
    h = shield_diagnostics.get(cfg, {}).get('history', [])
    if not h:
        ax.set_title(f'{cfg} (no history)'); continue
    rows = []
    for hh in h:
        for k, v in hh.get('alignment', {}).items():
            rows.append({'iter': hh['iteration'],
                         'pair': k.replace('/', '<->'), 'cosine': v})
    align = pd.DataFrame(rows)
    for p, sub in align.groupby('pair'):
        ax.plot(sub['iter'], sub['cosine'], marker='o', label=p)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('iteration'); ax.set_title(cfg); ax.legend(fontsize=7)
axes[0].set_ylabel('cosine similarity')
plt.tight_layout()
save_fig(fig, 'gradient_alignment_warm_vs_cold-both', where='main')
plt.show()


# In[32]:


# Cell 10.2 - composite-objective trajectories
fig, ax = plt.subplots(figsize=(11, 5))
for cfg, diag in shield_diagnostics.items():
    h = diag.get('history', [])
    if not h: continue
    xs = [hh['iteration'] for hh in h]
    ys_ = [hh['composite_objective'] for hh in h]
    ax.plot(xs, ys_, marker='o', label=cfg, lw=1.0, alpha=0.85)
ax.set_xlabel('iteration'); ax.set_ylabel('composite objective')
ax.set_title('Composite objective per SHIELD configuration')
ax.legend(fontsize=7, loc='best', ncol=2)
plt.tight_layout()
save_fig(fig, 'composite_objective_trajectories', where='si')
plt.show()


# ## 11. Dimension-reduction visualizations of splits

# In[33]:


# Cell 11.1 - embeddings
embeds = {}
try:
    embeds['tSNE'] = TSNE(n_components=2, perplexity=30, init='pca',
                          random_state=RANDOM_SEED).fit_transform(X_scaled)
    print('t-SNE done')
except Exception as exc:
    print(f't-SNE failed: {exc}')

for emb_name, emb in embeds.items():
    np.save(OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.npy', emb)
    pd.DataFrame(emb, columns=[f'{emb_name}_1', f'{emb_name}_2']).to_csv(
        OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.csv', index=True, index_label='row_id'
    )
    print(f'Saved {emb_name} embedding: {emb.shape}')


# In[34]:


# Cell 11.2 - split-colored scatter panels (+ target-colored panel)
showcase = ['RANDOM', 'STRATIFIED', 'PC1_SCAFFOLD',  'KMEANS', 'AGGLOMERATIVE', 
            'HMD-SHIELD_warm', 'H-SHIELD_warm', 'M-SHIELD_warm', 'D-SHIELD_warm',
            'SHIELD_NO_H_warm', 'SHIELD_NO_M_warm', 'SHIELD_NO_D_warm', 
            'HMD-SHIELD_cold-left', 'HMD-SHIELD_cold-right', 'HMD-SHIELD_cold-both']
showcase = [s for s in showcase if s in baseline_results or s in shield_results]

for emb_name, emb in embeds.items():
    n_panels = len(showcase) + 1
    cols = min(4, n_panels)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, name in zip(axes[:len(showcase)], showcase):
        fold = baseline_results.get(name, shield_results.get(name))
        ax.scatter(emb[fold == 0, 0], emb[fold == 0, 1], s=8, c='#4477AA',
                   alpha=0.55, label='train')
        ax.scatter(emb[fold == 1, 0], emb[fold == 1, 1], s=8, c='#CC6677',
                   alpha=0.55, label='test')
        scaled = lpi_df.set_index('config').loc[name, 'scaled_L_pi'] \
                 if name in lpi_df['config'].values else float('nan')
        ax.set_title(f'{name}\nscaled L(pi)={scaled:.3f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc='best', fontsize=7)

    ax_tgt = axes[len(showcase)]
    sc = ax_tgt.scatter(emb[:, 0], emb[:, 1], s=8, c=y_RD,
                        cmap='viridis', alpha=0.7, edgecolors='none')
    plt.colorbar(sc, ax=ax_tgt, label=TARGET_COL, fraction=0.046, pad=0.04)
    ax_tgt.set_title(f'Colored by\n{TARGET_COL}', fontsize=9)
    ax_tgt.set_xticks([]); ax_tgt.set_yticks([])

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f'{emb_name} of joint feature space, colored by fold', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_splits',
             where='main' if emb_name in ('PCA','UMAP') else 'si')
    plt.show()


# ## 12. OOD Assessment Metrics

# In[35]:


# # Cell 12.1 - OOD characterization metrics per split

from sklearn.neighbors  import NearestNeighbors
from scipy.spatial      import ConvexHull, Delaunay
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
import matplotlib.ticker as mticker
import warnings

# t-SNE embeddings 
_tsne_2d = embeds.get('tSNE')

# Helpers 

def _nn_distances(X_train, X_test):
    nn = NearestNeighbors(n_neighbors=1, n_jobs=N_JOBS)
    nn.fit(X_train)
    dists, _ = nn.kneighbors(X_test)
    return dists[:, 0]


def _convex_hull_outside_rate(emb_train, emb_test):
    if emb_train.shape[0] < 4:
        return float('nan')
    try:
        hull  = ConvexHull(emb_train)
        delan = Delaunay(emb_train[hull.vertices])
        return float((delan.find_simplex(emb_test) < 0).mean())
    except Exception:
        return float('nan')


# Main loop 
all_splits = {f'baseline::{k}': v for k, v in baseline_results.items()}
all_splits.update({f'SHIELD::{k}': v for k, v in shield_results.items()})

ood_rows_161 = []

for sp_name_161, fold_161 in all_splits.items():
    mask_tr_161 = fold_161 == 0
    mask_te_161 = fold_161 == 1
    Xtr_161 = X_scaled[mask_tr_161]
    Xte_161 = X_scaled[mask_te_161]

    if len(Xtr_161) < 10 or len(Xte_161) < 5:
        continue

    row_161 = {'split': sp_name_161}

    try:
        nn_d_161 = _nn_distances(Xtr_161, Xte_161)
        row_161['nn_mean'] = float(nn_d_161.mean())
        row_161['nn_p95']  = float(np.percentile(nn_d_161, 95))
        row_161['nn_max']  = float(nn_d_161.max())
    except Exception as e:
        print(f'  [NN] {sp_name_161}: {e}')
        row_161['nn_mean'] = row_161['nn_p95'] = row_161['nn_max'] = float('nan')

    if _tsne_2d is not None:
        try:
            soap_d_161 = _nn_distances(_tsne_2d[mask_tr_161], _tsne_2d[mask_te_161])
            row_161['soap_dist_mean'] = float(soap_d_161.mean())
        except Exception as e:
            print(f'  [SOAP] {sp_name_161}: {e}')
            row_161['soap_dist_mean'] = float('nan')
    else:
        row_161['soap_dist_mean'] = float('nan')

    if _tsne_2d is not None:
        try:
            row_161['hull_outside_rate'] = _convex_hull_outside_rate(
                _tsne_2d[mask_tr_161], _tsne_2d[mask_te_161]
            )
        except Exception as e:
            print(f'  [Hull] {sp_name_161}: {e}')
            row_161['hull_outside_rate'] = float('nan')
    else:
        row_161['hull_outside_rate'] = float('nan')

    ood_rows_161.append(row_161)
    print(
        f'{sp_name_161:40s}  '
        f'NN_mean={row_161["nn_mean"]:.3f}  '
        f'NN_p95={row_161["nn_p95"]:.3f}  '
        f'NN_max={row_161["nn_max"]:.3f}  '
        f'SOAP={row_161["soap_dist_mean"]:.3f}  '
        f'Hull={row_161["hull_outside_rate"]:.3f}',
        flush=True
    )

ood_df_161 = pd.DataFrame(ood_rows_161).set_index('split')

# Save results table 
save_table(ood_df_161.reset_index(), 'ood_characterization_metrics', where='main')
display(ood_df_161.round(4))


# Individual bar figures
_metric_meta_161 = {
    'nn_mean':          ('Mean NN Distance (feature space)',     'L2 distance',      'higher → more OOD'),
    'nn_p95':           ('95th-pct NN Distance (feature space)', 'L2 distance',      'higher → tail extrapolation'),
    'nn_max':           ('Max NN Distance (feature space)',       'L2 distance',      'higher → extreme extrapolation'),
    'soap_dist_mean':   ('SOAP Distance (t-SNE 2-D)',            'Euclidean (2-D)',   'higher → structural novelty'),
    'hull_outside_rate':('Convex Hull Outside Rate (t-SNE 2-D)', 'fraction outside', 'higher → more extrapolation'),
}

for col_161, (title_161, ylabel_161, note_161) in _metric_meta_161.items():
    if col_161 not in ood_df_161.columns:
        continue
    vals_161 = ood_df_161[col_161].dropna()
    colors_161 = ['#CC6677' if 'SHIELD' in s else '#4477AA' for s in vals_161.index]
    fig_161, ax_161 = plt.subplots(figsize=(max(8, 0.55 * len(vals_161)), 4))
    ax_161.barh(vals_161.index, vals_161.values, color=colors_161,
                edgecolor='white', height=0.65)
    ax_161.set_xlabel(ylabel_161)
    ax_161.set_title(f'{title_161}\n({note_161})', fontsize=10)
    ax_161.axvline(vals_161.mean(), color='k', lw=1.2, ls='--',
                   label=f'mean={vals_161.mean():.3f}')
    ax_161.legend(handles=[
        Patch(color='#4477AA', label='baseline'),
        Patch(color='#CC6677', label='SHIELD'),
        Line2D([0], [0], color='k', lw=1.2, ls='--', label=f'mean={vals_161.mean():.3f}'),
    ], fontsize=8)
    plt.tight_layout()
    save_fig(fig_161, f'ood_{col_161}', where='main')
    plt.show()


# MASTER FIGURE 1 - Bubble map

fig_m1, ax_m1 = plt.subplots(figsize=(11, 7))

_x_m1  = ood_df_161['nn_mean'].fillna(0)
_y_m1  = ood_df_161['soap_dist_mean'].fillna(0)
_sz_m1 = ood_df_161['hull_outside_rate'].fillna(0)
_c_m1  = ood_df_161['nn_p95'].fillna(0)
_mx_m1 = ood_df_161['nn_max'].fillna(0)

_sz_norm_m1 = (_sz_m1 - _sz_m1.min()) / (_sz_m1.max() - _sz_m1.min() + 1e-12)
_sz_plot_m1 = 70 + 560 * _sz_norm_m1

sc_m1 = ax_m1.scatter(
    _x_m1, _y_m1,
    s=_sz_plot_m1, c=_c_m1, cmap='plasma',
    edgecolors='white', linewidths=0.9, alpha=0.88, zorder=3,
)
cbar_m1 = plt.colorbar(sc_m1, ax=ax_m1, pad=0.01)
cbar_m1.set_label('95th-pct NN Distance (feature space)', fontsize=9)

for sp_161 in ood_df_161.index:
    ax_m1.annotate(
        f'{sp_161}\nmax={_mx_m1[sp_161]:.2f}',
        xy=(_x_m1[sp_161], _y_m1[sp_161]),
        fontsize=6.2, ha='center', va='bottom',
        xytext=(0, 7), textcoords='offset points', color='#1a1a1a',
    )

ax_m1.set_xlabel('Mean NN Distance  (feature space)  →  more OOD', fontsize=10)
ax_m1.set_ylabel('SOAP Distance  (t-SNE 2-D)  →  structural novelty', fontsize=10)

# Bubble size legend
for frac_leg, lbl_leg in [(0.0, '0 % outside hull'), (0.5, '50 %'), (1.0, '100 %')]:
    ax_m1.scatter([], [], s=70 + 560 * frac_leg, c='gray', alpha=0.5, label=lbl_leg)

ax_m1.set_title(
    'OOD Characterization Map\n'
    'x = Mean NN Dist  ·  y = SOAP Dist  ·  size = Hull Outside Rate  '
    '·  color = p95 NN Dist  ·  label = Max NN Dist',
    fontsize=9.5,
)
ax_m1.legend(title='Hull Outside Rate', fontsize=7, loc='upper left', framealpha=0.85)
plt.tight_layout()
save_fig(fig_m1, 'ood_master_bubble_map', where='main')
plt.show()


# MASTER FIGURE 2 - Parallel coordinates

_pc_cols  = ['nn_mean', 'nn_p95', 'nn_max', 'soap_dist_mean']
_pc_labels = ['Mean NN\n(feat.)', '95th-pct NN\n(feat.)', 'Max NN\n(feat.)', 'SOAP Dist\n(t-SNE)']

_pc_df = ood_df_161[_pc_cols].dropna()

_pc_norm = (_pc_df - _pc_df.min()) / (_pc_df.max() - _pc_df.min() + 1e-12)

_hull_vals = ood_df_161.loc[_pc_norm.index, 'hull_outside_rate'].fillna(0)
_hull_norm = (_hull_vals - _hull_vals.min()) / (_hull_vals.max() - _hull_vals.min() + 1e-12)
_lw_map    = 0.9 + 2.6 * _hull_norm

_palette_161 = {
    'baseline': '#4477AA',
    'SHIELD':   '#CC6677',
}
_baseline_splits = [s for s in _pc_norm.index if 'SHIELD' not in s]
_shield_splits   = [s for s in _pc_norm.index if 'SHIELD'     in s]

_blues  = plt.cm.Blues(np.linspace(0.45, 0.90, max(1, len(_baseline_splits))))
_reds   = plt.cm.Reds (np.linspace(0.45, 0.90, max(1, len(_shield_splits))))
_color_map_161 = {}
for i, s in enumerate(_baseline_splits): _color_map_161[s] = _blues[i]
for i, s in enumerate(_shield_splits):   _color_map_161[s] = _reds[i]

n_axes_161  = len(_pc_cols)
x_pos_161   = np.arange(n_axes_161)

fig_m2, axes_m2 = plt.subplots(
    1, n_axes_161 - 1,
    figsize=(13, 6),
    sharey=False,
)
fig_m2.subplots_adjust(wspace=0, left=0.06, right=0.82, top=0.88, bottom=0.12)

for sp_161 in _pc_norm.index:
    vals_sp = _pc_norm.loc[sp_161].values
    col_sp  = _color_map_161[sp_161]
    lw_sp   = float(_lw_map[sp_161])
    for seg in range(n_axes_161 - 1):
        axes_m2[seg].plot(
            [0, 1], [vals_sp[seg], vals_sp[seg + 1]],
            color=col_sp, lw=lw_sp, alpha=0.82, solid_capstyle='round',
        )
        axes_m2[seg].set_xlim(0, 1)
        axes_m2[seg].set_ylim(-0.05, 1.05)

for seg, ax_seg in enumerate(axes_m2):
    ax_seg.spines['top'].set_visible(False)
    ax_seg.spines['bottom'].set_visible(False)
    ax_seg.spines['right'].set_visible(False)
    ax_seg.spines['left'].set_linewidth(1.5)
    ax_seg.spines['left'].set_color('#555555')
    ax_seg.set_xticks([])
    ax_seg.yaxis.set_major_locator(mticker.LinearLocator(6))
    ax_seg.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    orig_min = _pc_df[_pc_cols[seg]].min()
    orig_max = _pc_df[_pc_cols[seg]].max()
    ax_seg.set_yticklabels(
        [f'{orig_min + t * (orig_max - orig_min):.3f}'
         for t in np.linspace(0, 1, 6)],
        fontsize=7,
    )
    ax_seg.set_xlabel(_pc_labels[seg], fontsize=9, labelpad=6)
    if seg > 0:
        ax_seg.yaxis.set_visible(False)

ax_last = axes_m2[-1].twinx()
ax_last.set_ylim(-0.05, 1.05)
ax_last.yaxis.set_major_locator(mticker.LinearLocator(6))
orig_min_last = _pc_df[_pc_cols[-1]].min()
orig_max_last = _pc_df[_pc_cols[-1]].max()
ax_last.set_yticklabels(
    [f'{orig_min_last + t * (orig_max_last - orig_min_last):.3f}'
     for t in np.linspace(0, 1, 6)],
    fontsize=7,
)
ax_last.set_xlabel(_pc_labels[-1], fontsize=9, labelpad=6)
ax_last.spines['top'].set_visible(False)
ax_last.spines['bottom'].set_visible(False)
ax_last.spines['left'].set_visible(False)

legend_handles = []
for s in _baseline_splits:
    legend_handles.append(Line2D([0], [0], color=_color_map_161[s], lw=1.8, label=s))
for s in _shield_splits:
    legend_handles.append(Line2D([0], [0], color=_color_map_161[s], lw=1.8, label=s))
legend_handles += [
    Line2D([0], [0], color='gray', lw=0.9, label='thin  = low hull-outside rate'),
    Line2D([0], [0], color='gray', lw=3.5, label='thick = high hull-outside rate'),
]
fig_m2.legend(
    handles=legend_handles,
    loc='center right', bbox_to_anchor=(1.0, 0.5),
    fontsize=7.5, framealpha=0.9, title='Split  (line width = Hull rate)',
    title_fontsize=8,
)

fig_m2.suptitle(
    'Distance Metric Profiles Across Splits  '
    '(axes: normalized to [0,1];  tick labels: original units)',
    fontsize=10.5, y=0.96,
)

save_fig(fig_m2, 'ood_master_parallel_coords', where='main')
plt.show()


# In[36]:


# Cell 12.2 - Export all splits to a single combined CSV

_splits_df = df.copy()

for split_name, fold_arr in all_splits.items():
    col = 'fold__' + split_name.replace('::', '__').replace(' ', '_')
    _splits_df[col] = fold_arr

_fold_cols = [c for c in _splits_df.columns if c.startswith('fold__')]
print(f'Splits embedded: {len(_fold_cols)}')
for c in _fold_cols:
    counts = _splits_df[c].value_counts().sort_index().to_dict()
    print(f'  {c:55s}  {counts}')

# Save
_out_path = OUT_INTER / 'splits' / 'all_splits_combined.csv'
_splits_df.to_csv(_out_path, index=True, index_label='row_id')
print(f'\nSaved: {_out_path}  (shape: {_splits_df.shape})')


# ## 13. SHIELD (alpha, beta, gamma) sweep, Pareto, Hamming stability

# In[37]:


# Cell 13.1 - sweep on warm mode
sweep_kwargs = dict(
    n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hierarchy_main, affinity=AM,
    affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE, kernel_matrix=K_mat,
    y_target=Y_TARGET,
    stratification_strategy=STRAT_STRATEGY,
    class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
    two_entity_mode='warm',
    eta=ETA, mu=MU, nu=NU, tau=TAU, 
    max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
    integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
    rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
    stage0_samples=SHIELD_STAGE0_SAMPLES,
    precomputed_normalizers=CACHED_NORMALIZERS,
    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
)
print(f'Running shield.sweep over '
      f'{len(SWEEP_GRID["alpha"])*len(SWEEP_GRID["beta"])*len(SWEEP_GRID["gamma"])} configs...')
t0 = time.time()
sweep_res = shield_sweep(
    alpha_grid=SWEEP_GRID['alpha'], beta_grid=SWEEP_GRID['beta'],
    gamma_grid=SWEEP_GRID['gamma'],
    metric_keys=('M_normalized', 'D_normalized', 'H_normalized'),
    aggregation='min_of_normalized',
    instability_threshold=HAMMING_STABILITY_THRESHOLD,
    **sweep_kwargs,
)
print(f'Sweep complete in {time.time()-t0:.1f}s; recommended '
      f'(alpha,beta,gamma)={sweep_res.grid_points[sweep_res.recommended_index]}')


# In[38]:


# Cell 13.2 - sweep table + plot
sweep_rows = []
for gp, r in zip(sweep_res.grid_points, sweep_res.results):
    last = r.history[-1]['terms'] if r.history else {}
    sweep_rows.append({
        'alpha': gp[0], 'beta': gp[1], 'gamma': gp[2],
        'H_normalized': last.get('H_normalized'),
        'M_normalized': last.get('M_normalized'),
        'D_normalized': last.get('D_normalized'),
        'strat_normalized': last.get('strat_normalized'),
        'composite_objective': r.history[-1]['composite_objective'] if r.history else None,
    })
sweep_df = pd.DataFrame(sweep_rows)
sweep_df['is_pareto'] = sweep_df.index.isin(sweep_res.pareto_indices)
sweep_df['is_recommended'] = (sweep_df.index == sweep_res.recommended_index)
save_table(sweep_df, 'shield_sweep_grid', where='si')

fig = plt.figure(figsize=(13, 4))
ax1 = fig.add_subplot(1, 3, 1); ax2 = fig.add_subplot(1, 3, 2); ax3 = fig.add_subplot(1, 3, 3)
sc = ax1.scatter(sweep_df['M_normalized'], sweep_df['D_normalized'],
                 c=sweep_df['H_normalized'], cmap='viridis', s=70,
                 edgecolors='black', linewidths=0.5)
plt.colorbar(sc, ax=ax1, label='H_norm')
par = sweep_df[sweep_df['is_pareto']]
ax1.scatter(par['M_normalized'], par['D_normalized'],
            s=180, facecolors='none', edgecolors='red', linewidths=1.5, label='Pareto')
rec = sweep_df.loc[sweep_res.recommended_index]
ax1.scatter([rec['M_normalized']], [rec['D_normalized']],
            s=300, marker='*', color='gold', edgecolors='black', linewidths=1, label='recommended')
ax1.set_xlabel('M_norm'); ax1.set_ylabel('D_norm'); ax1.set_title('Sweep: M vs D')
ax1.legend(fontsize=8)

ax2.scatter(range(len(sweep_df)), sweep_df['composite_objective'],
            c=['gold' if x == sweep_res.recommended_index else '#4477AA'
               for x in sweep_df.index], s=60, edgecolors='black')
ax2.set_xlabel('config idx'); ax2.set_ylabel('composite objective')
ax2.set_title('Composite across sweep')

sns.heatmap(sweep_res.stability.pairwise_hamming, ax=ax3, cmap='magma',
            cbar_kws={'label': 'frac differing atoms'})
ax3.set_title(f'Hamming ({len(sweep_res.stability.assignments)} points)')
plt.tight_layout()
save_fig(fig, 'sweep_pareto_hamming', where='si')
plt.show()


# ## 14. Hierarchy lambda ablation (alloy identity / family / solidification class)

# In[39]:


# Cell 14.1 - run SHIELD warm with each lambda variant
lambda_results = {}
for var_name, lam in LAMBDA_VARIANTS_SI.items():
    hier = build_hierarchy(df, lam)
    print(f'  lambda variant={var_name}, lambdas={lam}')
    t0 = time.time()
    try:
        res = shield_split(
            n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
            hierarchy=hier, affinity=AM, affinity_provenance='label-blind-handcrafted',
            d_mode=DIST_MODE, kernel_matrix=K_mat,
            y_target=Y_TARGET, stratification_strategy=STRAT_STRATEGY,
            class_delta=STRAT_TOL, balance_eps=BALANCE_TOL, two_entity_mode='warm',
            alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
            max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
            integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
            rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
            stage0_samples=SHIELD_STAGE0_SAMPLES,
            precomputed_normalizers=CACHED_NORMALIZERS,
            sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
        )
        lambda_results[var_name] = res.fold_assignment.cpu().numpy().astype(int)
        print(f'    wall={time.time()-t0:.1f}s')
    except Exception as exc:
        print(f'    FAILED: {exc}')

rows = []
for name, fold in lambda_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    rows.append({'variant': name, 'scaled_L_pi': cross_fold_similarity(fold)[1],
                 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
lam_df = pd.DataFrame(rows)
save_table(lam_df, 'lambda_variant_leakage', where='si')
lam_df


# ## 15. Can the model identify the printer machine from process parameters alone?

# In[40]:


# Cell 15.1 - classify machine ID from process parameters

mach_ids = machine_entity_of
mach_counts = pd.Series(mach_ids).value_counts()
big_machines = mach_counts[mach_counts >= 8].index
mask = pd.Series(mach_ids).isin(big_machines).values
if mask.sum() >= 50 and len(big_machines) >= 2:
    Xc = X_scaled[mask]
    yc = mach_ids[mask]
    clf = RandomForestClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, Xc, yc, cv=3, scoring='accuracy', n_jobs=N_JOBS)
    print(f'Machine-ID identifiability (3-fold CV accuracy, {len(big_machines)} '
          f'machines >= 8 samples): {scores.mean():.3f} +- {scores.std():.3f}')
    print('=> A high value means process parameters fingerprint the machine, '
          'so a random split lets the model memorize machine-specific parameters such as laser beam diameter.')

    yc_pred = cross_val_predict(clf, Xc, yc, cv=3, n_jobs=N_JOBS)

    mach_names = pd.factorize(df[MACHINE_COL])[1].tolist()

    confusion = pd.crosstab(pd.Series(yc, name='true').map(lambda i: mach_names[i]),
                            pd.Series(yc_pred, name='pred').map(lambda i: mach_names[i]))
    save_table(confusion.reset_index(), 'machine_id_confusion_matrix', where='si')
else:
    print('Insufficient per-machine support for identifiability test.')