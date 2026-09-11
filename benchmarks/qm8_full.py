# # SHIELD benchmark for QM8 (no coarsening)

# ## 1. Setup, imports, configuration, output directories

# In[1]:

# Prevent BLAS / OpenMP / MKL thread explosion
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# In[2]:

# Cell 1.1 - imports (standard + scientific)
import os, sys, json, time, math, hashlib, pickle, warnings, importlib, platform
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sps
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import cdist

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold

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

print('SHIELD version:', __import__('shield').__version__)
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, 'PyTorch:', torch.__version__)


# In[3]:


# Cell 1.2 - global configuration (every user-tunable knob)
# Dataset
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'qm8.csv')
DATASET_NAME = 'qm8'
SMILES_COL   = 'smiles'
TARGET_COLS_RAW = [
    'E1-CC2','E2-CC2','f1-CC2','f2-CC2',
    'E1-PBE0','E2-PBE0','f1-PBE0','f2-PBE0',
    'E1-PBE0.1','E2-PBE0.1','f1-PBE0.1','f2-PBE0.1',
    'E1-CAM','E2-CAM','f1-CAM','f2-CAM',
]
PRIMARY_TARGET = 'E1-CC2'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 42

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.05

# Fingerprints (shared by M term and D kernel)
ECFP_RADIUS  = 2
ECFP_NBITS   = 1024
AFFINITY_TYPES = ['tanimoto', 'morgan_cosine']
PRIMARY_AFFINITY = 'tanimoto'

# M term — epsilon-NN Tanimoto affinity
EPS_AFF_M    = 0.48
EPS_AFF_M_SI = [0.4, 0.425, 0.45, 0.475, 0.5]
T2_WEIGHT    = 0.25

# H term — two-level chromophore hierarchy
LAMBDA_MAIN = [1.0, 0.30]

# SHIELD coefficients (main configuration)
ALPHA, BETA, GAMMA, ETA, MU, NU, TAU = 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0

# D term — MMD shift in Tanimoto PSD kernel space
DIST_MODE = 'shift'

STRAT_TOL   = 0.10
# OT regularization (used only in D-match comparison cell)
OT_REG    = 0.10
# Stratification (MU=0 in main config)
STRAT_STRATEGY  = 'sliced_wasserstein'
SW_PROJECTIONS  = 100
SW_QUANTILE_PTS = 256

# kNN purity neighbourhood sizes (diagnostic only)
KNN_K = [5, 10, 25]

# SHIELD solver controls
SHIELD_MAX_ITER     = 15
SHIELD_STAGE0_SAMPLES  = 100
SHIELD_INIT_BACKEND = 'auto'
SHIELD_ROUNDING     = 'pipage'
SHIELD_MILP_SIZE    = 10_000
SHIELD_DEVICE       = 'cpu'
N_JOBS              = -1

# ML evaluation
ML_RUN = True

print(f'Config loaded. MAX_ROWS={MAX_ROWS}, K={N_FOLDS}, '
      f'DIST_MODE={DIST_MODE}, ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}, '
      f'EPS_AFF_M={EPS_AFF_M}, LAMBDA_MAIN={LAMBDA_MAIN}')


# In[4]:


# Cell 1.3 - output directories and figure-saving utility
OUT_DIR     = _HERE / 'outputs'
OUT_MAIN    = OUT_DIR / 'main_text'
OUT_SI      = OUT_DIR / 'supplementary'
OUT_INTER   = OUT_DIR / 'intermediate'
OUT_META    = OUT_DIR / 'metadata'

for d in [OUT_MAIN, OUT_SI, OUT_INTER, OUT_META]:
    for sub in ['figures', 'tables', 'data']:
        (d / sub).mkdir(parents=True, exist_ok=True)
for d in ['featurization', 'splits', 'shield_runs']:
    (OUT_INTER / d).mkdir(parents=True, exist_ok=True)

# Matplotlib defaults — Nature-friendly
sns.set_style('whitegrid')
mpl.rcParams.update({
    'figure.dpi':      300,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'font.family':     'sans-serif',
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'pdf.fonttype':    42,
    'svg.fonttype':    'none',
})

def save_fig(fig, name, where='main', subdir='figures', formats=('png','pdf')):
    """Save figure as PNG/PDF to the appropriate output folder ('main' or 'si')."""
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


# In[5]:


# Cell 1.4 - environment / optional-dependency check
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
    'rdkit':        _try('rdkit'),
    'xgboost':      _try('xgboost'),
    'umap':         _try('umap', 'umap'),
    'alphashape':   _try('alphashape'),
    'shapely':      _try('shapely'),
    'chemprop':     _try('chemprop'),
    'dgllife':      _try('dgllife'),
    'astartes':     _try('astartes'),
    'matminer':     _try('matminer'),
    'descriptastorus': _try('descriptastorus'),
}
for k, v in _versions.items():
    print(f'  {k:18s}  {v if v else "MISSING (optional)"}')

with open(OUT_META / 'environment.json', 'w') as fh:
    json.dump({
        'python':   sys.version,
        'platform': platform.platform(),
        'versions': _versions,
        'shield_version': __import__('shield').__version__,
    }, fh, indent=2)

if _OPT['rdkit'] is None:
    raise RuntimeError('RDKit is required for QM8 featurization. Install via conda-forge.')


# ## 2. Data loading and EDA

# In[6]:


# Cell 2.1 - load QM8 dataset
print(f'Loading from {DATASET_PATH}')
df_raw = pd.read_csv(DATASET_PATH)
TARGET_COLS = [c for c in TARGET_COLS_RAW if c in df_raw.columns]
print(f'Raw rows: {len(df_raw):,};  detected {len(TARGET_COLS)} target columns')

if MAX_ROWS is not None and len(df_raw) > MAX_ROWS:
    df = df_raw.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df):,} rows (seed={RANDOM_SEED})')
else:
    df = df_raw.copy()

n = len(df)
df.head()


# In[7]:


# Cell 2.2 - attach DataSAIL splits and filter dataset to matched molecules

from rdkit import Chem as _Chem
from rdkit.Chem.inchi import MolToInchi as _MolToInchi
import warnings as _warnings
_warnings.filterwarnings('ignore', category=UserWarning)

DATASAIL_PATH = str(_REPO_ROOT / 'splits' / 'qm8_DataSAIL.csv')

PRECOMPUTED_FOLD_MAP = {
    'train': 0,  # ~80 %  -> training fold
    'val':   1,  # ~10 %  -> holdout  (merged with test to give 80/20)
    'test':  1,  # ~10 %  -> holdout
}

def _conn_key(s):
    """Connectivity-only InChI match key (formula + bond topology only)."""
    m = _Chem.MolFromSmiles(s)
    if m is None: return None
    inc = _MolToInchi(m, options='-SNon')
    if inc is None: return None
    parts = inc.split('/')
    kept = [parts[0], parts[1]]
    for p in parts[2:]:
        if p.startswith('c'):
            kept.append(p)
            break
    return '/'.join(kept)

df_ds = pd.read_csv(DATASAIL_PATH)
_ds_keys = df_ds['smiles'].apply(_conn_key)
_ds_valid = _ds_keys.notna()
df_ds = df_ds[_ds_valid].copy()
df_ds['_ck'] = _ds_keys[_ds_valid].values
_ds_map = {row['_ck']: (row['split'], row['datasail']) for _, row in df_ds.iterrows()}

_src_keys = df[SMILES_COL].apply(_conn_key)
_matched  = _src_keys.apply(lambda k: k in _ds_map if k is not None else False)
n_dropped = int((~_matched).sum())
print(f'DataSAIL join: {_matched.sum():,} matched, {n_dropped} dropped (absent from DataSAIL)')

df          = df[_matched].copy().reset_index(drop=True)
_ck_vals    = _src_keys[_matched].values

df['_split_base']     = [_ds_map[k][0] for k in _ck_vals]
df['_split_datasail'] = [_ds_map[k][1] for k in _ck_vals]

n = len(df)
print(f'Working dataset after filter: {n:,} molecules')
print(f"  _split_base     : {pd.Series(df['_split_base']).value_counts().sort_index().to_dict()}")
print(f"  _split_datasail : {pd.Series(df['_split_datasail']).value_counts().sort_index().to_dict()}")


# In[8]:


# Cell 2.2 - target distribution panel 
fig, axes = plt.subplots(4, 4, figsize=(12, 10))
for ax, col in zip(axes.flat, TARGET_COLS):
    ax.hist(df[col].dropna(), bins=40, color='#4477AA', edgecolor='black', alpha=0.85)
    ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=8)
fig.suptitle('QM8 target distributions (n=%d)' % n)
plt.tight_layout()
save_fig(fig, 'eda_target_distributions', where='si')
plt.show()


# In[9]:


# Cell 2.3 - target correlation heatmap
corr = df[TARGET_COLS].corr().values
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(TARGET_COLS))); ax.set_xticklabels(TARGET_COLS, rotation=80, fontsize=8)
ax.set_yticks(range(len(TARGET_COLS))); ax.set_yticklabels(TARGET_COLS, fontsize=8)
ax.set_title('Target-target correlation')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
save_fig(fig, 'eda_target_correlation', where='si')
save_array(corr, 'target_correlation', where='si')
plt.show()


# ## 3. Featurization

# In[10]:


# Cell 3.1 - compute ECFP4 fingerprints and bit vectors
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

CACHE_FP = OUT_INTER / 'featurization' / f'fps_n{n}_r{ECFP_RADIUS}_b{ECFP_NBITS}.pkl'

if CACHE_FP.exists():
    with open(CACHE_FP, 'rb') as fh:
        cache_obj = pickle.load(fh)
    bv  = cache_obj['bv']
    fps = cache_obj['fps']
    print(f'Loaded cached fingerprints from {CACHE_FP.name}')
else:
    fps = []
    for s in df[SMILES_COL]:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, ECFP_RADIUS, nBits=ECFP_NBITS) if m else None)
    valid = np.array([fp is not None for fp in fps])
    if not valid.all():
        df = df[valid].reset_index(drop=True)
        fps = [f for f in fps if f is not None]
        n = len(df)
        print(f'Dropped {(~valid).sum()} invalid SMILES; new n={n}')
    bv = np.zeros((n, ECFP_NBITS), dtype=np.uint8)
    arr_buf = np.zeros((ECFP_NBITS,), dtype=np.int8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr_buf)
        bv[i] = arr_buf.astype(np.uint8)
    with open(CACHE_FP, 'wb') as fh:
        pickle.dump({'bv': bv, 'fps': fps}, fh)
    print(f'Computed and cached fingerprints: shape={bv.shape}')

print(f'Mean bits set per molecule: {bv.sum(axis=1).mean():.1f}')


# In[11]:


# Cell 3.2 - Murcko scaffolds, ring systems, atom-count bins
mols  = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]
scaffolds = []
rings     = []
for m in mols:
    if m is None:
        scaffolds.append(''); rings.append('acyclic'); continue
    sc = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(m), includeChirality=False)
    scaffolds.append(sc if sc else 'acyclic')
    msc = Chem.MolFromSmiles(sc) if sc else None
    rings.append(sc if (msc is not None and msc.GetRingInfo().NumRings() > 0) else 'acyclic')

natoms = np.array([m.GetNumHeavyAtoms() if m else 0 for m in mols])
atom_bin = pd.cut(natoms, bins=[-0.5, 4.5, 7.5, np.inf],
                  labels=['ha<=4', 'ha5-7', 'ha>=8']).astype(str)

df['_scaffold'] = scaffolds
df['_ring']     = rings
df['_atombin']  = atom_bin

print(f'Distinct scaffolds : {df["_scaffold"].nunique()}')
print(f'Distinct rings     : {df["_ring"].nunique()}')
print(f'Atom-count bins    : {df["_atombin"].value_counts().to_dict()}')


# In[12]:


# Cell 3.3 - M term: epsilon-NN Tanimoto affinity + Tier-2 singleton fallback

def _eps_sparsify(A, eps):
    out = np.where(A > eps, A, 0.0).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out

affinities = {}
for at in AFFINITY_TYPES:
    cache = OUT_INTER / 'featurization' / f'aff_{at}_n{n}.npy'
    if cache.exists():
        A_full = np.load(cache)
        print(f'  loaded cached affinity ({at}) shape={A_full.shape}')
    elif at == 'tanimoto':
        A_full = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            A_full[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        np.fill_diagonal(A_full, 0.0)
        np.save(cache, A_full)
    else:  # morgan_cosine
        A_full = cosine_similarity(bv.astype(np.float32)).astype(np.float32)
        np.fill_diagonal(A_full, 0.0)
        np.save(cache, A_full)
    affinities[at] = {
        'full': A_full,
        **{f'eps_{eps:.2f}': _eps_sparsify(A_full, eps) for eps in EPS_AFF_M_SI},
    }

# Primary full Tanimoto matrix (used for K_MATRIX and diagnostics)
A_full = affinities[PRIMARY_AFFINITY]['full']

# Tier-1: epsilon-NN sparsification at primary threshold
AM_tier1 = _eps_sparsify(A_full, EPS_AFF_M)

# Tier-2: singleton fallback grouped by (n_heavy_atoms, n_pi_bonds)
_n_pi = []
for m in mols:
    if m is None:
        _n_pi.append(0)
        continue
    _n_pi.append(sum(1 for b in m.GetBonds() if b.GetBondTypeAsDouble() >= 2))
n_pi_bonds  = np.array(_n_pi, dtype=int)
natoms_arr  = np.array([m.GetNumHeavyAtoms() if m else 0 for m in mols], dtype=int)
t2_key      = list(zip(natoms_arr.tolist(), n_pi_bonds.tolist()))

AM = AM_tier1.copy()
tier1_degree  = (AM_tier1 > 0).sum(axis=1)
singleton_idx = np.where(tier1_degree == 0)[0]

_t2_groups = {}
for idx in singleton_idx:
    _t2_groups.setdefault(t2_key[idx], []).append(int(idx))

for k, grp in _t2_groups.items():
    for i in range(len(grp)):
        for j in range(i + 1, len(grp)):
            ii, jj = grp[i], grp[j]
            AM[ii, jj] = max(AM[ii, jj], T2_WEIGHT)
            AM[jj, ii] = max(AM[jj, ii], T2_WEIGHT)

am_t2_info = {
    'n_singletons_tier1': int(len(singleton_idx)),
    'n_tier2_groups':     int(len(_t2_groups)),
    'tier2_group_sizes':  sorted([len(g) for g in _t2_groups.values()], reverse=True)[:20],
    'eps_used':           EPS_AFF_M,
    'tier2_weight':       T2_WEIGHT,
}
np.save(OUT_INTER / 'featurization' / 'AM.npy', AM)
print(f'M affinity (eps={EPS_AFF_M}):')
print(f'  Tier-1 density   : {(AM_tier1 > 0).sum() / AM_tier1.size:.5f}')
print(f'  Tier-1 singletons: {am_t2_info["n_singletons_tier1"]} / {n}')
print(f'  Tier-2 groups    : {am_t2_info["n_tier2_groups"]}')
print(f'  Tier-2 sizes     : {am_t2_info["tier2_group_sizes"][:10]}')
print(f'  AM density       : {(AM > 0).sum() / AM.size:.5f}')


# In[13]:


# Cell 3.4 - cost matrix for D-match
C = (1.0 - A_full).astype(np.float32)
C = np.clip((C + C.T) / 2.0, 0.0, 1.0)
np.fill_diagonal(C, 0.0)
np.save(OUT_INTER / 'featurization' / 'cost_matrix.npy', C)
print(f'Cost matrix: shape={C.shape}, mean={C.mean():.4f}, '
      f'symmetric? {np.allclose(C, C.T)}')


# In[14]:


# Cell 3.5 - Tanimoto PSD kernel for D-shift (label-independent, unit diagonal)

K_MATRIX = A_full.astype(np.float32).copy()
np.fill_diagonal(K_MATRIX, 1.0)
np.save(OUT_INTER / 'featurization' / 'kernel_matrix.npy', K_MATRIX)
print(f'D-shift kernel K_MATRIX (Tanimoto PSD):')
print(f'  shape          : {K_MATRIX.shape}')
print(f'  diag mean      : {K_MATRIX.diagonal().mean():.4f}  (should be 1.0)')
print(f'  off-diag mean  : {(K_MATRIX.sum() - K_MATRIX.trace()) / (n * (n - 1)):.4f}')
print(f'  symmetric?     : {np.allclose(K_MATRIX, K_MATRIX.T)}')
print(f'  min off-diag   : {K_MATRIX[~np.eye(n, dtype=bool)].min():.4f}')


# ## 4. Hierarchy construction

# In[15]:


# Cell 4.1 - H term: two-level SMARTS chromophore hierarchy (QM8-specific)

from rdkit.Chem import MolFromSmarts

# L2 chromophore groups
CHR_SMARTS = {
    # 1. Nitro
    'nitro': [
        '[NX3](=O)=O',
        '[N+](=O)[O-]',
    ],
    # 2. Nitrile
    'nitrile': [
        '[CX2]#[NX1]',
    ],
    # 3. Conjugated carbonyl
    'carbonyl_conj': [
        '[#6](=[OX1])[CX3;!a]=[CX3;!a]',
        '[#6]=[#6][CX3](=[OX1])',
        '[#6](=[OX1])[CX2]#[CX2]',
    ],
    # 4. Aldehyde
    'aldehyde': [
        '[CX3H1](=O)',
    ],
    # 5. Ketone / imine
    'ketone_imine': [
        '[#6][CX3](=O)[#6]',
        '[#6]=[NX2;!a][#6]',
        '[CX3H0](=O)[OX2H0]',
    ],
    # 6. Heteroaromatic
    'heteroaromatic': [
        '[n,o,s]',
    ],
    # 7. Benzene / carbocyclic aromatic
    'benzene': [
        'c1ccccc1',
        'c1cccc2ccccc12',
    ],
    # 8. Conjugated diene / triene
    'diene_triene': [
        '[#6]=[#6]-[#6]=[#6]',
        '[#6]=[#6]-[#6]=[#6]-[#6]=[#6]',
    ],
    # 9. Alkene
    'alkene': [
        '[CX3]=[CX3]',
        '[CX2]=[CX3]',
        '[CX3]=[CX2]',
    ],
    # Aliphatic sub-groups (priority N > O > S > F)
    # 10. Nitrogen aliphatic
    'nitrogen_aliphatic': [
        '[NX3;!a]',
    ],
    # 11. Oxygen aliphatic
    'oxygen_aliphatic': [
        '[OX2;!a]',
    ],
    # 12. Sulfur aliphatic
    'sulfur_aliphatic': [
        '[S;!a]',
    ],
    # 13. Fluorinated aliphatic
    'fluorinated': [
        '[F]',
    ],
    # 14. Aliphatic
    'aliphatic': [
        '[#6,#7,#8,#16,#9]',
    ],
}

CHR_GROUP_ORDER = [
    'nitro', 'nitrile', 'carbonyl_conj', 'aldehyde', 'ketone_imine',
    'heteroaromatic', 'benzene', 'diene_triene', 'alkene',
    'nitrogen_aliphatic', 'oxygen_aliphatic', 'sulfur_aliphatic', 'fluorinated',
    'aliphatic',
]

# L1 transition_type mega-groups
L1_MAP = {
    'nitro':               'charge_transfer',
    'nitrile':             'charge_transfer',
    'carbonyl_conj':       'n_pi_star',
    'aldehyde':            'n_pi_star',
    'ketone_imine':        'n_pi_star',
    'heteroaromatic':      'pi_pi_star',
    'benzene':             'pi_pi_star',
    'diene_triene':        'pi_pi_star',
    'alkene':              'pi_pi_star',
    'nitrogen_aliphatic':  'lone_pair_sigma',
    'oxygen_aliphatic':    'lone_pair_sigma',
    'sulfur_aliphatic':    'lone_pair_sigma',
    'fluorinated':         'aliphatic',
    'aliphatic':           'aliphatic',
}
L1_GROUPS = ['charge_transfer', 'n_pi_star', 'pi_pi_star', 'lone_pair_sigma', 'aliphatic']

# ── compile SMARTS patterns ──────────────────────────────────────────────────
_chr_compiled = {}
for grp in CHR_GROUP_ORDER:
    pats = []
    for sma in CHR_SMARTS[grp]:
        p = MolFromSmarts(sma)
        if p is not None:
            pats.append(p)
        else:
            print(f'  WARNING: invalid SMARTS for group {grp}: {sma}')
    _chr_compiled[grp] = pats

# ── assign L2 and L1 to every molecule ──────────────────────────────────────
def _assign_chromophore(mol):
    if mol is None:
        return 'aliphatic'
    for grp in CHR_GROUP_ORDER:
        if any(mol.HasSubstructMatch(p) for p in _chr_compiled[grp]):
            return grp
    return 'aliphatic'

chromophore_L2 = [_assign_chromophore(m) for m in mols]
chromophore_L1 = [L1_MAP[g] for g in chromophore_L2]
df['_chromophore_L2'] = chromophore_L2
df['_chromophore_L1'] = chromophore_L1

print('L2 chromophore_type distribution:')
for g in CHR_GROUP_ORDER:
    cnt = chromophore_L2.count(g)
    if cnt > 0:
        print(f'  {g:20s}: {cnt:4d} ({100*cnt/n:.1f}%)')

print('\nL1 transition_type distribution:')
for g in L1_GROUPS:
    cnt = chromophore_L1.count(g)
    print(f'  {g:20s}: {cnt:4d} ({100*cnt/n:.1f}%)')


# ── build_hierarchy ───────────────────────────────────────────────────────
def build_hierarchy(df_local, lambda_per_level):
    """Two-level QM8 chromophore hierarchy (lambda_per_level = [L2, L1], deepest first)."""
    def _grp(col):
        vals = df_local[col].tolist()
        uniq = list(dict.fromkeys(vals))
        idx  = {u: i for i, u in enumerate(uniq)}
        desc = [[] for _ in uniq]
        for i, v in enumerate(vals):
            desc[idx[v]].append(i)
        return uniq, desc

    L2_names, L2_desc = _grp('_chromophore_L2')
    L1_names, L1_desc = _grp('_chromophore_L1')

    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float64))

    return [
        {'name': 'chromophore_type', 'descendants': L2_desc,
         'similarity': empty, 'lambda': lambda_per_level[0]},
        {'name': 'transition_type',  'descendants': L1_desc,
         'similarity': empty, 'lambda': lambda_per_level[1]},
    ]

hierarchy_main = build_hierarchy(df, LAMBDA_MAIN)
for L in hierarchy_main:
    sizes = sorted([len(d) for d in L['descendants']], reverse=True)
    print(f"\n  {L['name']:22s}: {len(L['descendants'])} nodes, "
          f"lambda={L['lambda']}, top sizes={sizes[:8]}")

with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)
print('\nHierarchy saved to', OUT_INTER / 'featurization' / 'hierarchy_main.pkl')


# In[16]:


# Cell 4.2 - H term visualizations: chromophore hierarchy diagnostics

# ── Panels A + B: group sizes ────────────────────────────────────────────────
fig_ab, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4))

l2_counts = [(g, chromophore_L2.count(g)) for g in CHR_GROUP_ORDER if chromophore_L2.count(g) > 0]
labels_l2 = [x[0] for x in l2_counts]
vals_l2   = [x[1] for x in l2_counts]
cmap10    = plt.cm.tab20
colors_l2 = [cmap10(i % 10) for i in range(len(labels_l2))]
bars = ax_a.bar(range(len(labels_l2)), vals_l2, color=colors_l2, edgecolor='black', linewidth=0.5)
ax_a.set_xticks(range(len(labels_l2)))
ax_a.set_xticklabels(labels_l2, rotation=35, ha='right', fontsize=9)
ax_a.set_ylabel('n molecules')
ax_a.set_title('H: L2 chromophore_type group sizes\n(all groups with >= 1 member shown)')
for bar, v in zip(bars, vals_l2):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
              str(v), ha='center', va='bottom', fontsize=8)
ax_a.axhline(5, color='red', linestyle=':', linewidth=1, label='min=5 threshold')
ax_a.legend(fontsize=8)

l1_counts = [(g, chromophore_L1.count(g)) for g in L1_GROUPS if chromophore_L1.count(g) > 0]
labels_l1 = [x[0] for x in l1_counts]
vals_l1   = [x[1] for x in l1_counts]
l1_pal = ['#E64B35', '#4DBBD5', '#3C5488', '#44AA99', '#B0B0B0'][:len(labels_l1)]
wedges, texts, autotexts = ax_b.pie(
    vals_l1, labels=labels_l1, colors=l1_pal,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
ax_b.set_title('H: L1 transition_type proportions\n(all four paradigms should appear)')

plt.tight_layout()
save_fig(fig_ab, 'H_group_sizes_and_L1_pie', where='main')
plt.show()

# ── Panel C: t-SNE colored by L2 group ─────────────────────────────────────
cache_tsne = OUT_INTER / 'featurization' / f'tsne_n{n}.npy'
if cache_tsne.exists():
    tsne_emb = np.load(cache_tsne)
    print(f'Loaded cached t-SNE: shape={tsne_emb.shape}')
else:
    print(f'Computing t-SNE on {n} x {bv.shape[1]} bits ...')
    t0_tsne = time.time()
    tsne_emb = TSNE(n_components=2, perplexity=min(30, n // 4), init='pca',
                    random_state=RANDOM_SEED).fit_transform(bv.astype(np.float32))
    np.save(cache_tsne, tsne_emb)
    print(f'  done in {time.time()-t0_tsne:.1f}s')

fig_c, ax_c = plt.subplots(figsize=(9, 7))
for gi, grp in enumerate(CHR_GROUP_ORDER):
    mask = np.array(chromophore_L2) == grp
    if mask.sum() == 0:
        continue
    ax_c.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1],
                 s=15, alpha=0.75, color=cmap10(gi % 10),
                 label=f'{grp} (n={mask.sum()})', linewidths=0)
ax_c.legend(loc='upper right', fontsize=7, markerscale=2, framealpha=0.85)
ax_c.set_title('t-SNE of ECFP4 colored by H L2 chromophore group\n'
               'Partial (not full) separation confirms H is not redundant with M')
ax_c.set_xlabel('t-SNE 1')
ax_c.set_ylabel('t-SNE 2')
plt.tight_layout()
save_fig(fig_c, 'H_tsne_L2_groups', where='main')
plt.show()

# ── Panel D: E1-CC2 box plots by L2 group ───────────────────────────────────
e1_arr       = df['E1-CC2'].values
chr_l2_arr   = np.array(chromophore_L2)
active_grps  = [g for g in CHR_GROUP_ORDER if (chr_l2_arr == g).sum() >= 3]
box_data     = [e1_arr[chr_l2_arr == g] for g in active_grps]
color_map    = {g: cmap10(CHR_GROUP_ORDER.index(g) % 10) for g in CHR_GROUP_ORDER}

fig_d, ax_d = plt.subplots(figsize=(11, 5))
bp = ax_d.boxplot(box_data, patch_artist=True, notch=False,
                  medianprops=dict(color='black', linewidth=1.5),
                  whiskerprops=dict(linestyle='--', color='gray'),
                  capprops=dict(color='gray'))
for patch, grp in zip(bp['boxes'], active_grps):
    patch.set_facecolor(color_map[grp])
    patch.set_alpha(0.75)
ax_d.set_xticks(range(1, len(active_grps) + 1))
ax_d.set_xticklabels(active_grps, rotation=35, ha='right', fontsize=9)
ax_d.set_ylabel('E1-CC2 (eV)')
ax_d.set_title('E1-CC2 distribution by H L2 chromophore group\n'
               'Distinct distributions confirm H captures real photophysics')
ax_d.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig_d, 'H_E1_boxplots_by_L2', where='main')
plt.show()

# ── Orthogonality check: intra-group vs cross-group Tanimoto ─────────────────
print('\nOrthogonality check (mean intra-group Tanimoto vs cross-group):')
print('  (partial overlap expected; complete collapse => H redundant with M)')
for grp in active_grps[:7]:
    mask     = chr_l2_arr == grp
    in_idx   = np.where(mask)[0]
    out_idx  = np.where(~mask)[0]
    if len(in_idx) < 2:
        continue
    intra_T  = A_full[np.ix_(in_idx, in_idx)]
    intra_mu = intra_T[np.triu_indices(len(in_idx), k=1)].mean()
    cross_T  = A_full[np.ix_(in_idx, out_idx[:max(1, len(out_idx))])]
    cross_mu = cross_T.mean()
    print(f'  {grp:20s}: intra T={intra_mu:.3f},  cross T={cross_mu:.3f}  '
          f'(ratio={intra_mu/(cross_mu+1e-9):.2f}x)')


# ## 5. Baseline splitters 

# In[17]:


# Cell 5.1 - implement baseline splitters
from sklearn.cluster import KMeans, AgglomerativeClustering

def _greedy_pack(labels, n_clusters):
    """First-fit decreasing bin-packing of cluster groups into the train fold."""
    grp_to_idx = defaultdict(list)
    for i, c in enumerate(labels):
        grp_to_idx[c].append(i)
    order = sorted(grp_to_idx.values(), key=len, reverse=True)
    f = np.ones(n, dtype=int)
    target_train = int(round(FOLD_RATIOS[0] * n))
    cum = 0
    for g in order:
        if cum + len(g) <= target_train:
            for j in g:
                f[j] = 0
            cum += len(g)
        else:
            continue
    return f

def split_random(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int)
    f[perm[:cut]] = 0
    return f

def split_stratified(seed, target=PRIMARY_TARGET):
    y = df[target].values
    bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    rng = np.random.default_rng(seed)
    f = np.ones(n, dtype=int)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        cut = int(round(FOLD_RATIOS[0] * len(idx)))
        f[idx[:cut]] = 0
    return f

def split_scaffold(seed):
    labels_map = {s: i for i, s in enumerate(df['_scaffold'].unique())}
    labels = np.array([labels_map[s] for s in df['_scaffold']])
    return _greedy_pack(labels, len(labels_map))

def split_kmeans(seed, n_clusters=15):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(bv.astype(np.float32))
    return _greedy_pack(km.labels_, n_clusters)

def split_kennard_stone(seed):
    X = bv.astype(np.float32)
    centroid = X.mean(axis=0, keepdims=True)
    d = np.linalg.norm(X - centroid, axis=1)
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int)
    f[np.argsort(-d)[:n_test]] = 1
    return f

def split_agglomerative(seed, n_clusters=10):
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = ag.fit_predict(PCA(n_components=20, random_state=seed).fit_transform(bv.astype(np.float32)))
    return _greedy_pack(labels, n_clusters)

BASELINE_SPLITTERS = {
    'RANDOM':        split_random,
    'STRATIFIED':    split_stratified,
    'SCAFFOLD':      split_scaffold,
    'KMEANS':        split_kmeans,
    'KENNARDSTONE':  split_kennard_stone,
    'AGGLOMERATIVE': split_agglomerative,
}


# In[18]:


# Cell 5.2 - run all baseline splitters and persist splits
baseline_results = {}
for name, fn in BASELINE_SPLITTERS.items():
    f = fn(RANDOM_SEED)
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    out = OUT_INTER / 'splits' / f'{name}.csv'
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(out, index=False)
    print(f'  {name:14s}  sizes={sizes.round(3).tolist()}')


# In[19]:


# Cell 5.3 - precomputed splits from qm8_DataSAIL.csv

for _pc_name, _pc_col in [('SPLIT_BASE', '_split_base'), ('SPLIT_DATASAIL', '_split_datasail')]:
    _f = np.array([PRECOMPUTED_FOLD_MAP[lb] for lb in df[_pc_col].values], dtype=int)
    _sizes = np.bincount(_f, minlength=N_FOLDS) / n
    baseline_results[_pc_name] = _f
    _out = OUT_INTER / 'splits' / f'{_pc_name}.csv'
    pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(_out, index=False)
    print(f'  {_pc_name:16s}  sizes={_sizes.round(3).tolist()}')


# In[20]:


# Cell 5.4 - additional splitting tools

import warnings as _w54
_w54.filterwarnings('ignore')

_Y_PRIMARY = df[PRIMARY_TARGET].values.astype(np.float64)

def split_dc_butina(seed):
    """Sphere-exclusion (Leader) clustering on ECFP4 (cutoff 0.4); falls back to exact Butina."""
    try:
        from rdkit.SimDivFilters import rdSimDivPickers as _sdp
        lp = _sdp.LeaderPicker()
        leaders = list(lp.LazyBitVectorPick(fps, len(fps), 0.4))
        n_clusters = len(leaders)
        X = bv.astype(np.float32)
        cnt = X.sum(1)
        Lidx = np.asarray(leaders, dtype=np.int64)
        Lx = X[Lidx]
        Lc = cnt[Lidx]
        labels = np.empty(n, dtype=int)
        CH = 4096
        for s0 in range(0, n, CH):
            s1 = min(s0 + CH, n)
            inter = X[s0:s1] @ Lx.T
            den = cnt[s0:s1, None] + Lc[None, :] - inter
            labels[s0:s1] = (inter / np.clip(den, 1e-8, None)).argmax(1)
        print(f'    [DC_BUTINA] LeaderPicker: {n_clusters} clusters (cutoff 0.4)', flush=True)
        return _greedy_pack(labels, n_clusters)
    except Exception as _bex:
        from rdkit.ML.Cluster import Butina as _Butina
        print(f'    [DC_BUTINA] LeaderPicker unavailable ({_bex}); exact Butina (O(n^2))...', flush=True)
        X = bv.astype(np.float32)
        cnt = X.sum(1)
        dists = np.empty(n * (n - 1) // 2, dtype=np.float64)
        pos = 0
        for i in range(1, n):
            inter = X[i] @ X[:i].T
            den = cnt[i] + cnt[:i] - inter
            dists[pos:pos + i] = 1.0 - inter / np.clip(den, 1e-8, None)
            pos += i
        clusters = _Butina.ClusterData(dists, n, 0.4, isDistData=True)
        labels = np.empty(n, dtype=int)
        for cid, cl in enumerate(clusters):
            for idx in cl:
                labels[idx] = cid
        return _greedy_pack(labels, len(clusters))

def split_dc_fingerprint(seed):
    """Lexicographic sort on packed ECFP4 bits; first 80% train, last 20% test."""
    packed = np.packbits(bv, axis=1)
    order = np.lexsort(packed.T[::-1])
    n_train = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int)
    f[order[:n_train]] = 0
    return f

def split_dc_maxmin(seed):
    """MaxMin diversity picker (C++ MaxMinPicker); falls back to numpy greedy loop."""
    n_test = int(round(FOLD_RATIOS[1] * n))
    try:
        from rdkit.SimDivFilters import rdSimDivPickers as _sdp
        mmp = _sdp.MaxMinPicker()
        try:
            picks = mmp.LazyBitVectorPick(fps, len(fps), n_test, [], int(seed))
        except Exception:
            picks = mmp.LazyBitVectorPick(fps, len(fps), n_test)
        f = np.zeros(n, dtype=int)
        f[np.asarray(list(picks), dtype=np.int64)] = 1
        return f
    except Exception as _mex:
        print(f'    [DC_MAXMIN] MaxMinPicker unavailable ({_mex}); numpy greedy...', flush=True)
        rng = np.random.default_rng(seed)
        X = bv.astype(np.float32)
        counts = X.sum(axis=1)
        start = int(rng.integers(n))
        selected = [start]
        inter = X @ X[start]
        tani = inter / (counts + counts[start] - inter).clip(1e-8)
        min_dists = (1.0 - tani).astype(np.float64)
        min_dists[start] = -np.inf
        while len(selected) < n_test:
            nxt = int(np.argmax(min_dists))
            selected.append(nxt)
            inter = X @ X[nxt]
            tani = inter / (counts + counts[nxt] - inter).clip(1e-8)
            np.minimum(min_dists, 1.0 - tani, out=min_dists)
            min_dists[nxt] = -np.inf
        f = np.zeros(n, dtype=int)
        f[selected] = 1
        return f

def split_dc_scaffold(seed):
    """DeepChem scaffold split using the same _scaffold column as SCAFFOLD baseline."""
    grp = defaultdict(list)
    for i, sc in enumerate(df['_scaffold']):
        grp[sc].append(i)
    labels_map = {sc: cid for cid, sc in enumerate(grp)}
    labels = np.array([labels_map[sc] for sc in df['_scaffold']])
    return _greedy_pack(labels, len(grp))

def split_dc_weight(seed):
    """Molecular weight split: heaviest molecules go to test."""
    from rdkit.Chem.Descriptors import MolWt as _MolWt
    mw = np.array([_MolWt(m) if m else 0.0 for m in mols])
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int)
    f[np.argsort(-mw)[:n_test]] = 1
    return f

ADDITIONAL_SPLITTERS = {
    'DC_BUTINA':      split_dc_butina,
    'DC_FINGERPRINT': split_dc_fingerprint,
    'DC_MAXMIN':      split_dc_maxmin,
    'DC_SCAFFOLD':    split_dc_scaffold,
    'DC_WEIGHT':      split_dc_weight,
}

for _name, _fn in ADDITIONAL_SPLITTERS.items():
    try:
        _f = _fn(RANDOM_SEED)
        _sizes = np.bincount(_f, minlength=N_FOLDS) / n
        baseline_results[_name] = _f
        pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(
            OUT_INTER / 'splits' / f'{_name}.csv', index=False)
        print(f'  {_name:22s}  sizes={_sizes.round(3).tolist()}')
    except Exception as _exc:
        print(f'  {_name:22s}  FAILED: {_exc}')


# ## 6. SHIELD primary configurations and Stage-0 normalizer table

# In[21]:


# Cell 6.1 - SHIELD ablation configurations
SHIELD_CONFIGS = {
    'HMD-SHIELD':     dict(alpha=ALPHA, beta=BETA,  gamma=GAMMA, eta=ETA, mu=MU),
    'H-SHIELD':   dict(alpha=ALPHA, beta=0.0,   gamma=0.0,   eta=ETA, mu=MU),
    'M-SHIELD':   dict(alpha=0.0,   beta=BETA,  gamma=0.0,   eta=ETA, mu=MU),
    'D-SHIELD':   dict(alpha=0.0,   beta=0.0,   gamma=GAMMA, eta=ETA, mu=MU),
}
print('SHIELD config grid (D-shift, y_target=None):')
for k, c in SHIELD_CONFIGS.items():
    print(f'  {k:18s}  {c}')


# In[22]:


# Cell 6.2 - run all SHIELD configurations

# Y_TARGETS kept for diagnostic purposes (Wasserstein evaluation in section 7)
Y_TARGETS = df[TARGET_COLS].values.astype(np.float32)

shield_results    = {}
shield_diagnostics = {}
t_all = time.time()
CACHED_NORMALIZERS = None


for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    kwargs = dict(
        n=n, K=N_FOLDS,
        r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main       if cfg['alpha'] > 0 else None,
        affinity=AM                    if cfg['beta']  > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        kernel_matrix=K_MATRIX         if cfg['gamma'] > 0 else None,
        class_delta=STRAT_TOL,
        balance_eps=BALANCE_TOL,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'],     mu=cfg['mu'],     nu=NU, tau=TAU,
        max_iter=SHIELD_MAX_ITER,
        seed=RANDOM_SEED,
        device=SHIELD_DEVICE,
        nodes=N_JOBS,
        integer_z_final=True,
        init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=SHIELD_ROUNDING,
        milp_size_limit=SHIELD_MILP_SIZE,
        stage0_samples=SHIELD_STAGE0_SAMPLES,
        precomputed_normalizers=CACHED_NORMALIZERS,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        res  = shield_split(**kwargs)
        if CACHED_NORMALIZERS is None:                   
            CACHED_NORMALIZERS = res.metadata.get('stage0_normalizers')
        fold = res.fold_assignment.cpu().numpy().astype(int)
        shield_results[cfg_name]     = fold
        shield_diagnostics[cfg_name] = {
            'history':     res.history,
            'diagnostics': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                            for k, v in res.diagnostics.items() if k != 'history'},
            'metadata':    res.metadata,
        }
        (OUT_INTER / 'shield_runs' / cfg_name).mkdir(exist_ok=True)
        pd.DataFrame({'index': np.arange(n), 'fold': fold}).to_csv(
            OUT_INTER / 'shield_runs' / cfg_name / 'split.csv', index=False)
        print(f'  {cfg_name:18s}  wall={time.time()-t0:.1f}s  '
              f'sizes={(np.bincount(fold, minlength=N_FOLDS)/n).round(3).tolist()}')
    except Exception as exc:
        print(f'  {cfg_name:18s}  FAILED: {exc}')

print(f'\nTotal SHIELD wall time: {time.time()-t_all:.1f}s')


# In[23]:


# Cell 6.3 - Stage-0 normalizer table 
norm_rows = []
for cfg_name, diag in shield_diagnostics.items():
    md_ = diag['metadata']
    s0 = md_.get('stage0_normalizers', {})
    norm_rows.append({
        'config': cfg_name,
        'H_norm':     s0.get('H'),
        'M_norm':     s0.get('M'),
        'D_norm':     s0.get('D'),
        'strat_norm': s0.get('strat'),
        'bal_norm':   s0.get('bal'),
        'init':       md_.get('init_backend'),
    })
norm_df = pd.DataFrame(norm_rows)
save_table(norm_df, 'stage0_normalizers', where='si')
norm_df


# ## 7. Leakage diagnostics

# In[24]:


# Cell 7.1 - cross-fold L(pi) for every method
TOTAL_SIM = float(A_full.sum())

def cross_fold_similarity(fold, A=A_full):
    F = fold.reshape(-1, 1)
    cross = (F != F.T).astype(np.float32)
    raw = float((A * cross).sum()) / 2.0
    return raw, raw / max(TOTAL_SIM / 2.0, 1e-9)

lpi_rows = []
for name, fold in baseline_results.items():
    raw, scaled = cross_fold_similarity(fold)
    lpi_rows.append({'method': 'baseline', 'config': name, 'L_pi': raw, 'scaled_L_pi': scaled})
for name, fold in shield_results.items():
    raw, scaled = cross_fold_similarity(fold)
    lpi_rows.append({'method': 'SHIELD', 'config': name, 'L_pi': raw, 'scaled_L_pi': scaled})
lpi_df = pd.DataFrame(lpi_rows).sort_values('scaled_L_pi')
save_table(lpi_df, 'cross_fold_similarity', where='main')
lpi_df


# In[25]:


# Cell 7.2 - L(pi) bar chart
fig, ax = plt.subplots(figsize=(10, 4))
agg = lpi_df.set_index('config')['scaled_L_pi'].sort_values()
colors = ['#888888' if c in baseline_results else '#4477AA' for c in agg.index]
ax.bar(range(len(agg)), agg.values, color=colors, edgecolor='black')
ax.axhline(agg.loc['RANDOM'] if 'RANDOM' in agg.index else 1.0,
           color='red', ls='--', lw=1, label='RANDOM baseline')
ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg.index, rotation=45, ha='right')
ax.set_ylabel('scaled L(pi)'); ax.set_title('Cross-fold structural similarity (lower = stronger OOD)')
ax.legend()
plt.tight_layout()
save_fig(fig, 'L_pi_bar_chart', where='main')
plt.show()


# In[26]:


# Cell 7.3 - per-channel decomposition L_H, L_M, L_W
hier_levels_obj = build_hierarchy_levels(hierarchy_main)
AM_tensor = torch.from_numpy(AM).to(torch.float32)

def per_channel_leakage(fold):
    z = torch.zeros((n, N_FOLDS), dtype=torch.float32)
    z[np.arange(n), fold] = 1.0
    L_H = float(_evaluate_h_term_from_z(z, hier_levels_obj))
    L_M = float(cut_energy(z, AM_tensor))
    tr = Y_TARGETS[fold == 0]; te = Y_TARGETS[fold == 1]
    if len(tr) and len(te):
        L_W = float(np.mean([wasserstein_distance(tr[:, j], te[:, j])
                             for j in range(Y_TARGETS.shape[1])]))
    else:
        L_W = float('nan')
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


# In[27]:


# Cell 7.6 - kNN purity 
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
                            'knn_purity': knn_purity(fold, bv, k)})
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


# ## 11. Dimension-reduction visualizations (PCA / t-SNE / UMAP) of splits

# In[28]:


# Cell 11.1 - compute embeddings
embeds = {}
try:
    embeds['tSNE'] = TSNE(n_components=2, perplexity=30, init='pca',
                          random_state=RANDOM_SEED).fit_transform(bv.astype(np.float32))
    print('t-SNE done')
except Exception as exc:
    print(f't-SNE failed: {exc}')


# Export embeddings to disk for later reuse
for emb_name, emb in embeds.items():
    np.save(OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.npy', emb)
    pd.DataFrame(emb, columns=[f'{emb_name}_1', f'{emb_name}_2']).to_csv(
        OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.csv', index=True, index_label='row_id'
    )
    print(f'Saved {emb_name} embedding: {emb.shape}')


# In[29]:


# Cell 11.2 - split-colored scatter panels (all methods)
all_splits = {**baseline_results, **shield_results}
all_names  = list(all_splits.keys())

# ── (i) grid over every split method ─────────────────────────────────────────
for emb_name, emb in embeds.items():
    cols = 4
    rows = int(np.ceil(len(all_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, all_names):
        fold = all_splits[name]
        ax.scatter(emb[fold == 0, 0], emb[fold == 0, 1], s=6, c='#4477AA',
                   alpha=0.55, label='train')
        ax.scatter(emb[fold == 1, 0], emb[fold == 1, 1], s=6, c='#CC6677',
                   alpha=0.55, label='test')
        scaled = lpi_df.set_index('config').loc[name, 'scaled_L_pi'] \
                 if name in lpi_df['config'].values else float('nan')
        ax.set_title(f'{name}\nscaled L(pi)={scaled:.3f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(all_names):]:
        ax.set_visible(False)
    axes[0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{emb_name} embedding colored by fold (all methods)', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_splits',
             where='main' if emb_name in ('PCA', 'UMAP') else 'si')
    plt.show()

# ── (ii) t-SNE colored by PRIMARY_TARGET (E1-CC2) ───────────────────────────
_tsne_key = next((k for k in embeds if 'tsne' in k.lower() or 't-sne' in k.lower()), None)
if _tsne_key is not None:
    emb_t  = embeds[_tsne_key]
    y_col  = df[PRIMARY_TARGET].values.astype(np.float64)
    fig_t, ax_t = plt.subplots(figsize=(7, 6))
    sc = ax_t.scatter(emb_t[:, 0], emb_t[:, 1], c=y_col, cmap='viridis',
                      s=8, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax_t, label=f'{PRIMARY_TARGET} (eV)', shrink=0.85)
    ax_t.set_title(
        f't-SNE of ECFP4 colored by {PRIMARY_TARGET}\n'
        f'n={n},  mean={y_col.mean():.3f} eV,  std={y_col.std():.3f} eV',
        fontsize=10)
    ax_t.set_xlabel('t-SNE 1'); ax_t.set_ylabel('t-SNE 2')
    ax_t.set_xticks([]); ax_t.set_yticks([])
    plt.tight_layout()
    save_fig(fig_t, f'tsne_{PRIMARY_TARGET.replace("-", "_")}_target_color', where='si')
    plt.show()
else:
    print('t-SNE key not found in embeds; available keys:', list(embeds.keys()))


# ## 8. OOD Assessment Metrics

# In[30]:


# Cell 11.3 - OOD characterization metrics per split

from sklearn.neighbors  import NearestNeighbors
from scipy.spatial      import ConvexHull, Delaunay
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
import matplotlib.ticker as mticker
import warnings

bv32 = bv.astype(np.float32)

_tsne_2d = embeds.get('tSNE')

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nn_distances(X_train, X_test):
    nn = NearestNeighbors(n_neighbors=1, n_jobs=N_JOBS)
    nn.fit(X_train)
    return nn.kneighbors(X_test)[0][:, 0]

def _convex_hull_outside_rate(emb_train, emb_test):
    if emb_train.shape[0] < 4:
        return float('nan')
    try:
        hull  = ConvexHull(emb_train)
        delan = Delaunay(emb_train[hull.vertices])
        return float((delan.find_simplex(emb_test) < 0).mean())
    except Exception:
        return float('nan')

# ── Main loop ─────────────────────────────────────────────────────────────────
all_splits = {f'baseline::{k}': v for k, v in baseline_results.items()}
all_splits.update({f'SHIELD::{k}': v for k, v in shield_results.items()})

ood_rows = []
for sp_name, fold in all_splits.items():
    mask_tr = fold == 0
    mask_te = fold == 1
    Xtr = bv32[mask_tr]
    Xte = bv32[mask_te]

    if len(Xtr) < 10 or len(Xte) < 5:
        continue

    row = {'split': sp_name}

    # (1-3) NN distances in ECFP4 feature space
    try:
        nn_d = _nn_distances(Xtr, Xte)
        row['nn_mean'] = float(nn_d.mean())
        row['nn_p95']  = float(np.percentile(nn_d, 95))
        row['nn_max']  = float(nn_d.max())
    except Exception as e:
        print(f'  [NN] {sp_name}: {e}')
        row['nn_mean'] = row['nn_p95'] = row['nn_max'] = float('nan')

    # (4) SOAP Distance — mean NN in t-SNE 2-D space
    if _tsne_2d is not None:
        try:
            row['soap_dist_mean'] = float(
                _nn_distances(_tsne_2d[mask_tr], _tsne_2d[mask_te]).mean()
            )
        except Exception as e:
            print(f'  [SOAP] {sp_name}: {e}')
            row['soap_dist_mean'] = float('nan')
    else:
        row['soap_dist_mean'] = float('nan')

    # (5) Convex Hull Outside Rate — t-SNE 2-D space
    if _tsne_2d is not None:
        try:
            row['hull_outside_rate'] = _convex_hull_outside_rate(
                _tsne_2d[mask_tr], _tsne_2d[mask_te]
            )
        except Exception as e:
            print(f'  [Hull] {sp_name}: {e}')
            row['hull_outside_rate'] = float('nan')
    else:
        row['hull_outside_rate'] = float('nan')

    ood_rows.append(row)
    print(
        f'{sp_name:45s}  '
        f'NN_mean={row["nn_mean"]:.3f}  '
        f'NN_p95={row["nn_p95"]:.3f}  '
        f'NN_max={row["nn_max"]:.3f}  '
        f'SOAP={row["soap_dist_mean"]:.3f}  '
        f'Hull={row["hull_outside_rate"]:.3f}',
        flush=True,
    )

ood_df = pd.DataFrame(ood_rows).set_index('split')
save_table(ood_df.reset_index(), 'ood_characterization_metrics', where='main')
display(ood_df.round(4))

# ── Individual bar figures (one per metric) ───────────────────────────────────
_metric_meta = {
    'nn_mean':          ('Mean NN Distance (ECFP4 feature space)',      'L2 distance',      'higher → more OOD'),
    'nn_p95':           ('95th-pct NN Distance (ECFP4 feature space)',  'L2 distance',      'higher → tail extrapolation'),
    'nn_max':           ('Max NN Distance (ECFP4 feature space)',        'L2 distance',      'higher → extreme extrapolation'),
    'soap_dist_mean':   ('SOAP Distance (t-SNE 2-D)',                   'Euclidean (2-D)',   'higher → structural novelty'),
    'hull_outside_rate':('Convex Hull Outside Rate (t-SNE 2-D)',        'fraction outside', 'higher → more extrapolation'),
}

for col, (title, ylabel, note) in _metric_meta.items():
    if col not in ood_df.columns:
        continue
    vals = ood_df[col].dropna()
    colors = ['#CC6677' if 'SHIELD' in s else '#4477AA' for s in vals.index]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(vals)), 4))
    ax.barh(vals.index, vals.values, color=colors, edgecolor='white', height=0.65)
    ax.set_xlabel(ylabel)
    ax.set_title(f'Tox21 — {title}\n({note})', fontsize=10)
    ax.axvline(vals.mean(), color='k', lw=1.2, ls='--')
    ax.legend(handles=[
        Patch(color='#4477AA', label='baseline'),
        Patch(color='#CC6677', label='SHIELD'),
        Line2D([0], [0], color='k', lw=1.2, ls='--', label=f'mean={vals.mean():.3f}'),
    ], fontsize=8)
    plt.tight_layout()
    save_fig(fig, f'ood_{col}', where='main')
    plt.show()

# ── Master Figure 1 — Bubble map ─────────────────────────────────────────────

fig_bub, ax_bub = plt.subplots(figsize=(11, 7))

_x   = ood_df['nn_mean'].fillna(0)
_y   = ood_df['soap_dist_mean'].fillna(0)
_sz  = ood_df['hull_outside_rate'].fillna(0)
_c   = ood_df['nn_p95'].fillna(0)
_mx  = ood_df['nn_max'].fillna(0)

_sz_norm = (_sz - _sz.min()) / (_sz.max() - _sz.min() + 1e-12)
_sz_plot = 70 + 560 * _sz_norm

sc_bub = ax_bub.scatter(
    _x, _y, s=_sz_plot, c=_c, cmap='plasma',
    edgecolors='white', linewidths=0.9, alpha=0.88, zorder=3,
)
cbar = plt.colorbar(sc_bub, ax=ax_bub, pad=0.01)
cbar.set_label('95th-pct NN Distance (ECFP4 feature space)', fontsize=9)

for sp in ood_df.index:
    ax_bub.annotate(
        f'{sp}\nmax={_mx[sp]:.2f}',
        xy=(_x[sp], _y[sp]),
        fontsize=6.2, ha='center', va='bottom',
        xytext=(0, 7), textcoords='offset points', color='#1a1a1a',
    )

ax_bub.set_xlabel('Mean NN Distance  (ECFP4)  →  more OOD',         fontsize=10)
ax_bub.set_ylabel('SOAP Distance  (t-SNE 2-D)  →  structural novelty', fontsize=10)
for frac_leg, lbl_leg in [(0.0, '0 % outside hull'), (0.5, '50 %'), (1.0, '100 %')]:
    ax_bub.scatter([], [], s=70 + 560 * frac_leg, c='gray', alpha=0.5, label=lbl_leg)
ax_bub.set_title(
    'Tox21 OOD Characterization Bubble Map\n'
    'x = Mean NN Dist  ·  y = SOAP Dist  ·  size = Hull Outside Rate  '
    '·  color = p95 NN Dist  ·  label = Max NN Dist',
    fontsize=9.5,
)
ax_bub.legend(title='Hull Outside Rate', fontsize=7, loc='upper left', framealpha=0.85)
plt.tight_layout()
save_fig(fig_bub, 'ood_master_bubble_map', where='main')
plt.show()

# ── Master Figure 2 — Parallel coordinates ───────────────────────────────────

_pc_cols   = ['nn_mean', 'nn_p95', 'nn_max', 'soap_dist_mean']
_pc_labels = ['Mean NN\n(ECFP4)', '95th-pct NN\n(ECFP4)', 'Max NN\n(ECFP4)', 'SOAP Dist\n(t-SNE)']

_pc_df   = ood_df[_pc_cols].dropna()
_pc_norm = (_pc_df - _pc_df.min()) / (_pc_df.max() - _pc_df.min() + 1e-12)

_hull_vals = ood_df.loc[_pc_norm.index, 'hull_outside_rate'].fillna(0)
_hull_norm = (_hull_vals - _hull_vals.min()) / (_hull_vals.max() - _hull_vals.min() + 1e-12)
_lw_map    = 0.9 + 2.6 * _hull_norm

_baseline_splits = [s for s in _pc_norm.index if 'SHIELD' not in s]
_shield_splits   = [s for s in _pc_norm.index if 'SHIELD'     in s]
_blues = plt.cm.Blues(np.linspace(0.45, 0.90, max(1, len(_baseline_splits))))
_reds  = plt.cm.Reds (np.linspace(0.45, 0.90, max(1, len(_shield_splits))))
_color_map = {}
for i, s in enumerate(_baseline_splits): _color_map[s] = _blues[i]
for i, s in enumerate(_shield_splits):   _color_map[s] = _reds[i]

n_axes   = len(_pc_cols)
fig_pc, axes_pc = plt.subplots(1, n_axes - 1, figsize=(13, 6), sharey=False)
fig_pc.subplots_adjust(wspace=0, left=0.06, right=0.82, top=0.88, bottom=0.12)

for sp in _pc_norm.index:
    vals_sp = _pc_norm.loc[sp].values
    for seg in range(n_axes - 1):
        axes_pc[seg].plot(
            [0, 1], [vals_sp[seg], vals_sp[seg + 1]],
            color=_color_map[sp], lw=float(_lw_map[sp]),
            alpha=0.82, solid_capstyle='round',
        )
        axes_pc[seg].set_xlim(0, 1)
        axes_pc[seg].set_ylim(-0.05, 1.05)

for seg, ax_seg in enumerate(axes_pc):
    ax_seg.spines['top'].set_visible(False)
    ax_seg.spines['bottom'].set_visible(False)
    ax_seg.spines['right'].set_visible(False)
    ax_seg.spines['left'].set_linewidth(1.5)
    ax_seg.spines['left'].set_color('#555555')
    ax_seg.set_xticks([])
    ax_seg.yaxis.set_major_locator(mticker.LinearLocator(6))
    orig_min = _pc_df[_pc_cols[seg]].min()
    orig_max = _pc_df[_pc_cols[seg]].max()
    ax_seg.set_yticklabels(
        [f'{orig_min + t * (orig_max - orig_min):.3f}' for t in np.linspace(0, 1, 6)],
        fontsize=7,
    )
    ax_seg.set_xlabel(_pc_labels[seg], fontsize=9, labelpad=6)
    if seg > 0:
        ax_seg.yaxis.set_visible(False)

ax_last = axes_pc[-1].twinx()
ax_last.set_ylim(-0.05, 1.05)
ax_last.yaxis.set_major_locator(mticker.LinearLocator(6))
orig_min_last = _pc_df[_pc_cols[-1]].min()
orig_max_last = _pc_df[_pc_cols[-1]].max()
ax_last.set_yticklabels(
    [f'{orig_min_last + t * (orig_max_last - orig_min_last):.3f}' for t in np.linspace(0, 1, 6)],
    fontsize=7,
)
ax_last.spines['top'].set_visible(False)
ax_last.spines['bottom'].set_visible(False)
ax_last.spines['left'].set_visible(False)

legend_handles = (
    [Line2D([0], [0], color=_color_map[s], lw=1.8, label=s) for s in _baseline_splits] +
    [Line2D([0], [0], color=_color_map[s], lw=1.8, label=s) for s in _shield_splits]   +
    [
        Line2D([0], [0], color='gray', lw=0.9, label='thin  = low hull-outside rate'),
        Line2D([0], [0], color='gray', lw=3.5, label='thick = high hull-outside rate'),
    ]
)
fig_pc.legend(
    handles=legend_handles,
    loc='center right', bbox_to_anchor=(1.0, 0.5),
    fontsize=7.5, framealpha=0.9,
    title='Split  (line width = Hull rate)', title_fontsize=8,
)
fig_pc.suptitle(
    'Tox21 OOD Distance Metric Profiles Across Splits\n'
    '(axes normalized to [0,1];  tick labels in original units)',
    fontsize=10.5, y=0.96,
)
save_fig(fig_pc, 'ood_master_parallel_coords', where='main')
plt.show()


# In[31]:


# Cell 16.x - export SMILES + fold assignments for every split

_all = {}
for k, v in globals().get('baseline_results', {}).items():
    _all[k] = v                         # Section 5 (incl. SPLIT_BASE / SPLIT_DATASAIL)
for k, v in globals().get('shield_results', {}).items():
    _all[f'SHIELD6_{k}'] = v            # Section 6 SHIELD ablations
for k, v in globals().get('lambda_results', {}).items():
    _all[f'LAMBDA12_{k}'] = v           # Section 12 lambda ablation
for k, v in globals().get('rounding_results', {}).items():
    _all[f'ROUND14_{k}'] = v            # Section 14 rounding modes

export = pd.DataFrame({'smiles': df[SMILES_COL].values})

if 'mol_id' in df.columns:
    export.insert(0, 'mol_id', df['mol_id'].values)

# Raw DataSAIL 3-way labels (train / val / test strings) for reference
for _raw in ['_split_base', '_split_datasail']:
    if _raw in df.columns:
        export[f'RAW{_raw}'] = df[_raw].values

# Primary target for convenience
if PRIMARY_TARGET in df.columns:
    export[PRIMARY_TARGET] = df[PRIMARY_TARGET].values

for name, fold in _all.items():
    fold = np.asarray(fold)
    if fold.shape[0] != len(df):
        print(f'  SKIP {name}: length {fold.shape[0]} != n={len(df)}')
        continue
    export[name] = fold.astype(int)

_out = OUT_DIR / 'all_splits_folds.csv'
export.to_csv(_out, index=False)
print(f'Exported {export.shape[0]:,} rows x {export.shape[1]} cols -> {_out}')
print('Fold convention: 0 = train, 1 = holdout/test.')
_skip_cols = {'smiles', PRIMARY_TARGET, 'RAW_split_base', 'RAW_split_datasail'}
print('Split columns:', [c for c in export.columns if c not in _skip_cols])


# In[32]:


# Cell 16.y - export all splits + original data to a single combined CSV

_all_splits_export = {f'baseline__{k}': v for k, v in baseline_results.items()}
_all_splits_export.update({f'SHIELD__{k}': v for k, v in shield_results.items()})
for k, v in globals().get('lambda_results', {}).items():
    _all_splits_export[f'LAMBDA12__{k}'] = v
for k, v in globals().get('rounding_results', {}).items():
    _all_splits_export[f'ROUND14__{k}'] = v

_splits_df = df.copy()

for split_name, fold_arr in _all_splits_export.items():
    fold_arr = np.asarray(fold_arr)
    if fold_arr.shape[0] != len(df):
        print(f'  SKIP {split_name}: length mismatch ({fold_arr.shape[0]} != {len(df)})')
        continue
    col = 'fold__' + split_name.replace('::', '__').replace(' ', '_')
    _splits_df[col] = fold_arr.astype(int)

_fold_cols = [c for c in _splits_df.columns if c.startswith('fold__')]
print(f'Splits embedded: {len(_fold_cols)}')
for c in _fold_cols:
    counts = _splits_df[c].value_counts().sort_index().to_dict()
    print(f'  {c:60s}  {counts}')

_out_path = OUT_INTER / 'splits' / 'all_splits_combined.csv'
_splits_df.to_csv(_out_path, index=True, index_label='row_id')
print(f'\nSaved: {_out_path}  (shape: {_splits_df.shape})')



# ## 13. SI — stratification strategies on CC2-E1 (single coordinate)

# In[38]:


# Cell 13.1 - SI: stratification strategies on primary target E1-CC2

y_single = Y_TARGETS[:, TARGET_COLS.index(PRIMARY_TARGET)].reshape(-1, 1)
strat_results = {}
for strat in ['quantile', 'moment', 'sliced_wasserstein']:
    print(f'  strategy={strat}')
    t0 = time.time()
    try:
        res = shield_split(
            n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
            hierarchy=hierarchy_main, affinity=AM,
            affinity_provenance='label-blind-handcrafted',
            d_mode=DIST_MODE, kernel_matrix=K_MATRIX,
            y_target=y_single, stratification_strategy=strat,
            P_proj=SW_PROJECTIONS, Q_grid=SW_QUANTILE_PTS,
            class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
            alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=0.5, nu=NU, tau=TAU,
            max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
            integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
            rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        )
        strat_results[strat] = res.fold_assignment.cpu().numpy().astype(int)
        sizes = (np.bincount(strat_results[strat], minlength=N_FOLDS) / n).round(3).tolist()
        print(f'    wall={time.time()-t0:.1f}s,  sizes={sizes}')
    except Exception as exc:
        print(f'    FAILED: {exc}')


# In[44]:


# Cell 13.2 - strategy comparison: per-target Wasserstein on the primary target + Q-Q plot

_hmd_fold = shield_results.get('HMD-SHIELD')
_all_panels = dict(strat_results)
if _hmd_fold is not None:
    _all_panels['HMD-SHIELD'] = _hmd_fold
else:
    print('WARNING: HMD-SHIELD not found in shield_results — 4th panel skipped.')

fig, axes = plt.subplots(1, len(_all_panels), figsize=(4 * len(_all_panels), 4), sharey=True)
if len(_all_panels) == 1: axes = [axes]

_qq_rows = []
for ax, (strat, fold) in zip(axes, _all_panels.items()):
    tr = y_single[fold == 0].ravel()
    te = y_single[fold == 1].ravel()
    qq_tr = np.quantile(tr, np.linspace(0, 0.99, 100))
    qq_te = np.quantile(te, np.linspace(0, 0.99, 100))
    W = wasserstein_distance(tr, te)

    ax.plot(qq_tr, qq_te, marker='.', markersize=4)
    lo = min(qq_tr.min(), qq_te.min()); hi = max(qq_tr.max(), qq_te.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    display_name = f'{strat}\n(MU=0, no strat.)' if strat == 'HMD-SHIELD' else strat
    ax.set_title(f'{display_name}\nW_1 = {W:.4g}')
    ax.set_xlabel('train quantile')
    ax.set_ylabel('test quantile')

    for q, qtr, qte in zip(np.linspace(0, 0.99, 100), qq_tr, qq_te):
        _qq_rows.append({
            'strategy':       strat,
            'quantile':       float(q),
            'train_quantile': float(qtr),
            'test_quantile':  float(qte),
            'wasserstein_w1': float(W),
        })

plt.tight_layout()
save_fig(fig, 'strat_strategy_qq_plots', where='si')
plt.show()

# ── Export tables for reimporting / replotting ────────────────────────────────
_qq_df   = pd.DataFrame(_qq_rows)
_qq_path = OUT_SI / 'tables' / 'strat_qq_data.csv'
_qq_df.to_csv(_qq_path, index=False)

_w1_summary = (
    _qq_df[['strategy', 'wasserstein_w1']]
    .drop_duplicates()
    .sort_values('wasserstein_w1')
    .reset_index(drop=True)
)
_w1_path = OUT_SI / 'tables' / 'strat_w1_summary.csv'
_w1_summary.to_csv(_w1_path, index=False)

print(f'Q-Q data    →  {_qq_path}  ({len(_qq_df)} rows, {_qq_df["strategy"].nunique()} strategies × 100 quantile pts)')
print(f'W_1 summary →  {_w1_path}')
print()
display(_w1_summary)


# In[ ]:




