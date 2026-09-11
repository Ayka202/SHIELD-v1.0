# # SHIELD benchmark for Tox21

# ## 1. Setup, imports, configuration, output directories

# In[1]:


# Cell 1.1 - imports
import os, sys, json, time, math, hashlib, pickle, warnings, importlib, platform
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sps
from scipy.stats import spearmanr

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, balanced_accuracy_score, confusion_matrix
)
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


# In[56]:


# Cell 1.2 - global configuration
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'tox21.csv')
DATASET_NAME = 'tox21'
SMILES_COL   = 'smiles'
ID_COL       = 'mol_id'

# All 12 Tox21 binary assays
NR_ASSAYS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
             'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma']
SR_ASSAYS = ['SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
ASSAY_COLS = NR_ASSAYS + SR_ASSAYS

PRIMARY_ASSAY = 'NR-ER'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 15

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.1
STRAT_TOL   = 0.1

# Affinity / hierarchy
ECFP_RADIUS  = 2
ECFP_NBITS   = 1024
AFFINITY_TYPES = ['tanimoto', 'morgan_cosine']
PRIMARY_AFFINITY = 'morgan_cosine'
KNN_AFFINITY_K   = 20
KNN_AFFINITY_K_SI = [15, 20, 25]
EPS_AFF_M     = 0.72

# M Tier-2: Ring Topology Class clustering 
EPS_T2_FINE     = 0.35    
T2_MAX_COMP     = 1150      
T2_WEIGHT       = 0.2    
T2_SMALL_COMP_MAX = 9 

LAMBDA_MAIN = [1.0, 1.0]


# SHIELD coefficients (main configuration)
ALPHA, BETA, GAMMA, ETA, MU = 1.0, 1.0, 1.0, 2.0, 2.0
NU, TAU = 0.0, 0.0
DIST_MODE = 'shift'
OT_REG    = 0.1
SHIELD_SINKHORN_ITER = 100


# Stratification on the binary target
STRAT_STRATEGY = 'none'
STRAT_N_BINS   = 10


# Coefficient sweep
SWEEP_GRID = {
    'alpha': [0.2, 0.5, 1.0, 2.0, 5.0],
    'beta':  [0.2, 0.5, 1.0, 2.0, 5.0],
    'gamma': [0.2, 0.5, 1.0, 2.0, 5.0],
}
HAMMING_STABILITY_THRESHOLD = 0.10

# kNN purity and embedding diagnostic
KNN_K = [5, 10, 20]
KNN_YHAT_K = 20
ALPHA_SHAPE_PARAMS = [0.01, 0.03, 0.05, 0.1, 0.15, 0.25, 0.5, 2.0, 3.0] 

# Solver / computational
SHIELD_MAX_ITER     = 15
SHIELD_STAGE0_SAMPLES  = 100
SHIELD_INIT_BACKEND = 'auto'
PRIMARY_ROUNDING = 'confidence_gap'
SHIELD_MILP_SIZE    = 10_000
SHIELD_DEVICE       = 'cpu'
N_JOBS              = -1


# ML evaluation
ML_RUN = True


TOX_SMARTS_V2 = {
    'michael_acceptor': [
        '[#6](=[OX1])[CX3;!a]=[CX3;!a]',
        '[NX3][CX3](=[OX1])[CX3;!a]=[CX3;!a]',
        '[SX4](=[OX1])(=[OX1])[CX3;!a]=[CX3;!a]',
    ],
    'quinone': [
        'O=C1C=CC(=O)C=C1',
        'O=C1C(=O)C=CC=C1',
        'O=C1C=CC(=O)c2ccccc21',
    ],
    'aromatic_aldehyde': [
        '[c][CX3H1]=O',
    ],
    'aliphatic_aldehyde': [
        '[CX4][CX3H1]=O',
        '[CX3H1]=O',
    ],
    'primary_ar_amine': [
        '[NH2][c]',
    ],
    'secondary_ar_amine': [
        '[NH1]([c])[#6]',
    ],
    'nitro_aromatic': [
        '[c][N+](=O)[O-]',
    ],
    'nitro_aliphatic': [
        '[CX4][N+](=O)[O-]',
        '[NX2]=O',
    ],
    'alkyl_halide': [
        '[ClX1][CX4]',
        '[BrX1][CX4]',
        '[IX1][CX4]',
    ],
    'epoxide': [
        '[OX2r3]',
    ],
    'phenol': [
        '[OX2H][c]',
    ],
    'carboxylic_acid': [
        '[CX3](=O)[OX2H1]',
    ],
    'sulfonamide': [
        '[SX4](=O)(=O)[NX3]',
        '[SX4](=O)(=O)[CX4]',
    ],
    'thiol': [
        '[SX2H]',
    ],
    'aliphatic_amine': [
        '[NX3;!a]',
    ],
    'n_heterocycle': [
        '[n]',
    ],
    'aryl_halide': [
        '[F,Cl,Br,I][c]',
    ],
    'amide': [
        '[NX3][CX3](=O)',
    ],
    'carbonyl_inert': [
        '[CX4][CX3](=O)[CX4]',
        '[CX3](=O)[OX2H0]',
    ],
    'general': [
        '[#6]',
    ],
}
TOX_GROUP_ORDER_V2 = [
    'michael_acceptor', 'quinone', 'aromatic_aldehyde', 'aliphatic_aldehyde',
    'primary_ar_amine', 'secondary_ar_amine', 'nitro_aromatic', 'nitro_aliphatic',
    'alkyl_halide', 'epoxide', 'phenol', 'carboxylic_acid', 'sulfonamide',
    'thiol', 'amide', 'aliphatic_amine', 'n_heterocycle', 'aryl_halide', 'carbonyl_inert', 'general',
]

ATOM_BIN_EDGES  = [-0.5, 8.5, 12.5, 16.5, 20.5, np.inf]
ATOM_BIN_LABELS = ['ha<=8', 'ha9-12', 'ha13-16', 'ha17-20', 'ha>=21']



print(f'Config loaded. PRIMARY_ASSAY={PRIMARY_ASSAY}, MAX_ROWS={MAX_ROWS}, '
      f'K={N_FOLDS}, primary affinity={PRIMARY_AFFINITY}, '
      f'primary strat={STRAT_STRATEGY}, rounding={PRIMARY_ROUNDING}')


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


# Cell 1.4 - environment / optional dependencies
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
    'rdkit':                  _try('rdkit'),
    'xgboost':                _try('xgboost'),
    'umap':                   _try('umap', 'umap'),
    'alphashape':             _try('alphashape'),
    'shapely':                _try('shapely'),
    'chemprop':               _try('chemprop'),
    'dgllife':                _try('dgllife'),
    'iterative_stratification': _try('iterative_stratification') or _try('skmultilearn') or _try('iterstrat'),
}
for k, v in _versions.items():
    print(f'  {k:24s}  {v if v else "MISSING (optional)"}')

with open(OUT_META / 'environment.json', 'w') as fh:
    json.dump({'python': sys.version, 'platform': platform.platform(),
               'versions': _versions,
               'shield_version': __import__('shield').__version__}, fh, indent=2)

if _OPT['rdkit'] is None:
    raise RuntimeError('RDKit is required for Tox21 featurization. Install via conda-forge.')


# ## 2. Data loading and EDA

# In[5]:


# Cell 2.1 - load Tox21 dataset, drop mol_id, filter to primary-assay-labeled subset
print(f'Loading from {DATASET_PATH}')
df_raw = pd.read_csv(DATASET_PATH)
print(f'Raw shape: {df_raw.shape}')

if ID_COL in df_raw.columns:
    df_raw = df_raw.drop(columns=[ID_COL])
    print(f'Dropped column "{ID_COL}".')

# Per-assay missing-data audit (saved to SI tables)
missing_audit = []
for a in ASSAY_COLS:
    if a in df_raw.columns:
        sub = df_raw[a].dropna()
        missing_audit.append({
            'assay': a,
            'n_labeled': int(sub.notna().sum()),
            'n_missing':  int(df_raw[a].isna().sum()),
            'positive_count': int((sub == 1).sum()),
            'positive_rate':  float((sub == 1).mean()),
        })
audit_df = pd.DataFrame(missing_audit)
save_table(audit_df, 'per_assay_missing_and_imbalance', where='si')
display(audit_df)


# In[6]:


# Cell 2.2 - filter to compounds labeled on the primary assay
print(f'Filtering to rows where "{PRIMARY_ASSAY}" is labeled...')
df = df_raw.reset_index(drop=True)
print(f'After primary-assay filter: {len(df):,} rows')

if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df):,} rows (seed={RANDOM_SEED})')

n = len(df)

pos_rates = df[ASSAY_COLS].mean(skipna=True)
Y_FULL = df[ASSAY_COLS].fillna(pos_rates).values.astype(np.float32)
Y_PRIMARY    = df[PRIMARY_ASSAY].values.astype(np.float32)
prim_labeled = ~np.isnan(Y_PRIMARY)
print(f'Total rows: {n:,}; {PRIMARY_ASSAY} labeled: {prim_labeled.sum():,}')
y_int = Y_PRIMARY[prim_labeled].astype(np.int64)
print(f'Primary-assay class balance (labeled): {dict(zip(*np.unique(y_int, return_counts=True)))}')
print(f'Primary-assay positive rate (labeled): {y_int.mean()*100:.2f}%')


# In[7]:


# Cell 2.3 - attach DataSAIL splits and filter dataset to matched molecules

from rdkit import Chem as _Chem
from rdkit.Chem.inchi import MolToInchi as _MolToInchi
import warnings as _warnings
_warnings.filterwarnings('ignore', category=UserWarning)

DATASAIL_PATH = str(_REPO_ROOT / 'splits' / 'tox21_DataSAIL.csv')

PRECOMPUTED_FOLD_MAP = {
    'train': 0,
    'val':   1,
    'test':  1,
}

def _conn_key(s):
    """Connectivity-only InChI string used as a match key.
    Retains only formula + /c (bond topology) from the -SNon InChI,
    stripping stereo, H-position, and charge layers.  This bridges
    the stereo / protonation / tautomer differences between source and
    DataSAIL SMILES without risking false positives on connectivity.
    """
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


pos_rates     = df[ASSAY_COLS].mean(skipna=True)
Y_FULL        = df[ASSAY_COLS].fillna(pos_rates).values.astype(np.float32)
Y_PRIMARY     = df[PRIMARY_ASSAY].values.astype(np.float32)
prim_labeled = ~np.isnan(Y_PRIMARY)
y_int         = Y_PRIMARY[prim_labeled].astype(np.int64)
print(f'Post-filter: {n:,} rows; {PRIMARY_ASSAY} labeled: {prim_labeled.sum():,}')
print(f'Primary-assay class balance (labeled): {dict(zip(*np.unique(y_int, return_counts=True)))}')



# In[8]:


# Cell 2.4 - per-assay positive-rate bar chart (SI exhibit)
fig, ax = plt.subplots(figsize=(10, 4))
audit_sorted = audit_df.sort_values('positive_rate')
ax.bar(audit_sorted['assay'], audit_sorted['positive_rate'] * 100,
       color=['#332288' if 'NR-' in a else '#117733' for a in audit_sorted['assay']],
       edgecolor='black')
ax.axhline(50, color='red', ls='--', lw=0.5, alpha=0.6)
ax.set_ylabel('positive rate (%)')
ax.set_title('Tox21 per-assay positive rates (NR = nuclear receptor, SR = stress response)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'per_assay_positive_rates', where='si')
plt.show()


# In[9]:


# Cell 2.5 - assay correlation heatmap (drives label-augmentation justification)
corr = df[ASSAY_COLS].corr().values
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(ASSAY_COLS))); ax.set_xticklabels(ASSAY_COLS, rotation=80, fontsize=8)
ax.set_yticks(range(len(ASSAY_COLS))); ax.set_yticklabels(ASSAY_COLS, fontsize=8)
ax.set_title('Tox21 assay-assay correlations (pairwise complete)')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
save_fig(fig, 'assay_correlations', where='si')
save_array(corr, 'assay_correlations', where='si')
plt.show()

ASSAY_CORR = corr
np.fill_diagonal(ASSAY_CORR, 0.0)


# ## 3. Featurization

# In[10]:


# Cell 3.1 - compute ECFP4 fingerprints (cached)
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

CACHE_FP = OUT_INTER / 'featurization' / f'fps_n{n}_r{ECFP_RADIUS}_b{ECFP_NBITS}_{PRIMARY_ASSAY}.pkl'

if CACHE_FP.exists():
    with open(CACHE_FP, 'rb') as fh:
        cache_obj = pickle.load(fh)
    bv  = cache_obj['bv']
    fps = cache_obj['fps']
    if len(fps) != len(df):
        valid = np.array([Chem.MolFromSmiles(s) is not None for s in df[SMILES_COL]])
        df = df[valid].reset_index(drop=True)
        Y_FULL = Y_FULL[valid]; Y_PRIMARY = Y_PRIMARY[valid]
        prim_labeled = prim_labeled[valid]

        n = len(df)
        print(f'Aligned df/Y arrays to cached fps: new n={n}')
    print(f'Loaded cached fingerprints from {CACHE_FP.name}')

else:
    fps = []
    for s in df[SMILES_COL]:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, ECFP_RADIUS, nBits=ECFP_NBITS) if m else None)
    valid = np.array([fp is not None for fp in fps])
    if not valid.all():
        df = df[valid].reset_index(drop=True)
        Y_FULL = Y_FULL[valid]; Y_PRIMARY = Y_PRIMARY[valid]
        prim_labeled = prim_labeled[valid]

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

print(f'Mean bits per molecule: {bv.sum(axis=1).mean():.1f}')


# In[11]:


# Cell 3.2 - scaffolds, ring systems, atom-count bins
mols = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]
scaffolds, rings = [], []
for m in mols:
    if m is None:
        scaffolds.append(''); rings.append('acyclic'); continue
    sc = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(m), includeChirality=False)
    scaffolds.append(sc if sc else 'acyclic')
    msc = Chem.MolFromSmiles(sc) if sc else None
    rings.append(sc if (msc is not None and msc.GetRingInfo().NumRings() > 0) else 'acyclic')

natoms = np.array([m.GetNumHeavyAtoms() if m else 0 for m in mols])
atom_bin = pd.cut(natoms, bins=ATOM_BIN_EDGES, labels=ATOM_BIN_LABELS).astype(str)

df['_scaffold'] = scaffolds
df['_ring']     = rings
df['_atombin']  = atom_bin

print(f'Distinct scaffolds: {df["_scaffold"].nunique()}')
print(f'Distinct rings:     {df["_ring"].nunique()}')
print(f'Atom-count bins:    {df["_atombin"].value_counts().to_dict()}')


# In[12]:


# Cell 3.3 - affinity matrices: Tanimoto ε-NN (M term) + kNN variants for SI

def _knn_sparsify(A, k):
    """k-nearest-neighbor sparsification (kept for SI sweep comparisons)."""
    n_ = A.shape[0]; kk = min(k, n_ - 1)
    knn_idx = np.argpartition(-A, kth=kk, axis=1)[:, :kk]
    out = np.zeros_like(A)
    rr  = np.repeat(np.arange(n_), kk)
    cc  = knn_idx.reshape(-1)
    out[rr, cc] = A[rr, cc]
    return np.maximum(out, out.T)

def _eps_sparsify(A, eps, scaffolds=None):
    """Threshold (ε-NN) sparsification, optionally scaffold-gated.

    Base rule: retain edge (i,j) if Tanimoto(i,j) > eps.
    Scaffold gate (active when `scaffolds` is supplied):
        additionally require scaffold[i] == scaffold[j].
        Rationale: T>eps across *different* Murcko scaffolds reflects
        incidental ECFP4 bit overlap (shared aromatic environments) rather
        than genuine SAR-series membership.  Keeping those cross-scaffold
        edges creates transitive percolation chains that merge otherwise
        unrelated scaffold families into a single giant component, making
        M's LP gradient contradictory for 30%+ of the dataset.
        Removing them gives M purely scaffold-coherent components.
    Exception: acyclic molecules (scaffold == 'acyclic') are exempt from
        the scaffold gate — two ring-free aliphatics with T>eps are
        genuinely similar and have no ring scaffold to compare.
    Edge weights are kept (not binarized) so higher-similarity pairs exert
    stronger clustering pressure in the M LP gradient.
    """
    out = np.where(A > eps, A, 0.0).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    if scaffolds is not None:
        sc = np.asarray(scaffolds)
        rows, cols = np.nonzero(np.triu(out, k=1))
        same  = (sc[rows] == sc[cols])
        acyc  = (sc[rows] == 'acyclic') | (sc[cols] == 'acyclic')
        bad   = ~(same | acyc)
        out[rows[bad], cols[bad]] = 0.0
        out[cols[bad], rows[bad]] = 0.0
    return out

affinities = {}
for at in AFFINITY_TYPES:
    cache = OUT_INTER / 'featurization' / f'aff_{at}_n{n}_{PRIMARY_ASSAY}.npy'
    if cache.exists():
        A_full = np.load(cache)
        print(f'  loaded cached affinity ({at}) shape={A_full.shape}')
    elif at == 'tanimoto':
        A_full = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            A_full[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        np.fill_diagonal(A_full, 0.0)
        np.save(cache, A_full)
    else:
        A_full = cosine_similarity(bv.astype(np.float32)).astype(np.float32)
        np.fill_diagonal(A_full, 0.0)
        np.save(cache, A_full)
    affinities[at] = {'full': A_full,
                  **{f'knn_k{k}': _knn_sparsify(A_full, k) for k in KNN_AFFINITY_K_SI}}

A_full = affinities[PRIMARY_AFFINITY]['full']

_scaffolds = df['_scaffold'].values
AM = _eps_sparsify(A_full, EPS_AFF_M, scaffolds=_scaffolds)
_n_edges   = int((AM > 0).sum()) // 2
_n_raw     = int((np.triu(A_full > EPS_AFF_M, k=1)).sum())
_removed   = _n_raw - _n_edges
_density   = float((AM > 0).mean())
print(f'M affinity (scaffold-gated ε-NN, ε={EPS_AFF_M}): '
      f'shape={AM.shape}, unique edges={_n_edges:,}, density={_density:.5f}')
print(f'  ε-NN before scaffold gate: {_n_raw:,} edges')
print(f'  Cross-scaffold edges removed: {_removed:,} ({_removed/max(_n_raw,1)*100:.1f}%)')
print(f'A_full (full Tanimoto) retained for diagnostics and D kernel: shape={A_full.shape}')


# In[13]:


# Cell 3.4 - M Tier-2: Ring Topology Class clustering for Tier-1 singleton molecules

import scipy.sparse as _sp2
import scipy.sparse.csgraph as _scg2

_am_sp_t1 = _sp2.csr_matrix(AM.astype(np.float32))
_n_comp_t1, _comp_lab_t1 = _scg2.connected_components(_am_sp_t1, directed=False)
_comp_sizes_t1 = np.bincount(_comp_lab_t1)

_comp_is_small  = (_comp_sizes_t1 <= T2_SMALL_COMP_MAX)
_singleton_mask = _comp_is_small[_comp_lab_t1]
_singleton_idx  = np.where(_singleton_mask)[0]
n_singletons    = int(_singleton_mask.sum())
n_pure_isolated = int((_comp_sizes_t1 == 1).sum())
n_in_small_comps = n_singletons - n_pure_isolated

print(f'Tier-2 eligible (T1 comp size 1-{T2_SMALL_COMP_MAX}): {n_singletons} / {n}  '
      f'({n_singletons/n*100:.1f}%)')
print(f'  purely isolated (comp size=1) : {n_pure_isolated}  ({n_pure_isolated/n*100:.1f}%)')
print(f'  in small comps (2-{T2_SMALL_COMP_MAX})         : {n_in_small_comps}')

def _ring_topology_key(scaffold_smi: str):
    """Coarse ring-topology key from a Murcko scaffold SMILES.

    Returns (n_rings_capped, n_aromatic, ring_sizes_tuple).
    Acyclic scaffolds ('' or 'acyclic') return (0, 0, ()).
    """
    if not scaffold_smi or scaffold_smi == 'acyclic':
        return (0, 0, ())
    mol = Chem.MolFromSmiles(scaffold_smi)
    if mol is None:
        return (0, 0, ())
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return (0, 0, ())
    n_rings        = len(atom_rings)
    n_rings_capped = min(n_rings, 4)
    ring_sizes     = tuple(sorted(len(r) for r in atom_rings))
    n_aromatic = sum(
        1 for ring in atom_rings
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
    )
    return (n_rings_capped, n_aromatic, ring_sizes)


_scaffold_col = df['_scaffold'].values
_topo_keys = [_ring_topology_key(_scaffold_col[i]) for i in _singleton_idx]

from collections import defaultdict
_key_to_singletons = defaultdict(list)
for idx, key in zip(_singleton_idx, _topo_keys):
    _key_to_singletons[key].append(idx)

n_topo_classes = len(_key_to_singletons)
class_sizes    = np.array([len(v) for v in _key_to_singletons.values()])
print(f'Distinct ring-topology classes among singletons: {n_topo_classes}')
print(f'  size distribution: min={class_sizes.min()}, '
      f'median={int(np.median(class_sizes))}, '
      f'max={class_sizes.max()}, mean={class_sizes.mean():.1f}')
print(f'  classes with size >= 2  : {(class_sizes >= 2).sum()}  (will generate Tier-2 edges)')
print(f'  classes with size > {T2_MAX_COMP}: {(class_sizes > T2_MAX_COMP).sum()}  '
      f'(will be sub-divided by scaffold ECFP4 at EPS_T2_FINE={EPS_T2_FINE})')

_oversized_keys   = {k for k, v in _key_to_singletons.items() if len(v) > T2_MAX_COMP}
_needs_scaffold_fp = set()
for k in _oversized_keys:
    _needs_scaffold_fp.update(_key_to_singletons[k])

_scaffold_fps = {}
if _needs_scaffold_fp:
    print(f'Computing scaffold ECFP4 for {len(_needs_scaffold_fp)} singletons '
          f'in {len(_oversized_keys)} oversized class(es)...')
    for idx in _needs_scaffold_fp:
        sc_smi = _scaffold_col[idx]
        if not sc_smi or sc_smi == 'acyclic':
            _scaffold_fps[idx] = fps[idx]
        else:
            m_sc = Chem.MolFromSmiles(sc_smi)
            _scaffold_fps[idx] = (
                AllChem.GetMorganFingerprintAsBitVect(m_sc, ECFP_RADIUS, nBits=ECFP_NBITS)
                if m_sc is not None else fps[idx]
            )

_t2_rows, _t2_cols = [], []
n_t2_direct = 0
n_t2_fine   = 0

for topo_key, member_idx in _key_to_singletons.items():
    if len(member_idx) < 2:
        continue

    if len(member_idx) <= T2_MAX_COMP:
        for a in range(len(member_idx)):
            for b in range(a + 1, len(member_idx)):
                _t2_rows.append(member_idx[a])
                _t2_cols.append(member_idx[b])
                n_t2_direct += 1

    else:
        m_cls = len(member_idx)
        sc_fp_list = [_scaffold_fps.get(idx) for idx in member_idx]
        T_sc = np.zeros((m_cls, m_cls), dtype=np.float32)
        for a in range(m_cls):
            if sc_fp_list[a] is None:
                continue
            valid_pos = [j for j, fp in enumerate(sc_fp_list) if fp is not None]
            valid_fps_list = [sc_fp_list[j] for j in valid_pos]
            sims = DataStructs.BulkTanimotoSimilarity(sc_fp_list[a], valid_fps_list)
            for vp, sim in zip(valid_pos, sims):
                T_sc[a, vp] = float(sim)
        np.fill_diagonal(T_sc, 0.0)

        edge_mat = (T_sc > EPS_T2_FINE).astype(np.uint8)
        _, comp_labels_sc = _scg2.connected_components(
            _sp2.csr_matrix(edge_mat), directed=False)
        comp_sizes_sc = np.bincount(comp_labels_sc)

        for cid in range(len(comp_sizes_sc)):
            comp_pos = np.where(comp_labels_sc == cid)[0]
            if len(comp_pos) < 2:
                continue
            if len(comp_pos) > T2_MAX_COMP:
                mean_sim = T_sc[np.ix_(comp_pos, comp_pos)].mean(axis=1)
                order    = np.argsort(-mean_sim)
                comp_pos = comp_pos[order[:T2_MAX_COMP]]
            sub_idx = [member_idx[p] for p in comp_pos]
            for a in range(len(sub_idx)):
                for b in range(a + 1, len(sub_idx)):
                    _t2_rows.append(sub_idx[a])
                    _t2_cols.append(sub_idx[b])
                    n_t2_fine += 1

_t1_edges = int((AM > 0).sum()) // 2

AM_orig = AM.copy()
if _t2_rows:
    _r = np.array(_t2_rows, dtype=np.int64)
    _c = np.array(_t2_cols, dtype=np.int64)
    new_slot = (AM[_r, _c] == 0.0)
    AM[_r[new_slot], _c[new_slot]] = T2_WEIGHT
    AM[_c[new_slot], _r[new_slot]] = T2_WEIGHT

n_t2_inserted = int((AM > 0).sum() // 2) - _t1_edges

_degree_combined = (AM > 0).sum(axis=1).astype(int)
_still_isolated  = int((_degree_combined == 0).sum())
_newly_connected = n_singletons - _still_isolated

_am_sp2t = _sp2.csr_matrix(AM.astype(np.float32))
_n_comp2, _comp_lab2 = _scg2.connected_components(_am_sp2t, directed=False)
_comp_sizes2 = np.bincount(_comp_lab2)
_test_size2  = int(round(FOLD_RATIOS[1] * n))


_small_mask_t2 = (_comp_sizes2 <= T2_SMALL_COMP_MAX)
_small_cids    = np.where(_small_mask_t2)[0]

if len(_small_cids) >= 2:
    _reps_t3 = [int(np.where(_comp_lab2 == cid)[0][0]) for cid in _small_cids]

    _r3 = np.array(_reps_t3[:-1], dtype=np.int64)
    _c3 = np.array(_reps_t3[1:],  dtype=np.int64)
    _new3 = (AM[_r3, _c3] == 0.0)
    AM[_r3[_new3], _c3[_new3]] = T2_WEIGHT
    AM[_c3[_new3], _r3[_new3]] = T2_WEIGHT

    _am_sp3 = _sp2.csr_matrix(AM.astype(np.float32))
    _n_comp3, _comp_lab3 = _scg2.connected_components(_am_sp3, directed=False)
    _comp_sizes3 = np.bincount(_comp_lab3)

    _merged_size       = int(_comp_sizes3[int(_comp_lab3[_reps_t3[0]])])
    _n_nodes_before    = int(_small_mask_t2[_comp_lab2].sum())

    print(f'\nTier-3 merge (size ≤ {T2_SMALL_COMP_MAX}):')
    print(f'  Components merged          : {len(_small_cids)}')
    print(f'  Nodes in those components  : {_n_nodes_before}')
    print(f'  Merged component size      : {_merged_size}')
    print(f'  Chain edges added          : {int(_new3.sum())}')
    print(f'  Total components after T3  : {_n_comp3}')
    print(f'  Merged comp vs test fold   : {_merged_size} vs {_test_size2}  '
          f'{"(!) EXCEEDS — LP infeasible" if _merged_size > _test_size2 else "(ok)"}')
else:
    print(f'\nTier-3 merge: fewer than 2 small components (size ≤ {T2_SMALL_COMP_MAX}) — nothing to merge.')


def _comp_hist(sizes, test_sz):
    """Component-size histogram using user-defined breakpoints.
    Bins: 1 | 2-9 | 10-35 | 35-60 | 60-100 | 100-1565 | >1565
    Boundaries are right-inclusive; each atom counted in exactly one bin.
    """
    bins = {
        'size=1'             : int((sizes == 1).sum()),
        'size 2-9'           : int(((sizes >= 2)   & (sizes <= 9)).sum()),
        'size 10-35'         : int(((sizes >= 10)  & (sizes <= 35)).sum()),
        'size 35-60'         : int(((sizes >= 36)  & (sizes <= 60)).sum()),
        'size 60-100'        : int(((sizes >= 61)  & (sizes <= 100)).sum()),
        'size 100-250'      : int(((sizes >= 101) & (sizes <= 250)).sum()),
        'size 250-500'      : int(((sizes >= 251) & (sizes <= 500)).sum()),
        'size 500-1565'      : int(((sizes >= 501) & (sizes <= 1565)).sum()),
        'size >1565 (!)'     : int((sizes > 1565).sum()),
    }
    return bins

_hist_t1  = _comp_hist(_comp_sizes_t1, _test_size2)
_hist_t12 = _comp_hist(_comp_sizes2,   _test_size2)

# Top-5 largest topology classes by member count
_top5 = sorted(_key_to_singletons.items(), key=lambda kv: -len(kv[1]))[:5]

# Tier-2 class size buckets
_cls1   = sum(1 for v in _key_to_singletons.values() if len(v) == 1)
_cls2_5 = sum(1 for v in _key_to_singletons.values() if 2 <= len(v) <= 5)
_cls6_T = sum(1 for v in _key_to_singletons.values() if 6 <= len(v) <= T2_MAX_COMP)
_cls_ov = sum(1 for v in _key_to_singletons.values() if len(v) > T2_MAX_COMP)
_n_in_ov = sum(len(v) for v in _key_to_singletons.values() if len(v) > T2_MAX_COMP)

W = 68
print('=' * W)
print('  M AFFINITY: BEFORE vs AFTER TIER-2')
print('=' * W)
print(f'  {"":32s}  {"BEFORE (Tier-1)":>18s}  {"AFTER (T1+T2)":>14s}')
print('-' * W)
print(f'  {"Unique edges":32s}  {_t1_edges:>18,}  {int((AM>0).sum())//2:>14,}')
print(f'  {"Isolated nodes":32s}  '
      f'{n_singletons:>14,} ({n_singletons/n*100:4.1f}%)  '
      f'{_still_isolated:>8,} ({_still_isolated/n*100:4.1f}%)')
print(f'  {"Newly connected by Tier-2":32s}  {"":>18s}  '
      f'{_newly_connected:>8,} ({_newly_connected/max(n_singletons,1)*100:.1f}% of singletons)')
print(f'  {"M coverage (>=1 edge)":>32s}  '
      f'{(n-n_singletons)/n*100:>17.1f}%  {(n-_still_isolated)/n*100:>13.1f}%')
print(f'  {"Connected components":32s}  {_n_comp_t1:>18,}  {_n_comp2:>14,}')
print(f'  {"Largest component":32s}  '
      f'{_comp_sizes_t1.max():>12,} ({_comp_sizes_t1.max()/n*100:.1f}%n)  '
      f'{_comp_sizes2.max():>6,} ({_comp_sizes2.max()/n*100:.1f}%n)')
print(f'  {f"Components > test fold ({_test_size2})":>32s}  '
      f'{int((_comp_sizes_t1>_test_size2).sum()):>18,}  '
      f'{int((_comp_sizes2>_test_size2).sum()):>14,}  <- target: 0')
print()
print(f'  Component-size histogram:')
for band in _hist_t12:
    print(f'    {band:<28s}: T1 only {_hist_t1[band]:>5}  |  T1+T2 {_hist_t12[band]:>5}')

print()
print('=' * W)
print('  TIER-2 CLASS BREAKDOWN')
print('=' * W)
print(f'  Ring-topology classes (n_topo_classes)  : {n_topo_classes}')
print(f'  Classes size=1  (no edges added)        : {_cls1}')
print(f'  Classes size 2-5                        : {_cls2_5}')
print(f'  Classes size 6-{T2_MAX_COMP} (direct connect)    : {_cls6_T}')
print(f'  Classes size >{T2_MAX_COMP} (sub-divided)       : {_cls_ov}  '
      f'({_n_in_ov} singletons affected)')
print(f'  Tier-2 edges (direct)                   : {n_t2_direct:,}')
print(f'  Tier-2 edges (fine sub-partition)       : {n_t2_fine:,}')
print(f'  Tier-2 edges inserted into AM           : {n_t2_inserted:,}')
print()
print(f'  Top-5 largest topology classes:')
print(f'    {"key (n_rings,n_arom,sizes)":<35s} members')
for _k, _v in _top5:
    _label = f'({_k[0]} rings, {_k[1]} arom, sizes={_k[2]})'
    _status = 'sub-divided' if len(_v) > T2_MAX_COMP else 'direct'
    print(f'    {_label:<35s} {len(_v):>4}  [{_status}]')

print()
print('=' * W)
print('  TUNING GUIDE')
print('=' * W)
print(f'  T2_MAX_COMP  = {T2_MAX_COMP:<5}  max component size allowed from a single topology class')
print(f'               lower  -> smaller clusters, more uniform sizes, fewer oversized comps')
print(f'               higher -> larger clusters, stronger OOD constraint, more molecules connected')
print(f'               watch  : "Components > test fold" must stay 0')
print()
print(f'  EPS_T2_FINE  = {EPS_T2_FINE:<5}  scaffold-ECFP4 threshold for sub-dividing oversized classes')
print(f'               only affects the {_cls_ov} class(es) with >{T2_MAX_COMP} members ({_n_in_ov} singletons)')
print(f'               lower  -> more connections within oversized classes, larger sub-clusters')
print(f'               higher -> fewer connections, smaller/sparser sub-clusters, more still-isolated')
print(f'               watch  : "Isolated nodes AFTER" and "Largest component"')
print()
print(f'  T2_WEIGHT    = {T2_WEIGHT:<5}  constant edge weight for Tier-2 edges')
print(f'               Tier-1 edge weights range: [{EPS_AFF_M:.2f}, 1.0] (raw Tanimoto)')
print(f'               Tier-2 must be < {EPS_AFF_M:.2f} (min Tier-1 weight) to let Tier-1 dominate')
print(f'               lower  -> softer topology grouping; balance/D terms override more easily')
print(f'               higher -> stronger topology grouping; approach {EPS_AFF_M:.2f} carefully')
print('=' * W)

am_t2_info = {
    'n_singletons_before'  : n_singletons,
    'n_isolated_after'     : _still_isolated,
    'n_newly_connected'    : _newly_connected,
    'coverage_gain_frac'   : _newly_connected / max(n_singletons, 1),
    'n_t2_edges_direct'    : n_t2_direct,
    'n_t2_edges_fine'      : n_t2_fine,
    'n_t2_edges_inserted'  : n_t2_inserted,
    'n_topo_classes'       : n_topo_classes,
    'EPS_T2_FINE'          : EPS_T2_FINE,
    'T2_MAX_COMP'          : T2_MAX_COMP,
    'T2_WEIGHT'            : T2_WEIGHT,
    'n_components_t1'      : _n_comp_t1,
    'n_components_combined': _n_comp2,
    'largest_comp_t1'      : int(_comp_sizes_t1.max()),
    'largest_comp_combined': int(_comp_sizes2.max()),
    'n_oversized_comps'    : int((_comp_sizes2 > _test_size2).sum()),
}


# In[14]:


# Cell 3.3.1 - M affinity graph: OOD efficiency diagnostic (Tier-1 + Tier-2)

import scipy.sparse as _sp
import scipy.sparse.csgraph as _scg

cache_tsne = OUT_INTER / 'featurization' / f'tsne_tox_n{n}.npy'
if cache_tsne.exists():
    tsne_m = np.load(cache_tsne)
    print(f'Loaded cached t-SNE: shape={tsne_m.shape}')
else:
    print(f'Running t-SNE on {n} × {bv.shape[1]} bits — may take 2-5 min ...')
    t0 = time.time()
    tsne_m = TSNE(n_components=2, perplexity=30, init='pca',
                  random_state=RANDOM_SEED).fit_transform(bv.astype(np.float32))
    np.save(cache_tsne, tsne_m)
    print(f't-SNE done in {time.time()-t0:.1f}s — cached for cell 4.2.')

degree = (AM > 0).sum(axis=1).astype(int)

_am_sp = _sp.csr_matrix(AM.astype(np.float32))
n_components, comp_labels = _scg.connected_components(_am_sp, directed=False)
comp_sizes = np.bincount(comp_labels)

n_isolated   = int((comp_sizes == 1).sum())
frac_isolated = n_isolated / n
largest_comp  = int(comp_sizes.max())
_test_size    = int(round(FOLD_RATIOS[1] * n))

if frac_isolated >= 0.30 and largest_comp <= 0.10 * n:
    _ood_verdict = '✓  GOOD  — many singleton nodes + small components'
elif frac_isolated >= 0.10 or largest_comp <= 0.20 * n:
    _ood_verdict = '~  MODERATE — reasonable scaffold coverage'
else:
    _ood_verdict = '✗  POOR  — few isolated nodes / giant component dominates'

_deg_log = np.log1p(degree.astype(float))
_vmax    = float(np.percentile(_deg_log[degree > 0], 95)) if (degree > 0).any() else 1.0

MAX_EDGES = 3000
_am_coo   = _am_sp.tocoo()
_eidx     = np.where(_am_coo.row < _am_coo.col)[0]
rng_m     = np.random.default_rng(RANDOM_SEED)
if len(_eidx) > MAX_EDGES:
    _eidx = rng_m.choice(_eidx, MAX_EDGES, replace=False)
_src = _am_coo.row[_eidx];  _dst = _am_coo.col[_eidx]
_wt  = _am_coo.data[_eidx]

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(21, 7))

mask0 = (degree == 0)
ax_a.scatter(tsne_m[mask0, 0], tsne_m[mask0, 1],
             c='#c8c8c8', s=5, linewidths=0, alpha=0.40,
             label=f'degree-0  ({mask0.sum():,}, {frac_isolated*100:.1f}%)')
sc_a = ax_a.scatter(tsne_m[~mask0, 0], tsne_m[~mask0, 1],
                    c=_deg_log[~mask0], cmap='YlOrRd', vmin=0, vmax=_vmax,
                    s=8, linewidths=0, alpha=0.75)
cb_a = plt.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.04)
cb_a.set_label('log(1 + degree)', fontsize=9)
_tick_vals = [1, 5, 10, 20, 50]
cb_a.set_ticks([np.log1p(v) for v in _tick_vals if np.log1p(v) <= _vmax])
cb_a.set_ticklabels([str(v) for v in _tick_vals if np.log1p(v) <= _vmax])
ax_a.set_title(f'Panel A — Node degree in AM\n'
               f'ε={EPS_AFF_M}, gray = degree-0 ({frac_isolated*100:.1f}%)', fontsize=11)
ax_a.legend(markerscale=2, fontsize=8, loc='upper right')
ax_a.set_xticks([]); ax_a.set_yticks([])

ax_b.scatter(tsne_m[:, 0], tsne_m[:, 1],
             c='#d0d0d0', s=4, linewidths=0, alpha=0.35, zorder=1)
_cmap_e = plt.get_cmap('Blues')
for s_, d_, w_ in zip(_src, _dst, _wt):
    x0, y0 = tsne_m[s_]
    x1, y1 = tsne_m[d_]
    ax_b.plot([x0, x1], [y0, y1],
              color=_cmap_e(float(w_)), lw=0.4, alpha=0.35, zorder=2)
ax_b.scatter(tsne_m[:, 0], tsne_m[:, 1],
             c='#444444', s=3, linewidths=0, alpha=0.50, zorder=3)
_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(EPS_AFF_M, 1.0))
cb_b = plt.colorbar(_sm, ax=ax_b, fraction=0.046, pad=0.04)
cb_b.set_label(f'Tanimoto weight (>{EPS_AFF_M})', fontsize=9)
ax_b.set_title(f'Panel B — Sampled AM edges  ({min(len(_eidx), MAX_EDGES):,} of {len(_eidx):,})\n'
               f'Total unique edges: {int((AM > 0).sum())//2:,}', fontsize=11)
ax_b.set_xticks([]); ax_b.set_yticks([])

_bins  = np.logspace(0, np.log10(max(largest_comp, 2)), 30)
ax_c.hist(comp_sizes, bins=_bins, color='#3C5488', edgecolor='white', linewidth=0.4)
ax_c.axvline(_test_size, color='red', ls='--', lw=1.4,
             label=f'test-fold size ({_test_size})')
ax_c.set_xscale('log'); ax_c.set_yscale('log')
ax_c.set_xlabel('Component size (molecules)', fontsize=10)
ax_c.set_ylabel('Count (log scale)', fontsize=10)
ax_c.set_title(f'Panel C — Connected-component sizes\n'
               f'{n_components:,} components  |  largest: {largest_comp}', fontsize=11)
ax_c.legend(fontsize=9)

fig.suptitle(f'M affinity OOD diagnostic  (Tier-1 eps={EPS_AFF_M} + Tier-2 ring topology)  n={n}',
             fontsize=13, y=1.01)
plt.tight_layout()
save_fig(fig, 'M_affinity_ood_diagnostic', where='main')
plt.show()

print(f'\nM affinity OOD diagnostic  (Tier-1 eps={EPS_AFF_M} + Tier-2 T2_WEIGHT={T2_WEIGHT}):')
print(f'  Nodes              : {n:,}')
print(f'  Unique edges       : {int((AM > 0).sum())//2:,}  '
      f'(density={float((AM > 0).mean()):.4f})')
print(f'  cf. kNN k=20       : {n*20//2:,} edges')
print(f'  Degree-0 nodes     : {n_isolated:,}  ({frac_isolated*100:.1f}%)'
      f'  ← molecules unconstrained by M')
print(f'  Connected comps    : {n_components:,}')
print(f'  Largest component  : {largest_comp}  ({largest_comp/n*100:.1f}% of n)')
print(f'  Test-fold size     : {_test_size}  ({FOLD_RATIOS[1]*100:.0f}% of n)')
_oversized = int((comp_sizes > _test_size).sum())
print(f'  Components > test  : {_oversized}  '
      f'← cannot fit entirely in test fold')
print(f'\n  OOD verdict: {_ood_verdict}')


# In[15]:


# D1: Tanimoto kernel for MMD-based distributional shift (D-shift mode).
K_MATRIX = A_full.copy()
np.fill_diagonal(K_MATRIX, 1.0)
np.save(OUT_INTER / 'featurization' / 'kernel_matrix.npy', K_MATRIX)
print(f'D1 Tanimoto kernel: shape={K_MATRIX.shape}, '
      f'diag_mean={np.diag(K_MATRIX).mean():.4f}, '
      f'off_diag_mean={K_MATRIX[~np.eye(K_MATRIX.shape[0], dtype=bool)].mean():.4f}, '
      f'symmetric? {np.allclose(K_MATRIX, K_MATRIX.T)}')


# In[16]:


# Cell 3.5 - pooled target marginal for D-match (uniform 1/n)
TARGET_MARG = np.full((N_FOLDS, n), 1.0 / n, dtype=np.float32)


# ## 4. Hierarchy construction

# In[17]:


# Cell 4.1 (Option E — revised) — Scaffold-Defining SMARTS + MW Co-filter Hierarchy

from rdkit.Chem import DataStructs, Descriptors, MolFromSmarts

MW_SIMPLE_ELECTROPHILE = 220.0

def _compile_smarts(group_name, patterns):
    compiled = []
    for sma in patterns:
        p = MolFromSmarts(sma)
        if p is not None:
            compiled.append(p)
        else:
            print(f"  WARNING: invalid SMARTS {sma!r} in '{group_name}' — skipped")
    return compiled

_FLAVONOID_SMARTS_RAW = [
    'O=C1CC(c2ccccc2)Oc2ccccc21',
    'O=C1C=C(c2ccccc2)Oc2ccccc21',
    'O=C1C=C(Oc2ccccc21)c1ccccc1',
    'O=C1C=COc2ccccc21',
    'O=C1OC=Cc2ccccc21',
    'O=c1ccc2ccccc2o1',
    'O=c1ccoc2ccccc12',
    'c1ccc(C(=O)C=Cc2ccccc2)cc1',
    '[c]C(=O)C=C[c]',
    'O=C1C(=Cc2ccccc2)Oc2ccccc21',
    'O=c1c(-c2ccccc2)coc2ccccc12',
    'O=c1cc(-c2ccccc2)oc2ccccc12',
    'Oc1ccc(C2CC(=O)c3ccccc3O2)cc1',
]
_ORGP_SMARTS_RAW = [
    '[P](=O)([OX2][CX4])([OX2][CX4])',
    '[P](=S)([OX2][CX4])([OX2][CX4])',
    '[P](=O)([OX2][c])([OX2][CX4])',
    '[P](=S)([OX2][c])([OX2][CX4])',
    '[P](=O)([OX2][CX3])([OX2][CX4])',
    '[P](=O)([OX2][CX4])([OX2][CX4])[OX2]',
    '[P](=S)([OX2][CX4])([OX2][CX4])[SX2]',
]
_MICHAEL_SMARTS_RAW = [
    '[#6](=[OX1])[CX3;!a]=[CX3;!a]',
    '[NX3][CX3](=[OX1])[CX3;!a]=[CX3;!a]',
    '[SX4](=[OX1])(=[OX1])[CX3;!a]=[CX3;!a]',
    '[CX3;!a]=[CX3;!a][CX2]#[NX1]',
    'O=C1C=CC(=O)O1',
    '[#6](=[OX1])C=C[#6](=[OX1])',
]
_PHTHALATE_SMARTS_RAW = [
    'O=C(O[#6])[c]1ccccc1C(=O)O[#6]',
    'OC(=O)[c]1ccccc1C(=O)O',
    'OC(=O)[c]1ccccc1C(=O)OCC',
    'O=C1OC(=O)c2ccccc21',
    'O=C(OCCCC)[c]1ccccc1C(=O)OCCCC',
]

_BIPHENYL_SMARTS      = MolFromSmarts('c1ccc(-c2ccccc2)cc1')
_DIPHENYLETHER_SMARTS = MolFromSmarts('c1ccc(Oc2ccccc2)cc1')
PCB_MIN_HALOGENS      = 2
PAH_MIN_AROMATIC_RINGS = 3

FLAVONOID_PATS  = _compile_smarts('flavonoid_chromone_chalcone', _FLAVONOID_SMARTS_RAW)
ORGP_PATS       = _compile_smarts('organophosphate_insecticide', _ORGP_SMARTS_RAW)
MICHAEL_PATS    = _compile_smarts('simple_reactive_electrophile', _MICHAEL_SMARTS_RAW)
PHTHALATE_PATS  = _compile_smarts('phthalate_ortho_diester', _PHTHALATE_SMARTS_RAW)

SCAFFOLD_DEF_L1_ORDER = [
    'Steroid_Phytoestrogen_NR_Ligand',
    'Polycyclic_Aromatic_AhR_Inducer',
    'Halogenated_Environmental_Pollutant',
    'Reactive_Small_Molecule',
    'Drug_Like_Scaffold',
]

SCAFFOLD_DEF_L1_MAP = {
    'steroid_gonane':                    'Steroid_Phytoestrogen_NR_Ligand',
    'flavonoid_chromone_chalcone':       'Steroid_Phytoestrogen_NR_Ligand',
    'PAH_fused_aromatic_3ring_plus':     'Polycyclic_Aromatic_AhR_Inducer',
    'halogenated_biphenyl_PCB_PBDE':     'Halogenated_Environmental_Pollutant',
    'organophosphate_insecticide':       'Halogenated_Environmental_Pollutant',
    'phthalate_ortho_diester':           'Halogenated_Environmental_Pollutant',
    'simple_reactive_electrophile':      'Reactive_Small_Molecule',
    'halogenated_monocyclic_arene':  'Halogenated_Environmental_Pollutant',
    'organohalogen_aliphatic':       'Halogenated_Environmental_Pollutant',
    'acyclic_N_compound':            'Drug_Like_Scaffold',

    'fused_66_N_aromatic_drug':          'Drug_Like_Scaffold',
    'fused_65_N_aromatic_drug':          'Drug_Like_Scaffold',
    'nonaromatic_N_het_bicyclic':        'Drug_Like_Scaffold',
    'polycyclic_carbocyclic_aromatic':   'Drug_Like_Scaffold',
    'monocyclic_benzene_simple':         'Drug_Like_Scaffold',
    'small_N_O_S_heterocycle':           'Drug_Like_Scaffold',
    'saturated_carbocycle_simple':       'Drug_Like_Scaffold',
    'acyclic_aliphatic':                 'Drug_Like_Scaffold',
}

SCAFFOLD_DEF_L2_ORDER = [
    'steroid_gonane',
    'flavonoid_chromone_chalcone',
    'halogenated_biphenyl_PCB_PBDE',
    'PAH_fused_aromatic_3ring_plus',
    'organophosphate_insecticide',
    'simple_reactive_electrophile',
    'phthalate_ortho_diester',
    'fused_66_N_aromatic_drug',
    'fused_65_N_aromatic_drug',
    'nonaromatic_N_het_bicyclic',
    'polycyclic_carbocyclic_aromatic',
    'halogenated_monocyclic_arene',
    'monocyclic_benzene_simple',
    'small_N_O_S_heterocycle',
    'saturated_carbocycle_simple',
    'acyclic_N_compound',
    'organohalogen_aliphatic',
    'acyclic_aliphatic',
]

SMALL_L2_THRESHOLD_E = 80


def _ring_info_e(mol):
    """Return ring statistics including the flat set of ring atom indices."""
    ri           = mol.GetRingInfo()
    n_rings      = ri.NumRings()
    atom_rings   = ri.AtomRings()
    ring_is_arom = [
        all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
        for ring in atom_rings
    ]
    n_aromatic   = sum(ring_is_arom)
    n_aliphatic  = n_rings - n_aromatic
    ring_atom_set = set(a for ring in atom_rings for a in ring)
    return ri, n_rings, atom_rings, ring_is_arom, n_aromatic, n_aliphatic, ring_atom_set


def _is_steroid_gonane(atom_rings, ring_is_arom, n_rings, n_aliphatic):
    """≥4 rings, ≥2 non-aromatic, ≥3 six-membered + ≥1 five-membered.

    Covers: estradiol (1 arom + 3 aliphatic), cholesterol (4 aliphatic),
    bile acids, cardenolides. Rejects: pyrene (4 fully aromatic).
    """
    if n_rings < 4 or n_aliphatic < 2:
        return False
    ring_sizes = [len(r) for r in atom_rings]
    return sum(1 for s in ring_sizes if s == 6) >= 3 and \
           sum(1 for s in ring_sizes if s == 5) >= 1


def _is_pah_3ring_plus(n_aromatic):
    """≥3 aromatic rings (any heteroatom — acridines/carbazoles included)."""
    return n_aromatic >= PAH_MIN_AROMATIC_RINGS


def _is_halogenated_biphenyl(mol):
    """Biphenyl (C-C) or diphenyl ether (O-bridge) + ≥2 Cl or Br."""
    has_biaryl = (
        (_BIPHENYL_SMARTS    is not None and mol.HasSubstructMatch(_BIPHENYL_SMARTS)) or
        (_DIPHENYLETHER_SMARTS is not None and mol.HasSubstructMatch(_DIPHENYLETHER_SMARTS))
    )
    if not has_biaryl:
        return False
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (17, 35)) >= PCB_MIN_HALOGENS


def _classify_fused_N_het(mol, atom_rings, ring_is_arom, n_aromatic):
    """Sub-classify a bicyclic+ N-het compound into one of 3 drug-scaffold groups.

    Priority: 6-5 N-aromatic > 6-6 N-aromatic > non-aromatic N-het.
    The 6-5 priority ensures purines (N in both 5 and 6) go to the benzimidazole
    family rather than the quinoline family — purines share indole-like ECFP4
    environments at radius 2 due to the 5-ring bridgehead geometry.

    Scientific mapping to Tox21:
      fused_66_N_aromatic_drug  — antimalarials, FQ antibiotics, kinase inhibitors,
                                   camptothecin analogues, ellipticine, neocuproine.
      fused_65_N_aromatic_drug  — PPIs (omeprazole/lansoprazole), anthelmintics
                                   (mebendazole/albendazole), purines (theophylline/
                                   caffeine/adenosine), tryptophan metabolites.
      nonaromatic_N_het_bicyclic — benzodiazepines, penicillins/cephalosporins,
                                    Ca-channel blockers (nifedipine), ACE inhibitors,
                                    tricyclic antidepressants (N in saturated ring).
    """
    arom_N_ring_sizes = [
        len(r) for r, is_a in zip(atom_rings, ring_is_arom)
        if is_a and any(mol.GetAtomWithIdx(a).GetAtomicNum() == 7 for a in r)
    ]

    if arom_N_ring_sizes:
        if any(s == 5 for s in arom_N_ring_sizes):
            return 'fused_65_N_aromatic_drug'
        return 'fused_66_N_aromatic_drug'

    return 'nonaromatic_N_het_bicyclic'


def classify_scaffold_e(mol):
    """Assign a molecule to one of 15 L2 scaffold-class groups (Option E, revised).

    Priority order: first matching rule wins.  Groups 1-7 are specific OOD-
    informative scaffolds.  Groups 8-15 refine the former three large catch-alls
    into domain-coherent sub-groups to reduce centroid-centroid Tanimoto.
    """
    if mol is None:
        return 'acyclic_aliphatic'

    (ri, n_rings, atom_rings, ring_is_arom,
     n_aromatic, n_aliphatic, ring_atom_set) = _ring_info_e(mol)

    if _is_steroid_gonane(atom_rings, ring_is_arom, n_rings, n_aliphatic):
        return 'steroid_gonane'

    if any(mol.HasSubstructMatch(p) for p in FLAVONOID_PATS):
        return 'flavonoid_chromone_chalcone'

    if _is_halogenated_biphenyl(mol):
        return 'halogenated_biphenyl_PCB_PBDE'

    if _is_pah_3ring_plus(n_aromatic):
        return 'PAH_fused_aromatic_3ring_plus'

    if any(mol.HasSubstructMatch(p) for p in ORGP_PATS):
        return 'organophosphate_insecticide'

    if any(mol.HasSubstructMatch(p) for p in MICHAEL_PATS):
        if Descriptors.ExactMolWt(mol) <= MW_SIMPLE_ELECTROPHILE:
            return 'simple_reactive_electrophile'

    if any(mol.HasSubstructMatch(p) for p in PHTHALATE_PATS):
        return 'phthalate_ortho_diester'

    if n_rings >= 2:
        has_N_in_ring = any(mol.GetAtomWithIdx(a).GetAtomicNum() == 7
                            for a in ring_atom_set)
        if has_N_in_ring:
            return _classify_fused_N_het(mol, atom_rings, ring_is_arom, n_aromatic)

    has_ring_het = any(mol.GetAtomWithIdx(a).GetAtomicNum() in (7, 8, 16)
                       for a in ring_atom_set)
    if n_rings >= 1 and any(ring_is_arom) and not has_ring_het:
        if n_rings >= 2:
            return 'polycyclic_carbocyclic_aromatic'
        else:
            n_hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
            return 'halogenated_monocyclic_arene' if n_hal >= 1 else 'monocyclic_benzene_simple'

    if n_rings == 0:
        n_hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
        if n_hal >= 1:
            return 'organohalogen_aliphatic'
        if any(a.GetAtomicNum() == 7 for a in mol.GetAtoms()):
            return 'acyclic_N_compound'
        return 'acyclic_aliphatic'

    if has_ring_het:
        return 'small_N_O_S_heterocycle'

    return 'saturated_carbocycle_simple'


def build_hierarchy(df_local, mols_local, fps_local, lambda_per_level):
    """Two-level scaffold-defining hierarchy (Option E).

    Returns SHIELD-compatible list [L2_dict, L1_dict (if lambda_l1>0)].
    """
    lambda_l2 = float(lambda_per_level[0])
    lambda_l1 = float(lambda_per_level[1]) if len(lambda_per_level) > 1 else 0.0
    _nbits    = next((fp.GetNumBits() for fp in fps_local if fp is not None), 1024)

    def _centroid_fp_arr(indices):
        if not indices:
            return np.zeros(_nbits, dtype=np.float32)
        buf = np.zeros(_nbits, dtype=np.int8)
        mat = np.zeros((len(indices), _nbits), dtype=np.float32)
        for row, idx in enumerate(indices):
            if fps_local[idx] is not None:
                DataStructs.ConvertToNumpyArray(fps_local[idx], buf)
                mat[row] = buf.astype(np.float32)
        return mat.mean(axis=0)

    def _tanimoto_centroids(a, b):
        dot   = float(np.dot(a, b))
        denom = float(np.dot(a, a) + np.dot(b, b) - dot)
        return dot / max(denom, 1e-12)

    group_of   = [classify_scaffold_e(m) for m in mols_local]
    grp_to_idx = defaultdict(list)
    for i, g in enumerate(group_of):
        grp_to_idx[g].append(i)

    local_l1_map    = dict(SCAFFOLD_DEF_L1_MAP)
    l1_to_l2_groups = defaultdict(list)
    for g in SCAFFOLD_DEF_L2_ORDER:
        if g in grp_to_idx:
            l1_to_l2_groups[SCAFFOLD_DEF_L1_MAP[g]].append(g)

    groups_to_remove = set()
    for l1_cls in SCAFFOLD_DEF_L1_ORDER:
        small = [g for g in l1_to_l2_groups.get(l1_cls, [])
                 if len(grp_to_idx.get(g, [])) < SMALL_L2_THRESHOLD_E]
        if not small:
            continue
        merged_name = f'{l1_cls}_general'
        merged_idx  = []
        for g in small:
            merged_idx.extend(grp_to_idx[g])
            groups_to_remove.add(g)
        grp_to_idx[merged_name] = sorted(merged_idx)
        local_l1_map[merged_name] = l1_cls
        print(f"  [merge] {l1_cls}: collapsed {len(small)} small groups "
              f"({', '.join(small)}) → '{merged_name}' ({len(merged_idx)} mols)")

    for g in groups_to_remove:
        del grp_to_idx[g]

    l2_surviving = [g for g in SCAFFOLD_DEF_L2_ORDER if g in grp_to_idx]
    l2_merged    = [f'{cls}_general' for cls in SCAFFOLD_DEF_L1_ORDER
                    if f'{cls}_general' in grp_to_idx]
    l2_active    = l2_surviving + l2_merged
    l2_desc      = [grp_to_idx[g] for g in l2_active]
    n_l2         = len(l2_active)

    centroids_l2 = [_centroid_fp_arr(desc) for desc in l2_desc]
    rs, cs, vs = [], [], []
    for i in range(n_l2):
        for j in range(i + 1, n_l2):
            t = _tanimoto_centroids(centroids_l2[i], centroids_l2[j])
            if t > 0.05:
                rs += [i, j]; cs += [j, i]; vs += [t, t]
    rs_arr = np.array(rs, dtype=np.int64)
    cs_arr = np.array(cs, dtype=np.int64)
    vs_arr = np.array(vs, dtype=np.float64)

    l1_grp_to_idx = defaultdict(list)
    for l2_grp, desc in zip(l2_active, l2_desc):
        l1_grp_to_idx[local_l1_map.get(l2_grp, 'Drug_Like_Scaffold')].extend(desc)

    l1_active    = [g for g in SCAFFOLD_DEF_L1_ORDER if g in l1_grp_to_idx]
    l1_desc      = [sorted(l1_grp_to_idx[g]) for g in l1_active]
    centroids_l1 = [_centroid_fp_arr(desc) for desc in l1_desc]
    n_l1         = len(l1_active)

    r1, c1, v1 = [], [], []
    for i in range(n_l1):
        for j in range(i + 1, n_l1):
            t = _tanimoto_centroids(centroids_l1[i], centroids_l1[j])
            if t > 0.05:
                r1 += [i, j]; c1 += [j, i]; v1 += [t, t]
    r1_arr = np.array(r1, dtype=np.int64)
    c1_arr = np.array(c1, dtype=np.int64)
    v1_arr = np.array(v1, dtype=np.float64)

    print(f"\n  ── L2 scaffold_class: {n_l2} active groups, "
          f"{len(vs)//2} edges, λ={lambda_l2} ──")
    for g, desc in zip(l2_active, l2_desc):
        l1_cls = local_l1_map.get(g, 'Drug_Like_Scaffold')
        flag   = "  *** LARGE" if len(desc) > 1200 else (
                 "  *** SMALL (may merge)" if len(desc) < 100 else "")
        print(f"    [{l1_cls[:3]}] {g:48s}: {len(desc):5d}{flag}")

    print(f"\n  ── L1 mechanism_target_class: {n_l1} mega-groups, "
          f"{len(v1)//2} edges, λ={lambda_l1} ──")
    l2_map_inv = defaultdict(list)
    for g in l2_active:
        l2_map_inv[local_l1_map.get(g, 'Drug_Like_Scaffold')].append(g)
    for cls, desc in zip(l1_active, l1_desc):
        print(f"    {cls:42s}: {len(desc):5d}  ← [{', '.join(l2_map_inv[cls])}]")

    print("\n  ── L2 centroid-centroid Tanimoto (off-diagonal low = strong OOD) ──")
    disp_n = min(n_l2, 10)
    hdr = " " * 49
    for j in range(disp_n):
        hdr += f"  {l2_active[j][:7]:7s}"
    print(hdr)
    for i in range(disp_n):
        row = f"  {l2_active[i]:48s}"
        for j in range(disp_n):
            t = _tanimoto_centroids(centroids_l2[i], centroids_l2[j])
            row += f"  {t:7.3f}"
        print(row)

    levels = [{
        'name':        'scaffold_class',
        'descendants': l2_desc,
        'similarity':  (rs_arr, cs_arr, vs_arr),
        'lambda':      lambda_l2,
    }]
    if lambda_l1 > 0.0:
        levels.append({
            'name':        'mechanism_target_class',
            'descendants': l1_desc})
    return levels


hierarchy_main = build_hierarchy(df, mols, fps, LAMBDA_MAIN)
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)



# In[18]:


# Cell 4.2 (Option E) — Scaffold-Defining Hierarchy Visualization

group_of_l2_vis = [classify_scaffold_e(m) for m in mols]
group_of_l2_arr = np.array(group_of_l2_vis)

group_of_l1_arr = np.array([
    SCAFFOLD_DEF_L1_MAP.get(g, 'Drug_Like_Scaffold')
    for g in group_of_l2_vis
])

cache_tsne = OUT_INTER / 'featurization' / f'tsne_tox_n{n}.npy'
if cache_tsne.exists():
    tsne_tox = np.load(cache_tsne)
    print(f'Loaded cached t-SNE: shape={tsne_tox.shape}')
else:
    print(f'Running t-SNE on {n} × {bv.shape[1]} bits — may take 2–5 min ...')
    t0 = time.time()
    tsne_tox = TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=RANDOM_SEED).fit_transform(bv.astype(np.float32))
    np.save(cache_tsne, tsne_tox)
    print(f't-SNE done in {time.time()-t0:.1f}s.')

SCAFFOLD_DEF_L2_PALETTE = {
    'steroid_gonane':                '#E64B35',
    'flavonoid_chromone_chalcone':   '#F4A460',
    'halogenated_biphenyl_PCB_PBDE': '#8B4513',
    'PAH_fused_aromatic_3ring_plus': '#7B2D8B',
    'organophosphate_insecticide':   '#808000',
    'simple_reactive_electrophile':  '#ADFF2F',
    'phthalate_ortho_diester':       '#CD853F',
    'drug_like_N_heterocycle':       '#3C5488',
    'benzenic_aromatic_scaffold':    '#B0B0B0',
    'aliphatic_acyclic_general':     '#D8D8D8',
}
SCAFFOLD_DEF_L2_ALPHA = {
    'drug_like_N_heterocycle':    0.30,
    'benzenic_aromatic_scaffold': 0.18,
    'aliphatic_acyclic_general':  0.20,
}
SCAFFOLD_DEF_L2_SIZE = {
    'drug_like_N_heterocycle':    5,
    'benzenic_aromatic_scaffold': 4,
    'aliphatic_acyclic_general':  4,
}

SCAFFOLD_DEF_L1_PALETTE = {
    'Steroid_Phytoestrogen_NR_Ligand':      '#E64B35',
    'Polycyclic_Aromatic_AhR_Inducer':      '#7B2D8B',
    'Halogenated_Environmental_Pollutant':  '#8B4513',
    'Reactive_Small_Molecule':              '#ADFF2F',
    'Drug_Like_Scaffold':                   '#B0B0B0',
}
SCAFFOLD_DEF_L1_ALPHA = {'Drug_Like_Scaffold': 0.20}

markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
active_l2_vis  = [g for g in SCAFFOLD_DEF_L2_ORDER if (group_of_l2_arr == g).any()]
merged_l2_vis  = [g for g in set(group_of_l2_arr)
                  if g not in SCAFFOLD_DEF_L2_ORDER and (group_of_l2_arr == g).any()]

bg_groups = ['drug_like_N_heterocycle', 'benzenic_aromatic_scaffold',
             'aliphatic_acyclic_general']
fg_groups = [g for g in active_l2_vis + merged_l2_vis if g not in bg_groups]

fig, (ax_l2, ax_l1) = plt.subplots(1, 2, figsize=(22, 10))

for g in bg_groups:
    mask = group_of_l2_arr == g
    if not mask.any():
        continue
    ax_l2.scatter(
        tsne_tox[mask, 0], tsne_tox[mask, 1],
        color=SCAFFOLD_DEF_L2_PALETTE.get(g, '#D0D0D0'),
        s=SCAFFOLD_DEF_L2_SIZE.get(g, 5),
        alpha=SCAFFOLD_DEF_L2_ALPHA.get(g, 0.20),
        linewidths=0, zorder=1,
        label=f"{g.replace('_', ' ')} ({mask.sum()})",
    )

for i, g in enumerate(fg_groups):
    mask = group_of_l2_arr == g
    if not mask.any():
        continue
    l1_cls = SCAFFOLD_DEF_L1_MAP.get(g, 'Drug_Like_Scaffold')
    edge   = SCAFFOLD_DEF_L1_PALETTE.get(l1_cls, '#555555')
    ax_l2.scatter(
        tsne_tox[mask, 0], tsne_tox[mask, 1],
        color=SCAFFOLD_DEF_L2_PALETTE.get(g, '#888888'),
        marker=markers[i % len(markers)],
        edgecolors=edge, linewidths=0.5,
        s=20, alpha=0.88, zorder=2,
        label=f"{g.replace('_', ' ')} ({mask.sum()})",
    )

ax_l2.set_title(f'L2 scaffold_class  ({len(active_l2_vis)} groups,  n={n})',
                fontsize=12)
ax_l2.set_xlabel('t-SNE 1'); ax_l2.set_ylabel('t-SNE 2')
ax_l2.set_xticks([]); ax_l2.set_yticks([])
ax_l2.legend(markerscale=1.8, fontsize=7.5, loc='upper right', ncol=2,
             framealpha=0.90, edgecolor='#bbbbbb',
             title='scaffold group (count)', title_fontsize=8)

bg_l1 = ['Drug_Like_Scaffold']
fg_l1 = [g for g in SCAFFOLD_DEF_L1_ORDER if g not in bg_l1]

for cls in bg_l1:
    mask = group_of_l1_arr == cls
    if not mask.any():
        continue
    ax_l1.scatter(
        tsne_tox[mask, 0], tsne_tox[mask, 1],
        color=SCAFFOLD_DEF_L1_PALETTE[cls],
        s=5, alpha=SCAFFOLD_DEF_L1_ALPHA.get(cls, 0.20),
        linewidths=0,
        label=f"{cls.replace('_', ' ')} ({mask.sum()})",
        zorder=1,
    )

for cls in fg_l1:
    mask = group_of_l1_arr == cls
    if not mask.any():
        continue
    ax_l1.scatter(
        tsne_tox[mask, 0], tsne_tox[mask, 1],
        color=SCAFFOLD_DEF_L1_PALETTE[cls],
        s=12, alpha=0.80, linewidths=0,
        label=f"{cls.replace('_', ' ')} ({mask.sum()})",
        zorder=2,
    )

ax_l1.set_title(f'L1 mechanism_target_class  (5 mega-groups,  n={n})', fontsize=12)
ax_l1.set_xlabel('t-SNE 1'); ax_l1.set_ylabel('t-SNE 2')
ax_l1.set_xticks([]); ax_l1.set_yticks([])
ax_l1.legend(markerscale=2.5, fontsize=8.5, loc='upper right',
             framealpha=0.92, edgecolor='#bbbbbb',
             title='mechanism class (count)', title_fontsize=9)

plt.suptitle(
    'Tox21 ECFP4 t-SNE  —  Option E: Scaffold-Defining SMARTS + MW Co-filter Hierarchy\n'
    'Edge color on L2 dots = parent L1 mechanism class  |  '
    'Red (steroids) and purple (PAHs) should be isolated islands from gray drug background',
    fontsize=11, y=1.01,
)
plt.tight_layout()
save_fig(fig, 'tsne_scaffold_defining_groups', where='main')
plt.show()

print('\nL2 scaffold_class group counts:')
for g in SCAFFOLD_DEF_L2_ORDER:
    mask = group_of_l2_arr == g
    if not mask.any():
        continue
    cnt    = int(mask.sum())
    l1cls  = SCAFFOLD_DEF_L1_MAP.get(g, 'Drug_Like_Scaffold')
    bar    = '█' * (cnt // 80)
    flag   = '  *** CHECK: below 80' if cnt < 80 else ''
    print(f"  [{l1cls[:3]}] {g:45s}: {cnt:5d}  {bar}{flag}")
for g in merged_l2_vis:
    mask = group_of_l2_arr == g
    cnt  = int(mask.sum())
    print(f"  [MRG] {g:45s}: {cnt:5d}  {'█' * (cnt // 80)}")

print('\nL1 mechanism_target_class group counts:')
for cls in SCAFFOLD_DEF_L1_ORDER:
    mask = group_of_l1_arr == cls
    if not mask.any():
        continue
    cnt = int(mask.sum())
    print(f"  {cls:42s}: {cnt:5d}  {'█' * (cnt // 80)}")

print('\nWithin-group mean Tanimoto (ECFP4, sampled 100 pairs) — higher = more coherent:')
rng = np.random.default_rng(RANDOM_SEED)
for g in SCAFFOLD_DEF_L2_ORDER:
    mask = group_of_l2_arr == g
    idx  = np.where(mask)[0]
    if len(idx) < 10:
        continue
    n_pairs = min(100, len(idx) * (len(idx) - 1) // 2)
    si = rng.integers(0, len(idx), size=n_pairs)
    sj = rng.integers(0, len(idx), size=n_pairs)
    same = si == sj
    si[same] = (si[same] + 1) % len(idx)
    tans = [
        DataStructs.TanimotoSimilarity(fps[idx[ii]], fps[idx[jj]])
        for ii, jj in zip(si, sj)
        if fps[idx[ii]] is not None and fps[idx[jj]] is not None
    ]
    if tans:
        print(f"  {g:45s}: {np.mean(tans):.3f} ± {np.std(tans):.3f}  (n={len(tans)})")


# ## 5. Baseline splitters

# In[19]:


# Cell 5.1 - implement baseline splitters
from sklearn.cluster import KMeans, AgglomerativeClustering

def _greedy_pack(labels, n_clusters):
    """First-fit decreasing bin-packing of cluster groups into the train fold.

    Sorts clusters by descending size, then iterates ALL clusters: adds a
    cluster to train if it fits within the remaining budget, skips it otherwise.
    This fills the budget much more tightly than stopping at the first
    oversized cluster (the old break behavior which left ~2.5% unfilled).
    Keeps every cluster intact (no partial splits).
    """
    grp_to_idx = defaultdict(list)
    for i, c in enumerate(labels): grp_to_idx[c].append(i)
    order = sorted(grp_to_idx.values(), key=len, reverse=True)
    f = np.ones(n, dtype=int)
    target_train = int(round(FOLD_RATIOS[0] * n)); cum = 0
    for g in order:
        if cum + len(g) <= target_train:
            for j in g: f[j] = 0
            cum += len(g)
        else: continue
    return f

def split_random(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[perm[:cut]] = 0
    return f

def split_class_stratified(seed, y=None):
    rng = np.random.default_rng(seed)
    if y is None:
        y = Y_PRIMARY
    f = np.ones(n, dtype=int)
    labeled = ~np.isnan(y.astype(np.float32))
    for c in np.unique(y[labeled]):
        idx = np.where((y == c) & labeled)[0]
        rng.shuffle(idx)
        cut = int(round(FOLD_RATIOS[0] * len(idx)))
        f[idx[:cut]] = 0
    unlabelled_idx = np.where(~labeled)[0]
    if len(unlabelled_idx):
        rng.shuffle(unlabelled_idx)
        cut = int(round(FOLD_RATIOS[0] * len(unlabelled_idx)))
        f[unlabelled_idx[:cut]] = 0
    return f

def split_scaffold(seed):
    labels_map = {s: i for i, s in enumerate(df['_scaffold'].unique())}
    labels = np.array([labels_map[s] for s in df['_scaffold']])
    return _greedy_pack(labels, len(labels_map))

def split_kmeans(seed, n_clusters=20):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(bv.astype(np.float32))
    return _greedy_pack(km.labels_, n_clusters)

def split_kennard_stone(seed):
    X = bv.astype(np.float32)
    centroid = X.mean(axis=0, keepdims=True)
    d = np.linalg.norm(X - centroid, axis=1)
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int); f[np.argsort(-d)[:n_test]] = 1
    return f

def split_multilabel_stratified(seed):
    """Multilabel-aware stratified split using the iterative-stratification
    library if available, otherwise fall back to primary-assay class strat."""
    if _OPT['iterative_stratification'] is not None:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as MSKF
        Y_bin = (Y_FULL >= 0.5).astype(int)
        mskf = MSKF(n_splits=int(round(1.0 / FOLD_RATIOS[1])),
                    shuffle=True, random_state=seed)
        for _, test_idx in mskf.split(np.arange(n).reshape(-1, 1), Y_bin):
            f = np.zeros(n, dtype=int); f[test_idx] = 1; return f
    return split_class_stratified(seed)

BASELINE_SPLITTERS = {
    'RANDOM':               split_random,
    'CLASS_STRATIFIED':     split_class_stratified,
    'MULTILABEL_STRATIFIED':split_multilabel_stratified,
    'SCAFFOLD':             split_scaffold,
    'KMEANS':               split_kmeans,
    'KENNARDSTONE':         split_kennard_stone,
}


# In[20]:


# Cell 5.2 - run all baseline splitters
baseline_results = {}

for name, fn in BASELINE_SPLITTERS.items():
    f = fn(RANDOM_SEED)
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(
        OUT_INTER / 'splits' / f'{name}.csv', index=False)
    print(f'  {name:22s}  sizes={sizes.round(3).tolist()}')


# In[21]:


# Cell 5.3 - precomputed splits from tox21_DataSAIL.csv

for _pc_name, _pc_col in [('SPLIT_BASE', '_split_base'), ('SPLIT_DATASAIL', '_split_datasail')]:
    _f = np.array([PRECOMPUTED_FOLD_MAP[lb] for lb in df[_pc_col].values], dtype=int)
    _sizes = np.bincount(_f, minlength=N_FOLDS) / n
    baseline_results[_pc_name] = _f
    _out = OUT_INTER / 'splits' / f'{_pc_name}.csv'
    pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(_out, index=False)
    print(f'  {_pc_name:16s}  sizes={_sizes.round(3).tolist()}')


# In[22]:


# Cell 5.4 - additional splitting tools: DeepChem-style (DC_BUTINA, DC_FINGERPRINT, DC_MAXMIN, DC_SCAFFOLD, DC_WEIGHT).

import warnings as _w54
_w54.filterwarnings('ignore')


def split_dc_butina(seed):
    """Sphere-exclusion (Leader) clustering on ECFP4 — C++ LeaderPicker, distance cutoff 0.4
    (Tanimoto >= 0.6), then greedy-pack clusters into folds. Falls back to exact
    Taylor-Butina (vectorized distance build) if the picker is unavailable."""
    try:
        from rdkit.SimDivFilters import rdSimDivPickers as _sdp
        lp = _sdp.LeaderPicker()
        leaders = list(lp.LazyBitVectorPick(fps, len(fps), 0.4))
        n_clusters = len(leaders)
        X = bv.astype(np.float32); cnt = X.sum(1)
        Lidx = np.asarray(leaders, dtype=np.int64)
        Lx = X[Lidx]; Lc = cnt[Lidx]
        labels = np.empty(n, dtype=int); CH = 4096
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
        X = bv.astype(np.float32); cnt = X.sum(1)
        dists = np.empty(n * (n - 1) // 2, dtype=np.float64); pos = 0
        for i in range(1, n):
            inter = X[i] @ X[:i].T
            den = cnt[i] + cnt[:i] - inter
            dists[pos:pos + i] = 1.0 - inter / np.clip(den, 1e-8, None); pos += i
        clusters = _Butina.ClusterData(dists, n, 0.4, isDistData=True)
        labels = np.empty(n, dtype=int)
        for cid, cl in enumerate(clusters):
            for idx in cl: labels[idx] = cid
        return _greedy_pack(labels, len(clusters))

def split_dc_fingerprint(seed):
    """Lexicographic sort on packed ECFP4 bits; first 80% train, last 20% test."""
    packed  = np.packbits(bv, axis=1)
    order   = np.lexsort(packed.T[::-1])
    n_train = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[order[:n_train]] = 0
    return f

def split_dc_maxmin(seed):
    """MaxMin diversity picker — C++ MaxMinPicker (lazy distances); test = the diverse
    picked set. Falls back to vectorized numpy greedy loop."""
    n_test = int(round(FOLD_RATIOS[1] * n))
    try:
        from rdkit.SimDivFilters import rdSimDivPickers as _sdp
        mmp = _sdp.MaxMinPicker()
        try:
            picks = mmp.LazyBitVectorPick(fps, len(fps), n_test, [], int(seed))
        except Exception:
            picks = mmp.LazyBitVectorPick(fps, len(fps), n_test)
        f = np.zeros(n, dtype=int); f[np.asarray(list(picks), dtype=np.int64)] = 1
        return f
    except Exception as _mex:
        print(f'    [DC_MAXMIN] MaxMinPicker unavailable ({_mex}); numpy greedy...', flush=True)
        rng = np.random.default_rng(seed)
        X = bv.astype(np.float32); counts = X.sum(axis=1)
        start = int(rng.integers(n)); selected = [start]
        inter = X @ X[start]
        tani = inter / (counts + counts[start] - inter).clip(1e-8)
        min_dists = (1.0 - tani).astype(np.float64); min_dists[start] = -np.inf
        while len(selected) < n_test:
            nxt = int(np.argmax(min_dists)); selected.append(nxt)
            inter = X @ X[nxt]
            tani = inter / (counts + counts[nxt] - inter).clip(1e-8)
            np.minimum(min_dists, 1.0 - tani, out=min_dists); min_dists[nxt] = -np.inf
        f = np.zeros(n, dtype=int); f[selected] = 1
        return f

def split_dc_scaffold(seed):
    """DeepChem scaffold split using the same _scaffold column as SCAFFOLD baseline."""
    from collections import defaultdict as _dd
    grp = _dd(list)
    for i, sc in enumerate(df['_scaffold']): grp[sc].append(i)
    labels_map = {sc: cid for cid, sc in enumerate(grp)}
    labels = np.array([labels_map[sc] for sc in df['_scaffold']])
    return _greedy_pack(labels, len(grp))

def split_dc_weight(seed):
    """Molecular-weight split: heaviest molecules go to test."""
    from rdkit.Chem.Descriptors import MolWt as _MolWt
    mw = np.array([_MolWt(m) if m else 0.0 for m in mols])
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int); f[np.argsort(-mw)[:n_test]] = 1
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
        _f     = _fn(RANDOM_SEED)
        _sizes = np.bincount(_f, minlength=N_FOLDS) / n
        baseline_results[_name] = _f
        pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(
            OUT_INTER / 'splits' / f'{_name}.csv', index=False)
        print(f'  {_name:22s}  sizes={_sizes.round(3).tolist()}')
    except Exception as _exc:
        print(f'  {_name:22s}  FAILED: {_exc}')


# ## 6. SHIELD primary configurations and Stage-0 normalizer table

# In[23]:


# Cell 6.1 - SHIELD ablation grid
SHIELD_CONFIGS = {
    'HMD-SHIELD':     dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU),
    'H-SHIELD':   dict(alpha=ALPHA, beta=0.0,  gamma=0.0,   eta=ETA, mu=MU),
    'M-SHIELD':   dict(alpha=0.0,   beta=BETA, gamma=0.0,   eta=ETA, mu=MU),
    'D-SHIELD':   dict(alpha=0.0,   beta=0.0,  gamma=GAMMA, eta=ETA, mu=MU),
}
print('SHIELD config grid:')
for k, c in SHIELD_CONFIGS.items():
    print(f'  {k:18s}  {c}')


# In[24]:


# Cell 6.2 - run all SHIELD configurations
shield_results = {}
shield_diagnostics = {}
t_all = time.time()
CACHED_NORMALIZERS = None

for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    kwargs = dict(
        n=n, K=N_FOLDS,
        r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main if cfg['alpha'] > 0 else None,
        affinity=AM if cfg['beta'] > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        stratification_strategy=STRAT_STRATEGY if cfg['mu'] > 0 else 'none',
        n_bins=STRAT_N_BINS,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'], mu=cfg['mu'], nu=NU, tau=TAU,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=PRIMARY_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        stage0_samples=SHIELD_STAGE0_SAMPLES,
        precomputed_normalizers=CACHED_NORMALIZERS,
        sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
        kernel_matrix=K_MATRIX if cfg['gamma'] > 0 else None,
        y_target=None,
        class_labels=None,
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
        print(f'  {cfg_name:18s}  wall={time.time()-t0:.1f}s  '
              f'sizes={(np.bincount(fold, minlength=N_FOLDS)/n).round(3).tolist()}')
    except Exception as exc:
        print(f'  {cfg_name:18s}  FAILED: {exc}')

print(f'Total SHIELD wall time: {time.time()-t_all:.1f}s')


# In[25]:


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

# In[26]:


# Cell 7.1 - cross-fold structural similarity L(pi)

A_pos = np.clip(A_full, 0.0, 1.0)

kappa = np.ones(A_pos.shape[0])

kappa_matrix = kappa.reshape(-1, 1) * kappa

TOTAL_SIM = float((A_pos * kappa_matrix).sum())

def cross_fold_similarity(fold, A=A_pos, kappa=kappa_matrix):
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


# In[27]:


# Cell 7.2 - L(pi) bar chart
fig, ax = plt.subplots(figsize=(10, 4))
agg = lpi_df.set_index('config')['scaled_L_pi'].sort_values()
colors = ['#888888' if c in baseline_results else '#4477AA' for c in agg.index]
ax.bar(range(len(agg)), agg.values, color=colors, edgecolor='black')
ax.axhline(agg.loc['RANDOM'] if 'RANDOM' in agg.index else 1.0,
           color='red', ls='--', lw=1, label='RANDOM baseline')
ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg.index, rotation=45, ha='right')
ax.set_ylabel('scaled L(pi)')
ax.set_title('Cross-fold structural similarity (lower = stronger OOD)')
ax.legend()
plt.tight_layout()
save_fig(fig, 'L_pi_bar_chart', where='main')
plt.show()


# In[28]:


# Cell 7.3 - per-channel decomposition L_H, L_M, L_W
hier_levels_obj = build_hierarchy_levels(hierarchy_main)
AM_tensor = torch.from_numpy(AM).to(torch.float32)
from scipy.stats import wasserstein_distance

def per_channel_leakage(fold):
    z = torch.zeros((n, N_FOLDS), dtype=torch.float32)
    z[np.arange(n), fold] = 1.0
    L_H = float(_evaluate_h_term_from_z(z, hier_levels_obj))
    L_M = float(cut_energy(z, AM_tensor))
    tr = Y_FULL[fold == 0]; te = Y_FULL[fold == 1]
    if len(tr) and len(te):
        L_W = float(np.mean([wasserstein_distance(tr[:, j], te[:, j])
                             for j in range(Y_FULL.shape[1])]))
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


# In[29]:


# Cell 7.4 - per-assay class retention heatmap (KEY main-text figure)
methods = list(baseline_results.keys()) + list(shield_results.keys())
retention = np.full((len(methods), len(ASSAY_COLS)), np.nan, dtype=np.float64)

for mi, name in enumerate(methods):
    fold = baseline_results.get(name, shield_results.get(name))
    for aj, a in enumerate(ASSAY_COLS):
        col = df[a].values
        labeled = ~np.isnan(col)
        if labeled.sum() == 0: continue
        pooled = np.nanmean(col)
        deviations = []
        for k in range(N_FOLDS):
            m = (fold == k) & labeled
            if m.sum() < 5: continue
            deviations.append(abs(np.mean(col[m]) - pooled))
        if deviations:
            retention[mi, aj] = np.mean(deviations)

retention_df = pd.DataFrame(retention, index=methods, columns=ASSAY_COLS)
save_table(retention_df.reset_index().rename(columns={'index': 'method'}),
           'per_assay_class_retention', where='main')

fig, ax = plt.subplots(figsize=(11, 0.5 * len(methods) + 1))
sns.heatmap(retention_df, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f',
            cbar_kws={'label': 'mean |fold positive rate - pooled| (lower=better)'})
ax.set_title(f'Per-assay class retention (class_delta tolerance = {STRAT_TOL})')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'per_assay_class_retention_heatmap', where='main')
plt.show()


# In[30]:


# Cell 5.5 - class balance diagnostic across all baseline splits (primary assay: NR-ER)

global_rate = float(np.nanmean(Y_PRIMARY))
print(f'Global {PRIMARY_ASSAY} positive rate: {global_rate*100:.2f}%')

retention_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    tr_mask = fold == 0
    te_mask = fold == 1
    tr_act = float(np.nanmean(Y_PRIMARY[tr_mask]))
    te_act = float(np.nanmean(Y_PRIMARY[te_mask]))
    retention_rows.append({
        'config':            name,
        'train_positive_rate': tr_act,
        'test_positive_rate':  te_act,
        'dev_train': abs(tr_act - global_rate),
        'dev_test':  abs(te_act - global_rate),
        'mean_dev':  (abs(tr_act - global_rate) + abs(te_act - global_rate)) / 2.0,
    })

ret_df = pd.DataFrame(retention_rows).sort_values('mean_dev').reset_index(drop=True)
save_table(ret_df, 'splits_class_retention', where='si')
display(ret_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(max(14, len(retention_rows) * 0.7), 4))


x = np.arange(len(ret_df))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, ret_df['train_positive_rate'] * 100, w,
       label='train', color='#4477AA', edgecolor='black')
ax.bar(x + w/2, ret_df['test_positive_rate']  * 100, w,
       label='test',  color='#CC6677', edgecolor='black')
ax.axhline(global_rate * 100, color='black', ls='--', lw=1.5,
           label=f'global rate ({100*global_rate:.1f}%)')
ax.set_xticks(x)
ax.set_xticklabels(ret_df['config'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel(f'{PRIMARY_ASSAY} positive rate (%)')
ax.set_title(f'{PRIMARY_ASSAY} positive rate per fold (dashed = global)')
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.bar(x, ret_df['mean_dev'] * 100, color='#EE7733', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(ret_df['config'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('mean |fold_rate − global_rate| (%)')
ax2.set_title('Class balance deviation — all splits (lower = better)')
ax2.axhline(STRAT_TOL * 100, color='red', ls=':', lw=1,
            label=f'STRAT_TOL = {STRAT_TOL}')
ax2.legend(fontsize=8)

plt.tight_layout()
save_fig(fig, 'splits_class_retention', where='si')
plt.show()

mean_class_retention_dev = ret_df.set_index('config')['mean_dev']
print('\nClass balance deviation (mean_dev), sorted best→worst:')
print(mean_class_retention_dev.round(4).to_string())


# In[31]:


# Cell 7.6 - kNN purity at k=5, 10, 25
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


# ## 8. Gradient alignment and convergence diagnostics

# In[32]:


# Cell 8.1 - gradient-alignment cosine per iteration (main text)
full_hist = shield_diagnostics.get('HMD-SHIELD', {}).get('history', [])
if full_hist:
    rows = []
    for h in full_hist:
        for k, v in h.get('alignment', {}).items():
            rows.append({'iteration': h['iteration'], 'pair': k.replace('/', '<->'), 'cosine': v})
    align_df = pd.DataFrame(rows)
    save_table(align_df, 'gradient_alignment_cosine', where='main')
    fig, ax = plt.subplots(figsize=(9, 4))
    for pair, sub in align_df.groupby('pair'):
        ax.plot(sub['iteration'], sub['cosine'], marker='o', label=pair)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('iteration'); ax.set_ylabel('cosine similarity')
    ax.set_title('HMD-SHIELD gradient alignment per iteration')
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    save_fig(fig, 'gradient_alignment_per_iteration', where='main')
    plt.show()


# In[33]:


# Cell 8.2 - composite-objective trajectory across configs (SI)
fig, ax = plt.subplots(figsize=(10, 5))
for cfg, diag in shield_diagnostics.items():
    h = diag.get('history', [])
    if not h: continue
    xs = [hh['iteration'] for hh in h]
    ys = [hh['composite_objective'] for hh in h]
    ax.plot(xs, ys, marker='o', label=cfg, lw=1.2)
ax.set_xlabel('iteration'); ax.set_ylabel('composite objective')
ax.set_title('Composite objective trajectory per SHIELD configuration')
ax.legend(fontsize=8, loc='best')
plt.tight_layout()
save_fig(fig, 'composite_objective_trajectories', where='si')
plt.show()


# In[34]:


# Cell 8.3 - per-iteration class-stratification residual diagnostic
rows = []
for cfg_name, diag in shield_diagnostics.items():
    md_ = diag['metadata']
    warns = md_.get('warnings', [])
    rows.append({'config': cfg_name,
                 'n_warnings': len(warns),
                 'milp_invoked': md_.get('milp_polish', {}).get('invoked', False),
                 'milp_trigger': md_.get('milp_polish', {}).get('trigger', ''),
                 })
strat_warn_df = pd.DataFrame(rows)
save_table(strat_warn_df, 'class_strat_audit', where='si')
strat_warn_df


# ## 11. Dimension-reduction visualizations of splits

# In[35]:


# Cell 11.1 - embeddings
import pacmap
import trimap
from sklearn.decomposition import KernelPCA

embeds = {}
try:
    embeds['tSNE'] = TSNE(n_components=2, perplexity=30, init='random',
                          random_state=RANDOM_SEED).fit_transform(bv.astype(np.float32))
    print('t-SNE done')
except Exception as exc:
    print(f't-SNE failed: {exc}')

for emb_name, emb in embeds.items():
    np.save(OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.npy', emb)
    pd.DataFrame(emb, columns=[f'{emb_name}_1', f'{emb_name}_2']).to_csv(
        OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.csv', index=True, index_label='row_id'
    )
    print(f'Saved {emb_name} embedding: {emb.shape}')


# In[36]:


# Cell 11.2 - t-SNE by primary target + split-colored scatter panels (all splits)

_tsne_emb = embeds.get('tSNE')
if _tsne_emb is not None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(_tsne_emb[prim_labeled & (Y_PRIMARY == 0), 0],
               _tsne_emb[prim_labeled & (Y_PRIMARY == 0), 1],
               s=5, c='#4477AA', alpha=0.55, label='inactive (0)', rasterized=True)
    ax.scatter(_tsne_emb[prim_labeled & (Y_PRIMARY == 1), 0],
               _tsne_emb[prim_labeled & (Y_PRIMARY == 1), 1],
               s=5, c='#CC6677', alpha=0.75, label='active (1)',  rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f't-SNE colored by {PRIMARY_ASSAY}', fontsize=11)
    ax.legend(loc='best', fontsize=8, markerscale=2)
    plt.tight_layout()
    save_fig(fig, 'tsne_target_class', where='main')
    plt.show()
else:
    print('t-SNE embedding not available in embeds dict.')

showcase = list(baseline_results.keys()) + list(shield_results.keys())

for emb_name, emb in embeds.items():
    cols = min(5, len(showcase))
    rows = int(np.ceil(len(showcase) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, showcase):
        fold = baseline_results.get(name, shield_results.get(name))
        ax.scatter(emb[fold == 0, 0], emb[fold == 0, 1], s=6, c='#4477AA',
                   alpha=0.55, label='train', rasterized=True)
        ax.scatter(emb[fold == 1, 0], emb[fold == 1, 1], s=6, c='#CC6677',
                   alpha=0.55, label='test',  rasterized=True)
        scaled = lpi_df.set_index('config').loc[name, 'scaled_L_pi'] \
                 if name in lpi_df['config'].values else float('nan')
        ax.set_title(f'{name}\nscaled L(π)={scaled:.3f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(showcase):]:
        ax.set_visible(False)
    axes[0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{emb_name} embedding colored by fold — all splits', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_splits',
             where='main' if emb_name in ('PCA', 'UMAP') else 'si')
    plt.show()


# ## 8. OOD Assessment Metrics

# In[38]:


# Cell 11.3 - OOD characterization metrics per split

from sklearn.neighbors  import NearestNeighbors
from scipy.spatial      import ConvexHull, Delaunay
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
import matplotlib.ticker as mticker
import warnings

bv32 = bv.astype(np.float32)

_tsne_2d = embeds.get('tSNE')

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

    try:
        nn_d = _nn_distances(Xtr, Xte)
        row['nn_mean'] = float(nn_d.mean())
        row['nn_p95']  = float(np.percentile(nn_d, 95))
        row['nn_max']  = float(nn_d.max())
    except Exception as e:
        print(f'  [NN] {sp_name}: {e}')
        row['nn_mean'] = row['nn_p95'] = row['nn_max'] = float('nan')

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


# In[39]:


# Cell 16.x - export SMILES + fold assignments for every split

_all = {}
for k, v in globals().get('baseline_results', {}).items():
    _all[k] = v
for k, v in globals().get('shield_results', {}).items():
    _all[f'SHIELD6_{k}'] = v
for k, v in globals().get('lambda_results', {}).items():
    _all[f'LAMBDA12_{k}'] = v
for k, v in globals().get('rounding_results', {}).items():
    _all[f'ROUND14_{k}'] = v

export = pd.DataFrame({'smiles': df[SMILES_COL].values})

if 'mol_id' in df.columns:
    export.insert(0, 'mol_id', df['mol_id'].values)

for _raw in ['_split_base', '_split_datasail']:
    if _raw in df.columns:
        export[f'RAW{_raw}'] = df[_raw].values

if PRIMARY_ASSAY in df.columns:
    export[PRIMARY_ASSAY] = df[PRIMARY_ASSAY].values

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
_skip_cols = {'smiles', PRIMARY_ASSAY, 'RAW_split_base', 'RAW_split_datasail'}
print('Split columns:', [c for c in export.columns if c not in _skip_cols])


# In[40]:


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


# ## 9. SHIELD (alpha, beta, gamma) sweep, Pareto frontier, Hamming stability

# In[41]:


# Cell 9.1 - shield.sweep over alpha x beta x gamma
sweep_kwargs = dict(
    n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hierarchy_main, affinity=AM,
    affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE, 
    stratification_strategy=STRAT_STRATEGY, n_bins=STRAT_N_BINS,
    class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
    eta=ETA, mu=MU, nu=NU, tau=TAU,
    max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
    integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
    rounding_mode=PRIMARY_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
    stage0_samples=SHIELD_STAGE0_SAMPLES,
    precomputed_normalizers=CACHED_NORMALIZERS,
    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
    kernel_matrix=K_MATRIX,
    y_target=None,
    class_labels=None,
)
print(f'Running shield.sweep over '
      f'{len(SWEEP_GRID["alpha"])*len(SWEEP_GRID["beta"])*len(SWEEP_GRID["gamma"])} '
      'configurations...')
t0 = time.time()
sweep_res = shield_sweep(
    alpha_grid=SWEEP_GRID['alpha'], beta_grid=SWEEP_GRID['beta'],
    gamma_grid=SWEEP_GRID['gamma'],
    metric_keys=('M_normalized', 'D_normalized', 'H_normalized'),
    aggregation='min_of_normalized',
    instability_threshold=HAMMING_STABILITY_THRESHOLD,
    **sweep_kwargs,
)
print(f'Sweep complete in {time.time()-t0:.1f}s; '
      f'recommended (alpha,beta,gamma)={sweep_res.grid_points[sweep_res.recommended_index]}')


# In[42]:


# Cell 9.2 - sweep results table
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
save_table(sweep_df, 'shield_sweep_grid', where='main')
sweep_df


# In[43]:


# Cell 9.3 - Pareto frontier plot
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
sc1 = ax1.scatter(sweep_df['M_normalized'], sweep_df['D_normalized'],
                  c=sweep_df['H_normalized'], cmap='viridis', s=60,
                  edgecolors='black', linewidths=0.5)
plt.colorbar(sc1, ax=ax1, label='H_normalized')
par = sweep_df[sweep_df['is_pareto']]
ax1.scatter(par['M_normalized'], par['D_normalized'],
            s=180, facecolors='none', edgecolors='red', linewidths=1.5, label='Pareto')
rec = sweep_df.loc[sweep_res.recommended_index]
ax1.scatter([rec['M_normalized']], [rec['D_normalized']],
            s=300, marker='*', color='gold', edgecolors='black', linewidths=1, label='recommended')
ax1.set_xlabel('M_normalized'); ax1.set_ylabel('D_normalized')
ax1.set_title('Sweep: M vs D, color = H, gold star = recommendation'); ax1.legend()

ax2.scatter(range(len(sweep_df)), sweep_df['composite_objective'],
            c=['gold' if x == sweep_res.recommended_index else '#4477AA'
               for x in sweep_df.index], s=60, edgecolors='black')
ax2.set_xlabel('config index'); ax2.set_ylabel('composite objective')
ax2.set_title('Composite objective across sweep')
plt.tight_layout()
save_fig(fig, 'sweep_pareto', where='main')
plt.show()


# In[44]:


# Cell 9.4 - Hamming-stability matrix
stab = sweep_res.stability
H = stab.pairwise_hamming
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(H, ax=ax, cmap='magma', annot=False,
            cbar_kws={'label': 'fraction differing atoms'})
ax.set_title(f'Hamming distance matrix ({len(stab.assignments)} sweep points; '
             f'threshold={HAMMING_STABILITY_THRESHOLD})')
plt.tight_layout()
save_fig(fig, 'hamming_stability_matrix', where='main')
print(f'Flagged unstable pairs: {len(stab.flagged_unstable)} of '
      f'{len(stab.assignments)*(len(stab.assignments)-1)//2}')
plt.show()


# ## 14. SI — confidence-gap rounding vs pipage rounding

# In[45]:


# Cell 14.1 - run SHIELD_FULL with both rounding modes
rounding_results = {}
for rmode in ['pipage', 'confidence_gap']:
    print(f'  rounding={rmode}')
    t0 = time.time()
    res = shield_split(
        n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main, affinity=AM,
        affinity_provenance='label-blind-handcrafted',
        d_mode=DIST_MODE, 
        stratification_strategy=STRAT_STRATEGY, n_bins=STRAT_N_BINS,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=rmode, milp_size_limit=SHIELD_MILP_SIZE,
        stage0_samples=SHIELD_STAGE0_SAMPLES,
        precomputed_normalizers=CACHED_NORMALIZERS,
        sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
        kernel_matrix=K_MATRIX,
        y_target=None,
        class_labels=None,
    )
    rounding_results[rmode] = res.fold_assignment.cpu().numpy().astype(int)
    print(f'    wall={time.time()-t0:.1f}s')

if len(rounding_results) == 2:
    f1, f2 = rounding_results['pipage'], rounding_results['confidence_gap']
    hamming = int((f1 != f2).sum())
    print(f'\nHamming distance pipage vs confidence_gap: {hamming} / {n} '
          f'({100 * hamming / n:.2f}%)')

cg_rep1 = rounding_results['confidence_gap']
res_cg2 = shield_split(
    n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hierarchy_main, affinity=AM, affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE, 
    stratification_strategy=STRAT_STRATEGY, n_bins=STRAT_N_BINS,
    class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
    alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
    max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED + 1, device=SHIELD_DEVICE, nodes=N_JOBS,
    integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
    rounding_mode='confidence_gap', milp_size_limit=SHIELD_MILP_SIZE,
    stage0_samples=SHIELD_STAGE0_SAMPLES,
    precomputed_normalizers=CACHED_NORMALIZERS,
    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
    kernel_matrix=K_MATRIX,
    y_target=None,
    class_labels=None,
)
cg_rep2 = res_cg2.fold_assignment.cpu().numpy().astype(int)
print(f'Hamming distance two confidence_gap seeds: {int((cg_rep1 != cg_rep2).sum())} '
      f'(zero would mean fully seed-independent; non-zero reflects upstream Sinkhorn / Stage-0 randomness)')

rows = []
for rmode, fold in rounding_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    rows.append({'rounding': rmode,
                 'scaled_L_pi': cross_fold_similarity(fold)[1],
                 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
rdf = pd.DataFrame(rows)
save_table(rdf, 'rounding_mode_comparison', where='si')
rdf
