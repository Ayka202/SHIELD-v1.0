# ## 1. Setup, imports, configuration, output directories

# In[1]:

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
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'qm9.csv')
DATASET_NAME = 'qm9'
SMILES_COL   = 'smiles'
TARGET_COLS_RAW = [
    'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
    'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom',
]
PRIMARY_TARGET = 'u0'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 42

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.05
STRAT_TOL   = 0.05

# Fingerprints (shared by M term and D kernel)
ECFP_RADIUS  = 2
ECFP_NBITS   = 1024
PRIMARY_AFFINITY = 'descriptor_rbf'
AFFINITY_TYPES   = ['descriptor_rbf']

# M term — epsilon-NN descriptor-RBF affinity
EPS_AFF_M    = 0.78
EPS_AFF_M_SI = [0.7, 0.75, 0.78, 0.8]
T2_WEIGHT    = 0.30

LAMBDA_MAIN = [1.0, 1.0]   # [L2_property_family, L1_size_bin]

# SHIELD coefficients (main configuration)
ALPHA, BETA, GAMMA, ETA, MU, NU, TAU = 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0

# D term — MMD shift in descriptor-RBF PSD kernel space (label-independent)
DIST_MODE = 'shift'

# OT regularization (used only in SI D-match comparison cell)
OT_REG    = 0.05

# Stratification (MU=0 in main config; SI exploration on HOMO-LUMO gap)
STRAT_STRATEGY  = 'sliced_wasserstein'
SW_PROJECTIONS  = 100
SW_QUANTILE_PTS = 256

# Sweep grid for the (alpha, beta, gamma) Pareto study
SWEEP_GRID = {
    'alpha': [0.2, 0.5, 1.0, 2.0, 3.0],
    'beta':  [0.2, 0.5, 1.0, 2.0, 3.0],
    'gamma': [0.2, 0.5, 1.0, 2.0, 3.0],
}
HAMMING_STABILITY_THRESHOLD = 0.10   # acceptable fraction of changing atoms

# kNN purity neighbourhood sizes (diagnostic only)
KNN_K = [5, 10, 20, 30]

# SHIELD solver controls
SHIELD_MAX_ITER     = 25
SHIELD_STAGE0_SAMPLES  = 100
SHIELD_INIT_BACKEND = 'auto'
SHIELD_ROUNDING     = 'pipage'
SHIELD_MILP_SIZE    = 25_000
SHIELD_DEVICE       = 'cpu'
N_JOBS              = -1

RUN_COARSENING_PRIMARY = True
COARSEN_N_SUPER_PRIMARY = 512


COARSEN_METHOD       = 'H_first'   # 'H_first' | 'LSH'
LSH_BALANCE_BUCKETS  = True
LSH_NUM_PERM         = 128
LSH_THRESHOLD        = 0.40        # Jaccard threshold (matches EPS_AFF_M)
D_KERNEL_POWER = 2.0

COARSEN_TRIALS = ['H_GROUPS', 128, 256, 512]


# Epsilon threshold for the super-node M term (centroid eps-NN graph).
EPS_AFF_SUPER = None   # None -> inherit EPS_AFF_M at runtime

EMB_SUBSAMPLE        = 10000
KNN_PURITY_SUBSAMPLE = 10000

# Master flag for the external-coarsening pipeline.
RUN_EXTERNAL_COARSENING = True

# Solver time limit + FIXED leakage-evaluation reference resolution
SHIELD_MILP_TIME = 120.0   # MILP polish time limit per solve (seconds)
EVAL_REF_N_SUPER = 4096

SWEEP_EPS_AFF_M      = [0.75, 0.77, 0.78, 0.79, 0.8]   # M: eps-NN descriptor-RBF threshold
SWEEP_T2_WEIGHT      = [0.30]   # M: Tier-2 fallback edge weight
SWEEP_T2_K           = [3]                   # M: Tier-2 nearest-neighbor count (1 value = fixed)
SWEEP_D_KERNEL_POWER = [1.0, 2.0, 3.0]


# ML evaluation
ML_RUN = True

print(f'Config loaded. MAX_ROWS={MAX_ROWS}, K={N_FOLDS}, '
      f'DIST_MODE={DIST_MODE}, ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}, '
      f'EPS_AFF_M={EPS_AFF_M}, LAMBDA_MAIN={LAMBDA_MAIN}, '
      f'COARSEN_PRIMARY=n_super={COARSEN_N_SUPER_PRIMARY}')


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
    """Save figure as PNG, PDF, and SVG to the appropriate output folder.
    `where` is 'main' or 'si'."""
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
    raise RuntimeError('RDKit is required for QM9 featurization. Install via conda-forge.')


# ## 2. Data loading and EDA

# In[6]:


# Cell 2.1 - load QM9 dataset
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
from rdkit import RDLogger as _RDLogger
import re as _re
import warnings as _warnings
_warnings.filterwarnings('ignore', category=UserWarning)
_RDLogger.DisableLog('rdApp.*')

DATASAIL_PATH = str(_REPO_ROOT / 'splits' / 'qm9_DataSAIL.csv')

PRECOMPUTED_FOLD_MAP = {
    'train': 0,  # ~80 %  -> training fold
    'val':   1,  # ~10 %  -> holdout  (merged with test to give 80/20)
    'test':  1,  # ~10 %  -> holdout
}

def _neutralize(m):
    """Zero every formal charge, compensating via the per-atom H count, so a
    protonated / deprotonated / charged variant collapses onto the neutral molecule."""
    m = _Chem.RWMol(m)
    for a in m.GetAtoms():
        chg = a.GetFormalCharge()
        if chg:
            a.SetFormalCharge(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(max(a.GetTotalNumHs() - chg, 0))
    mm = m.GetMol()
    try:
        mm.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return mm

def _match_keys(s):
    """Return (connectivity_key, skeleton_key) for a SMILES.
    connectivity_key = neutral InChI formula + /c  (stereo stripped) -> specific, ~1:1
    skeleton_key     = heavy-atom formula (H stripped) + /c          -> charge/H/bond-order invariant
    """
    m = _Chem.MolFromSmiles(s)
    if m is None:
        return (None, None)
    inc = None
    try:
        inc = _MolToInchi(_neutralize(m), options='-SNon')
    except Exception:
        inc = None
    if not inc:
        try:
            inc = _MolToInchi(m, options='-SNon')
        except Exception:
            inc = None
    if not inc:
        return (None, None)
    parts = inc.split('/')
    formula = parts[1]
    conn = ''
    for p in parts[2:]:
        if p.startswith('c'):
            conn = p
            break
    return (formula + '|' + conn, _re.sub(r'H\d*', '', formula) + '|' + conn)

df_ds = pd.read_csv(DATASAIL_PATH)
_ds_keys = df_ds['smiles'].apply(_match_keys)
df_ds['_k_conn'] = _ds_keys.apply(lambda t: t[0])
df_ds['_k_skel'] = _ds_keys.apply(lambda t: t[1])

_map_conn = {}
for _k, _g in df_ds[df_ds['_k_conn'].notna()].groupby('_k_conn'):
    _map_conn[_k] = (_g['split'].mode().iat[0], _g['datasail'].mode().iat[0])
_map_skel, _skel_ambiguous = {}, set()
for _k, _g in df_ds[df_ds['_k_skel'].notna()].groupby('_k_skel'):
    if _g['split'].nunique() == 1 and _g['datasail'].nunique() == 1:
        _map_skel[_k] = (_g['split'].iat[0], _g['datasail'].iat[0])
    else:
        _skel_ambiguous.add(_k)

_src_keys = df[SMILES_COL].apply(_match_keys)
_labels, _tier = [], []
for _k1, _k2 in _src_keys.values:
    if _k1 is not None and _k1 in _map_conn:
        _labels.append(_map_conn[_k1]); _tier.append(1)
    elif _k2 is not None and _k2 in _map_skel:
        _labels.append(_map_skel[_k2]); _tier.append(2)
    else:
        _labels.append(None)
        _tier.append(-1 if (_k2 is not None and _k2 in _skel_ambiguous) else 0)

_matched   = pd.Series([l is not None for l in _labels], index=df.index)
_n_t1      = _tier.count(1)
_n_t2      = _tier.count(2)
_n_amb     = _tier.count(-1)
_n_absent  = _tier.count(0)
print(f'DataSAIL join: {int(_matched.sum()):,} matched '
      f'(tier-1 connectivity={_n_t1:,}, tier-2 heavy-skeleton={_n_t2:,})')
print(f'  dropped {_n_amb + _n_absent:,}: '
      f'{_n_absent:,} genuinely absent from DataSAIL, '
      f'{_n_amb:,} match only an ambiguous skeleton (conflicting DataSAIL folds -> not assigned)')

df = df[_matched].copy().reset_index(drop=True)
_kept = [l for l in _labels if l is not None]
df['_split_base']     = [l[0] for l in _kept]
df['_split_datasail'] = [l[1] for l in _kept]

n = len(df)
print(f'Working dataset after filter: {n:,} molecules')
print(f"  _split_base     : {pd.Series(df['_split_base']).value_counts().sort_index().to_dict()}")
print(f"  _split_datasail : {pd.Series(df['_split_datasail']).value_counts().sort_index().to_dict()}")


# In[8]:


# Cell 2.2 - target distribution panel (saved as SI Fig. B.2.1)
ncols = 4
nrows = int(np.ceil(len(TARGET_COLS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
for ax, col in zip(axes.flat, TARGET_COLS):
    ax.hist(df[col].dropna(), bins=40, color='#4477AA', edgecolor='black', alpha=0.85)
    ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=8)
for ax in axes.flat[len(TARGET_COLS):]:
    ax.set_visible(False)
fig.suptitle('QM9 target distributions (n=%d)' % n)
plt.tight_layout()
save_fig(fig, 'eda_target_distributions', where='si')
plt.show()


# In[9]:


# Cell 2.3 - target correlation heatmap
corr = df[TARGET_COLS].corr().values
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(TARGET_COLS))); ax.set_xticklabels(TARGET_COLS, rotation=60, fontsize=8)
ax.set_yticks(range(len(TARGET_COLS))); ax.set_yticklabels(TARGET_COLS, fontsize=8)
ax.set_title('QM9 target-target correlation (12 DFT properties)')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
save_fig(fig, 'eda_target_correlation', where='si')
save_array(corr, 'target_correlation', where='si')
plt.show()


# ## 3. Featurization — ECFP4, scaffolds, affinity, cost, kernel

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


_rng_emb = np.random.default_rng(RANDOM_SEED + 7)
if EMB_SUBSAMPLE is None or n <= EMB_SUBSAMPLE:
    EMB_IDX = np.arange(n)
else:
    EMB_IDX = np.sort(_rng_emb.choice(n, size=int(EMB_SUBSAMPLE), replace=False))
print(f'EMB_IDX: {len(EMB_IDX)} atoms sampled for embeddings/diagnostics (of n={n})')


# In[12]:


# Cell 3.2b - Physicochemical descriptors (QM9 property drivers) -- STRUCTURAL FEATURE
from rdkit.Chem import Descriptors as _Desc, rdMolDescriptors as _rdMD

_DESC_NAMES = ['n_heavy','n_C','n_N','n_O','n_F','n_H','DBE','n_rings','n_arom_rings',
               'n_rotatable','n_HBD','n_HBA','frac_csp3','TPSA','MolWt','n_double','n_triple']

def _mol_descriptors(m):
    if m is None:
        return [0.0] * len(_DESC_NAMES)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    nC, nN, nO, nF = syms.count('C'), syms.count('N'), syms.count('O'), syms.count('F')
    nHeavy = m.GetNumHeavyAtoms()
    nH = int(sum(a.GetTotalNumHs() for a in m.GetAtoms()))
    dbe = (2*nC + 2 + nN - nH - nF) / 2.0
    ndb = sum(1 for b in m.GetBonds() if b.GetBondTypeAsDouble() == 2.0)
    ntb = sum(1 for b in m.GetBonds() if b.GetBondTypeAsDouble() == 3.0)
    return [float(nHeavy), float(nC), float(nN), float(nO), float(nF), float(nH), float(dbe),
            float(_rdMD.CalcNumRings(m)), float(_rdMD.CalcNumAromaticRings(m)),
            float(_rdMD.CalcNumRotatableBonds(m)), float(_rdMD.CalcNumHBD(m)),
            float(_rdMD.CalcNumHBA(m)), float(_rdMD.CalcFractionCSP3(m)),
            float(_rdMD.CalcTPSA(m)), float(_Desc.MolWt(m)), float(ndb), float(ntb)]

CACHE_DESC = OUT_INTER / 'featurization' / f'descriptors_n{n}.npy'
if CACHE_DESC.exists():
    desc_raw = np.load(CACHE_DESC); print(f'Loaded cached descriptors {desc_raw.shape}')
else:
    desc_raw = np.array([_mol_descriptors(m) for m in mols], dtype=np.float64)
    np.save(CACHE_DESC, desc_raw); print(f'Computed descriptors {desc_raw.shape}')

_mu = desc_raw.mean(axis=0); _sd = desc_raw.std(axis=0)
D_feat = ((desc_raw - _mu) / np.where(_sd < 1e-9, 1.0, _sd)).astype(np.float32)
D_feat[:, _sd < 1e-9] = 0.0

_rng_sig = np.random.default_rng(RANDOM_SEED + 5)
_si = _rng_sig.choice(n, size=min(n, 2000), replace=False)
_Xs = D_feat[_si].astype(np.float64)
_sq = (_Xs * _Xs).sum(1)
_d2 = np.clip(_sq[:, None] + _sq[None, :] - 2.0 * (_Xs @ _Xs.T), 0.0, None)
_med = float(np.median(_d2[np.triu_indices(len(_si), k=1)]))
RBF_SIGMA = float(np.sqrt(max(_med, 1e-6)) / np.sqrt(2.0))
print(f'D_feat (standardized, {D_feat.shape[1]} descriptors); RBF_SIGMA={RBF_SIGMA:.4f}')

_nheavy = desc_raw[:, 0].astype(int)
_nN = desc_raw[:, 2].astype(int); _nO = desc_raw[:, 3].astype(int); _nF = desc_raw[:, 4].astype(int)
_dbe = np.rint(desc_raw[:, 6]).astype(int); _narom = desc_raw[:, 8].astype(int)

def _size_bin(h):     return 'sz<=5' if h <= 5 else f'sz{int(h)}'
def _comp_class(nN, nO, nF):
    het = ('N' if nN > 0 else '') + ('O' if nO > 0 else '') + ('F' if nF > 0 else '')
    return het if het else 'C_only'
def _unsat_class(dbe, narom):
    if narom > 0: return 'aromatic'
    if dbe <= 0:  return 'saturated'
    if dbe == 1:  return 'unsat1'
    return 'unsat2+'

df['_size_bin']    = [_size_bin(h) for h in _nheavy]
df['_comp_class']  = [_comp_class(a, b, c) for a, b, c in zip(_nN, _nO, _nF)]
df['_unsat_class'] = [_unsat_class(d, a) for d, a in zip(_dbe, _narom)]
df['_h_L1'] = df['_size_bin']                                              # coarse paradigm = SIZE
df['_h_L2'] = df['_size_bin'] + '|' + df['_comp_class'] + '|' + df['_unsat_class']  # fine families

print('Size bins   :', df['_size_bin'].value_counts().sort_index().to_dict())
print('Composition :', df['_comp_class'].value_counts().to_dict())
print('Unsaturation:', df['_unsat_class'].value_counts().to_dict())
print(f'H families: {df["_h_L2"].nunique()} property families (L2) nested in '
      f'{df["_h_L1"].nunique()} size bins (L1).')


# In[13]:


# Cell 3.3 - M term: super-node RBF affinity on physicochemical descriptors

def _rbf_sim(centroids, sigma=None):
    sigma = RBF_SIGMA if sigma is None else sigma
    C = np.asarray(centroids, dtype=np.float32)
    sq = (C * C).sum(axis=1); ns = C.shape[0]
    K = np.zeros((ns, ns), dtype=np.float32)
    for s in range(0, ns, 1024):
        e = min(s + 1024, ns)
        d2 = np.clip(sq[s:e, None] + sq[None, :] - 2.0 * (C[s:e] @ C.T), 0.0, None)
        K[s:e] = np.exp(-d2 / (2.0 * sigma * sigma + 1e-12)).astype(np.float32)
    return K

def build_super_affinity(centroids, eps=EPS_AFF_M, t2_weight=T2_WEIGHT):
    eps = EPS_AFF_SUPER if EPS_AFF_SUPER is not None else eps
    K = _rbf_sim(centroids); np.fill_diagonal(K, 0.0)
    A = np.where(K > eps, K, 0.0).astype(np.float32); np.fill_diagonal(A, 0.0)
    for i in np.where((A > 0).sum(axis=1) == 0)[0]:
        order = np.argsort(-K[i]); cnt = 0
        for j in order:
            if int(j) == int(i): continue
            A[i, j] = max(A[i, j], t2_weight); A[j, i] = max(A[j, i], t2_weight); cnt += 1
            if cnt >= 3: break
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0.0)
    return A

_Ks = _rbf_sim(D_feat[EMB_IDX]); _up = _Ks[np.triu_indices(_Ks.shape[0], k=1)]
print('M term (descriptor RBF) -- sampled calibration:')
for ev in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f'  eps={ev:.2f}: {100*float((_up > ev).mean()):.1f}% of sampled pairs kept')


# In[14]:


# Cell 3.3v - M term visualizations (descriptor RBF, sampled)
Ks = _rbf_sim(D_feat[EMB_IDX]); m = Ks.shape[0]; upper = Ks[np.triu_indices(m, k=1)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(upper, bins=60, color='#332288', alpha=0.85)
for ev in EPS_AFF_M_SI:
    axes[0].axvline(ev, ls='--', lw=1, label=f'eps={ev:.2f} ({100*float((upper>ev).mean()):.0f}%)')
axes[0].set_xlabel('descriptor RBF similarity'); axes[0].set_ylabel('pair count (atom sample)')
axes[0].set_title('Descriptor-RBF similarity distribution'); axes[0].legend(fontsize=7)
deg = (Ks > EPS_AFF_M).sum(1)
axes[1].hist(deg, bins=40, color='#117733', alpha=0.85)
axes[1].set_xlabel(f'RBF eps-NN degree (eps={EPS_AFF_M})'); axes[1].set_ylabel('count')
axes[1].set_title('eps-NN degree distribution')
_pt = PRIMARY_TARGET if PRIMARY_TARGET in TARGET_COLS else TARGET_COLS[0]
yv = df[_pt].values[EMB_IDX]
rng = np.random.default_rng(RANDOM_SEED + 2); npair = min(3000, m*(m-1)//2)
pi = rng.integers(0, m, npair*2); pj = rng.integers(0, m, npair*2); vp = pi != pj
pi, pj = pi[vp][:npair], pj[vp][:npair]
axes[2].scatter(Ks[pi, pj], np.abs(yv[pi]-yv[pj]), s=4, alpha=0.3, color='#882255')
r_val, _ = spearmanr(Ks[pi, pj], np.abs(yv[pi]-yv[pj]))
axes[2].set_xlabel('descriptor RBF similarity'); axes[2].set_ylabel(f'|delta {_pt}|')
axes[2].set_title(f'RBF sim vs |delta {_pt}|\nSpearman r={r_val:.3f}')
plt.tight_layout(); save_fig(fig, 'M_affinity_diagnostics', where='main'); plt.show()
print('Unlike ECFP, descriptor-RBF similarity should NEGATIVELY correlate with target')
print(f'distance (similar descriptors -> similar {_pt}); i.e. M now TRACKS the QM9 property')
print('structure, so a low-L(pi) (structurally-dissimilar) split is a real property-OOD.')


# In[15]:


# Cell 3.4 - cost matrix for D-match (descriptor space): 1 - RBF similarity
def build_super_cost(centroids):
    K = _rbf_sim(centroids)
    cost = (1.0 - K).astype(np.float32)
    cost = np.clip((cost + cost.T) / 2.0, 0.0, 1.0); np.fill_diagonal(cost, 0.0)
    return cost
print('D-match cost builder defined (descriptor RBF; used in cell 14.1).')


# In[16]:


# Cell 3.5 - D-shift kernel (descriptor space): Gaussian RBF (PSD Mercer kernel)
def build_super_kernel(centroids):
    K = np.clip(_rbf_sim(centroids), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(K, 1.0)
    return K
print('D-shift RBF kernel builder defined (descriptor space; K_MATRIX built in cell 4.5).')


# In[17]:


# Cell 3.5v - D term visualizations (descriptor RBF, sampled)
Ks = _rbf_sim(D_feat[EMB_IDX]); off = Ks[~np.eye(Ks.shape[0], dtype=bool)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sub = np.sort(np.random.default_rng(RANDOM_SEED + 3).choice(Ks.shape[0], min(80, Ks.shape[0]), replace=False))
im = axes[0].imshow(Ks[np.ix_(sub, sub)], cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im, ax=axes[0], label='RBF kernel')
axes[0].set_title('D-shift RBF kernel (descriptor space, sample)')
axes[1].hist(off, bins=60, color='#4477AA', alpha=0.85)
axes[1].axvline(off.mean(), color='red', ls='--', label=f'mean={off.mean():.3f}')
axes[1].set_xlabel('off-diagonal RBF kernel'); axes[1].set_ylabel('count')
axes[1].set_title('Off-diagonal kernel distribution'); axes[1].legend()
plt.tight_layout(); save_fig(fig, 'D_kernel_diagnostics', where='main'); plt.show()
print('D-shift kernel is now a Gaussian RBF on standardized QM9 descriptors (PSD).')


# ## 4. Hierarchy construction — two-level functional-class taxonomy
# 

# In[18]:


# Cell 4.1 - Candidate H axes + property-driver validation  (NOT the active H)

from rdkit.Chem import MolFromSmarts

FC_SMARTS = {
    'nitro':                    ['[NX3](=O)=O', '[N+](=O)[O-]'],
    'nitroso':                  ['[NX2]=O'],
    'azide':                    ['[#7]=[#7]=[#7]'],
    'cyanamide_carbodiimide':   ['[NX3]C#N', '[NX2]=C=[NX2]'],
    'nitrile':                  ['[NX1]#[CX2]'],
    'carbamate':                ['[NX3][CX3](=[OX1])[OX2]'],
    'urea':                     ['[NX3][CX3](=[OX1])[NX3]'],
    'carbonate':                ['[OX2][CX3](=[OX1])[OX2]'],
    'carboxylic_acid':          ['[CX3](=[OX1])[OX2H1]'],
    'anhydride':                ['[CX3](=[OX1])[OX2][CX3]=[OX1]'],
    'ester':                    ['[CX3](=[OX1])[OX2H0][#6]'],
    'lactam':                   ['[NX3;R][CX3;R]=[OX1]'],
    'amide':                    ['[NX3][CX3](=[OX1])[#6]', '[CX3H1](=[OX1])[NX3]'],
    'conjugated_carbonyl':      ['[#6]=[#6][CX3]=[OX1]', '[OX1]=[CX3][#6]=[#6]'],
    'aldehyde':                 ['[CX3H1]=[OX1]'],
    'ketone':                   ['[#6][CX3](=[OX1])[#6]'],
    'amidine_guanidine':        ['[NX3][CX3]=[NX2]'],
    'imine':                    ['[CX3]=[NX2][#6]', '[CX3H]=[NX2]'],
    'pyridine_type':            ['[nX2;r6]'],
    'azole_type':               ['[n;r5]'],
    'furan_type':               ['[o;r5]'],
    'aromatic_carbocyclic':     ['c1ccccc1'],
    'hydrazine_hydrazone':      ['[NX3][NX3]', '[#6]=[NX2][NX3]'],
    'hydroxylamine':            ['[NX3][OX2H1]', '[NX3][OX2][#6]'],
    'enamine':                  ['[NX3;!$(NC=O)][CX3]=[CX3]'],
    'tertiary_amine':           ['[NX3H0;!a]([#6])([#6])[#6]'],
    'secondary_amine':          ['[NX3H1;!a]'],
    'primary_amine':            ['[NX3H2;!a]'],
    'epoxide':                  ['[OX2;r3]'],
    'oxetane':                  ['[OX2;r4]'],
    'acetal_ketal':             ['[CX4]([OX2])[OX2]'],
    'enol':                     ['[OX2H1][CX3]=[CX3]'],
    'enol_ether':               ['[OX2]([#6])[CX3]=[CX3]'],
    'diol':                     ['[OX2H1][CX4][CX4][OX2H1]', '[OX2H1][CX4][CX4][CX4][OX2H1]'],
    'unsaturated_alcohol':      ['[OX2H1][CX4][CX3]=[CX3]', '[OX2H1][CX4][CX2]#[CX2]'],
    'cyclic_ether':             ['[OX2;R;!r3;!r4]'],
    'alcohol':                  ['[OX2H1;!a][#6]'],
    'ether':                    ['[OX2H0;!a]([#6])[#6]'],
    'alkyne':                   ['[CX2]#[CX2]'],
    'conjugated_diene':         ['[CX3]=[CX3][CX3]=[CX3]'],
    'alkene':                   ['[CX3]=[CX3]', '[CX2]=[CX3]'],
    'cyclopropane':             ['[CX4;r3]'],
    'fluorinated':              ['[F]'],
    'saturated_hydrocarbon':    ['[#6,#7,#8,#9]'],
}

FC_GROUP_ORDER = [
    # strong EWG
    'nitro', 'nitroso', 'azide', 'cyanamide_carbodiimide', 'nitrile',
    # carbonyl / acyl
    'carbamate', 'urea', 'carbonate', 'carboxylic_acid', 'anhydride', 'ester',
    'lactam', 'amide', 'conjugated_carbonyl', 'aldehyde', 'ketone',
    # conjugated N
    'amidine_guanidine', 'imine',
    # aromatic pi
    'pyridine_type', 'azole_type', 'furan_type', 'aromatic_carbocyclic',
    # saturated N donors
    'hydrazine_hydrazone', 'hydroxylamine', 'enamine',
    'tertiary_amine', 'secondary_amine', 'primary_amine',
    # oxygen functionalities
    'epoxide', 'oxetane', 'acetal_ketal', 'enol', 'enol_ether', 'diol',
    'unsaturated_alcohol', 'cyclic_ether', 'alcohol', 'ether',
    # unsaturation
    'alkyne', 'conjugated_diene', 'alkene', 'cyclopropane',
    # halogen + catch-all
    'fluorinated', 'saturated_hydrocarbon',
]

L1_MAP = {
    'carbamate': 'carbonyl_acyl', 'urea': 'carbonyl_acyl', 'carbonate': 'carbonyl_acyl',
    'carboxylic_acid': 'carbonyl_acyl', 'anhydride': 'carbonyl_acyl', 'ester': 'carbonyl_acyl',
    'lactam': 'carbonyl_acyl', 'amide': 'carbonyl_acyl',
    # pure carbonyl (oxo)
    'conjugated_carbonyl': 'carbonyl_oxo', 'aldehyde': 'carbonyl_oxo', 'ketone': 'carbonyl_oxo',
    # nitrile / N electron-withdrawing / C=N
    'nitro': 'nitrile_ewN_imine', 'nitroso': 'nitrile_ewN_imine', 'azide': 'nitrile_ewN_imine',
    'cyanamide_carbodiimide': 'nitrile_ewN_imine', 'nitrile': 'nitrile_ewN_imine',
    'amidine_guanidine': 'nitrile_ewN_imine', 'imine': 'nitrile_ewN_imine',
    # saturated nitrogen lone-pair donors
    'hydrazine_hydrazone': 'amine', 'hydroxylamine': 'amine', 'enamine': 'amine',
    'tertiary_amine': 'amine', 'secondary_amine': 'amine', 'primary_amine': 'amine',
    # aromatic pi-systems
    'pyridine_type': 'heteroaromatic', 'azole_type': 'heteroaromatic',
    'furan_type': 'heteroaromatic', 'aromatic_carbocyclic': 'heteroaromatic',
    # O-H hydrogen-bond donors
    'enol': 'hydroxyl', 'diol': 'hydroxyl', 'unsaturated_alcohol': 'hydroxyl',
    'alcohol': 'hydroxyl',
    # ethers / cyclic & strained O / acetals
    'epoxide': 'ether_oxa', 'oxetane': 'ether_oxa', 'acetal_ketal': 'ether_oxa',
    'enol_ether': 'ether_oxa', 'cyclic_ether': 'ether_oxa', 'ether': 'ether_oxa',
    # non-heteroatom-functional hydrocarbons + halogen
    'alkyne': 'hydrocarbon_halide', 'conjugated_diene': 'hydrocarbon_halide',
    'alkene': 'hydrocarbon_halide', 'cyclopropane': 'hydrocarbon_halide',
    'fluorinated': 'hydrocarbon_halide', 'saturated_hydrocarbon': 'hydrocarbon_halide',
}
L1_GROUPS = ['carbonyl_acyl', 'carbonyl_oxo', 'nitrile_ewN_imine', 'amine',
             'heteroaromatic', 'hydroxyl', 'ether_oxa', 'hydrocarbon_halide']

_fc_compiled = {}
for grp in FC_GROUP_ORDER:
    pats = []
    for sma in FC_SMARTS[grp]:
        p = MolFromSmarts(sma)
        if p is not None:
            pats.append(p)
        else:
            print(f'  WARNING: invalid SMARTS for group {grp}: {sma}')
    _fc_compiled[grp] = pats

def _assign_functional_class(mol):
    if mol is None:
        return 'saturated_hydrocarbon'
    for grp in FC_GROUP_ORDER:
        if any(mol.HasSubstructMatch(p) for p in _fc_compiled[grp]):
            return grp
    return 'saturated_hydrocarbon'

functional_class_L2 = [_assign_functional_class(m) for m in mols]
functional_class_L1 = [L1_MAP.get(g, 'hydrocarbon_halide') for g in functional_class_L2]
df['_functional_class_L2'] = functional_class_L2
df['_functional_class_L1'] = functional_class_L1

print('=' * 78)
print('DEPRECATED functional-class baseline below -- this is NOT the active H.')
print('The active H is built in cell 4.2 as size x comp x unsat property families.')
print('Shown only as the eta^2 comparison baseline.')
print('=' * 78)
print(f'[baseline] L2 functional_class distribution ({len(set(functional_class_L2))} of '
      f'{len(FC_GROUP_ORDER)} families populated):')
for g in FC_GROUP_ORDER:
    cnt = functional_class_L2.count(g)
    if cnt > 0:
        print(f'  {g:25s}: {cnt:6d} ({100*cnt/n:.1f}%)')

print('\nL1 heteroatom_class distribution:')
for g in L1_GROUPS:
    cnt = functional_class_L1.count(g)
    if cnt > 0:
        print(f'  {g:25s}: {cnt:6d} ({100*cnt/n:.1f}%)')


def _eta_squared(y, groups):
    y = np.asarray(y, dtype=np.float64)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0: return 0.0
    g = np.asarray(groups); ssw = 0.0
    for lev in pd.unique(g):
        yi = y[g == lev]
        if yi.size: ssw += float(((yi - yi.mean()) ** 2).sum())
    return max(0.0, 1.0 - ssw / sst)

_Yt = df[TARGET_COLS].values.astype(np.float64)
_axes = {'functional_class_L2 (old H)': df['_functional_class_L2'].values,
         'size_bin':   df['_size_bin'].values,
         'comp_class': df['_comp_class'].values,
         'unsat_class':df['_unsat_class'].values,
         'property_family (size x comp x unsat)': df['_h_L2'].values}
_eta = np.array([[_eta_squared(_Yt[:, ti], av) for ti in range(len(TARGET_COLS))]
                 for av in _axes.values()])
eta_df = pd.DataFrame(_eta, index=list(_axes.keys()), columns=TARGET_COLS)
save_table(eta_df.reset_index().rename(columns={'index': 'axis'}),
           'property_driver_eta_squared', where='main')

fig, (axH, axB) = plt.subplots(1, 2, figsize=(17, 4.5), gridspec_kw={'width_ratios': [3, 1]})
sns.heatmap(eta_df, ax=axH, cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f',
            cbar_kws={'label': 'eta^2 (variance explained)'})
axH.set_title('Variance of each QM9 target explained by each candidate split axis\n'
              '(higher = the axis separates that property -> stronger OOD/generalization barrier)')
axH.set_xlabel(''); axH.tick_params(axis='x', labelrotation=45)
_mean_eta = eta_df.mean(axis=1).sort_values()
axB.barh(range(len(_mean_eta)), _mean_eta.values, color='#4477AA')
axB.set_yticks(range(len(_mean_eta))); axB.set_yticklabels(_mean_eta.index, fontsize=8)
axB.set_xlabel('mean eta^2 over 12 targets'); axB.set_title('Average explanatory power')
for i, v in enumerate(_mean_eta.values): axB.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=8)
plt.tight_layout(); save_fig(fig, 'property_driver_validation', where='main'); plt.show()

print('HOW TO READ: each cell = fraction of that QM9 target''s variance explained by')
print('  grouping molecules on that axis. A GOOD OOD axis has HIGH eta^2 (holding a')
print('  group out genuinely shifts the property). Compare functional_class_L2 (old H)')
print('  against size / composition / unsaturation:')
print(f'  -> Highest mean explanatory power: "{_mean_eta.index[-1]}" (mean eta^2={_mean_eta.iloc[-1]:.2f}).')
print(f'  -> Lowest: "{_mean_eta.index[0]}" (mean eta^2={_mean_eta.iloc[0]:.2f}).')
print('  size/composition dominate the extensive targets (U0/U/H/G/ZPVE/Cv/alpha/R^2);')
print('  unsaturation/composition drive the electronic ones (HOMO/LUMO/gap/mu).')
print('  functional_class explains little -> a weak OOD axis. Cell 4.2 therefore rebuilds')
print('  H on size x composition x unsaturation, with SIZE as the coarse (L1) axis.')


# In[19]:


# Cell 4.2 - H term: PROPERTY-DRIVER hierarchy (size x composition x unsaturation)

def build_hierarchy(df_local, lambda_per_level):
    def _grp(col):
        vals = df_local[col].tolist(); uniq = list(dict.fromkeys(vals))
        idx = {u: i for i, u in enumerate(uniq)}; desc = [[] for _ in uniq]
        for i, v in enumerate(vals): desc[idx[v]].append(i)
        return uniq, desc
    _, L2d = _grp('_h_L2'); _, L1d = _grp('_h_L1')
    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float64))
    return [{'name': 'property_family', 'descendants': L2d, 'similarity': empty, 'lambda': lambda_per_level[0]},
            {'name': 'size_bin',        'descendants': L1d, 'similarity': empty, 'lambda': lambda_per_level[1]}]

hierarchy_main = build_hierarchy(df, LAMBDA_MAIN)
for L in hierarchy_main:
    sizes = sorted([len(d) for d in L['descendants']], reverse=True)
    print(f"  {L['name']:16s}: {len(L['descendants'])} families, lambda={L['lambda']}, top sizes={sizes[:8]}")
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)

_tgt = eta_df.loc['size_bin'].idxmax()
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (col, ttl) in zip(axes, [('_size_bin', f'NEW axis: {_tgt} by size bin'),
                                  ('_functional_class_L2', f'OLD axis: {_tgt} by functional class')]):
    order = sorted(df[col].unique())
    data = [df[_tgt].values[df[col].values == g] for g in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for p in bp['boxes']: p.set_facecolor('#4477AA'); p.set_alpha(0.6)
    ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(order, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel(_tgt); ax.set_title(ttl)
plt.tight_layout(); save_fig(fig, 'H_property_family_coherence', where='main'); plt.show()
print(f'ACTIVE H (this is the hierarchy passed to SHIELD): {len(hierarchy_main[0]["descendants"])} property families (L2) nested in '
      f'{len(hierarchy_main[1]["descendants"])} size bins (L1).')
print(f'The LEFT panel ({_tgt} by size bin) should show clear, ordered separation; the')
print('RIGHT panel (by functional class) should be flat/overlapping -- confirming the new')
print('axis is a far stronger generalization barrier. Increase LAMBDA_MAIN[1] (size) to')
print('push harder toward pure size-extrapolation OOD.')


# ## 4.5 External coarsening — super-node construction (PRIMARY for QM9)

# In[20]:


# Cell 4.5.1 - external coarsening algorithms (H-first + optional LSH)
from shield import coarsening as _coarsening
from shield.coarsening import lift_assignment_to_atoms


def h_first_coarsen(hierarchy_levels, feat, n_target, seed=RANDOM_SEED):
    '''H-group-first coarsening on physicochemical descriptors (feat = D_feat).
    Each L2 property family is a super-node; families larger than ceil(n/n_target)
    are split by balanced K-means (Euclidean on descriptors) within the family; if
    n_target < n_families, families are greedily merged by size-bin match + descriptor
    RBF similarity. Returns (labels, centroids (n_super,d), super_weights, n_super).'''
    n_atoms, d = feat.shape
    feat_t = torch.as_tensor(np.asarray(feat), dtype=torch.float32)
    level0 = hierarchy_levels[0]['descendants']
    families = [list(map(int, desc)) for desc in level0 if len(desc) > 0]
    covered = set()
    for fam in families:
        covered.update(fam)
    residual = [i for i in range(n_atoms) if i not in covered]
    if residual:
        families.append(residual)
    n_families = len(families)

    L1_of_atom = np.full(n_atoms, -1, dtype=np.int64)
    if len(hierarchy_levels) > 1:
        for gid, desc in enumerate(hierarchy_levels[1]['descendants']):
            for i in desc:
                L1_of_atom[int(i)] = gid

    if n_target >= n_families:
        max_super_size = max(1, math.ceil(n_atoms / max(1, int(n_target))))
        labels = np.full(n_atoms, -1, dtype=np.int64)
        next_id = 0
        for fam in families:
            fam_arr = np.asarray(fam, dtype=np.int64)
            fsize = int(fam_arr.shape[0])
            if fsize <= max_super_size:
                labels[fam_arr] = next_id
                next_id += 1
            else:
                k_sub = min(fsize, int(math.ceil(fsize / max_super_size)))
                res = _coarsening.coarsen_balanced_kmeans(
                    feat_t[fam_arr], n_clusters=k_sub, seed=seed, device='cpu')
                sub = res.labels.cpu().numpy().astype(np.int64)
                uniq = np.unique(sub)
                remap = {int(u): next_id + j for j, u in enumerate(uniq)}
                labels[fam_arr] = np.array([remap[int(s)] for s in sub], dtype=np.int64)
                next_id += len(uniq)
        n_super = int(next_id)
    else:
        groups = [list(map(int, fam)) for fam in families]
        gcent = [feat_t[np.asarray(fam, dtype=np.int64)].mean(dim=0) for fam in families]
        gL1 = []
        for fam in families:
            vals = L1_of_atom[np.asarray(fam, dtype=np.int64)]
            vals = vals[vals >= 0]
            gL1.append(int(np.bincount(vals).argmax()) if vals.size else -1)
        active = list(range(n_families))

        def _pair_sim(a, b):
            ca, cb = gcent[a], gcent[b]
            d2 = float(((ca - cb) ** 2).sum())
            rbf = math.exp(-d2 / (2.0 * RBF_SIGMA * RBF_SIGMA + 1e-12))
            same = 1.0 if (gL1[a] >= 0 and gL1[a] == gL1[b]) else 0.0
            return same + rbf

        while len(active) > int(n_target):
            best, best_s = None, -1.0
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    s = _pair_sim(active[ii], active[jj])
                    if s > best_s:
                        best_s, best = s, (ii, jj)
            ii, jj = best
            a, b = active[ii], active[jj]
            na, nb = len(groups[a]), len(groups[b])
            groups[a] = groups[a] + groups[b]
            gcent[a] = (gcent[a] * na + gcent[b] * nb) / (na + nb)
            active.pop(jj)
        labels = np.full(n_atoms, -1, dtype=np.int64)
        for new_id, gidx in enumerate(active):
            labels[np.asarray(groups[gidx], dtype=np.int64)] = new_id
        n_super = len(active)

    labels_t = torch.as_tensor(labels, dtype=torch.long)
    agg = _coarsening.aggregate_super_nodes(
        labels_t, torch.ones(n_atoms, dtype=torch.float32), features=feat_t, n_clusters=n_super)
    return (labels, agg['features'].cpu().numpy().astype(np.float32),
            agg['super_weights'].cpu().numpy().astype(np.float32), n_super)


def lsh_coarsen(bv, n_target, threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM,
                balance=LSH_BALANCE_BUCKETS, seed=RANDOM_SEED):
    '''Optional MinHash-LSH coarsening; falls back to balanced K-means if
    datasketch is unavailable. Returns the same 4-tuple as h_first_coarsen.'''
    n_atoms, d = bv.shape
    bv_np = np.asarray(bv)
    bv_t = torch.as_tensor(bv_np, dtype=torch.float32)

    def _finalize(labels):
        labels = np.asarray(labels, dtype=np.int64)
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
        ns = int(labels.max()) + 1
        agg = _coarsening.aggregate_super_nodes(
            torch.as_tensor(labels, dtype=torch.long),
            torch.ones(n_atoms, dtype=torch.float32), features=bv_t, n_clusters=ns)
        return (labels, agg['features'].cpu().numpy().astype(np.float32),
                agg['super_weights'].cpu().numpy().astype(np.float32), ns)

    try:
        from datasketch import MinHash, MinHashLSH
    except Exception:
        print('[lsh_coarsen] datasketch not installed; falling back to balanced K-means.')
        res = _coarsening.coarsen_balanced_kmeans(
            bv_t, n_clusters=min(int(n_target), n_atoms), seed=seed, device='cpu')
        return _finalize(res.labels.cpu().numpy())

    lsh = MinHashLSH(threshold=float(threshold), num_perm=int(num_perm))
    minhashes = []
    for i in range(n_atoms):
        mh = MinHash(num_perm=int(num_perm))
        for b in bv_np[i].nonzero()[0].tolist():
            mh.update(str(int(b)).encode('utf8'))
        minhashes.append(mh)
        lsh.insert(str(i), mh)
    parent = list(range(n_atoms))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n_atoms):
        for key in lsh.query(minhashes[i]):
            j = int(key)
            if j != i:
                ra, rb = _find(i), _find(j)
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([_find(i) for i in range(n_atoms)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.int64)
    if not balance:
        return _finalize(labels)

    target_size = max(1, int(round(n_atoms / max(1, int(n_target)))))
    new_labels = np.full(n_atoms, -1, dtype=np.int64)
    nxt = 0
    for b in range(int(labels.max()) + 1):
        idx = np.where(labels == b)[0]
        if idx.size == 0:
            continue
        if idx.size <= int(1.5 * target_size):
            new_labels[idx] = nxt
            nxt += 1
        else:
            k_sub = min(idx.size, max(1, int(round(idx.size / target_size))))
            res = _coarsening.coarsen_balanced_kmeans(
                bv_t[idx], n_clusters=k_sub, seed=seed, device='cpu')
            sub = res.labels.cpu().numpy().astype(np.int64)
            uniq = np.unique(sub)
            remap = {int(u): nxt + j for j, u in enumerate(uniq)}
            new_labels[idx] = np.array([remap[int(s)] for s in sub], dtype=np.int64)
            nxt += len(uniq)
    labels = new_labels
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    ns = int(labels.max()) + 1
    while ns > int(n_target):
        agg = _coarsening.aggregate_super_nodes(
            torch.as_tensor(labels, dtype=torch.long),
            torch.ones(n_atoms, dtype=torch.float32), features=bv_t, n_clusters=ns)
        cent = agg['features']
        sizes = np.bincount(labels, minlength=ns)
        small = int(sizes.argmin())
        cs = cent[small]
        sq = (cent * cent).sum(dim=1)
        inter = cent @ cs
        denom = (sq + float((cs * cs).sum()) - inter).clamp_min(1e-12)
        tani = (inter / denom).cpu().numpy()
        tani[small] = -1.0
        labels[labels == small] = int(tani.argmax())
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
        ns = int(labels.max()) + 1
    return _finalize(labels)


def remap_hierarchy_to_super_nodes(hierarchy_levels_atom, labels):
    '''Rewrite each family's descendant atom indices as super-node IDs, preserving
    name/similarity/lambda. Each super-node is assigned to exactly ONE family per
    level by MAJORITY atom vote, so the result is a valid partition even in the
    merge regime (n_target < n_families), where a merged super-node straddles
    families. In the normal regime (each super-node within one family) this is
    identical to a plain set-map. A valid partition is required for the H/LP
    containment term to be well posed.'''
    labels = np.asarray(labels)
    n_super = int(labels.max()) + 1 if labels.size else 0
    new_levels = []
    for level in hierarchy_levels_atom:
        assign = np.full(n_super, -1, dtype=np.int64)
        best = np.zeros(n_super, dtype=np.float64)
        for gid, desc in enumerate(level['descendants']):
            if len(desc) == 0:
                continue
            sn = labels[np.asarray(desc, dtype=np.int64)]
            cnt = np.bincount(sn, minlength=n_super).astype(np.float64)
            take = cnt > best
            assign[take] = gid
            best[take] = cnt[take]
        new_desc = [[] for _ in level['descendants']]
        for s in range(n_super):
            if assign[s] >= 0:
                new_desc[assign[s]].append(int(s))
        new_levels.append({'name': level['name'], 'descendants': [sorted(d) for d in new_desc],
                           'similarity': level['similarity'], 'lambda': level['lambda']})
    return new_levels

print('External coarsening algorithms defined: h_first_coarsen, lsh_coarsen, '
      'remap_hierarchy_to_super_nodes.')


# In[21]:


# Cell 4.5.2 - PRIMARY coarsening: atoms -> super-nodes (H-first, <= 1024)
N_L2_FAMILIES = len(hierarchy_main[0]['descendants'])
COARSEN_TRIALS_RESOLVED = [N_L2_FAMILIES if t == 'H_GROUPS' else int(t) for t in COARSEN_TRIALS]
print(f'L2 families (H_GROUPS): {N_L2_FAMILIES}')
print(f'Resolved COARSEN_TRIALS: {COARSEN_TRIALS_RESOLVED}')
print(f'Active COARSEN_METHOD  : {COARSEN_METHOD}')
print(f'PRIMARY n_super target : {COARSEN_N_SUPER_PRIMARY}')

t0 = time.time()
if COARSEN_METHOD == 'H_first':
    labels_super, centroids, super_weights, n_super = h_first_coarsen(
        hierarchy_main, D_feat, COARSEN_N_SUPER_PRIMARY, seed=RANDOM_SEED)
elif COARSEN_METHOD == 'LSH':
    labels_super, centroids, super_weights, n_super = lsh_coarsen(
        bv, COARSEN_N_SUPER_PRIMARY, seed=RANDOM_SEED)
else:
    raise ValueError(f"Unknown COARSEN_METHOD '{COARSEN_METHOD}'")

_sz = np.bincount(labels_super)
print(f'\nPRIMARY coarsening ({COARSEN_METHOD}): {n} atoms -> {n_super} super-nodes '
      f'({time.time()-t0:.1f}s)')
print(f'  super-node sizes: min={_sz.min()}, mean={_sz.mean():.1f}, max={_sz.max()}')
np.save(OUT_INTER / 'featurization' / 'labels_super.npy', labels_super)
np.save(OUT_INTER / 'featurization' / 'centroids_super.npy', centroids)


# In[22]:


# Cell 4.5.2b - super-node M-graph eps calibration (descriptor RBF) + guidance
_Krbf = _rbf_sim(centroids); np.fill_diagonal(_Krbf, 0.0)
_eps_grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
_deg_at = np.zeros((len(_eps_grid), n_super))
for gi, ev in enumerate(_eps_grid):
    _deg_at[gi] = (_Krbf > ev).sum(axis=1)
_density = _deg_at.sum(1) / max(n_super * (n_super - 1), 1)
_meandeg = _deg_at.mean(1); _isofrac = (_deg_at == 0).mean(1)
_floor = max(4.0, math.log2(max(n_super, 2)))
_ok = np.where((_isofrac <= 0.05) & (_meandeg >= _floor))[0]
_rec = int(_ok[-1]) if len(_ok) else int(np.argmin(np.abs(_density - 0.01)))
_eps_rec = float(_eps_grid[_rec])
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4))
axL.semilogy(_eps_grid, np.clip(_density, 1e-7, None), 'o-', color='#4477AA')
axL.axvspan(0.002, 0.02, color='green', alpha=0.12, label='target density 0.2-2%')
axL.axvline(_eps_rec, color='red', ls='--', label=f'recommended {_eps_rec:g}')
axL.set_xlabel('EPS_AFF_SUPER (descriptor-RBF threshold)'); axL.set_ylabel('M-graph density (log)')
axL.set_title('Super-node M-graph DENSITY vs eps'); axL.legend(fontsize=8)
axR.plot(_eps_grid, _meandeg, 'o-', color='#117733', label='mean degree')
axR.plot(_eps_grid, 100 * _isofrac, 's--', color='#CC6677', label='% isolated')
axR.axhline(_floor, color='gray', ls=':', label=f'degree floor {_floor:.0f}')
axR.axvline(_eps_rec, color='red', ls='--')
axR.set_xlabel('EPS_AFF_SUPER'); axR.set_ylabel('mean degree / % isolated')
axR.set_title('Super-node M-graph CONNECTIVITY vs eps'); axR.legend(fontsize=8)
plt.tight_layout(); save_fig(fig, 'super_node_eps_calibration', where='si'); plt.show()
print('HOW TO READ: pick the LARGEST eps (sparsest graph) whose % isolated stays < ~5%')
print('  and mean degree stays above the floor (gray dotted). Too LOW -> near-complete')
print('  graph (M cut non-discriminative); too HIGH -> isolated nodes (M loses signal).')
print(f'  Recommended EPS_AFF_SUPER = {_eps_rec:g}  (density={_density[_rec]:.4f}, '
      f'mean_deg={_meandeg[_rec]:.1f}, isolated={100*_isofrac[_rec]:.1f}%).')
if EPS_AFF_SUPER is None:
    EPS_AFF_SUPER = _eps_rec
    print(f'  => EPS_AFF_SUPER was None; AUTO-SET to {EPS_AFF_SUPER:g}.')
else:
    print(f'  => EPS_AFF_SUPER already set to {EPS_AFF_SUPER:g}; keeping user value.')


# In[23]:


# Cell 4.5.3 - super-node relational matrices (M, D, cost) + hierarchy + helpers
AM       = build_super_affinity(centroids)
AM = (AM * np.outer(super_weights, super_weights)).astype(np.float32)
np.fill_diagonal(AM, 0.0)
AM_super_tensor = torch.from_numpy(AM).to(torch.float32)




A_full   = build_super_kernel(centroids); np.fill_diagonal(A_full, 0.0)
K_MATRIX = build_super_kernel(centroids)
if D_KERNEL_POWER != 1.0:
    K_MATRIX = np.clip(K_MATRIX ** D_KERNEL_POWER, 0.0, 1.0).astype(np.float32)
np.fill_diagonal(K_MATRIX, 1.0)
C        = build_super_cost(centroids)
hier_super = remap_hierarchy_to_super_nodes(hierarchy_main, labels_super)
hier_levels_obj_super = build_hierarchy_levels(hier_super)




TOTAL_SIM = float(A_full.sum())

for _lvl in hier_super:
    _ids = [x for desc in _lvl['descendants'] for x in desc]
    assert len(_ids) == len(set(_ids)), f"super-node hierarchy level {_lvl['name']} not disjoint"

np.save(OUT_INTER / 'featurization' / 'AM_super.npy', AM)
np.save(OUT_INTER / 'featurization' / 'K_super.npy', K_MATRIX)
print(f'Super-node matrices: AM {AM.shape} (density {(AM>0).sum()/max(AM.size,1):.4f}), '
      f'K_MATRIX {K_MATRIX.shape}, C {C.shape}')
print(f'  AM off-diag mean (nonzero): {AM[AM>0].mean() if (AM>0).any() else 0:.3f}')


def lift_super_to_atoms(fold_super):
    '''Scatter a super-node fold assignment back to atom-level (length n).'''
    return lift_assignment_to_atoms(
        torch.as_tensor(fold_super, dtype=torch.long),
        torch.as_tensor(labels_super, dtype=torch.long)).cpu().numpy().astype(int)


def atom_fold_to_super(fold_atoms):
    '''Project an atom-level fold (length n) to super-nodes by weighted majority.
    For a lifted SHIELD fold this is exact (all atoms in a super-node share a fold).'''
    fold_atoms = np.asarray(fold_atoms)
    if fold_atoms.shape[0] == n_super:
        return fold_atoms.astype(int)
    tally = np.zeros((n_super, N_FOLDS), dtype=np.float64)
    np.add.at(tally, (labels_super, fold_atoms.astype(int)), 1.0)
    return tally.argmax(axis=1).astype(int)


def aggregate_targets_to_super(y_atom):
    '''Weight-average atom-level targets onto super-nodes (for stratification).'''
    y_atom = np.asarray(y_atom, dtype=np.float32)
    if y_atom.ndim == 1:
        y_atom = y_atom.reshape(-1, 1)
    agg = _coarsening.aggregate_super_nodes(
        torch.as_tensor(labels_super, dtype=torch.long),
        torch.ones(n, dtype=torch.float32),
        targets=torch.as_tensor(y_atom, dtype=torch.float32), n_clusters=n_super)
    return agg['targets'].cpu().numpy()

print('Helpers ready: lift_super_to_atoms, atom_fold_to_super, aggregate_targets_to_super.')


# In[24]:


# Cell 4.6 - FIXED evaluation reference (scale-invariant), in DESCRIPTOR space
_eval_ref_target = int(min(EVAL_REF_N_SUPER, n))
labels_ref, centroids_ref, super_weights_ref, n_ref = h_first_coarsen(
    hierarchy_main, D_feat, _eval_ref_target, seed=RANDOM_SEED)
K_ref = build_super_kernel(centroids_ref)
A_ref = K_ref.copy(); np.fill_diagonal(A_ref, 0.0)
_egr = np.round(np.arange(0.10, 0.91, 0.05), 2); _floor = max(4.0, math.log2(max(n_ref, 2)))
EVAL_REF_EPS = float(EPS_AFF_M)
for _ev in _egr:
    _d = (A_ref > _ev).sum(1)
    if (_d == 0).mean() <= 0.05 and _d.mean() >= _floor: EVAL_REF_EPS = float(_ev)
AM_ref = np.where(A_ref > EVAL_REF_EPS, A_ref, 0.0).astype(np.float32); np.fill_diagonal(AM_ref, 0.0)
for i in np.where((AM_ref > 0).sum(1) == 0)[0]:
    order = np.argsort(-A_ref[i]); cnt = 0
    for j in order:
        if int(j) == int(i): continue
        AM_ref[i, j] = max(AM_ref[i, j], T2_WEIGHT); AM_ref[j, i] = max(AM_ref[j, i], T2_WEIGHT); cnt += 1
        if cnt >= 3: break
AM_ref = np.maximum(AM_ref, AM_ref.T); np.fill_diagonal(AM_ref, 0.0)
w_ref = super_weights_ref.astype(np.float64)
TOTAL_SIM_REF = float(w_ref @ (A_ref @ w_ref)); TOTAL_M_REF = float(w_ref @ (AM_ref @ w_ref))

def _fold_to_ref(fold_atoms):
    fold = np.asarray(fold_atoms).astype(int)
    tally = np.zeros((n_ref, N_FOLDS), dtype=np.float64)
    np.add.at(tally, (labels_ref, fold), 1.0)
    return tally.argmax(axis=1).astype(int)

print(f'Fixed evaluation reference (DESCRIPTOR RBF): {n} atoms -> {n_ref} ref nodes '
      f'(EVAL_REF_N_SUPER={EVAL_REF_N_SUPER}, ref eps={EVAL_REF_EPS:g}; independent of '
      f'COARSEN_N_SUPER_PRIMARY={COARSEN_N_SUPER_PRIMARY}).')
print('  L(pi)/L_M scored here (mass-weighted, descriptor space); L_H/L_W exact atom-level.')


# ## 5. Baseline splitters 

# In[25]:


# Cell 5.1 - implement baseline splitters

from sklearn.cluster import KMeans, AgglomerativeClustering

def _greedy_pack(labels, n_clusters):
    """First-fit decreasing bin-packing of cluster groups into the train fold.

    Sorts clusters by descending size, then iterates ALL clusters: adds a
    cluster to train if it fits within the remaining budget, skips it otherwise.
    This fills the budget much more tightly than stopping at the first
    oversized cluster (the old break behavior which left budget unfilled).
    Keeps every cluster intact (no partial splits).
    """
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
    split_pt = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int)
    f[perm[:split_pt]] = 0
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
    rng = np.random.default_rng(seed)
    scaffold_to_idx = defaultdict(list)
    for i, s in enumerate(df['_scaffold']):
        scaffold_to_idx[s].append(i)
    order = list(scaffold_to_idx.values())
    rng.shuffle(order)
    f = np.ones(n, dtype=int)
    cum = 0
    target_train = int(round(FOLD_RATIOS[0] * n))
    for group in order:
        if cum + len(group) <= target_train:
            for j in group: f[j] = 0
            cum += len(group)
        else:
            break
    return f

def split_kmeans(seed, n_clusters=10):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(bv.astype(np.float32))
    return _greedy_pack(km.labels_, n_clusters)

def split_kennard_stone(seed):
    X = bv.astype(np.float32)
    centroid = X.mean(axis=0, keepdims=True)
    d = np.linalg.norm(X - centroid, axis=1)
    n_test = int(round(FOLD_RATIOS[1] * n))
    test_idx = np.argsort(-d)[:n_test]
    f = np.zeros(n, dtype=int)
    f[test_idx] = 1
    return f

def split_agglomerative(seed):
    pca = PCA(n_components=20, random_state=seed)
    X20_sub = pca.fit_transform(bv[EMB_IDX].astype(np.float32))
    ag = AgglomerativeClustering(n_clusters=N_FOLDS, linkage='ward')
    sub_lab = ag.fit_predict(X20_sub)
    cents = np.stack([X20_sub[sub_lab == c].mean(axis=0) for c in range(N_FOLDS)])
    X20_all = pca.transform(bv.astype(np.float32))
    d = ((X20_all[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1)
    sizes = np.bincount(labels, minlength=N_FOLDS)
    order = np.argsort(-sizes)
    fold_of_cluster = np.zeros(N_FOLDS, dtype=int)
    fold_of_cluster[order[0]] = 0
    fold_of_cluster[order[1:]] = 1
    return fold_of_cluster[labels]

BASELINE_SPLITTERS = {
    'RANDOM':       split_random,
    'STRATIFIED':   split_stratified,
    'SCAFFOLD':     split_scaffold,
    'KMEANS':       split_kmeans,
    'KENNARDSTONE': split_kennard_stone,
    'AGGLOMERATIVE':split_agglomerative,
}


# In[26]:


# Cell 5.2 - run all baseline splitters and persist splits
baseline_results = {}
for name, fn in BASELINE_SPLITTERS.items():
    f = fn(RANDOM_SEED)
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    out = OUT_INTER / 'splits' / f'{name}.csv'
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(out, index=False)
    print(f'  {name:14s}  sizes={sizes.round(3).tolist()}')


# In[27]:


# Cell 5.3 - precomputed splits from qm9_DataSAIL.csv

for _pc_name, _pc_col in [('SPLIT_BASE', '_split_base'), ('SPLIT_DATASAIL', '_split_datasail')]:
    _f = np.array([PRECOMPUTED_FOLD_MAP[lb] for lb in df[_pc_col].values], dtype=int)
    _sizes = np.bincount(_f, minlength=N_FOLDS) / n
    baseline_results[_pc_name] = _f
    _out = OUT_INTER / 'splits' / f'{_pc_name}.csv'
    pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(_out, index=False)
    print(f'  {_pc_name:16s}  sizes={_sizes.round(3).tolist()}')


# In[28]:


# Cell 5.4 - additional splitting tools

import warnings as _w54
_w54.filterwarnings('ignore')

_Y_PRIMARY = df[PRIMARY_TARGET].values.astype(np.float64)


def split_dc_butina(seed):
    """Sphere-exclusion (Leader) clustering on ECFP4 — C++ LeaderPicker, distance cutoff 0.4
    (Tanimoto >= 0.6), then greedy-pack clusters into folds. Falls back to exact
    Taylor-Butina (vectorized distance build) if the picker is unavailable."""
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

# In[29]:


# Cell 6.1 - SHIELD ablation configurations
SHIELD_CONFIGS = {
    'HMD-SHIELD':     dict(alpha=ALPHA, beta=BETA,  gamma=GAMMA, eta=ETA, mu=MU),
    'MD-SHIELD':     dict(alpha=0.0,   beta=BETA,  gamma=GAMMA, eta=ETA, mu=MU),
    'HD-SHIELD':     dict(alpha=ALPHA, beta=0.0,   gamma=GAMMA, eta=ETA, mu=MU),
    'HM-SHIELD':     dict(alpha=ALPHA, beta=BETA,  gamma=0.0,   eta=ETA, mu=MU),
    'H-SHIELD':   dict(alpha=ALPHA, beta=0.0,   gamma=0.0,   eta=ETA, mu=MU),
    'M-SHIELD':   dict(alpha=0.0,   beta=BETA,  gamma=0.0,   eta=ETA, mu=MU),
    'D-SHIELD':   dict(alpha=0.0,   beta=0.0,   gamma=GAMMA, eta=ETA, mu=MU),
}
print('SHIELD config grid (D-shift, y_target=None):')
for k, c in SHIELD_CONFIGS.items():
    print(f'  {k:18s}  {c}')


# In[30]:


# Cell 6.2 - run all SHIELD configurations at SUPER-NODE scale, then lift to atoms

Y_TARGETS = df[TARGET_COLS].values.astype(np.float32)
_Y_STD = Y_TARGETS.std(axis=0, keepdims=True); _Y_STD[_Y_STD == 0] = 1.0


shield_results     = {}
shield_diagnostics = {}
t_all = time.time()
CACHED_NORMALIZERS = None

for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    kwargs = dict(
        n=n_super, 
        K=N_FOLDS,
        weights=super_weights,
        r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hier_super           if cfg['alpha'] > 0 else None,
        affinity=AM                    if cfg['beta']  > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        kernel_matrix=K_MATRIX         if cfg['gamma'] > 0 else None,
        class_delta=STRAT_TOL,
        balance_eps=BALANCE_TOL,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'],     mu=cfg['mu'],     nu=NU, tau=TAU,
        coarsen=False,
        max_iter=SHIELD_MAX_ITER,
        seed=RANDOM_SEED,
        device=SHIELD_DEVICE,
        nodes=N_JOBS,
        integer_z_final=True,
        init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=SHIELD_ROUNDING,
        milp_size_limit=SHIELD_MILP_SIZE,
        milp_time_limit_s=SHIELD_MILP_TIME,
        stage0_samples=SHIELD_STAGE0_SAMPLES,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        res = shield_split(**kwargs)
        if CACHED_NORMALIZERS is None:
            CACHED_NORMALIZERS = res.metadata.get('stage0_normalizers')
        fold = lift_super_to_atoms(res.fold_assignment)
        shield_results[cfg_name] = fold
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


# In[31]:


# Cell 6.3 - Stage-0 normalizer table (main-text exhibit + SI table)
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

# In[32]:


# Cell 7.1 - cross-fold L(pi) on the FIXED scale-invariant reference (Cell 4.6)


def _fold_mass_to_ref(fold_atoms):
    fold = np.asarray(fold_atoms).astype(int)
    Mref = np.zeros((n_ref, N_FOLDS), dtype=np.float64)
    np.add.at(Mref, (labels_ref, fold), 1.0)
    return Mref

def cross_fold_similarity(fold, A=None):
    Mref = _fold_mass_to_ref(fold)
    within = sum(float(Mref[:, k] @ (A_ref @ Mref[:, k])) for k in range(N_FOLDS))
    raw = max(TOTAL_SIM_REF - within, 0.0) / 2.0
    return raw, raw / max(TOTAL_SIM_REF / 2.0, 1e-9)

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


# In[33]:


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


# In[34]:


# Cell 7.3 - per-channel decomposition L_H, L_M, L_W (fixed reference; Cell 4.6)
_L2_DESC_ATOM = [np.asarray(d, dtype=np.int64)
                 for d in hierarchy_main[0]['descendants'] if len(d) > 0]






def per_channel_leakage(fold):
    fold = np.asarray(fold).astype(int)
    minority = 0
    for desc in _L2_DESC_ATOM:
        fc = np.bincount(fold[desc], minlength=N_FOLDS)
        minority += int(fc.sum() - fc.max())
    L_H = float(minority) / max(n, 1)

    Mref = _fold_mass_to_ref(fold)
    within_m = sum(float(Mref[:, k] @ (AM_ref @ Mref[:, k])) for k in range(N_FOLDS))
    L_M = max(TOTAL_M_REF - within_m, 0.0) / 2.0

    tr = Y_TARGETS[fold == 0] / _Y_STD; te = Y_TARGETS[fold == 1] / _Y_STD
    L_W = float(np.mean([wasserstein_distance(tr[:, j], te[:, j])
                         for j in range(Y_TARGETS.shape[1])])) if len(tr) and len(te) else float('nan')
    return L_H, L_M, L_W

ch_rows = []
for name, fold in baseline_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'baseline', 'config': name, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
for name, fold in shield_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'SHIELD', 'config': name, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
channel_df = pd.DataFrame(ch_rows)
save_table(channel_df, 'channel_decomposition', where='main')
channel_df


# In[35]:


# Cell 7.5 - per-target Wasserstein heatmap (KEY main-text figure)
methods = list(baseline_results.keys()) + list(shield_results.keys())
W_mat = np.zeros((len(methods), len(TARGET_COLS)), dtype=np.float64)
for mi, name in enumerate(methods):
    fold = baseline_results.get(name) if name in baseline_results else shield_results[name]
    tr = Y_TARGETS[fold == 0]; te = Y_TARGETS[fold == 1]
    for tj, col in enumerate(TARGET_COLS):
        if len(tr) and len(te):
            W_mat[mi, tj] = wasserstein_distance(tr[:, tj], te[:, tj])
        else:
            W_mat[mi, tj] = np.nan

W_df = pd.DataFrame(W_mat, index=methods, columns=TARGET_COLS)
save_table(W_df.reset_index().rename(columns={'index': 'method'}),
           'per_target_wasserstein', where='main')

fig, ax = plt.subplots(figsize=(13, 0.5 * len(methods) + 1))
sns.heatmap(W_df, ax=ax, cmap='YlOrRd', annot=False, cbar_kws={'label': 'W_1(train, test)'})
ax.set_title('Per-target 1-D Wasserstein distance between train and test marginals')
plt.tight_layout()
save_fig(fig, 'per_target_wasserstein_heatmap', where='main')
plt.show()


# In[36]:


# Cell 7.6 - kNN purity diagnostic (atom subsample; avoids O(n_test*n_train) blow-up)

import os
from threadpoolctl import threadpool_limits

_N_THREADS = os.cpu_count() if N_JOBS in (None, -1) else max(1, N_JOBS)

_rng_knn = np.random.default_rng(RANDOM_SEED + 11)
if KNN_PURITY_SUBSAMPLE is None or n <= KNN_PURITY_SUBSAMPLE:
    KNN_IDX = np.arange(n)
else:
    KNN_IDX = np.sort(_rng_knn.choice(n, size=int(KNN_PURITY_SUBSAMPLE), replace=False))

_Xs = bv[KNN_IDX].astype(np.float32)
_Xsq = (_Xs * _Xs).sum(1)
_k_max = max(KNN_K)

def knn_purity_all_k(fold, ks=KNN_K, idx=KNN_IDX, Xs=_Xs, Xsq=_Xsq, k_max=_k_max):
    fold_s = np.asarray(fold)[idx]
    test_idx = np.where(fold_s == 1)[0]
    if not len(test_idx):
        return {k: float('nan') for k in ks}

    Xt = Xs[test_idx]
    with threadpool_limits(limits=_N_THREADS, user_api='blas'):
        d = (Xt * Xt).sum(1, keepdims=True) + Xsq[None, :] - 2 * Xt @ Xs.T
    d[np.arange(len(test_idx)), test_idx] = np.inf

    kth = min(k_max, Xs.shape[0] - 2)
    part = np.argpartition(d, kth=kth, axis=1)[:, :kth + 1]
    d_part = np.take_along_axis(d, part, axis=1)
    order = np.argsort(d_part, axis=1)
    nn_sorted = np.take_along_axis(part, order, axis=1)

    out = {}
    for k in ks:
        neigh = nn_sorted[:, :k]
        out[k] = float(np.mean(fold_s[neigh] == 0))
    return out

purity_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    per_k = knn_purity_all_k(fold)
    for k in KNN_K:
        purity_rows.append({'config': name, 'k': k, 'knn_purity': per_k[k]})
purity_df = pd.DataFrame(purity_rows)
save_table(purity_df, 'knn_purity', where='main')

fig, axes = plt.subplots(1, len(KNN_K), figsize=(5 * len(KNN_K), 4))
if len(KNN_K) == 1: axes = [axes]
for ax, k in zip(axes, KNN_K):
    sub = purity_df[purity_df['k'] == k].set_index('config')['knn_purity'].sort_values()
    colors = ['#888888' if c in baseline_results else '#4477AA' for c in sub.index]
    ax.bar(range(len(sub)), sub.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.index, rotation=45, ha='right')
    ax.axhline(FOLD_RATIOS[0], color='red', ls='--', lw=1, label='random baseline')
    ax.set_title(f'kNN purity (k={k}, sample={len(KNN_IDX)})'); ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
save_fig(fig, 'knn_purity_panels', where='main')
plt.show()


# ## 11. Dimension-reduction visualizations (PCA / t-SNE / UMAP) of splits

# In[37]:


# Cell 11.1 - compute embeddings
embeds = {}
try:
    embeds['tSNE'] = TSNE(
                            n_components=2,
                            perplexity=50,
                            early_exaggeration=24,
                            learning_rate='auto',
                            max_iter=1000,
                            init='pca',
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                        ).fit_transform(bv.astype(np.float32))
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


# In[38]:


# Cell 11.2 - split-colored scatter panels (all methods)
all_splits = {**baseline_results, **shield_results}
all_names  = list(all_splits.keys())

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

# In[ ]:


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


# In[40]:


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


# In[41]:

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


# ## 9. M/D term-construction sweep: vary the settings that define the M and D terms

# In[42]:


# Cell 9.1 - M/D term-construction sweep (descriptor RBF; one SHIELD solve per setting)

import itertools as _it

def _build_AM_md(centroids, eps, t2_weight, t2_k):
    K = _rbf_sim(centroids); np.fill_diagonal(K, 0.0)
    A = np.where(K > eps, K, 0.0).astype(np.float32); np.fill_diagonal(A, 0.0)
    for i in np.where((A > 0).sum(axis=1) == 0)[0]:
        order = np.argsort(-K[i]); cnt = 0
        for j in order:
            if int(j) == int(i): continue
            A[i, j] = max(A[i, j], t2_weight); A[j, i] = max(A[j, i], t2_weight); cnt += 1
            if cnt >= int(t2_k): break
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0.0); return A

def _build_K_md(centroids, power):
    K = np.clip(_rbf_sim(centroids), 0.0, 1.0).astype(np.float32)
    if float(power) != 1.0: K = np.clip(K ** float(power), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(K, 1.0); return K

md_grid = list(_it.product(SWEEP_EPS_AFF_M, SWEEP_T2_WEIGHT, SWEEP_T2_K, SWEEP_D_KERNEL_POWER))
print(f'M/D term-construction sweep: {len(md_grid)} combination(s) on {n_super} super-nodes '
      f'(eps x t2w x t2k x Dpow = {len(SWEEP_EPS_AFF_M)} x {len(SWEEP_T2_WEIGHT)} x '
      f'{len(SWEEP_T2_K)} x {len(SWEEP_D_KERNEL_POWER)})')
md_sweep_results = {}; md_sweep_rows = []; t_all = time.time()
for (eps, t2w, t2k, dpow) in md_grid:
    label = f'eps{eps:g}_t2w{t2w:g}_k{int(t2k)}_p{dpow:g}'; t0 = time.time()
    try:
        AM_s = _build_AM_md(centroids, eps, t2w, t2k); K_s = _build_K_md(centroids, dpow)
        res = shield_split(
            n=n_super, weights=super_weights, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
            hierarchy=hier_super, affinity=AM_s, affinity_provenance='label-blind-handcrafted',
            d_mode=DIST_MODE, kernel_matrix=K_s, class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
            alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
            coarsen=False, max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED,
            device=SHIELD_DEVICE, nodes=N_JOBS, integer_z_final=True,
            init_backend=SHIELD_INIT_BACKEND, rounding_mode=SHIELD_ROUNDING,
            milp_size_limit=SHIELD_MILP_SIZE, milp_time_limit_s=SHIELD_MILP_TIME,
            stage0_samples=SHIELD_STAGE0_SAMPLES)
        fold = lift_super_to_atoms(res.fold_assignment); md_sweep_results[label] = fold
        L_H, L_M, L_W = per_channel_leakage(fold); _, scaled = cross_fold_similarity(fold)
        md_sweep_rows.append({'label': label, 'EPS_AFF_M': eps, 'T2_WEIGHT': t2w, 'T2_K': int(t2k),
            'D_KERNEL_POWER': dpow, 'AM_density': float((AM_s > 0).sum() / max(AM_s.size, 1)),
            'scaled_L_pi': scaled, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W,
            'frac_test': float((fold == 1).mean()), 'wall_s': time.time() - t0})
        print(f'  {label:30s} scaled_L_pi={scaled:.4f}  L_M={L_M:.1f}  L_W={L_W:.4f}  wall={time.time()-t0:.1f}s')
    except Exception as exc:
        print(f'  {label:30s} FAILED: {exc}')
md_sweep_df = pd.DataFrame(md_sweep_rows)
save_table(md_sweep_df, 'md_term_sweep_grid', where='main')
print(f'\nM/D sweep complete in {time.time()-t_all:.1f}s; {len(md_sweep_results)} split(s).')


# In[43]:


# Cell 9.2 - M/D sweep results table + recommended term-construction setting
if not md_sweep_df.empty:
    ranked = md_sweep_df.sort_values('scaled_L_pi').reset_index(drop=True)
    best = ranked.iloc[0]
    print('Recommended M/D setting (lowest scaled L(pi) = strongest structural OOD):')
    print(f"  EPS_AFF_M={best['EPS_AFF_M']:g}, T2_WEIGHT={best['T2_WEIGHT']:g}, "
          f"T2_K={int(best['T2_K'])}, D_KERNEL_POWER={best['D_KERNEL_POWER']:g}  ->  "
          f"scaled_L_pi={best['scaled_L_pi']:.4f}, L_M={best['L_M']:.1f}, L_W={best['L_W']:.4f}")
    save_table(ranked, 'md_term_sweep_ranked', where='main')
    display(ranked.head(-1))
else:
    print('M/D sweep produced no rows (check SHIELD / sweep ranges).')


# In[44]:


# Cell 9.3 - M/D sweep diagnostics: OOD strength vs the term-construction settings
if not md_sweep_df.empty:
    swept = [p for p in ['EPS_AFF_M', 'T2_WEIGHT', 'T2_K', 'D_KERNEL_POWER']
             if md_sweep_df[p].nunique() > 1]
    if not swept:
        swept = ['EPS_AFF_M']
    fig, axes = plt.subplots(1, len(swept), figsize=(5 * len(swept), 4), squeeze=False)
    for ax, p in zip(axes[0], swept):
        g = md_sweep_df.groupby(p)['scaled_L_pi'].mean()
        ax.plot(g.index, g.values, marker='o', color='#4477AA')
        ax.set_xlabel(p); ax.set_ylabel('mean scaled L(pi)')
        ax.set_title(f'OOD strength vs {p}\n(lower = stronger OOD)')
    plt.tight_layout()
    save_fig(fig, 'md_sweep_leakage_vs_params', where='main')
    plt.show()

    if md_sweep_df['EPS_AFF_M'].nunique() > 1 and md_sweep_df['T2_WEIGHT'].nunique() > 1:
        pvt = md_sweep_df.pivot_table(index='T2_WEIGHT', columns='EPS_AFF_M',
                                      values='scaled_L_pi', aggfunc='mean')
        fig2, ax2 = plt.subplots(figsize=(1.3 * pvt.shape[1] + 2, 1.0 * pvt.shape[0] + 2))
        sns.heatmap(pvt, ax=ax2, annot=True, fmt='.3f', cmap='viridis_r',
                    cbar_kws={'label': 'scaled L(pi)'})
        ax2.set_title('M-term sweep: scaled L(pi) over EPS_AFF_M x T2_WEIGHT\n(lower = stronger OOD)')
        plt.tight_layout()
        save_fig(fig2, 'md_sweep_eps_t2_heatmap', where='main')
        plt.show()

    # super-node M graph density vs eps threshold
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for t2w, sub in md_sweep_df.groupby('T2_WEIGHT'):
        s = sub.groupby('EPS_AFF_M')['AM_density'].mean()
        ax3.plot(s.index, s.values, marker='s', label=f'T2_WEIGHT={t2w:g}')
    ax3.set_xlabel('EPS_AFF_M'); ax3.set_ylabel('super-node AM density')
    ax3.set_title('M graph density vs eps threshold'); ax3.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig3, 'md_sweep_am_density', where='si')
    plt.show()
else:
    print('No M/D sweep rows to plot.')


# In[45]:


# Cell 9.4 - sensitivity of the split to the M/D term settings (pairwise fold diff)
if len(md_sweep_results) >= 2:
    labels = list(md_sweep_results.keys()); m = len(labels)
    D = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i != j:
                D[i, j] = 100.0 * (md_sweep_results[labels[i]] != md_sweep_results[labels[j]]).mean()
    Dd = pd.DataFrame(D, index=labels, columns=labels)
    save_table(Dd.reset_index().rename(columns={'index': 'label'}),
               'md_sweep_fold_diff_pct', where='si')
    fig, ax = plt.subplots(figsize=(max(6, m * 0.7), max(5, m * 0.6)))
    sns.heatmap(Dd, ax=ax, cmap='magma', annot=(m <= 12), fmt='.0f',
                cbar_kws={'label': '% molecules in different fold'})
    ax.set_title(f'Split sensitivity to M/D term settings ({m} combinations)\n'
                 f'(higher % = more sensitive to the setting)')
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)
    ax.tick_params(axis='y', labelrotation=0, labelsize=6)
    plt.tight_layout()
    save_fig(fig, 'md_sweep_sensitivity_heatmap', where='main')
    plt.show()
    iu = np.triu_indices(m, k=1)
    print(f'Mean pairwise fold difference across M/D settings: {D[iu].mean():.1f}%  '
          f'(0% = setting-insensitive split; higher = more sensitive)')
else:
    print('Need >= 2 sweep settings for a sensitivity heatmap.')


# ## 15. SI — Coarsening experiment

# In[39]:


# Cell 15.0 - external-coarsening sensitivity: run SHIELD over varying n_super

coarsen_results  = {}
coarsen_metadata = {}

if RUN_EXTERNAL_COARSENING:
    for n_target in COARSEN_TRIALS_RESOLVED:
        label = f'ext_coarsen_{n_target}'
        print(f'\n{"="*60}\n  Trial: {label}  (n_target={n_target})\n{"="*60}')
        try:
            if COARSEN_METHOD == 'H_first':
                labels_c, centroids_c, sw_c, n_super_c = h_first_coarsen(
                    hierarchy_main, D_feat, n_target, seed=RANDOM_SEED)
            else:
                labels_c, centroids_c, sw_c, n_super_c = lsh_coarsen(
                    bv, n_target, seed=RANDOM_SEED)
            sizes = np.bincount(labels_c)
            print(f'  {n} atoms -> {n_super_c} super-nodes (requested {n_target}); '
                  f'sizes min/mean/max={sizes.min()}/{sizes.mean():.1f}/{sizes.max()}')

            AM_c   = build_super_affinity(centroids_c)
            AM_c = (AM_c * np.outer(sw_c, sw_c)).astype(np.float32); np.fill_diagonal(AM_c, 0.0)
            K_c = build_super_kernel(centroids_c)
            if D_KERNEL_POWER != 1.0:
                K_c = np.clip(K_c ** D_KERNEL_POWER, 0.0, 1.0).astype(np.float32)
            np.fill_diagonal(K_c, 1.0)
            hier_c = remap_hierarchy_to_super_nodes(hierarchy_main, labels_c)

            t0 = time.time()
            res_c = shield_split(
                n=n_super_c,
                weights=sw_c,
                K=N_FOLDS,
                r=np.array(FOLD_RATIOS, dtype=np.float32),
                hierarchy=hier_c,
                affinity=AM_c,
                affinity_provenance='label-blind-handcrafted',
                d_mode=DIST_MODE, kernel_matrix=K_c,
                class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
                alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
                coarsen=False,
                max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
                integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
                rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
                milp_time_limit_s=SHIELD_MILP_TIME,
                stage0_samples=SHIELD_STAGE0_SAMPLES,
            )
            fold_atoms = lift_assignment_to_atoms(
                res_c.fold_assignment,
                torch.as_tensor(labels_c, dtype=torch.long)).cpu().numpy().astype(int)
            coarsen_results[label] = fold_atoms
            coarsen_metadata[label] = {
                'n_super': n_super_c, 'n_target': n_target, 'method': COARSEN_METHOD,
                'size_min': int(sizes.min()), 'size_mean': float(sizes.mean()),
                'size_max': int(sizes.max()), 'sizes': sizes.astype(int).tolist(),
                'wall_s': time.time() - t0, 'history': res_c.history, 'metadata': res_c.metadata,
            }
            fold_sizes = (np.bincount(fold_atoms, minlength=N_FOLDS) / n).round(3).tolist()
            print(f'  Atom-level fold sizes: {fold_sizes}  wall={time.time()-t0:.1f}s')
            (OUT_INTER / 'shield_runs' / label).mkdir(exist_ok=True, parents=True)
            pd.DataFrame({'index': np.arange(n), 'fold': fold_atoms}).to_csv(
                OUT_INTER / 'shield_runs' / label / 'split.csv', index=False)
        except Exception as exc:
            import traceback
            print(f'  FAILED: {exc}'); traceback.print_exc()

    print(f'\nCompleted {len(coarsen_results)}/{len(COARSEN_TRIALS_RESOLVED)} trials.')


# In[40]:


# Cell 15.1 - leakage diagnostics for external-coarsening trials
coarsen_rows = []
_full = channel_df[channel_df['config'] == 'HMD-SHIELD']
coarsen_rows.append({
    'config': 'HMD-SHIELD', 'L_H': float(_full['L_H'].iloc[0]),
    'L_M': float(_full['L_M'].iloc[0]), 'L_W': float(_full['L_W'].iloc[0]),
    'scaled_L_pi': float(lpi_df.set_index('config').loc['HMD-SHIELD', 'scaled_L_pi']),
})
for label, fold in coarsen_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    coarsen_rows.append({'config': label, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W,
                         'scaled_L_pi': cross_fold_similarity(fold)[1]})
coarsen_df = pd.DataFrame(coarsen_rows, columns=['config', 'L_H', 'L_M', 'L_W', 'scaled_L_pi'])
save_table(coarsen_df, 'external_coarsening_leakage', where='si')
print(f'External-coarsening leakage ({len(coarsen_df)} rows: HMD-SHIELD + {len(coarsen_results)} trials):')
display(coarsen_df)


# In[41]:


# Cell 15.2 - embedding scatter: external-coarsening trials vs HMD-SHIELD
if coarsen_results and embeds:
    panel = {'HMD-SHIELD': shield_results['HMD-SHIELD']}
    panel.update(coarsen_results)
    _lpi = {row['config']: row['scaled_L_pi'] for _, row in coarsen_df.iterrows()}
    for emb_name, emb in embeds.items():
        ncfg = len(panel); cols = min(4, ncfg); rows = int(np.ceil(ncfg / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (name, fold) in zip(axes, panel.items()):
            fs = np.asarray(fold)
            ax.scatter(emb[fs == 0, 0], emb[fs == 0, 1], s=6, c='#4477AA', alpha=0.55, label='train')
            ax.scatter(emb[fs == 1, 0], emb[fs == 1, 1], s=6, c='#CC6677', alpha=0.55, label='test')
            tag = ' (reference)' if name == 'HMD-SHIELD' else ''
            ax.set_title(f'{name}{tag}\nscaled L(pi)={_lpi.get(name, float("nan")):.3f}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[ncfg:]: ax.set_visible(False)
        axes[0].legend(loc='best', fontsize=7)
        fig.suptitle(f'{emb_name} - external coarsening vs HMD-SHIELD', fontsize=11)
        plt.tight_layout()
        save_fig(fig, f'external_coarsening_{emb_name.lower()}_scatter', where='si')
        plt.show()
else:
    print('No coarsen_results or embeddings; skipping scatter.')


# In[42]:


# Cell 15.4 - pairwise fold-assignment difference heatmap (external coarsening)
if coarsen_results:
    configs = {'HMD-SHIELD': shield_results['HMD-SHIELD']}
    for lb in sorted(coarsen_results.keys(), key=lambda x: coarsen_metadata.get(x, {}).get('n_super', 0)):
        configs[lb] = coarsen_results[lb]
    names = list(configs.keys()); nc = len(names)
    diff = np.zeros((nc, nc), dtype=np.float64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i != j:
                diff[i, j] = 100.0 * (configs[ni] != configs[nj]).mean()
    diff_df = pd.DataFrame(diff, index=names, columns=names)
    save_table(diff_df.reset_index().rename(columns={'index': 'config'}),
               'external_coarsening_fold_diff_pct', where='si')
    fig, ax = plt.subplots(figsize=(max(6, nc * 1.4), max(5, nc * 1.15)))
    sns.heatmap(diff_df, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd',
                vmin=0, vmax=(diff[diff < 100].max() if diff.max() > 0 else 1.0),
                linewidths=0.5, linecolor='#cccccc',
                cbar_kws={'label': '% molecules in different fold', 'shrink': 0.8})
    ax.set_title('Pairwise fold-assignment difference (% molecules)\nHMD-SHIELD = reference',
                 fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelrotation=30, labelsize=9); ax.tick_params(axis='y', labelrotation=0, labelsize=9)
    plt.tight_layout(); save_fig(fig, 'external_coarsening_fold_diff_heatmap', where='si'); plt.show()
    print('\nDifference vs HMD-SHIELD:')
    ref = configs['HMD-SHIELD']
    for name, fold in list(configs.items())[1:]:
        pct = 100.0 * (fold != ref).mean()
        print(f'  {name:20s}: {pct:5.1f}%  ({int((fold != ref).sum()):,} / {len(ref):,})')
else:
    print('No coarsen_results; skipping difference heatmap.')


# In[43]:


# Cell 15.5 - super-node size distribution diagnostics
if coarsen_metadata:
    order = sorted(coarsen_metadata.keys(), key=lambda lb: coarsen_metadata[lb].get('n_super', 0))
    npan = len(order); cols = min(4, npan); rows = int(np.ceil(npan / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, label in zip(axes, order):
        sizes = np.asarray(coarsen_metadata[label].get('sizes', []), dtype=float)
        n_super_c = coarsen_metadata[label].get('n_super', int(sizes.size))
        if sizes.size:
            ax.hist(sizes, bins=min(40, max(1, int(sizes.max()))), color='#117733', edgecolor='none', alpha=0.85)
            ax.axvline(sizes.mean(), color='red', linestyle='--', linewidth=1.2, label=f'mean={sizes.mean():.1f}')
            ax.legend(fontsize=7)
        ax.set_title(f'{label}\nn_super={n_super_c}', fontsize=9)
        ax.set_xlabel('atoms per super-node', fontsize=8); ax.set_ylabel('count', fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[npan:]: ax.set_visible(False)
    fig.suptitle('Super-node size distributions across external-coarsening trials', fontsize=11)
    plt.tight_layout(); save_fig(fig, 'external_coarsening_supernode_sizes', where='si'); plt.show()
else:
    print('No coarsen_metadata; skipping size diagnostics.')

