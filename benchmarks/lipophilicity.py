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

# Repo path discovery
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


# In[ ]:


# Cell 1.2 - global configuration (every user-tunable knob)
# Dataset
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'Lipophilicity.csv')
DATASET_NAME = 'lipophilicity'
SMILES_COL   = 'smiles'
TARGET_COLS_RAW = ['exp']
PRIMARY_TARGET  = 'exp'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 42

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.05
STRAT_TOL   = 0.05

# Fingerprints (shared by M term and D cost matrix)
ECFP_RADIUS  = 2
ECFP_NBITS   = 1024
PRIMARY_AFFINITY = 'tanimoto'

# M term
EPS_SERIES    = 0.4

# H term
LAMBDA_MAIN = [1.0, 1.0]


# SHIELD coefficients
ALPHA, BETA, GAMMA, ETA, MU, NU, TAU = 1.0, 0.2, 3.0, 1.0, 0.5, 0.0, 0.0

# D term
DIST_MODE    = 'match'
OT_REG       = 0.05

# Stratification on logD
STRAT_STRATEGY  = 'quantile'
STRAT_N_BINS    = 10
SW_PROJECTIONS  = 100
SW_QUANTILE_PTS = 256

# Sweep grid
SWEEP_GRID = {
    'alpha': [0.2, 0.5, 1.0, 3.0],
    'beta':  [0.2, 0.5, 1.0, 3.0],
    'gamma': [0.2, 0.5, 1.0, 3.0],
}
HAMMING_STABILITY_THRESHOLD = 0.10

# kNN purity
KNN_K = [5, 10, 25]

# SHIELD solver controls
SHIELD_MAX_ITER     = 15
SHIELD_STAGE0_SAMPLES  = 100
SHIELD_SINKHORN_ITER = 50
SHIELD_INIT_BACKEND = 'auto'
SHIELD_ROUNDING     = 'pipage'
SHIELD_MILP_SIZE    = 5_000
SHIELD_DEVICE       = 'cpu'
N_JOBS              = -1

# ML evaluation
ML_RUN = True


# In[3]:


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


# In[4]:


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
    raise RuntimeError('RDKit is required for featurization. Install via conda-forge.')


# ## 2. Data loading and EDA

# In[5]:


# Cell 2.1 - load Lipophilicity dataset
print(f'Loading from {DATASET_PATH}')
df_raw = pd.read_csv(DATASET_PATH)
TARGET_COLS = [c for c in TARGET_COLS_RAW if c in df_raw.columns]
print(f'Raw rows: {len(df_raw):,};  detected {len(TARGET_COLS)} target column(s)')
print(f'Target stats:\n{df_raw[TARGET_COLS].describe().round(3)}')

if MAX_ROWS is not None and len(df_raw) > MAX_ROWS:
    df = df_raw.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df):,} rows (seed={RANDOM_SEED})')
else:
    df = df_raw.copy()

n = len(df)
df.head()


# In[6]:


# Cell 2.2 - attach DataSAIL splits and filter dataset to matched molecules

from rdkit import Chem as _Chem
from rdkit.Chem.inchi import MolToInchi as _MolToInchi
import warnings as _warnings
_warnings.filterwarnings('ignore', category=UserWarning)

DATASAIL_PATH = str(_REPO_ROOT / 'splits' / 'lipophilicity_DataSAIL.csv')

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
    if len(parts) < 2: return None  
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


# In[7]:


# Cell 2.2 - logD distribution 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.hist(df[PRIMARY_TARGET].dropna(), bins=50, color='#4477AA',
        edgecolor='black', alpha=0.85)
ax.axvline(df[PRIMARY_TARGET].mean(), color='red', ls='--', lw=1.5,
           label=f'mean = {df[PRIMARY_TARGET].mean():.2f}')
ax.axvline(0, color='gray', ls=':', lw=1, alpha=0.5, label='logD=0')
ax.set_xlabel('logD (experimental, pH 7.4)')
ax.set_ylabel('count')
ax.set_title(f'Lipophilicity logD distribution (n={n:,})')
ax.legend(fontsize=9)

ax2 = axes[1]
from scipy.stats import probplot
probplot(df[PRIMARY_TARGET].dropna(), dist='norm', plot=ax2)
ax2.set_title('logD Q-Q plot (normality check)')

fig.suptitle('Lipophilicity target distribution (n=%d)' % n)
plt.tight_layout()
save_fig(fig, 'eda_target_distribution', where='si')
plt.show()


# In[8]:


# Cell 2.3 - logD vs molecular weight scatter
try:
    from rdkit import Chem
    from rdkit.Chem.Descriptors import MolWt, NumHDonors, NumHAcceptors
    mw_vals  = [MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.nan
                for s in df[SMILES_COL]]
    hbd_vals = [NumHDonors(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.nan
                for s in df[SMILES_COL]]
    logd_vals = df[PRIMARY_TARGET].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(mw_vals, logd_vals, s=8, alpha=0.4, c='#4477AA', edgecolors='none')
    ax.set_xlabel('Molecular weight (Da)'); ax.set_ylabel('logD (exp)')
    ax.set_title('logD vs MW')

    ax2 = axes[1]
    ax2.scatter(hbd_vals, logd_vals, s=8, alpha=0.5, c='#CC3311', edgecolors='none')
    ax2.set_xlabel('H-bond donors'); ax2.set_ylabel('logD (exp)')
    ax2.set_title('logD vs H-bond donors\n(HBD strongly reduces logD — motivates H term)')

    plt.tight_layout()
    save_fig(fig, 'eda_logd_vs_descriptors', where='si')
    plt.show()
except Exception as exc:
    print(f'EDA scatter skipped: {exc}')


# ## 3. Featurization 

# In[9]:


# Cell 3.1 - compute ECFP4 fingerprints and bit vectors
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

CACHE_FP = OUT_INTER / 'featurization' / f'fps_lipo_n{n}_r{ECFP_RADIUS}_b{ECFP_NBITS}.pkl'

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


# In[10]:


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
atom_bin = pd.cut(natoms, bins=[-0.5, 19.5, 29.5, np.inf],
                  labels=['ha<=19', 'ha20-29', 'ha>=30']).astype(str)

df['_scaffold'] = scaffolds
df['_ring']     = rings
df['_atombin']  = atom_bin

print(f'Distinct scaffolds : {df["_scaffold"].nunique()}')
print(f'Distinct rings     : {df["_ring"].nunique()}')
print(f'Atom-count bins    : {df["_atombin"].value_counts().to_dict()}')


# In[11]:


# Cell 3.3 - lipophilicity descriptor space + ACTUAL functional-group counts
from rdkit.Chem import Descriptors as _Desc, rdMolDescriptors as _rdMD, Crippen as _Crip
from rdkit.Chem import MolFromSmarts as _Sma

_ACID = [_Sma(s) for s in ['[CX3](=O)[OX2H1]', '[CX3](=O)[O-]',
                           '[SX4](=[OX1])(=[OX1])[OX2H1]',
                           '[PX4](=[OX1])([OX2H1])[OX2H1]', 'c1nnn[nH]1']]
_BASE = [_Sma(s) for s in ['[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$([N+]);!$(Nc);!$(NS=O)][CX4]',
                           '[NX3][CX3]=[NX2]', '[NX3]C(=[NX2])[NX3]']]
_FUNC = {'COOH': '[CX3](=O)[OX2H1]', 'sulfonic': '[SX4](=[OX1])(=[OX1])[OX2H1]',
         'phenol': '[c][OX2H1]', 'alcohol': '[OX2H1][CX4]', 'amide': '[CX3](=O)[NX3]',
         'ether': '[#6][OX2H0][#6]', 'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
         'sulfonamide': '[SX4](=[OX1])(=[OX1])[NX3]', 'CF3': '[CX4]([F])([F])[F]',
         'arom_halide': '[c][F,Cl,Br,I]', 'arom_amine': '[c][NX3;H1,H2]'}
_FUNC_C = {k: _Sma(v) for k, v in _FUNC.items()}

def _count(m, pats):
    return sum(len(m.GetSubstructMatches(p)) for p in pats if p is not None) if m is not None else 0

_DESC_NAMES = (['MolLogP', 'TPSA', 'NumHBD', 'NumHBA', 'MolWt', 'nAromRings', 'nRotB',
                'FracCsp3', 'nHalogen', 'n_acid', 'n_base'] + list(_FUNC_C))

def _featurize(m):
    if m is None:
        return [0.0] * len(_DESC_NAMES), 'neutral_lipophilic'
    logp = _Crip.MolLogP(m); tpsa = _rdMD.CalcTPSA(m)
    hbd = _rdMD.CalcNumHBD(m); hba = _rdMD.CalcNumHBA(m); mw = _Desc.MolWt(m)
    narom = _rdMD.CalcNumAromaticRings(m); nrot = _rdMD.CalcNumRotatableBonds(m)
    fcsp3 = _rdMD.CalcFractionCSP3(m)
    nhal = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F', 'Cl', 'Br', 'I'))
    nacid = _count(m, _ACID); nbase = _count(m, _BASE)
    fc = [_count(m, [p]) for p in _FUNC_C.values()]
    vec = [logp, tpsa, float(hbd), float(hba), mw, float(narom), float(nrot), fcsp3,
           float(nhal), float(nacid), float(nbase)] + [float(x) for x in fc]
    if nacid > 0 and nbase > 0: ion = 'zwitterionic'
    elif nacid > 0:             ion = 'anionic_acidic'
    elif nbase > 0:             ion = 'strongly_cationic'
    elif tpsa >= 40 or hbd >= 1: ion = 'neutral_polar'
    else:                        ion = 'neutral_lipophilic'
    return vec, ion

mols = mols if 'mols' in globals() else [Chem.MolFromSmiles(s) for s in df[SMILES_COL].tolist()]
_rows, _ion = [], []
for m in mols:
    v, io_ = _featurize(m); _rows.append(v); _ion.append(io_)
desc_raw = np.array(_rows, dtype=np.float64)
_mu = desc_raw.mean(0); _sd = desc_raw.std(0)
L_feat = ((desc_raw - _mu) / np.where(_sd < 1e-9, 1.0, _sd)).astype(np.float32)
L_feat[:, _sd < 1e-9] = 0.0
df['_ion_class'] = _ion
df['_logp_band'] = pd.cut(desc_raw[:, 0], bins=[-1e9, 1, 2, 3, 1e9],
                          labels=['logP<1', 'logP1-2', 'logP2-3', 'logP>3']).astype(str)

_rng_s = np.random.default_rng(RANDOM_SEED + 5)
_si = _rng_s.choice(n, size=min(n, 2000), replace=False)
_X = L_feat[_si].astype(np.float64); _sq = (_X * _X).sum(1)
_d2 = np.clip(_sq[:, None] + _sq[None, :] - 2 * (_X @ _X.T), 0, None)
RBF_SIGMA = float(np.sqrt(max(np.median(_d2[np.triu_indices(len(_si), 1)]), 1e-6)) / np.sqrt(2))
print(f'L_feat: {L_feat.shape[1]} lipophilicity descriptors (incl. actual functional-group '
      f'counts); RBF_SIGMA={RBF_SIGMA:.4f}')
print('Ionization class:', df['_ion_class'].value_counts().to_dict())
for _nm, _c in [('MolLogP', 0), ('TPSA', 1), ('NumHBD', 2)]:
    _r = float(np.corrcoef(desc_raw[:, _c], df[PRIMARY_TARGET].values)[0, 1])
    print(f'  Pearson r({_nm}, logD) = {_r:+.3f}')
print('  (logP should correlate strongly POSITIVELY with logD; TPSA/HBD negatively -- '
      'confirming the descriptor space tracks the target.)')


# In[12]:


# Cell 3.4 - M term: congeneric-series (matched-analog) affinity for GRAPH-CUT
from rdkit import DataStructs
from collections import defaultdict
import scipy.sparse as _sps, scipy.sparse.csgraph as _csg

EPS_SERIES = globals().get('EPS_SERIES', 0.50)

# M term: block-sparse analog graph 
_series = defaultdict(list)
for _i, _sc in enumerate(df['_scaffold'].tolist()):
    if _sc and _sc != 'acyclic':
        _series[_sc].append(_i)

AM = np.zeros((n, n), dtype=np.float32); _npairs = 0
for _sc, _mem in _series.items():
    if len(_mem) < 2:
        continue
    _fg = [fps[i] for i in _mem]
    for _a in range(len(_mem)):
        _sims = DataStructs.BulkTanimotoSimilarity(_fg[_a], _fg)
        for _b in range(_a + 1, len(_mem)):
            if _sims[_b] >= EPS_SERIES:
                AM[_mem[_a], _mem[_b]] = _sims[_b]; AM[_mem[_b], _mem[_a]] = _sims[_b]; _npairs += 1
np.fill_diagonal(AM, 0.0)
np.save(OUT_INTER / 'featurization' / 'AM.npy', AM)

def _lipo_rbf(X, sigma=None):
    sigma = RBF_SIGMA if sigma is None else sigma
    X = np.asarray(X, dtype=np.float32); sq = (X * X).sum(1); m = X.shape[0]
    K = np.zeros((m, m), dtype=np.float32)
    for s in range(0, m, 512):
        e = min(s + 512, m)
        d2 = np.clip(sq[s:e, None] + sq[None, :] - 2.0 * (X[s:e] @ X.T), 0.0, None)
        K[s:e] = np.exp(-d2 / (2.0 * sigma * sigma + 1e-12)).astype(np.float32)
    return K
A_full = _lipo_rbf(L_feat); np.fill_diagonal(A_full, 0.0)
K_MATRIX = A_full.copy(); np.fill_diagonal(K_MATRIX, 1.0)

# connectivity report 
_nc, _lab = _csg.connected_components(_sps.csr_matrix(AM > 0), directed=False)
_csize = np.bincount(_lab); _multi = _csize[_csize >= 2]
_in_series = int((AM > 0).any(1).sum()); _test_sz = int(round(FOLD_RATIOS[1] * n))
print(f'M = congeneric-series analog graph (scaffold blocks, intra-Tanimoto >= {EPS_SERIES}):')
print(f'  scaffolds with >=2 members : {sum(1 for m in _series.values() if len(m) >= 2)}')
print(f'  analog pairs (edges)       : {_npairs:,}   density={float((AM>0).mean()):.5f}')
print(f'  molecules in a series      : {_in_series}/{n} ({100*_in_series/n:.0f}%) -- singletons '
      f'have NO analogs and are CORRECTLY left unconstrained (no Tier-2 needed)')
print(f'  series clusters (>=2)      : {len(_multi)}   largest = {int(_csize.max())}  '
      f'(test fold ~ {_test_sz})')
if int(_csize.max()) > _test_sz:
    print(f'  WARNING: largest series ({int(_csize.max())}) > test fold ({_test_sz}); raise '
          f'EPS_SERIES to fragment it so a series can be held out whole.')
print('  A_full / K_MATRIX rebuilt as the descriptor-RBF similarity (for the D term + L(pi)).')


# In[13]:


# Cell 3.5 - M term diagnostics & tuning: congeneric-series analog graph
from rdkit import DataStructs
from collections import defaultdict
import scipy.sparse as _sps, scipy.sparse.csgraph as _csg
from scipy.stats import spearmanr
_y = df[PRIMARY_TARGET].values

_S = defaultdict(list)
for _i, _sc in enumerate(df['_scaffold'].tolist()):
    if _sc and _sc != 'acyclic': _S[_sc].append(_i)
_pi, _pj, _ps, _pdy = [], [], [], []
for _sc, _mem in _S.items():
    if len(_mem) < 2: continue
    _fg = [fps[i] for i in _mem]; _yv = _y[_mem]
    for _a in range(len(_mem)):
        _sims = DataStructs.BulkTanimotoSimilarity(_fg[_a], _fg)
        for _b in range(_a + 1, len(_mem)):
            _pi.append(_mem[_a]); _pj.append(_mem[_b]); _ps.append(_sims[_b]); _pdy.append(abs(_yv[_a]-_yv[_b]))
_pi=np.array(_pi); _pj=np.array(_pj); _ps=np.array(_ps); _pdy=np.array(_pdy)
_rng=np.random.default_rng(RANDOM_SEED); _ri=_rng.integers(0,n,30000); _rj=_rng.integers(0,n,30000)
_vv=_ri!=_rj; _cross_dy=np.abs(_y[_ri[_vv]]-_y[_rj[_vv]]); _cross_mu=float(_cross_dy.mean())
_test_sz=int(round(FOLD_RATIOS[1]*n))

# sweep EPS_SERIES
print(f'{len(_ps):,} intra-scaffold analog pairs found.  Cross-scaffold baseline mean |dlogD| = {_cross_mu:.2f}\n')
print('   eps   pairs  %mol_in_series  largest_series  mean|dlogD|_in  ratio_in/cross  spearman  verdict')
print('  ' + '-'*94)
_grid = sorted(set([0.0, 0.1, 0.2, 0.25, 0.30,0.40,0.50,0.60,0.70] + [round(EPS_SERIES,2)]))
for _e in _grid:
    _m = _ps >= _e
    if int(_m.sum()) < 5:
        print(f'  {_e:.2f}     <5 pairs (too sparse)'); continue
    _g = _sps.csr_matrix((np.ones(int(_m.sum())), (_pi[_m], _pj[_m])), shape=(n, n)); _g = _g + _g.T
    _ncc, _lb = _csg.connected_components(_g, directed=False); _cs = np.bincount(_lb)
    _largest = int(_cs[_cs >= 2].max()) if (_cs >= 2).any() else 0
    _inmol = int(np.unique(np.concatenate([_pi[_m], _pj[_m]])).size)
    _muin = float(_pdy[_m].mean()); _ratio = _muin / (_cross_mu + 1e-9)
    _r, _ = spearmanr(_ps[_m], -_pdy[_m])
    _vd = ('OK   ' if (_largest <= _test_sz and 0.15 <= _inmol/n <= 0.85 and _ratio < 0.90)
           else ('WARN series>fold' if _largest > _test_sz else '~ check'))
    _mk = '  <-- current' if abs(_e - EPS_SERIES) < 1e-9 else ''
    print(f'  {_e:.2f}  {int(_m.sum()):6d}   {100*_inmol/n:9.0f}%  {_largest:13d}  {_muin:13.2f}  {_ratio:13.2f}  {_r:8.3f}  {_vd}{_mk}')
print('\n  READ THIS: a GOOD M (a) makes within-series |dlogD| MUCH smaller than the cross baseline')
print('  (ratio < ~0.8 => analogs really do share logD), (b) keeps the largest series <= the test')
print('  fold (else it cannot be held out whole), (c) puts ~15-80% of molecules in some series.')
print('  Raise EPS_SERIES => tighter, smaller, more logD-coherent series; lower => looser analogs.')

# 4-panel figure at current EPS_SERIES 
_m = _ps >= EPS_SERIES
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0,0].hist(_ps, bins=50, color='#4477AA', alpha=0.85)
ax[0,0].axvline(EPS_SERIES, ls='--', lw=1.6, color='red', label=f'EPS_SERIES={EPS_SERIES}')
ax[0,0].set_xlabel('intra-scaffold Tanimoto'); ax[0,0].set_ylabel('pair count')
ax[0,0].set_title('A. Within-series analog similarity'); ax[0,0].legend(fontsize=8)
_g=_sps.csr_matrix((np.ones(int(_m.sum())),(_pi[_m],_pj[_m])),shape=(n,n)); _g=_g+_g.T
_ncc,_lb=_csg.connected_components(_g,directed=False); _cs=np.bincount(_lb); _ms=_cs[_cs>=2]
ax[0,1].hist(_ms, bins=30, color='#117733', alpha=0.85); ax[0,1].set_yscale('log')
ax[0,1].axvline(_test_sz, color='red', ls='--', label=f'test fold ~{_test_sz}')
ax[0,1].set_xlabel('series (cluster) size'); ax[0,1].set_ylabel('count (log)')
ax[0,1].set_title(f'B. Congeneric-series sizes ({len(_ms)} series)'); ax[0,1].legend(fontsize=8)
ax[1,0].hist(_cross_dy, bins=50, density=True, alpha=0.5, color='#BBBBBB', label=f'random pairs (mu={_cross_mu:.2f})')
ax[1,0].hist(_pdy[_m], bins=50, density=True, alpha=0.6, color='#CC6677', label=f'within-series (mu={_pdy[_m].mean():.2f})')
ax[1,0].set_xlabel('|delta logD|'); ax[1,0].set_ylabel('density')
ax[1,0].set_title('C. Analogs share logD (within-series << random)'); ax[1,0].legend(fontsize=8)
_r,_=spearmanr(_ps[_m], -_pdy[_m])
_sub=np.random.default_rng(0).choice(np.where(_m)[0], min(int(_m.sum()),4000), replace=False)
ax[1,1].scatter(_ps[_sub], _pdy[_sub], s=6, alpha=0.3, color='#882255')
ax[1,1].set_xlabel('edge Tanimoto'); ax[1,1].set_ylabel('|delta logD|')
ax[1,1].set_title(f'D. Stronger analog edge -> smaller |dlogD| (Spearman r={_r:.3f})')
plt.tight_layout(); save_fig(fig,'M_congeneric_series_diagnostics',where='main'); plt.show()

_frac_iso = 1 - np.unique(np.concatenate([_pi[_m],_pj[_m]])).size/n
print('\nGUIDANCE:')
print(f'  - {100*_frac_iso:.0f}% of molecules are singletons (no analogs) -> EXPECTED & fine; M leaves')
print('    them free (no Tier-2 rescue needed, unlike an RBF graph).')
print('  - If panel C shows within-series ~ random (ratio ~1), EPS_SERIES is too low (off-target series).')
print('  - If panel D Spearman is weak/negative, the scaffold blocks are not logD-coherent: raise EPS_SERIES.')
print('  - Tune M strength via BETA (cell 1.2 / 6.1); try BETA in [1,3] if the M channel stays weak in the sweep.')


# In[14]:


# Cell 3.6 - D-match cost C + target marginals (lipophilicity descriptor space)
C = (1.0 - A_full).astype(np.float32); C = np.clip((C + C.T) / 2.0, 0.0, 1.0); np.fill_diagonal(C, 0.0)
np.save(OUT_INTER / 'featurization' / 'cost_matrix.npy', C)
TARGET_MARG = np.tile(np.full(n, 1.0 / n, dtype=np.float32), (N_FOLDS, 1))
print(f'D-match cost C (1 - descriptor-RBF): shape={C.shape}, mean={C.mean():.4f}, '
      f'symmetric={np.allclose(C, C.T)}, diag={C.diagonal().mean():.3f}')
print(f'Target marginal: uniform 1/{n} (each fold should be a representative lipophilicity sample)')


# In[15]:


# Cell 3.7 - D-match cost matrix diagnostics

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: C heatmap (PCA-sorted subsample)
rng_d = np.random.default_rng(RANDOM_SEED + 3)
sub_size = min(80, n)
sub_idx  = rng_d.choice(n, size=sub_size, replace=False)
pca_1d   = PCA(n_components=1, random_state=RANDOM_SEED).fit_transform(
                bv[sub_idx].astype(np.float32)).ravel()
order    = np.argsort(pca_1d)
C_sub    = C[np.ix_(sub_idx[order], sub_idx[order])]
ax = axes[0]
im = ax.imshow(C_sub, cmap='magma', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='C(i,j) = 1 - Tanimoto')
ax.set_title(f'D-match cost matrix C\n({sub_size}-mol subsample, PCA-sorted)')
ax.set_xlabel('molecule index (PCA order)')
ax.set_ylabel('molecule index (PCA order)')

# Panel 2: off-diagonal cost distribution
off_diag = C[~np.eye(n, dtype=bool)]
ax2 = axes[1]
ax2.hist(off_diag, bins=60, color='#4477AA', edgecolor='none', alpha=0.85)
ax2.axvline(off_diag.mean(), color='red', ls='--',
            label=f'mean = {off_diag.mean():.3f}')
ax2.axvline(np.median(off_diag), color='orange', ls=':',
            label=f'median = {np.median(off_diag):.3f}')
ax2.set_xlabel('C(i,j) = 1 - Tanimoto similarity')
ax2.set_ylabel('count')
ax2.set_title('Off-diagonal cost distribution\n(D-match: lower cost = more similar pairs)')
ax2.legend(fontsize=9)

plt.tight_layout()
save_fig(fig, 'D_cost_diagnostics', where='main')
plt.show()
print('D-match: label-blind OT cost; y_target passed separately for stratification.')


# ## 4. Hierarchy construction 
# 

# In[16]:


# Cell 4.1 - H term: lipophilicity functional / ionization taxonomy

import re as _re, math as _math
from rdkit.Chem import MolFromSmarts, Crippen, rdMolDescriptors

MAX_FAMILY = globals().get('MAX_FAMILY', 350)
MIN_FAMILY = globals().get('MIN_FAMILY', 25)

# (i) ionization state at pH 7.4
_ACID_SMARTS = [
    '[CX3](=O)[OX2H1]', '[CX3](=O)[O-]',
    '[$([SX4](=O)(=O)[OX2H1]),$([SX4](=O)(=O)[O-])]',
    '[$([PX4](=O)([OX2H1])[OX2H1]),$([PX4](=O)([O-])[OX2H1]),$([PX4](=O)([O-])[O-])]',
    'c1[nH]nnn1', 'c1n[nH]nn1',
    '[SX4](=O)(=O)[NX3H1][CX3]=[OX1]',
]
_BASE_SMARTS = [
    '[NX3;!$(NC=O);!$(N=*);!$(N#*);!$(NS(=O)=O);!$([NX3]a);!$([N+])][CX4]',
    '[NX3][CX3]=[NX2]',
    '[NX3]C(=[NX2])[NX3]',
]
_acid_pats = [p for p in (MolFromSmarts(s) for s in _ACID_SMARTS) if p is not None]
_base_pats = [p for p in (MolFromSmarts(s) for s in _BASE_SMARTS) if p is not None]
_cf3_pat   = MolFromSmarts('[CX4](F)(F)F')

def _has(m, pats):
    return any(m.HasSubstructMatch(p) for p in pats)

def _ionization(m):
    if m is None: return 'neutral'
    a, b = _has(m, _acid_pats), _has(m, _base_pats)
    if a and b: return 'zwitterion'
    if a:       return 'acid'
    if b:       return 'base'
    return 'neutral'

# (ii)+(iii) lipophilic character
def _character(m):
    if m is None: return 'aliphatic'
    nhal  = sum(1 for at in m.GetAtoms() if at.GetSymbol() in ('F', 'Cl', 'Br', 'I'))
    cf3   = m.HasSubstructMatch(_cf3_pat) if _cf3_pat is not None else False
    tpsa  = rdMolDescriptors.CalcTPSA(m)
    narom = rdMolDescriptors.CalcNumAromaticRings(m)
    if cf3 or nhal >= 2: return 'halogen_rich'
    if tpsa >= 90:       return 'very_polar'
    if tpsa >= 50:       return 'polar'
    if narom >= 3:       return 'aromatic_rich'
    if narom >= 1:       return 'aromatic'
    return 'aliphatic'

ion   = [_ionization(m) for m in mols]
charc = [_character(m)  for m in mols]
logp  = np.array([Crippen.MolLogP(m) if m is not None else 0.0 for m in mols])
base_label = np.array([f'{i}|{c}' for i, c in zip(ion, charc)], dtype=object)

# size-balance
fam = base_label.copy()
_sizes = pd.Series(fam).value_counts()
_tiny  = set(_sizes[_sizes < MIN_FAMILY].index)
for j in range(n):
    if fam[j] in _tiny:
        fam[j] = f'{ion[j]}|minor'
_sizes = pd.Series(fam).value_counts()
for lab in [l for l in _sizes.index if l.endswith('|minor') and _sizes[l] < MIN_FAMILY]:
    io_ = lab.split('|')[0]
    sib = [l for l in _sizes.index if l.startswith(io_ + '|') and l != lab]
    if sib:
        fam[fam == lab] = max(sib, key=lambda l: _sizes[l])
# cap
out = fam.copy()
for lab in pd.unique(fam):
    idx = np.where(fam == lab)[0]
    if len(idx) > MAX_FAMILY:
        k = int(_math.ceil(len(idx) / MAX_FAMILY))
        order = idx[np.argsort(logp[idx])]
        for b, chunk in enumerate(np.array_split(order, k)):
            out[chunk] = f'{lab}|lp{b}'
fam = out

df['_polarity_L2'] = fam
df['_polarity_L1'] = np.array([_re.sub(r'\|lp\d+$', '', f) for f in fam], dtype=object)

# report summary
_l2 = pd.Series(fam); _szl2 = _l2.value_counts()
print(f'H L2 (fine, lambda={LAMBDA_MAIN[0]}): {_szl2.size} families | '
      f'largest={_szl2.max()} | smallest={_szl2.min()} | singletons={(_szl2==1).sum()}')
print('  family                              n     mean_logD')
for lab, cnt in _szl2.sort_values(ascending=False).items():
    print(f'  {lab:34s} {cnt:4d}   {df[PRIMARY_TARGET].values[fam==lab].mean():+.2f}')
print(f'\nH L1 (coarse, lambda={LAMBDA_MAIN[1]}) ionization x character:')
for lab, cnt in pd.Series(df["_polarity_L1"]).value_counts().items():
    print(f'  {lab:28s} {cnt:5d}   mean_logD={df[PRIMARY_TARGET].values[df["_polarity_L1"].values==lab].mean():+.2f}')
assert _szl2.max() <= MAX_FAMILY and (_szl2 == 1).sum() == 0

# build hierarchy (deepest first) 
def build_hierarchy(df_local, lambda_per_level):
    def _grp(col):
        vals = df_local[col].tolist(); uniq = list(dict.fromkeys(vals))
        idx = {u: i for i, u in enumerate(uniq)}; desc = [[] for _ in uniq]
        for i, v in enumerate(vals): desc[idx[v]].append(i)
        return desc
    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float64))
    return [{'name': 'polarity_functional_class', 'descendants': _grp('_polarity_L2'),
             'similarity': empty, 'lambda': lambda_per_level[0]},
            {'name': 'ionization_class', 'descendants': _grp('_polarity_L1'),
             'similarity': empty, 'lambda': lambda_per_level[1]}]

hierarchy_main = build_hierarchy(df, LAMBDA_MAIN)
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)

# guiding visuals 
def _eta_sq(y, g):
    y = np.asarray(y, float); sst = float(((y - y.mean())**2).sum())
    if sst <= 0: return 0.0
    g = np.asarray(g); ssw = sum(float(((y[g==lv]-y[g==lv].mean())**2).sum()) for lv in pd.unique(g))
    return max(0.0, 1.0 - ssw/sst)
_y = df[PRIMARY_TARGET].values
fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
_s = _szl2.sort_values(ascending=False)
ax[0].bar(range(len(_s)), _s.values, color='#4477AA')
ax[0].axhline(MAX_FAMILY, color='red', ls='--', label=f'MAX_FAMILY={MAX_FAMILY}')
ax[0].set_xlabel('L2 family rank'); ax[0].set_ylabel('n molecules')
ax[0].set_title(f'L2 family sizes ({len(_s)} families, max {_s.max()})'); ax[0].legend(fontsize=8)
_o = df.groupby('_polarity_L1')[PRIMARY_TARGET].median().sort_values().index.tolist()
bp = ax[1].boxplot([_y[df['_polarity_L1'].values==g] for g in _o], patch_artist=True, showfliers=False)
for p in bp['boxes']: p.set_facecolor('#88CCEE')
ax[1].axhline(_y.mean(), color='red', ls='--', lw=1)
ax[1].set_xticks(range(1, len(_o)+1)); ax[1].set_xticklabels(_o, rotation=30, ha='right', fontsize=7)
ax[1].set_ylabel('logD'); ax[1].set_title('logD by ionization x character (L1)')
_ax = {'ionization_class (L1)': df['_polarity_L1'].values,
       'functional_family (L2)': df['_polarity_L2'].values,
       'logP_band': df['_logp_band'].values}
_e = {k: _eta_sq(_y, v) for k, v in _ax.items()}
ax[2].barh(range(len(_e)), list(_e.values()), color='#117733')
ax[2].set_yticks(range(len(_e))); ax[2].set_yticklabels(list(_e.keys()), fontsize=8)
ax[2].set_xlabel('eta^2 (logD variance explained)'); ax[2].set_title('Axis vs logD')
for i, v in enumerate(_e.values()): ax[2].text(v+0.005, i, f'{v:.2f}', va='center', fontsize=8)
plt.tight_layout(); save_fig(fig, 'H_lipophilicity_taxonomy', where='main'); plt.show()
print('\nThe L2 families are all <= MAX_FAMILY with no singletons, each labeled by an actual')
print('logD determinant (ionization x lipophilic character x logP band). The boxplot shows')
print('ionization/character separates logD; eta^2 quantifies how much logD each axis explains.')


# In[17]:


# Cell 4.2 - H term visualizations: lipophilicity taxonomy diagnostics
from sklearn.decomposition import PCA as _PCA

_y    = df[PRIMARY_TARGET].values
_L1   = df['_polarity_L1'].astype(str).values
_L2   = df['_polarity_L2'].astype(str).values
_ionp = np.array([s.split('|')[0] for s in _L1])
_char = np.array([s.split('|')[1] if '|' in s else 'na' for s in _L1])
if 'logp' in globals():
    _logp = np.asarray(logp, dtype=float)
else:
    from rdkit.Chem import Crippen as _Cr
    _logp = np.array([_Cr.MolLogP(m) if m is not None else np.nan for m in mols])

_ION_ORDER = [c for c in ['acid', 'zwitterion', 'base', 'neutral'] if c in set(_ionp)]
_ICOL = {'acid': '#CC6677', 'zwitterion': '#AA4499', 'base': '#4477AA', 'neutral': '#117733'}

fig, axes = plt.subplots(2, 2, figsize=(15, 11)); axA, axB, axC, axD = axes.ravel()

# Panel A: ionization x character grid (mean logD, annotated with counts) 
_tmp = pd.DataFrame({'ion': _ionp, 'char': _char, 'logD': _y})
_mean = _tmp.pivot_table(index='ion', columns='char', values='logD', aggfunc='mean')
_cnt  = _tmp.pivot_table(index='ion', columns='char', values='logD', aggfunc='size')
_mean = _mean.reindex(index=[i for i in _ION_ORDER if i in _mean.index])
_cnt  = _cnt.reindex(index=_mean.index, columns=_mean.columns)
_annot = np.where(_cnt.values > 0,
                  np.char.add(np.char.add(np.round(_mean.values, 2).astype('U6'), '\n n='),
                              np.nan_to_num(_cnt.values).astype(int).astype('U6')), '')
sns.heatmap(_mean, ax=axA, cmap='RdYlBu_r', annot=_annot, fmt='', annot_kws={'fontsize': 8},
            cbar_kws={'label': 'mean logD'}, linewidths=0.5, linecolor='white')
axA.set_title('A. Functional matrix: mean logD by ionization x lipophilic character')
axA.set_xlabel('lipophilic character'); axA.set_ylabel('ionization @ pH 7.4')

# Panel B: logD vs cLogP colored by ionization (the ionization penalty)
for c in _ION_ORDER:
    m = _ionp == c
    axB.scatter(_logp[m], _y[m], s=10, alpha=0.45, color=_ICOL.get(c, '#888888'),
                label=f'{c} (n={int(m.sum())}, ΔlogD={np.nanmean(_y[m]-_logp[m]):+.2f})', linewidths=0)
_lo = float(np.nanmin([_logp.min(), _y.min()])); _hi = float(np.nanmax([_logp.max(), _y.max()]))
axB.plot([_lo, _hi], [_lo, _hi], 'k--', lw=1, label='logD = cLogP (neutral)')
axB.set_xlabel('cLogP (Crippen)'); axB.set_ylabel('experimental logD(7.4)')
axB.set_title('B. logD vs cLogP -- acids/bases fall below y=x (ionization penalty)')
axB.legend(fontsize=7, loc='upper left')

# Panel C: PCA of the lipophilicity descriptor space, colored by ionization 
_pc = _PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(np.asarray(L_feat))
for c in _ION_ORDER:
    m = _ionp == c
    axC.scatter(_pc[m, 0], _pc[m, 1], s=10, alpha=0.5, color=_ICOL.get(c, '#888888'),
                label=f'{c}', linewidths=0)
axC.set_xlabel('descriptor PC1'); axC.set_ylabel('descriptor PC2')
axC.set_title('C. Lipophilicity-descriptor space (PCA) by ionization class')
axC.legend(fontsize=8, markerscale=2)

# Panel D: mean logD per L2 family (sorted), marker size = family size
_g = pd.DataFrame({'fam': _L2, 'logD': _y}).groupby('fam')['logD']
_stat = pd.DataFrame({'mean': _g.mean(), 'std': _g.std().fillna(0.0), 'n': _g.size()}).sort_values('mean')
_fcol = [_ICOL.get(f.split('|')[0], '#888888') for f in _stat.index]
axD.errorbar(range(len(_stat)), _stat['mean'], yerr=_stat['std'], fmt='none', ecolor='#cccccc', zorder=1)
axD.scatter(range(len(_stat)), _stat['mean'], s=np.clip(_stat['n'].values, 10, 300),
            c=_fcol, alpha=0.8, edgecolors='black', linewidths=0.4, zorder=2)
axD.axhline(_y.mean(), color='red', ls='--', lw=1, label='global mean logD')
axD.set_xlabel(f'L2 family rank ({len(_stat)} families)'); axD.set_ylabel('mean logD (± std)')
axD.set_title('D. logD gradient across L2 families (marker size = family n)')
axD.legend(fontsize=8)

plt.tight_layout(); save_fig(fig, 'H_lipophilicity_taxonomy_diagnostics', where='main'); plt.show()

# logD - cLogP ionization penalty table (the mechanism, quantified) 
print('Ionization penalty (mean logD - cLogP per ionization class):')
print('  neutral ~ 0 (logD≈logP); acids most negative; bases/zwitterions negative -- as expected.')
for c in _ION_ORDER:
    m = _ionp == c
    print(f'  {c:11s}: n={int(m.sum()):4d}  mean(logD-cLogP)={np.nanmean(_y[m]-_logp[m]):+.2f}  '
          f'mean logD={_y[m].mean():+.2f}')

# H-vs-M coherence
print('\nH/M coherence (mean intra-group vs cross-group descriptor-RBF similarity):')
print('  intra > cross confirms H groups are coherent in M''s space without being identical to it.')
_top = pd.Series(_L1).value_counts().head(6).index.tolist()
for grp in _top:
    ii = np.where(_L1 == grp)[0]
    if ii.size < 2: continue
    oo = np.where(_L1 != grp)[0]
    _intra = A_full[np.ix_(ii, ii)][np.triu_indices(ii.size, k=1)].mean()
    _cross = A_full[np.ix_(ii, oo[:min(oo.size, 2000)])].mean()
    print(f'  {grp:26s}: intra={_intra:.3f}  cross={_cross:.3f}  ratio={_intra/(_cross+1e-9):.2f}x')


# ## 5. Baseline splitters

# In[18]:


# Cell 5.1 - implement baseline splitters
from sklearn.cluster import KMeans, AgglomerativeClustering

def _greedy_pack(labels, n_clusters):
    """First-fit decreasing bin-packing of cluster groups into the train fold.

    Sorts clusters by descending size, then iterates ALL clusters: adds a
    cluster to train if it fits within the remaining budget, skips it otherwise.
    This fills the budget much more tightly than stopping at the first oversized
    cluster (the old break behavior which left part of the train quota unfilled).
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
    split_pt = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[perm[:split_pt]] = 0
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

def split_kmeans(seed, n_clusters=10):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(bv.astype(np.float32))
    return _greedy_pack(km.labels_, n_clusters)

def split_kennard_stone(seed):
    X = bv.astype(np.float32)
    centroid = X.mean(axis=0, keepdims=True)
    d = np.linalg.norm(X - centroid, axis=1)
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int); f[np.argsort(-d)[:n_test]] = 1
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


# In[19]:


# Cell 5.2 - run all baseline splitters and persist splits
baseline_results = {}
for name, fn in BASELINE_SPLITTERS.items():
    f = fn(RANDOM_SEED)
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(
        OUT_INTER / 'splits' / f'{name}.csv', index=False)
    print(f'  {name:16s}  sizes={sizes.round(3).tolist()}')


# In[20]:


# Cell 5.3 - precomputed splits from lipophilicity_DataSAIL.csv

for _pc_name, _pc_col in [('SPLIT_BASE', '_split_base'), ('SPLIT_DATASAIL', '_split_datasail')]:
    _f = np.array([PRECOMPUTED_FOLD_MAP[lb] for lb in df[_pc_col].values], dtype=int)
    _sizes = np.bincount(_f, minlength=N_FOLDS) / n
    baseline_results[_pc_name] = _f
    pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(
        OUT_INTER / 'splits' / f'{_pc_name}.csv', index=False)
    print(f'  {_pc_name:16s}  sizes={_sizes.round(3).tolist()}')


# In[21]:


# Cell 5.4 - additional splitting tools

import warnings as _w54
_w54.filterwarnings('ignore')

_y_logd = df[PRIMARY_TARGET].values


# DeepChem-style splits

def split_dc_butina(seed):
    """Sphere-exclusion (Leader) clustering on ECFP4 — C++ LeaderPicker, distance cutoff 0.4
    (Tanimoto >= 0.6), then assign each molecule to its most-similar leader and greedy-pack
    clusters. Falls back to exact Butina with vectorized distance build if picker missing."""
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
    """Fingerprint lexicographic-sort split: train = first 80% by ECFP4 bit order."""
    packed = np.packbits(bv, axis=1)
    order  = np.lexsort(packed.T[::-1])
    n_train = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[order[:n_train]] = 0
    return f

def split_dc_maxmin(seed):
    """MaxMin diversity picker — C++ MaxMinPicker (lazy distances); falls back to
    vectorized numpy greedy loop. Test set = the maximally diverse picked subset."""
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
    """DeepChem scaffold split: Murcko groups packed greedily into folds."""
    grp = defaultdict(list)
    for i, sc in enumerate(df['_scaffold']): grp[sc].append(i)
    labels_map = {sc: cid for cid, sc in enumerate(grp)}
    labels = np.array([labels_map[sc] for sc in df['_scaffold']])
    return _greedy_pack(labels, len(grp))

def split_dc_weight(seed):
    """Molecular-weight split: test = heaviest 20% of molecules."""
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
        print(f'  {_name:16s}  sizes={_sizes.round(3).tolist()}')
    except Exception as _exc:
        print(f'  {_name:16s}  FAILED: {_exc}')


# ## 6. SHIELD primary configurations and Stage-0 normalizer table

# In[22]:


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


# In[23]:


# Cell 6.2 - run all SHIELD configurations

Y_TARGETS = df[TARGET_COLS].values.astype(np.float32)
y_strat   = Y_TARGETS[:, 0:1]

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
        cost=C                         if cfg['gamma'] > 0 else None,
        target_marginals=TARGET_MARG   if cfg['gamma'] > 0 else None,
        epsilon_OT=OT_REG,
        y_target=y_strat               if cfg['mu']    > 0 else None,
        stratification_strategy=STRAT_STRATEGY if cfg['mu'] > 0 else 'none',
        n_bins=STRAT_N_BINS,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
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
        sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
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
        logd_tr = float(Y_TARGETS[fold == 0, 0].mean())
        logd_te = float(Y_TARGETS[fold == 1, 0].mean())
        print(f'  {cfg_name:18s}  wall={time.time()-t0:.1f}s  '
              f'sizes={(np.bincount(fold, minlength=N_FOLDS)/n).round(3).tolist()}  '
              f'logD_mean=[{logd_tr:.2f},{logd_te:.2f}]')
    except Exception as exc:
        print(f'  {cfg_name:18s}  FAILED: {exc}')

print(f'\nTotal SHIELD wall time: {time.time()-t_all:.1f}s')


# In[24]:


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


# ## 7. Leakage diagnostics — L(pi), channel decomposition, kNN purity, alpha-shape IoU, per-target Wasserstein

# In[25]:


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


# In[26]:


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


# In[27]:


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


# In[28]:


# Cell 7.5 - kNN purity 
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

# Display the table before plotting
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


# Cell 7.6 - Train Potential Leakage (TPL)
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors

bv32 = bv.astype(np.float32)

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

    # Build ANN index on TRAIN set only
    if use_hnsw:
        idx = hnswlib.Index(space='l2', dim=d)
        idx.init_index(max_elements=len(train_idx), ef_construction=200, M=32)
        idx.add_items(Xtr, np.arange(len(train_idx)))
        idx.set_ef(max(m + 16, 64))

        # Compute bandwidth
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

    # Compute TPL via Mann-Whitney U statistic
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

# Compute TPL for all splits
tpl_rows = []

for name, fold in {**baseline_results, **shield_results}.items():
    tpl_val, tpl_lo, tpl_hi, (phi_tr, phi_te, h) = tpl(bv32, fold)
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

# Compute error bars from CI
err_lower = sub.values - tpl_df.set_index('config').loc[sub.index, 'tpl_ci_lower'].values
err_upper = tpl_df.set_index('config').loc[sub.index, 'tpl_ci_upper'].values - sub.values

bars = ax.bar(range(len(sub)), sub.values, yerr=np.array([err_lower, err_upper]),
              color=colors, edgecolor='black', linewidth=1.2, capsize=5, alpha=0.8)
ax.set_xticks(range(len(sub)))
ax.set_xticklabels(sub.index, rotation=45, ha='right', fontsize=9)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.axhline(y=0.1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Good separation threshold')
ax.axhline(y=-0.1, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Severe leakage threshold')
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


# ## 7. Dimension-reduction 

# In[30]:


# Cell 7.1 - embeddings
import pacmap
import trimap
from sklearn.decomposition import KernelPCA

bv32 = bv.astype(np.float32)

embeds = {}
try:
    embeds['tSNE'] = TSNE(n_components=2, perplexity=50, init='pca', learning_rate="auto",
                          method="barnes_hut", random_state=RANDOM_SEED).fit_transform(bv32)

    print('t-SNE done')
except Exception as exc:
    print(f't-SNE failed: {exc}')

# Export embeddings to disk for later reuse
for emb_name, emb in embeds.items():
    # Save as .npy
    np.save(OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.npy', emb)
    # Save as CSV with row index for easy reloading
    pd.DataFrame(emb, columns=[f'{emb_name}_1', f'{emb_name}_2']).to_csv(
        OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.csv', index=True, index_label='row_id'
    )
    print(f'Saved {emb_name} embedding: {emb.shape}')


# In[31]:


# Cell 7.2 - split-colored scatter panels
showcase = ['RANDOM', 'STRATIFIED', 'SCAFFOLD', 'KMEANS', 'KENNARDSTONE', 'AGGLOMERATIVE', 'SPLIT_BASE', 'SPLIT_DATASAIL',
            'DC_BUTINA', 'DC_FINGERPRINT', 'DC_MAXMIN', 'DC_SCAFFOLD', 'DC_WEIGHT',
            'HMD-SHIELD', 'H-SHIELD', 'M-SHIELD', 'D-SHIELD']
showcase = [s for s in showcase if s in baseline_results or s in shield_results]
for emb_name, emb in embeds.items():
    cols = min(4, len(showcase)); rows = int(np.ceil(len(showcase) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, showcase):
        fold = baseline_results.get(name, shield_results.get(name))
        ax.scatter(emb[fold == 0, 0], emb[fold == 0, 1], s=6, c='#4477AA',
                   alpha=0.55, label='train')
        ax.scatter(emb[fold == 1, 0], emb[fold == 1, 1], s=6, c='#CC6677',
                   alpha=0.55, label='test')
        scaled = lpi_df.set_index('config').loc[name, 'scaled_L_pi'] \
                 if name in lpi_df['config'].values else float('nan')
        ax.set_title(f'{name}\nscaled L(pi)={scaled:.3f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(showcase):]: ax.set_visible(False)
    axes[0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{emb_name} embedding colored by fold', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_splits',
             where='main' if emb_name in ('PCA','UMAP') else 'si')
    plt.show()



# tsne panel colored by target value
import matplotlib.cm as cm
import matplotlib.colors as mcolors

for emb_name, emb in embeds.items():
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=Y_TARGETS, cmap='RdYlBu_r',
        s=6, alpha=0.65,
        vmin=np.percentile(Y_TARGETS, 2),
        vmax=np.percentile(Y_TARGETS, 98),
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('TARGET', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{emb_name} — TARGET', fontsize=10)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_target',
             where='main' if emb_name in ('PCA', 'UMAP') else 'si')
    plt.show()


# In[32]:


# 8.3 diagnostic only
showcase = ['RANDOM', 'STRATIFIED', 'SCAFFOLD', 'KMEANS', 'KENNARDSTONE', 'AGGLOMERATIVE', 'SPLIT_BASE', 'SPLIT_DATASAIL',
            'DC_BUTINA', 'DC_FINGERPRINT', 'DC_MAXMIN', 'DC_SCAFFOLD', 'DC_WEIGHT',
            'HMD-SHIELD', 'H-SHIELD', 'M-SHIELD', 'D-SHIELD']

for name in showcase:
    fold = baseline_results.get(name, shield_results.get(name))

    n_train = np.sum(fold == 0)
    n_test = np.sum(fold == 1)
    n_total = len(fold)

    p_train = 100 * n_train / n_total
    p_test = 100 * n_test / n_total

    print(
        f"{name}: {n_train} train / {n_test} test "
        f"({p_train:.1f}%, {p_test:.1f}%)"
    )


# ## 9. SHIELD (alpha, beta, gamma) sweep, Pareto frontier, Hamming stability

# In[33]:


# Cell 9.1 - sweep using shield.sweep
sweep_kwargs = dict(
    n=n, K=N_FOLDS,
    r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hierarchy_main,
    affinity=AM,
    affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE,
    cost=C,
    target_marginals=TARGET_MARG,
    epsilon_OT=OT_REG,
    y_target=y_strat,
    stratification_strategy=STRAT_STRATEGY,
    n_bins=STRAT_N_BINS,
    class_delta=STRAT_TOL,
    balance_eps=BALANCE_TOL,
    eta=ETA, mu=MU, nu=NU, tau=TAU,
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
    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
)
n_cfgs = (len(SWEEP_GRID['alpha']) * len(SWEEP_GRID['beta'])
          * len(SWEEP_GRID['gamma']))
print(f'Running shield.sweep over alpha x beta x gamma ({n_cfgs} configs)...')
t0 = time.time()
sweep_res = shield_sweep(
    alpha_grid=SWEEP_GRID['alpha'],
    beta_grid=SWEEP_GRID['beta'],
    gamma_grid=SWEEP_GRID['gamma'],
    metric_keys=('M_normalized', 'D_normalized', 'H_normalized'),
    aggregation='min_of_normalized',
    instability_threshold=HAMMING_STABILITY_THRESHOLD,
    **sweep_kwargs,
)
print(f'Sweep complete in {time.time()-t0:.1f}s; {len(sweep_res.results)} runs.')
print(f'Recommendation: alpha,beta,gamma = {sweep_res.grid_points[sweep_res.recommended_index]}')
print(f'Recommendation info: {sweep_res.recommendation_info}')


# In[34]:


# Cell 9.2 - sweep results table
sweep_rows = []
for gp, r in zip(sweep_res.grid_points, sweep_res.results):
    last = r.history[-1]['terms'] if r.history else {}
    _, scaled_lpi = cross_fold_similarity(r.fold_assignment.cpu().numpy().astype(int))
    sweep_rows.append({
        'alpha': gp[0], 'beta': gp[1], 'gamma': gp[2],
        'H_normalized': last.get('H_normalized'),
        'M_normalized': last.get('M_normalized'),
        'D_normalized': last.get('D_normalized'),
        'strat_normalized': last.get('strat_normalized'),
        'composite_objective': (
            gp[0] * (last.get('H_normalized') or 0.0)
            + gp[1] * (last.get('M_normalized') or 0.0)
            + gp[2] * (last.get('D_normalized') or 0.0)
            + MU    * (last.get('strat_normalized') or 0.0)
        ) if last else None,
        'scaled_L_pi': scaled_lpi,
    })
sweep_df = pd.DataFrame(sweep_rows)
sweep_df['is_pareto'] = sweep_df.index.isin(sweep_res.pareto_indices)
sweep_df['is_recommended'] = (sweep_df.index == sweep_res.recommended_index)
save_table(sweep_df, 'shield_sweep_grid', where='main')
sweep_df


# In[35]:


# Cell 9.3 - Pareto frontier plot (3D projected to 2 with color)
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Panel 1: M vs D colored by H
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
ax1.set_title('Sweep: M vs D, color = H, gold star = recommendation')
ax1.legend()

# Panel 2: composite objective vs scalar score
ax2.scatter(range(len(sweep_df)), sweep_df['composite_objective'],
            c=['gold' if x == sweep_res.recommended_index else ('red' if i else '#4477AA')
               for i, x in enumerate(sweep_df.index)], s=60, edgecolors='black')
ax2.set_xlabel('config index'); ax2.set_ylabel('composite objective')
ax2.set_title('Composite objective across sweep')
plt.tight_layout()
save_fig(fig, 'sweep_pareto', where='main')
plt.show()


# In[36]:


# Cell 9.4 - Hamming-stability matrix across sweep
stab = sweep_res.stability
H = stab.pairwise_hamming
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(H, ax=ax, cmap='magma', annot=False, cbar_kws={'label': 'fraction differing atoms'})
ax.set_title(f'Hamming distance matrix across {len(stab.assignments)} sweep points\n'
             f'(threshold={HAMMING_STABILITY_THRESHOLD})')
plt.tight_layout()
save_fig(fig, 'hamming_stability_matrix', where='main')
print(f'Flagged unstable pairs: {len(stab.flagged_unstable)} of '
      f'{len(stab.assignments)*(len(stab.assignments)-1)//2}')
plt.show()


# ## 9. OOD Assessment Metrics

# In[37]:


# # Cell 8.1 - OOD characterization metrics per split

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
    Xtr_161 = bv32[mask_tr_161]
    Xte_161 = bv32[mask_te_161]

    if len(Xtr_161) < 10 or len(Xte_161) < 5:
        continue

    row_161 = {'split': sp_name_161}

    # (1-3) NN distances in raw feature space
    try:
        nn_d_161 = _nn_distances(Xtr_161, Xte_161)
        row_161['nn_mean'] = float(nn_d_161.mean())
        row_161['nn_p95']  = float(np.percentile(nn_d_161, 95))
        row_161['nn_max']  = float(nn_d_161.max())
    except Exception as e:
        print(f'  [NN] {sp_name_161}: {e}')
        row_161['nn_mean'] = row_161['nn_p95'] = row_161['nn_max'] = float('nan')

    # (4) SOAP Distance — mean NN in t-SNE 2-D space
    if _tsne_2d is not None:
        try:
            soap_d_161 = _nn_distances(_tsne_2d[mask_tr_161], _tsne_2d[mask_te_161])
            row_161['soap_dist_mean'] = float(soap_d_161.mean())
        except Exception as e:
            print(f'  [SOAP] {sp_name_161}: {e}')
            row_161['soap_dist_mean'] = float('nan')
    else:
        row_161['soap_dist_mean'] = float('nan')

    # (5) Convex Hull Outside Rate — t-SNE 2-D space
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


# MASTER FIGURE 1 — Bubble map

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


# MASTER FIGURE 2 — Parallel coordinates

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

# ── Legend ──
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


# In[43]:


# Cell 9.2 - Export all splits to a single combined CSV

_splits_df = df.copy()

# Add all fold assignment columns
for split_name, fold_arr in all_splits.items():
    col = 'fold__' + split_name.replace('::', '__').replace(' ', '_')
    _splits_df[col] = fold_arr

# Confirm fold columns created
_fold_cols = [c for c in _splits_df.columns if c.startswith('fold__')]
print(f'Splits embedded: {len(_fold_cols)}')
for c in _fold_cols:
    counts = _splits_df[c].value_counts().sort_index().to_dict()
    print(f'  {c:55s}  {counts}')

# Save
_out_path = OUT_INTER / 'splits' / 'all_splits_combined.csv'
_splits_df.to_csv(_out_path, index=True, index_label='row_id')
print(f'\nSaved: {_out_path}  (shape: {_splits_df.shape})')


# ## 13. SI — stratification strategy comparison on logD (single coordinate)
# 

# In[47]:


# Cell 13.1 - SI: stratification strategies on primary target logD

y_single = Y_TARGETS[:, 0:1]
strat_results = {}
for strat in ['quantile', 'moment', 'sliced_wasserstein']:
    print(f'  strategy={strat}')
    t0 = time.time()
    try:
        res = shield_split(
            n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
            hierarchy=hierarchy_main, affinity=AM,
            affinity_provenance='label-blind-handcrafted',
            d_mode=DIST_MODE, cost=C, target_marginals=TARGET_MARG, epsilon_OT=OT_REG,
            y_target=y_single, stratification_strategy=strat,
            n_bins=STRAT_N_BINS,
            P_proj=SW_PROJECTIONS, Q_grid=SW_QUANTILE_PTS,
            class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
            alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=0.5, nu=NU, tau=TAU,
            max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
            integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
            rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
            stage0_samples=SHIELD_STAGE0_SAMPLES,
            precomputed_normalizers=CACHED_NORMALIZERS,
            sinkhorn_max_iter=SHIELD_SINKHORN_ITER,

        )
        strat_results[strat] = res.fold_assignment.cpu().numpy().astype(int)
        sizes = (np.bincount(strat_results[strat], minlength=N_FOLDS) / n).round(3).tolist()
        print(f'    wall={time.time()-t0:.1f}s,  sizes={sizes}')
    except Exception as exc:
        print(f'    FAILED: {exc}')


# In[48]:


# Cell 13.2 - strategy comparison: per-target Wasserstein on the primary target + Q-Q plot
fig, axes = plt.subplots(1, len(strat_results), figsize=(4 * len(strat_results), 4), sharey=True)
if len(strat_results) == 1: axes = [axes]
for ax, (strat, fold) in zip(axes, strat_results.items()):
    tr = y_single[fold == 0].ravel(); te = y_single[fold == 1].ravel()
    qq_tr = np.quantile(tr, np.linspace(0, 1, 100))
    qq_te = np.quantile(te, np.linspace(0, 1, 100))
    ax.plot(qq_tr, qq_te, marker='.')
    lo = min(qq_tr.min(), qq_te.min()); hi = max(qq_tr.max(), qq_te.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    W = wasserstein_distance(tr, te)
    ax.set_title(f'{strat}\nW_1 = {W:.4g}')
    ax.set_xlabel('train quantile'); ax.set_ylabel('test quantile')
plt.tight_layout()
save_fig(fig, 'strat_strategy_qq_plots', where='si')
plt.show()


# ## 14. SI — D-match (OT, main) vs D-shift (MMD, SI comparison)
# 

# In[62]:


# Cell 14.1 - SI: D-shift (MMD) as comparison to main D-match (OT)

K_MATRIX_SI = A_full.astype(np.float32).copy()
np.fill_diagonal(K_MATRIX_SI, 1.0)

try:
    res_dshift = shield_split(
        n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main, affinity=AM,
        affinity_provenance='label-blind-handcrafted',
        d_mode='shift', kernel_matrix=K_MATRIX_SI,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
        y_target=y_strat, stratification_strategy='none', n_bins=STRAT_N_BINS,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        stage0_samples=SHIELD_STAGE0_SAMPLES,
        precomputed_normalizers=CACHED_NORMALIZERS,
        sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
    )
    fold_dshift = res_dshift.fold_assignment.cpu().numpy().astype(int)
    print(f'D-shift (SI): sizes={(np.bincount(fold_dshift, minlength=N_FOLDS)/n).round(3).tolist()}')
except Exception as exc:
    fold_dshift = None
    print(f'D-shift FAILED: {exc}')

if fold_dshift is not None:
    from scipy.stats import wasserstein_distance
    rows = [
        {'mode': 'D-match (main)', **dict(zip(['L_H','L_M','L_W'], per_channel_leakage(shield_results['HMD-SHIELD']))),
         'scaled_L_pi': cross_fold_similarity(shield_results['HMD-SHIELD'])[1],
         'logD_W1': wasserstein_distance(Y_TARGETS[shield_results['HMD-SHIELD']==0, 0],
                                         Y_TARGETS[shield_results['HMD-SHIELD']==1, 0])},
        {'mode': 'D-shift (SI)',  **dict(zip(['L_H','L_M','L_W'], per_channel_leakage(fold_dshift))),
         'scaled_L_pi': cross_fold_similarity(fold_dshift)[1],
         'logD_W1': wasserstein_distance(Y_TARGETS[fold_dshift==0, 0],
                                         Y_TARGETS[fold_dshift==1, 0])},
    ]
    dmatch_cmp_df = pd.DataFrame(rows)
    save_table(dmatch_cmp_df, 'd_match_vs_d_shift', where='si')
    display(dmatch_cmp_df)


# In[63]:


# Cell 14.2 - SI: t-SNE visualization and export of D-match vs D-shift splits

# ── t-SNE embedding ──────────────────────────────────────────────────────────
if 'tSNE' in embeds:
    _emb = embeds['tSNE']
    _emb_label = 't-SNE'
else:
    print('t-SNE not in embeds, recomputing...')
    from sklearn.manifold import TSNE as _TSNE
    _emb = _TSNE(n_components=2, perplexity=25, init='random',
                 random_state=RANDOM_SEED).fit_transform(bv.astype(np.float32))
    _emb_label = 't-SNE (recomputed)'

_fold_dmatch = shield_results['HMD-SHIELD']
_COLORS = {0: '#4477AA', 1: '#CC6677'}
_LABELS = {0: 'train (fold 0)', 1: 'test  (fold 1)'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (split_label, fold_arr) in zip(axes, [
    ('D-match  [OT, main]', _fold_dmatch),
    ('D-shift  [MMD, SI]',  fold_dshift),
]):
    for fv in [0, 1]:
        mask = fold_arr == fv
        ax.scatter(_emb[mask, 0], _emb[mask, 1],
                   s=7, alpha=0.55, c=_COLORS[fv],
                   label=f'{_LABELS[fv]}  (n={mask.sum():,})',
                   linewidths=0)
    _, lpi = cross_fold_similarity(fold_arr)
    w1 = wasserstein_distance(Y_TARGETS[fold_arr == 0, 0],
                              Y_TARGETS[fold_arr == 1, 0])
    ax.set_title(f'{split_label}\nscaled L(π)={lpi:.4f}   logD W₁={w1:.4f}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2.5, loc='best')

fig.suptitle(f'{_emb_label} — D-match (OT) vs D-shift (MMD) fold assignments',
             fontsize=11, y=1.01)
plt.tight_layout()
save_fig(fig, 'si_dmatch_vs_dshift_tsne', where='si')
plt.show()

# ── Export table ─────────────────────────────────────────────────────────────
_export_dshift = pd.DataFrame({
    SMILES_COL:       df[SMILES_COL].values,
    PRIMARY_TARGET:   df[PRIMARY_TARGET].values,
    'fold_dmatch_OT': _fold_dmatch.astype(int),
    'fold_dshift_MMD': fold_dshift.astype(int),
})
_export_path = OUT_SI / 'data' / 'dmatch_vs_dshift_folds.csv'
_export_dshift.to_csv(_export_path, index=False)
print(f'Exported {len(_export_dshift):,} rows -> {_export_path}')
print(_export_dshift.head())
print(f'\nFold counts:')
print(f"  D-match : {pd.Series(_fold_dmatch).value_counts().sort_index().to_dict()}")
print(f"  D-shift : {pd.Series(fold_dshift).value_counts().sort_index().to_dict()}")


# In[66]:


# Cell 13.2 - strategy comparison: per-target Wasserstein on the primary target + Q-Q plot

_qq_splits = dict(strat_results)
if fold_dshift is not None:
    _qq_splits['D-shift (MMD)'] = fold_dshift

fig, axes = plt.subplots(1, len(_qq_splits), figsize=(4 * len(_qq_splits), 4), sharey=True)
if len(_qq_splits) == 1: axes = [axes]

_qq_rows = []
for ax, (label, fold) in zip(axes, _qq_splits.items()):
    tr = y_single[fold == 0].ravel(); te = y_single[fold == 1].ravel()
    qq_tr = np.quantile(tr, np.linspace(0, 1, 100))
    qq_te = np.quantile(te, np.linspace(0, 1, 100))
    is_dshift = label.startswith('D-shift')
    ax.plot(qq_tr, qq_te, marker='.', color='#CC6677' if is_dshift else '#4477AA')
    lo = min(qq_tr.min(), qq_te.min()); hi = max(qq_tr.max(), qq_te.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    W = wasserstein_distance(tr, te)
    ax.set_title(f'{label}\nW₁ = {W:.4g}', fontsize=10,
                 fontweight='bold' if is_dshift else 'normal')
    ax.set_xlabel('train quantile')
    ax.set_ylabel('test quantile')

    for q, qtr, qte in zip(np.linspace(0, 1, 100), qq_tr, qq_te):
        _qq_rows.append({
            'strategy':       label,
            'quantile':       float(q),
            'train_quantile': float(qtr),
            'test_quantile':  float(qte),
            'wasserstein_w1': float(W),
        })

plt.suptitle('logD Q-Q plot: train vs test quantiles (stratification strategies + D-shift)',
             fontsize=10, y=1.02)
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
_w1_summary.to_csv(OUT_SI / 'tables' / 'strat_w1_summary.csv', index=False)

print(f'Q-Q data    →  {_qq_path}  ({len(_qq_df)} rows, {_qq_df["strategy"].nunique()} strategies × 100 pts)')
print(f'W_1 summary →  {OUT_SI / "tables" / "strat_w1_summary.csv"}')
display(_w1_summary)
