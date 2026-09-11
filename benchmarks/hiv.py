# ## 1. Setup — imports, configuration, output directories

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


# In[ ]:


# Cell 1.2 - global configuration
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'HIV.csv')
DATASET_NAME = 'hiv'
SMILES_COL   = 'smiles'
TARGET_COL   = 'HIV_active'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 42

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.05
STRAT_TOL   = 0.05

# Fingerprints
ECFP_RADIUS  = 2
ECFP_NBITS   = 1024
PRIMARY_AFFINITY = 'tanimoto'

# M term
EPS_AFF_M     = 0.35
T2_WEIGHT     = 0.20
T2_MAX_COMP   = 250
EPS_T2_FINE   = 0.30

# H term
LAMBDA_MAIN = [1.0, 0.50]


# D term
DIST_MODE       = 'shift'
STRAT_STRATEGY  = 'none'       

# Atom count bins (for Tier-2 and baseline splitters)
ATOM_BIN_EDGES  = [-0.5, 20.5, 35.5, float('inf')]
ATOM_BIN_LABELS = ['ha<=20', 'ha21-35', 'ha>35']

# SHIELD coefficients
ALPHA, BETA, GAMMA, ETA, MU = 1.0, 0.2, 3.0, 1.0, 0.0
NU, TAU = 0.0, 0.0

# Sweep grid
SWEEP_GRID = {
    'alpha': [0.2, 0.5, 1.0, 2.0, 3.0],
    'beta':  [0.2, 0.5, 1.0, 2.0, 3.0],
    'gamma': [0.2, 0.5, 1.0, 2.0, 3.0],
}
HAMMING_STABILITY_THRESHOLD = 0.10

# kNN purity / embedding
KNN_K           = [5, 10, 25]

# SHIELD solver controls
SHIELD_MAX_ITER     = 15
SHIELD_STAGE0_SAMPLES  = 100
SHIELD_SINKHORN_ITER = 500
SHIELD_INIT_BACKEND = 'auto'
SHIELD_ROUNDING     = 'pipage'
PRIMARY_ROUNDING    = SHIELD_ROUNDING
SHIELD_MILP_SIZE    = 45_000
SHIELD_DEVICE       = 'cpu'
N_JOBS              = -1

# External coarsening
COARSEN_N_SUPER_PRIMARY = 1024
COARSEN_BACKEND         = 'h_first'
COARSEN_CLASS_PURE      = True
EVAL_REF_N_SUPER        = 4096
EPS_AFF_SUPER           = None
T2_K_SUPER              = 3
D_KERNEL_POWER          = 1.0
SHIELD_MILP_TIME        = 120.0

# ML evaluation
ML_RUN = True

print(f'Config: n_folds={N_FOLDS}, dist_mode={DIST_MODE}, '
      f'eps={EPS_AFF_M}, lambda_main={LAMBDA_MAIN}, '
      f'alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, mu={MU}')


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
    raise RuntimeError('RDKit is required for HIV featurization. Install via conda-forge.')


# ## 2. Data loading and EDA 
# 

# In[5]:


# Cell 2.1 - load HIV dataset
print(f'Loading from {DATASET_PATH}')
df_raw = pd.read_csv(DATASET_PATH)
print(f'Raw shape: {df_raw.shape}')
print(f'Columns: {list(df_raw.columns)}')

for drop_col in ['mol_id', 'activity']:
    if drop_col in df_raw.columns:
        df_raw = df_raw.drop(columns=[drop_col])
        print(f'Dropped column "{drop_col}".')

# Class balance audit
if TARGET_COL in df_raw.columns:
    y_raw = df_raw[TARGET_COL].dropna()
    n_pos = int((y_raw == 1).sum())
    n_neg = int((y_raw == 0).sum())
    pos_rate = float(n_pos / len(y_raw))
    print(f'\nClass balance: {n_pos} active ({100*pos_rate:.2f}%) / {n_neg} inactive')
    audit_df = pd.DataFrame([{'target': TARGET_COL, 'n_labeled': len(y_raw),
                               'n_active': n_pos, 'n_inactive': n_neg,
                               'positive_rate': pos_rate}])
    save_table(audit_df, 'class_balance_audit', where='si')
    display(audit_df)

df = df_raw.reset_index(drop=True)

if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df):,} rows (seed={RANDOM_SEED})')

n = len(df)
print(f'\nFinal n={n:,}')
df.head()


# In[6]:


# Cell 2.2 - attach DataSAIL splits and filter dataset to matched molecules

from rdkit import Chem as _Chem
from rdkit.Chem.inchi import MolToInchi as _MolToInchi
from rdkit import RDLogger as _RDLogger
import re as _re
import warnings as _warnings
_warnings.filterwarnings('ignore', category=UserWarning)
_RDLogger.DisableLog('rdApp.*')

DATASAIL_PATH = str(_REPO_ROOT / 'splits' / 'hiv_DataSAIL.csv')

PRECOMPUTED_FOLD_MAP = {
    'train': 0,
    'val':   1,
    'test':  1,
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
    if len(parts) < 2:
        return (None, None)
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

df          = df[_matched].copy().reset_index(drop=True)
_kept = [l for l in _labels if l is not None]
df['_split_base']     = [l[0] for l in _kept]
df['_split_datasail'] = [l[1] for l in _kept]

n = len(df)
print(f'Working dataset after filter: {n:,} molecules')
print(f"  _split_base     : {pd.Series(df['_split_base']).value_counts().sort_index().to_dict()}")
print(f"  _split_datasail : {pd.Series(df['_split_datasail']).value_counts().sort_index().to_dict()}")


# In[7]:


# Cell 2.3 - prepare target arrays
Y_PRIMARY = df[TARGET_COL].values.astype(np.float32)
prim_labeled = ~np.isnan(Y_PRIMARY)
print(f'Total rows: {n:,}; labeled: {prim_labeled.sum():,}')

Y_FULL = Y_PRIMARY.reshape(-1, 1)

y_int = Y_PRIMARY[prim_labeled].astype(np.int64)
print(f'Class balance: {dict(zip(*np.unique(y_int, return_counts=True)))}')
print(f'Positive rate: {y_int.mean()*100:.2f}%')


# In[8]:


# Cell 2.4 - HIV class imbalance bar chart and MW distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel A: class balance
ax = axes[0]
counts = {'Inactive (0)': int((Y_PRIMARY == 0).sum()),
          'Active (1)':   int((Y_PRIMARY == 1).sum())}
bars = ax.bar(list(counts.keys()), list(counts.values()),
              color=['#4477AA', '#CC3311'], edgecolor='black')
for bar, v in zip(bars, counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{v:,}\n({100*v/n:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title('HIV dataset class balance')
ax.set_ylabel('molecule count')

# Panel B: molecular weight distribution by class
try:
    from rdkit import Chem
    from rdkit.Chem.Descriptors import MolWt
    mw_active   = [MolWt(Chem.MolFromSmiles(s)) for s, y in zip(df[SMILES_COL], Y_PRIMARY)
                   if y == 1 and Chem.MolFromSmiles(s) is not None]
    mw_inactive = [MolWt(Chem.MolFromSmiles(s)) for s, y in zip(df[SMILES_COL], Y_PRIMARY)
                   if y == 0 and Chem.MolFromSmiles(s) is not None]
    ax2 = axes[1]
    ax2.hist(mw_inactive, bins=50, color='#4477AA', alpha=0.6, label=f'Inactive (n={len(mw_inactive):,})', density=True)
    ax2.hist(mw_active,   bins=50, color='#CC3311', alpha=0.7, label=f'Active   (n={len(mw_active):,})',   density=True)
    ax2.set_xlabel('Molecular weight (Da)'); ax2.set_ylabel('density')
    ax2.set_title('MW distribution by HIV activity')
    ax2.legend()
except Exception as exc:
    axes[1].text(0.5, 0.5, f'MW panel skipped:\n{exc}', ha='center', va='center',
                 transform=axes[1].transAxes)
plt.tight_layout()
save_fig(fig, 'eda_class_balance_and_mw', where='si')
plt.show()


# In[9]:


# Cell 2.5 - Intra-class vs inter-class Tanimoto similarity
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

_mols_eda = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]
_fps_eda  = [AllChem.GetMorganFingerprintAsBitVect(m, ECFP_RADIUS, nBits=ECFP_NBITS)
             if m is not None else None for m in _mols_eda]
_valid    = [i for i, fp in enumerate(_fps_eda) if fp is not None]

rng = np.random.default_rng(RANDOM_SEED)
_sample_n = min(300, n)
_si = rng.choice(_valid, size=_sample_n, replace=False)

act_idx  = [i for i in _si if Y_PRIMARY[i] == 1][:50]
inact_idx = [i for i in _si if Y_PRIMARY[i] == 0][:200]

def _sim_pairs(idx_a, idx_b, fps_list, max_pairs=5000):
    sims = []
    for i in idx_a:
        if _fps_eda[i] is None: continue
        for j in idx_b:
            if i >= j or _fps_eda[j] is None: continue
            sims.append(DataStructs.TanimotoSimilarity(_fps_eda[i], _fps_eda[j]))
            if len(sims) >= max_pairs: return sims
    return sims

sim_aa = _sim_pairs(act_idx, act_idx, _fps_eda)
sim_ii = _sim_pairs(inact_idx, inact_idx, _fps_eda)
sim_ai = _sim_pairs(act_idx, inact_idx, _fps_eda)

fig, ax = plt.subplots(figsize=(8, 4))
for sims, label, color in [
    (sim_aa, f'Active-Active (n_pairs={len(sim_aa)})',     '#CC3311'),
    (sim_ai, f'Active-Inactive (n_pairs={len(sim_ai)})',   '#EE7733'),
    (sim_ii, f'Inactive-Inactive (n_pairs={len(sim_ii)})', '#4477AA'),
]:
    if sims:
        ax.hist(sims, bins=40, alpha=0.65, density=True, label=label, color=color)
ax.set_xlabel('ECFP4 Tanimoto similarity'); ax.set_ylabel('density')
ax.set_title('Intra/inter-class Tanimoto similarity\n(confirms structural clustering of actives)')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, 'eda_intraclass_tanimoto', where='si')
plt.show()
print('Intra-active mean Tanimoto:', f'{np.mean(sim_aa):.3f}' if sim_aa else 'N/A')
print('Inactive-inactive mean Tanimoto:', f'{np.mean(sim_ii):.3f}' if sim_ii else 'N/A')


# ## 3. Featurization 

# In[10]:


# Cell 3.1 - compute ECFP4 fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

CACHE_FP = OUT_INTER / 'featurization' / f'fps_n{n}_r{ECFP_RADIUS}_b{ECFP_NBITS}_{TARGET_COL}.pkl'

if CACHE_FP.exists():
    with open(CACHE_FP, 'rb') as fh:
        cache_obj = pickle.load(fh)
    bv  = cache_obj['bv']
    fps = cache_obj['fps']
    if len(fps) != len(df):
        valid = np.array([Chem.MolFromSmiles(s) is not None for s in df[SMILES_COL]])
        df = df[valid].reset_index(drop=True)
        Y_PRIMARY = Y_PRIMARY[valid]; Y_FULL = Y_PRIMARY.reshape(-1, 1)
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
        Y_PRIMARY = Y_PRIMARY[valid]; Y_FULL = Y_PRIMARY.reshape(-1, 1)
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


# Cell 3.3 - chemotype axes for the H term (Bemis-Murcko scaffold taxonomy)

def _framework_class(sc_smiles):
    if not sc_smiles or sc_smiles == 'acyclic':
        return 'acyclic'
    msc = Chem.MolFromSmiles(sc_smiles)
    if msc is None or msc.GetRingInfo().NumRings() == 0:
        return 'acyclic'
    ri = msc.GetRingInfo(); rings_at = [set(r) for r in ri.AtomRings()]
    nring = min(ri.NumRings(), 5)
    narom = sum(1 for ring in ri.AtomRings()
                if all(msc.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    fused = any(len(rings_at[a] & rings_at[b]) >= 2
                for a in range(len(rings_at)) for b in range(a + 1, len(rings_at)))
    has_n = any(at.GetSymbol() == 'N' for at in msc.GetAtoms())
    has_s = any(at.GetSymbol() == 'S' for at in msc.GetAtoms())
    het = ('N' if has_n else '') + ('S' if has_s else '') or 'C'
    return f'{nring}R_{narom}ar_{"fused" if fused else "iso"}_{het}'

_scaf = df['_scaffold'].tolist()
df['_framework'] = [_framework_class(sc) for sc in _scaf]

MIN_SCAFFOLD_SIZE = 10
_sc_counts = df['_scaffold'].value_counts().to_dict()
def _chemotype_L2(sc, fw):
    if not sc or sc == 'acyclic':
        return f'acyclic::{fw}'
    return sc if _sc_counts.get(sc, 0) >= MIN_SCAFFOLD_SIZE else f'rare::{fw}'
df['_chemotype_L2'] = [_chemotype_L2(sc, fw) for sc, fw in zip(_scaf, df['_framework'])]
df['_chemotype_L1'] = df['_framework']
df['_h_L2'] = df['_chemotype_L2']
df['_h_L1'] = df['_chemotype_L1']

_n_rare = int(df['_h_L2'].astype(str).str.startswith('rare::').sum())
print(f'Chemotype L2 (scaffold) families: {df["_h_L2"].nunique()} '
      f'(rare scaffolds <{MIN_SCAFFOLD_SIZE} mols pooled per framework; {_n_rare} mols pooled)')
print(f'Framework L1 classes: {df["_h_L1"].nunique()}')
print('Top L1 frameworks (count):', df['_h_L1'].value_counts().head(8).to_dict())
_ar = df.groupby('_h_L1')[TARGET_COL].mean().sort_values(ascending=False)
print(f'Most-active framework: {_ar.index[0]} ({100*_ar.iloc[0]:.1f}% active vs '
      f'{100*Y_PRIMARY.mean():.1f}% base) -> chemotype carries activity signal.')


# In[13]:


# Cell 3.4 - M term: super-node Tanimoto affinity builders (HIV)
def _tani_sim(centroids):
    '''Generalized Tanimoto T(a,b)=<a,b>/(|a|^2+|b|^2-<a,b>) on (mean-ECFP) centroids;
    reduces to standard Tanimoto for binary vectors. PSD; values in [0,1].'''
    C = np.asarray(centroids, dtype=np.float32); sq = (C * C).sum(1); ns = C.shape[0]
    K = np.zeros((ns, ns), dtype=np.float32)
    for s in range(0, ns, 1024):
        e = min(s + 1024, ns); inter = C[s:e] @ C.T
        den = sq[s:e, None] + sq[None, :] - inter
        K[s:e] = np.where(den > 1e-9, inter / np.maximum(den, 1e-9), 0.0).astype(np.float32)
    return np.clip(K, 0.0, 1.0)

def build_super_affinity(centroids, eps=None, t2_k=None):
    '''M graph: Tanimoto eps-NN on super-node centroids + Tier-2 k-NN rescue so no
    super-node is left isolated (keeps the graph-cut well-posed).'''
    eps = (EPS_AFF_SUPER if EPS_AFF_SUPER is not None else EPS_AFF_M) if eps is None else eps
    t2_k = T2_K_SUPER if t2_k is None else t2_k
    K = _tani_sim(centroids); np.fill_diagonal(K, 0.0)
    A = np.where(K > eps, K, 0.0).astype(np.float32); np.fill_diagonal(A, 0.0)
    for i in np.where((A > 0).sum(1) == 0)[0]:
        order = np.argsort(-K[i]); cnt = 0
        for j in order:
            if int(j) == int(i): continue
            A[i, j] = max(A[i, j], T2_WEIGHT); A[j, i] = max(A[j, i], T2_WEIGHT); cnt += 1
            if cnt >= int(t2_k): break
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0.0); return A
print('M builders defined (_tani_sim, build_super_affinity); applied to centroids in cell 4.5.3.')


# In[14]:


# Cell 3.5 - D-match cost builder (super-node; 1 - Tanimoto).
def build_super_cost(centroids):
    K = _tani_sim(centroids); cost = np.clip(1.0 - K, 0.0, 1.0).astype(np.float32)
    cost = (cost + cost.T) / 2.0; np.fill_diagonal(cost, 0.0); return cost
print('D-match cost builder defined (1 - Tanimoto on centroids).')


# In[15]:


# Cell 3.6 - D-shift kernel builder (super-node Tanimoto PSD kernel)
def build_super_kernel(centroids):
    K = np.clip(_tani_sim(centroids), 0.0, 1.0).astype(np.float32); np.fill_diagonal(K, 1.0); return K
print('D-shift kernel builder defined (Tanimoto PSD on centroids).')


# ## 4. Hierarchy construction
# 

# In[16]:


# Cell 4.1 - Candidate H axes + activity validation 
from rdkit.Chem import MolFromSmarts

#  L1/L2 membership map
L1_MAP = {
    'nucleoside_analog':     'NRTI_like',

    'diaryl_pyrimidine':     'NNRTI_like',
    'halogenated_alkyl':     'NNRTI_like',

    'peptidomimetic':        'PI_like',

    'diketo_acid':           'Integrase_like',
    'polycyclic_heterocyclic': 'Integrase_like',

    'catechol':              'Other_antiviral',
    'sulfonamide':           'Other_antiviral',
    'guanidine_urea':        'Other_antiviral',
    'macrolide_like':        'Other_antiviral',
    'general_aromatic':      'Other_antiviral',
}

L1_GROUPS = ['NRTI_like', 'NNRTI_like', 'PI_like', 'Integrase_like', 'Other_antiviral']

# L2 SMARTS definitions (priority-ordered)
PHARM_SMARTS = {
    'nucleoside_analog': [
        '[C@@H]1OCC[C@H]1n',
        '[C@H]1OC[C@@H]1n',
        '[C@H]1O[C@@H](CO)[C@@H]1n',
        'O1[CH]CC([CH2]O1)n',
    ],

    'diaryl_pyrimidine': [
        'c1cc(-c2ncnc(n2))cc1',
        'c1cnc(nc1)-c2ccccc2',
        'n1ccnc(c1)-c2cccc2',
        'c1nc(nc(n1)N)-c2cccc2',
    ],

    'peptidomimetic': [
        '[NX3H][CX3](=O)[CX4]',
    ],

    'diketo_acid': [
        '[CX3](=O)[CX4H][CX3](=O)[OX2H1]',
        '[CX3](=O)[CX3](=O)',
        '[CX3](=O)[CX4][CX3](=O)',
    ],

    'catechol': [
        'c1cc(O)c(O)cc1',
    ],

    'sulfonamide': [
        '[SX4](=[OX1])(=[OX1])[NX3]',
    ],

    'guanidine_urea': [
        '[NX3][CX3](=[NX2])[NX3]',
        '[NX3H][CX3](=[OX1])[NX3]',
    ],

    'macrolide_like': [
        'O=C1CCCCCCCO1',
        'O=C1CCCCCCCCO1',
    ],

    'polycyclic_heterocyclic': [
        '[r5,r6]1~[r5,r6]~[r5,r6]1',
        'n1cc2ccccc2c1',
        'n1cnc2ccccc12',
        'c1cnc2ncccc2n1',
    ],

    'halogenated_alkyl': [
        '[CX4]([F])([F])[F]',
        '[CX4]([Cl])',
        '[CX4]([Br])',
        '[CX4]([F])[F]',
    ],

    'general_aromatic': [
        'c1ccccc1',
        '[nH]',
        '[n]1cccc1',
    ],
}

TOX_SMARTS = PHARM_SMARTS
TOX_GROUP_ORDER = [
    'nucleoside_analog', 'diaryl_pyrimidine', 'peptidomimetic',
    'diketo_acid', 'catechol', 'sulfonamide', 'guanidine_urea',
    'macrolide_like', 'polycyclic_heterocyclic', 'halogenated_alkyl',
    'general_aromatic',
]

L1_MEGA_GROUP_MAP = L1_MAP

# compile SMARTS patterns
_pharm_compiled = []
for g in TOX_GROUP_ORDER:
    pats = [MolFromSmarts(s) for s in PHARM_SMARTS[g]]
    pats = [p for p in pats if p is not None]
    _pharm_compiled.append((g, pats))

def _assign_pharmacophore(mol):
    if mol is None:
        return 'general_aromatic'
    for group_name, pats in _pharm_compiled:
        if any(mol.HasSubstructMatch(p) for p in pats):
            return group_name
    return 'general_aromatic'

pharmacophore_L2 = [_assign_pharmacophore(m) for m in mols]
pharmacophore_L1 = [L1_MAP.get(g, 'Other_antiviral') for g in pharmacophore_L2]
df['_pharmacophore_L2'] = pharmacophore_L2
df['_pharmacophore_L1'] = pharmacophore_L1

print('L2 pharmacophore_class distribution:')
for g in TOX_GROUP_ORDER:
    cnt = pharmacophore_L2.count(g)
    if cnt > 0:
        n_active = int(sum(1 for i, pg in enumerate(pharmacophore_L2)
                           if pg == g and Y_PRIMARY[i] == 1))
        print(f'  {g:25s}: {cnt:5d} total  ({n_active} active, '
              f'{100*n_active/max(cnt,1):.1f}% active rate)')

print('\nL1 mechanism_class distribution:')
for g in L1_GROUPS:
    cnt = pharmacophore_L1.count(g)
    n_active = int(sum(1 for i, pg in enumerate(pharmacophore_L1)
                       if pg == g and Y_PRIMARY[i] == 1))
    print(f'  {g:20s}: {cnt:5d} ({n_active} active, {100*n_active/max(cnt,1):.1f}% active rate)')


# In[17]:


# Cell 4.2 - H term: ACTIVE chemotype/scaffold hierarchy 
def build_hierarchy(df_local, mols_local, fps_local, lambda_per_level):
    '''Two-level HIV chemotype/scaffold hierarchy. mols_local/fps_local are unused
    (kept for call-compatibility with the SI lambda-ablation cell).
    lambda_per_level = [lambda_L2_chemotype_scaffold, lambda_L1_framework_class].'''
    def _grp(col):
        vals = df_local[col].tolist(); uniq = list(dict.fromkeys(vals))
        idx = {u: i for i, u in enumerate(uniq)}; desc = [[] for _ in uniq]
        for i, v in enumerate(vals): desc[idx[v]].append(i)
        return desc
    L2 = _grp('_h_L2'); L1 = _grp('_h_L1')
    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float64))
    return [{'name': 'chemotype_scaffold', 'descendants': L2, 'similarity': empty, 'lambda': lambda_per_level[0]},
            {'name': 'framework_class',    'descendants': L1, 'similarity': empty, 'lambda': lambda_per_level[1]}]

hierarchy_main = build_hierarchy(df, mols, fps, LAMBDA_MAIN)
for L in hierarchy_main:
    sizes = sorted([len(d) for d in L['descendants']], reverse=True)
    print(f"  {L['name']:20s}: {len(L['descendants'])} groups, lambda={L['lambda']}, top sizes={sizes[:8]}")
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)

_fw = df.groupby('_h_L1').agg(n=('_h_L1', 'size'), act=(TARGET_COL, 'sum'))
_fw['active_rate'] = _fw['act'] / _fw['n']; _fw = _fw.sort_values('n', ascending=False).head(15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.bar(range(len(_fw)), _fw['active_rate'] * 100, color='#CC6677')
ax1.axhline(100 * Y_PRIMARY.mean(), color='k', ls='--', lw=1, label=f'base rate {100*Y_PRIMARY.mean():.1f}%')
ax1.set_xticks(range(len(_fw))); ax1.set_xticklabels(_fw.index, rotation=70, ha='right', fontsize=6)
ax1.set_ylabel('% active'); ax1.set_title('HIV active rate by framework class (L1)'); ax1.legend(fontsize=8)
_sz = sorted([len(d) for d in hierarchy_main[0]['descendants']], reverse=True)
ax2.plot(range(len(_sz)), _sz, '.-', color='#4477AA'); ax2.set_yscale('log')
ax2.set_xlabel('scaffold-chemotype rank'); ax2.set_ylabel('molecules (log)')
ax2.set_title(f'L2 scaffold-chemotype sizes ({len(_sz)} groups)')
plt.tight_layout(); save_fig(fig, 'H_chemotype_hierarchy', where='main'); plt.show()
print(f'ACTIVE H (passed to SHIELD): {len(hierarchy_main[0]["descendants"])} scaffold chemotypes '
      f'(L2) nested in {len(hierarchy_main[1]["descendants"])} framework classes (L1).')
print('Framework classes differ markedly in active rate (left) -> chemotype is a genuine')
print('activity-bearing axis; holding chemotypes out creates real HIV OOD folds.')


# ## 4.3 External coarsening — chemotype-first, class-pure super-nodes (Tanimoto)

# In[18]:


# Cell 4.3.1 - external coarsening algorithms 
import math
from shield import coarsening as _coarsening
from shield.coarsening import lift_assignment_to_atoms

def h_first_coarsen(hierarchy_levels, feat, class_labels, n_target,
                    seed=RANDOM_SEED, class_pure=COARSEN_CLASS_PURE):
    '''Chemotype-first coarsening on ECFP. See original docstring; behavior is
    unchanged, only the merge regime is vectorized for speed.'''
    n_atoms, d = feat.shape
    feat_np = np.asarray(feat).astype(np.float32)
    feat_t  = torch.as_tensor(feat_np, dtype=torch.float32)
    cls = np.asarray(class_labels).astype(int)
    level0 = hierarchy_levels[0]['descendants']
    fams = [list(map(int, desc)) for desc in level0 if len(desc) > 0]
    covered = set()
    for f in fams: covered.update(f)
    residual = [i for i in range(n_atoms) if i not in covered]
    if residual: fams.append(residual)
    base = []
    for fam in fams:
        a = np.asarray(fam, dtype=np.int64)
        if class_pure:
            for c in np.unique(cls[a]): base.append(a[cls[a] == c])
        else:
            base.append(a)
    n_base = len(base)
    L1_of = np.full(n_atoms, -1, dtype=np.int64)
    if len(hierarchy_levels) > 1:
        for gid, desc in enumerate(hierarchy_levels[1]['descendants']):
            for i in desc: L1_of[int(i)] = gid

    if n_target >= n_base:
        # SPLIT regime
        max_sz = max(1, int(math.ceil(n_atoms / max(1, int(n_target)))))
        labels = np.full(n_atoms, -1, dtype=np.int64); nid = 0
        for a in base:
            if a.shape[0] <= max_sz:
                labels[a] = nid; nid += 1
            else:
                ks = min(a.shape[0], int(math.ceil(a.shape[0] / max_sz)))
                res = _coarsening.coarsen_balanced_kmeans(feat_t[a], n_clusters=ks, seed=seed, device='cpu')
                sub = res.labels.cpu().numpy().astype(np.int64); uniq = np.unique(sub)
                rm = {int(u): nid + j for j, u in enumerate(uniq)}
                labels[a] = np.array([rm[int(s)] for s in sub], dtype=np.int64); nid += len(uniq)
        n_super = int(nid)
    else:
        # MERGE regime
        groups = [a.tolist() for a in base]
        C   = np.stack([feat_np[a].mean(0) for a in base]).astype(np.float32)
        sq  = (C * C).sum(1).astype(np.float64)
        cnt = np.array([len(g) for g in groups], dtype=np.float64)
        cls_g = np.array([int(cls[a][0]) for a in base], dtype=np.int64)
        L1_g = []
        for a in base:
            v = L1_of[a]; v = v[v >= 0]; L1_g.append(int(np.bincount(v).argmax()) if v.size else -1)
        L1_g = np.asarray(L1_g, dtype=np.int64)
        G = n_base
        alive = np.ones(G, dtype=bool)
        idx_all = np.arange(G)
        NEG = -1.0e30

        def _row(i):
            dots = C @ C[i]
            den  = sq + sq[i] - dots
            t    = np.where(den > 1e-9, dots / np.maximum(den, 1e-30), 0.0)
            s    = t + ((L1_g == L1_g[i]) & (L1_g[i] >= 0)).astype(np.float64)
            s[~alive] = NEG
            s[cls_g != cls_g[i]] = NEG
            s[i] = NEG
            return s

        nn  = np.full(G, -1,  dtype=np.int64)
        nns = np.full(G, NEG, dtype=np.float64)
        for i in range(G):
            s = _row(i); j = int(s.argmax())
            if s[j] > NEG: nn[i] = j; nns[i] = s[j]

        active_count = G
        while active_count > int(n_target):
            i = int(np.where(alive, nns, NEG).argmax())
            j = int(nn[i])
            if j < 0 or nns[i] <= NEG:
                break
            a, b = (i, j) if i < j else (j, i)
            na, nb = cnt[a], cnt[b]
            C[a]  = (C[a] * na + C[b] * nb) / (na + nb)
            sq[a] = float((C[a] * C[a]).sum())
            cnt[a] = na + nb
            groups[a] = groups[a] + groups[b]
            alive[b] = False; nn[b] = -1; nns[b] = NEG
            active_count -= 1

            # 1) survivor's own best
            s_a = _row(a); ja = int(s_a.argmax())
            nn[a]  = ja if s_a[ja] > NEG else -1
            nns[a] = s_a[ja] if s_a[ja] > NEG else NEG

            # 2) groups that pointed at the merged pair must be recomputed
            aff = np.where(alive & np.isin(nn, (a, b)) & (idx_all != a))[0]
            if aff.size:
                D   = C[aff] @ C.T
                den = sq[aff][:, None] + sq[None, :] - D
                T   = np.where(den > 1e-9, D / np.maximum(den, 1e-30), 0.0)
                Sx  = T + ((L1_g[aff][:, None] == L1_g[None, :]) & (L1_g[aff][:, None] >= 0))
                Sx[:, ~alive] = NEG
                Sx[cls_g[aff][:, None] != cls_g[None, :]] = NEG
                Sx[np.arange(aff.size), aff] = NEG
                jb = Sx.argmax(1); sb = Sx[np.arange(aff.size), jb]; ok = sb > NEG
                nn[aff]  = np.where(ok, jb, -1)
                nns[aff] = np.where(ok, sb, NEG)

            # 3) the new survivor may now be the best partner for others
            better = np.where(alive & (idx_all != a) & (s_a > nns))[0]
            nn[better] = a; nns[better] = s_a[better]

        labels = np.full(n_atoms, -1, dtype=np.int64); nidx = 0
        for g in range(G):
            if alive[g]:
                labels[np.asarray(groups[g], dtype=np.int64)] = nidx; nidx += 1
        n_super = nidx

    labels_t = torch.as_tensor(labels, dtype=torch.long)
    agg = _coarsening.aggregate_super_nodes(labels_t, torch.ones(n_atoms, dtype=torch.float32),
                                            features=feat_t, n_clusters=n_super)
    cent = agg['features'].cpu().numpy().astype(np.float32)
    w = agg['super_weights'].cpu().numpy().astype(np.float32)
    csup = np.zeros(n_super, dtype=np.int64)
    for s in range(n_super):
        m = labels == s
        if m.any(): csup[s] = int(np.round(cls[m].mean()))
    return labels, cent, w, csup, n_super

def remap_hierarchy_to_super_nodes(levels_atom, labels):
    '''Unchanged from original.'''
    labels = np.asarray(labels); n_super = int(labels.max()) + 1 if labels.size else 0
    out = []
    for lvl in levels_atom:
        assign = np.full(n_super, -1, dtype=np.int64); best = np.zeros(n_super)
        for gid, desc in enumerate(lvl['descendants']):
            if len(desc) == 0: continue
            sn = labels[np.asarray(desc, dtype=np.int64)]; cnt = np.bincount(sn, minlength=n_super).astype(float)
            take = cnt > best; assign[take] = gid; best[take] = cnt[take]
        nd = [[] for _ in lvl['descendants']]
        for s in range(n_super):
            if assign[s] >= 0: nd[assign[s]].append(int(s))
        out.append({'name': lvl['name'], 'descendants': [sorted(d) for d in nd],
                    'similarity': lvl['similarity'], 'lambda': lvl['lambda']})
    return out
print('Coarsening algorithms defined (FAST merge): h_first_coarsen, remap_hierarchy_to_super_nodes.')


# In[19]:


# Cell 4.3.2 - PRIMARY coarsening (chemotype-first, class-pure) on ECFP
_class_atom = Y_PRIMARY.astype(np.int64)
labels_super, centroids, super_weights, class_super, n_super = h_first_coarsen(
    hierarchy_main, bv, _class_atom, COARSEN_N_SUPER_PRIMARY, seed=RANDOM_SEED)
print(f'Primary coarsening: {n:,} molecules -> {n_super} super-nodes '
      f'(target {COARSEN_N_SUPER_PRIMARY}, class_pure={COARSEN_CLASS_PURE})')
print(f'  super-node mass: min={super_weights.min():.0f} max={super_weights.max():.0f} '
      f'sum={super_weights.sum():.0f} (=n)')
_act = int((class_super == 1).sum())
print(f'  active super-nodes: {_act}/{n_super} ({100*_act/n_super:.1f}%); active mass '
      f'{100*super_weights[class_super==1].sum()/super_weights.sum():.2f}% (atom rate {100*Y_PRIMARY.mean():.2f}%)')
np.save(OUT_INTER / 'featurization' / 'labels_super.npy', labels_super)


# In[20]:


# Cell 4.3.3 - super-node M-graph eps calibration (Tanimoto) + connectivity diagnostic
import math
_Kt = _tani_sim(centroids); np.fill_diagonal(_Kt, 0.0)
_grid = np.round(np.arange(0.20, 0.91, 0.05), 2)
_deg = np.array([(_Kt > ev).sum(1) for ev in _grid])
_dens = _deg.sum(1) / max(n_super * (n_super - 1), 1); _md = _deg.mean(1); _iso = (_deg == 0).mean(1)
_floor = max(4.0, math.log2(max(n_super, 2)))
_ok = np.where((_iso <= 0.05) & (_md >= _floor))[0]
_rec = int(_ok[-1]) if len(_ok) else int(np.argmin(np.abs(_dens - 0.01)))
_eps_rec = float(_grid[_rec])
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4))
axL.semilogy(_grid, np.clip(_dens, 1e-7, None), 'o-', color='#4477AA')
axL.axvline(_eps_rec, color='red', ls='--', label=f'recommended {_eps_rec:g}')
axL.set_xlabel('EPS_AFF_SUPER (centroid Tanimoto)'); axL.set_ylabel('M-graph density (log)')
axL.set_title('Super-node M-graph density vs eps'); axL.legend(fontsize=8)
axR.plot(_grid, _md, 'o-', color='#117733', label='mean degree')
axR.plot(_grid, 100 * _iso, 's--', color='#CC6677', label='% isolated')
axR.axhline(_floor, color='gray', ls=':', label=f'degree floor {_floor:.0f}'); axR.axvline(_eps_rec, color='red', ls='--')
axR.set_xlabel('EPS_AFF_SUPER'); axR.set_title('Super-node M-graph connectivity vs eps'); axR.legend(fontsize=8)
plt.tight_layout(); save_fig(fig, 'super_node_eps_calibration', where='si'); plt.show()
print('HOW TO READ: pick the LARGEST eps (sparsest M graph) keeping % isolated < ~5% and')
print('  mean degree above the floor -> a discriminative but connected cut.')
print(f'  Recommended EPS_AFF_SUPER={_eps_rec:g} (density={_dens[_rec]:.4f}, '
      f'mean_deg={_md[_rec]:.1f}, isolated={100*_iso[_rec]:.1f}%).')
if EPS_AFF_SUPER is None:
    EPS_AFF_SUPER = _eps_rec; print(f'  => EPS_AFF_SUPER was None; AUTO-SET to {EPS_AFF_SUPER:g}.')
else:
    print(f'  => keeping user EPS_AFF_SUPER={EPS_AFF_SUPER:g}.')


# In[21]:


# Cell 4.3.4 - super-node relational matrices (M, D, cost) + hierarchy + helpers
AM       = build_super_affinity(centroids)                                 
AM = (AM * np.outer(super_weights, super_weights)).astype(np.float32); np.fill_diagonal(AM, 0.0)
A_full   = build_super_kernel(centroids); np.fill_diagonal(A_full, 0.0)
K_MATRIX = build_super_kernel(centroids)
if D_KERNEL_POWER != 1.0:
    K_MATRIX = np.clip(K_MATRIX ** D_KERNEL_POWER, 0.0, 1.0).astype(np.float32)
np.fill_diagonal(K_MATRIX, 1.0)
C        = build_super_cost(centroids)
TARGET_MARG  = np.tile(super_weights / super_weights.sum(), (N_FOLDS, 1)).astype(np.float32)
CLASS_LABELS = class_super.astype(np.int64)
hier_super = remap_hierarchy_to_super_nodes(hierarchy_main, labels_super)
hier_levels_obj_super = build_hierarchy_levels(hier_super)
AM_super_tensor = torch.from_numpy(AM).to(torch.float32)
TOTAL_SIM = float(A_full.sum())
for _lvl in hier_super:
    _ids = [x for d in _lvl['descendants'] for x in d]
    assert len(_ids) == len(set(_ids)), f"super hierarchy level {_lvl['name']} not disjoint"
np.save(OUT_INTER / 'featurization' / 'AM_super.npy', AM)
np.save(OUT_INTER / 'featurization' / 'K_super.npy', K_MATRIX)
print(f'Super matrices: AM {AM.shape} (density {(AM>0).sum()/max(AM.size,1):.4f}), '
      f'K {K_MATRIX.shape}, C {C.shape}; CLASS_LABELS active={int((CLASS_LABELS==1).sum())}/{n_super}')

def lift_super_to_atoms(fold_super):
    '''Scatter a super-node fold assignment (len n_super) back to atoms (len n).'''
    return lift_assignment_to_atoms(torch.as_tensor(fold_super, dtype=torch.long),
        torch.as_tensor(labels_super, dtype=torch.long)).cpu().numpy().astype(int)

def atom_fold_to_super(fold_atoms):
    fa = np.asarray(fold_atoms)
    if fa.shape[0] == n_super: return fa.astype(int)
    tally = np.zeros((n_super, N_FOLDS)); np.add.at(tally, (labels_super, fa.astype(int)), 1.0)
    return tally.argmax(1).astype(int)
print('Helpers ready: lift_super_to_atoms, atom_fold_to_super.')


# In[22]:


# Cell 4.3.5 - scale-invariant evaluation reference (Tanimoto, super-node)
_eval_ref_target = int(min(EVAL_REF_N_SUPER, n))
labels_ref, centroids_ref, super_weights_ref, class_ref, n_ref = h_first_coarsen(
    hierarchy_main, bv, _class_atom, _eval_ref_target, seed=RANDOM_SEED)
A_ref = build_super_kernel(centroids_ref); np.fill_diagonal(A_ref, 0.0)
_egr = np.round(np.arange(0.20, 0.91, 0.05), 2); _floor = max(4.0, np.log2(max(n_ref, 2)))
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
        if cnt >= T2_K_SUPER: break
AM_ref = np.maximum(AM_ref, AM_ref.T); np.fill_diagonal(AM_ref, 0.0)
w_ref = super_weights_ref.astype(np.float64)
TOTAL_SIM_REF = float(w_ref @ (A_ref @ w_ref)); TOTAL_M_REF = float(w_ref @ (AM_ref @ w_ref))

def _fold_to_ref(fold_atoms):
    fold = np.asarray(fold_atoms).astype(int); tally = np.zeros((n_ref, N_FOLDS))
    np.add.at(tally, (labels_ref, fold), 1.0); return tally.argmax(1).astype(int)
print(f'Fixed eval reference (Tanimoto): {n} atoms -> {n_ref} ref nodes '
      f'(EVAL_REF_N_SUPER={EVAL_REF_N_SUPER}, ref eps={EVAL_REF_EPS:g}; '
      f'independent of COARSEN_N_SUPER_PRIMARY={COARSEN_N_SUPER_PRIMARY}).')


# ## 5. Baseline splitters
# 

# In[23]:


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

def _split_class_stratified_basic(seed, y=None):
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

def split_class_stratified(seed):
    """Multilabel-aware stratified split using the iterative-stratification
    library if available, otherwise fall back to primary-assay class strat."""
    if _OPT['iterative_stratification'] is not None:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as MSKF
        Y_bin = (Y_FULL >= 0.5).astype(int)
        mskf = MSKF(n_splits=int(round(1.0 / FOLD_RATIOS[1])),
                    shuffle=True, random_state=seed)
        for _, test_idx in mskf.split(np.arange(n).reshape(-1, 1), Y_bin):
            f = np.zeros(n, dtype=int); f[test_idx] = 1; return f
    return _split_class_stratified_basic(seed)

BASELINE_SPLITTERS = {
    'RANDOM':               split_random,
    'CLASS_STRATIFIED':     split_class_stratified,
    'SCAFFOLD':             split_scaffold,
    'KMEANS':               split_kmeans,
    'KENNARDSTONE':         split_kennard_stone,
}


# In[24]:


# Cell 5.2 - run all baseline splitters
baseline_results = {}

for name, fn in BASELINE_SPLITTERS.items():
    f = fn(RANDOM_SEED)
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(
        OUT_INTER / 'splits' / f'{name}.csv', index=False)
    print(f'  {name:22s}  sizes={sizes.round(3).tolist()}')


# In[25]:


# Cell 5.3 - precomputed splits from hiv_DataSAIL.csv

for _pc_name, _pc_col in [('SPLIT_BASE', '_split_base'), ('SPLIT_DATASAIL', '_split_datasail')]:
    _f = np.array([PRECOMPUTED_FOLD_MAP[lb] for lb in df[_pc_col].values], dtype=int)
    _sizes = np.bincount(_f, minlength=N_FOLDS) / n
    baseline_results[_pc_name] = _f
    _out = OUT_INTER / 'splits' / f'{_pc_name}.csv'
    pd.DataFrame({'index': np.arange(n), 'fold': _f}).to_csv(_out, index=False)
    print(f'  {_pc_name:16s}  sizes={_sizes.round(3).tolist()}')


# In[26]:


# Cell 5.4 - additional splitting tools: DeepChem-style

import warnings as _w54
_w54.filterwarnings('ignore')


# DeepChem-style splits 

def split_dc_butina(seed):
    """Sphere-exclusion (Leader) clustering on ECFP4 — C++ LeaderPicker, distance cutoff 0.4
    (Tanimoto >= 0.6), then assign each molecule to its most-similar leader and greedy-pack
    clusters into folds. Same intent as Taylor–Butina but ~linear instead of O(n^2).
    Falls back to exact Butina (with a vectorized distance build) if the picker is missing."""
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
    """Fingerprint-sorted split (unchanged)."""
    packed  = np.packbits(bv, axis=1)
    order   = np.lexsort(packed.T[::-1])
    n_train = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[order[:n_train]] = 0
    return f

def split_dc_maxmin(seed):
    """MaxMin diversity picker — C++ MaxMinPicker (lazy distances); falls back to the
    vectorized numpy greedy loop. Same objective: test = the picked diverse set."""
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
    """DeepChem scaffold split (unchanged)."""
    from collections import defaultdict as _dd
    grp = _dd(list)
    for i, sc in enumerate(df['_scaffold']): grp[sc].append(i)
    labels_map = {sc: cid for cid, sc in enumerate(grp)}
    labels = np.array([labels_map[sc] for sc in df['_scaffold']])
    return _greedy_pack(labels, len(grp))

def split_dc_weight(seed):
    """Molecular-weight split (unchanged)."""
    from rdkit.Chem.Descriptors import MolWt as _MolWt
    mw = np.array([_MolWt(m) if m else 0.0 for m in mols])
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int); f[np.argsort(-mw)[:n_test]] = 1
    return f

# register and run

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



# ## 6. SHIELD primary configurations

# In[27]:


# Cell 6.1 - SHIELD ablation grid
SHIELD_CONFIGS = {
    'HMD-SHIELD':     dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU),
    'MD-SHIELD':     dict(alpha=0.0,   beta=BETA, gamma=GAMMA, eta=ETA, mu=MU),
    'HD-SHIELD':     dict(alpha=ALPHA, beta=0.0,  gamma=GAMMA, eta=ETA, mu=MU),
    'HM-SHIELD':     dict(alpha=ALPHA, beta=BETA, gamma=0.0,   eta=ETA, mu=MU),
    'H-SHIELD':   dict(alpha=ALPHA, beta=0.0,  gamma=0.0,   eta=ETA, mu=MU),
    'M-SHIELD':   dict(alpha=0.0,   beta=BETA, gamma=0.0,   eta=ETA, mu=MU),
    'D-SHIELD':   dict(alpha=0.0,   beta=0.0,  gamma=GAMMA, eta=ETA, mu=MU),
}
print('SHIELD config grid:')
for k, c in SHIELD_CONFIGS.items():
    print(f'  {k:18s}  {c}')


# In[28]:


# Cell 6.2 - run all SHIELD configurations on SUPER-NODES (classification)
shield_results = {}
shield_diagnostics = {}
t_all = time.time()
for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    kwargs = dict(
        n=n_super, weights=super_weights, K=N_FOLDS,
        r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hier_super if cfg['alpha'] > 0 else None,
        affinity=AM if cfg['beta'] > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        kernel_matrix=K_MATRIX if cfg['gamma'] > 0 else None,
        class_labels=CLASS_LABELS if cfg['mu'] > 0 else None,
        stratification_strategy='none',
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'], mu=cfg['mu'], nu=NU, tau=TAU,
        coarsen=False,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=PRIMARY_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        milp_time_limit_s=SHIELD_MILP_TIME, stage0_samples=SHIELD_STAGE0_SAMPLES,
        sinkhorn_max_iter=SHIELD_SINKHORN_ITER, 
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        res = shield_split(**kwargs)
        fold_super = res.fold_assignment.cpu().numpy().astype(int)
        fold = lift_super_to_atoms(fold_super)
        shield_results[cfg_name] = fold
        shield_diagnostics[cfg_name] = {
            'history': res.history,
            'diagnostics': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                            for k, v in res.diagnostics.items() if k != 'history'},
            'metadata': res.metadata}
        (OUT_INTER / 'shield_runs' / cfg_name).mkdir(exist_ok=True, parents=True)
        pd.DataFrame({'index': np.arange(n), 'fold': fold}).to_csv(
            OUT_INTER / 'shield_runs' / cfg_name / 'split.csv', index=False)
        rate_tr = float(Y_PRIMARY[fold == 0].mean()) if (fold == 0).any() else 0.0
        rate_te = float(Y_PRIMARY[fold == 1].mean()) if (fold == 1).any() else 0.0
        print(f'  {cfg_name:18s} wall={time.time()-t0:.1f}s '
              f'sizes={(np.bincount(fold, minlength=N_FOLDS)/n).round(3).tolist()} '
              f'active_rate=[{rate_tr:.3f},{rate_te:.3f}]')
    except Exception as exc:
        print(f'  {cfg_name:18s} FAILED: {exc}')
print(f'\nTotal SHIELD wall time: {time.time()-t_all:.1f}s  '
      f'(solved on {n_super} super-nodes, lifted to {n} atoms)')


# In[29]:


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

# In[30]:


# Cell 7.1 - cross-fold structural similarity L(pi) on the FIXED scale-invariant reference
_wref = w_ref

def _fold_mass_to_ref(fold_atoms):
    '''Molecule mass of each reference node split across folds; rows sum to w_ref.'''
    fold = np.asarray(fold_atoms).astype(int)
    Mref = np.zeros((n_ref, N_FOLDS), dtype=np.float64)
    np.add.at(Mref, (labels_ref, fold), 1.0)
    return Mref

def cross_fold_similarity(fold):
    Mref = _fold_mass_to_ref(fold)
    within = sum(float(Mref[:, k] @ (A_ref @ Mref[:, k])) for k in range(N_FOLDS))
    raw = max(TOTAL_SIM_REF - within, 0.0)
    scaled = raw / max(TOTAL_SIM_REF, 1e-9)
    return raw, scaled

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


# In[31]:


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


# In[32]:


# Cell 7.3 - per-channel decomposition L_H (atom-exact), L_M (scale-invariant ref), L_W (binary)
from scipy.stats import wasserstein_distance
hier_levels_obj = build_hierarchy_levels(hierarchy_main)   
_wref = w_ref

def per_channel_leakage(fold):
    fold = np.asarray(fold).astype(int)
    z = torch.zeros((n, N_FOLDS), dtype=torch.float32)
    z[np.arange(n), fold] = 1.0
    L_H = float(_evaluate_h_term_from_z(z, hier_levels_obj))
    Mref = _fold_mass_to_ref(fold)
    within_m = sum(float(Mref[:, k] @ (AM_ref @ Mref[:, k])) for k in range(N_FOLDS))
    L_M = max(TOTAL_M_REF - within_m, 0.0)
    tr = Y_PRIMARY[fold == 0]; te = Y_PRIMARY[fold == 1]
    L_W = float(wasserstein_distance(tr, te)) if (len(tr) and len(te)) else float('nan')
    return L_H, L_M, L_W

ch_rows = []
for name, fold in baseline_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'baseline', 'config': name, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
for name, fold in shield_results.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'SHIELD', 'config': name, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
channel_df = pd.DataFrame(ch_rows)
save_table(channel_df, 'per_channel_leakage', where='main')
display(channel_df)


# In[33]:


# Cell 7.4 - HIV class retention diagnostic

global_rate = float(np.nanmean(Y_PRIMARY))
methods = list(baseline_results.keys()) + list(shield_results.keys())
retention_rows = []
for name in methods:
    fold = baseline_results.get(name, shield_results.get(name))
    tr_act = float(np.nanmean(Y_PRIMARY[fold == 0]))
    te_act = float(np.nanmean(Y_PRIMARY[fold == 1]))
    retention_rows.append({'config': name,
                           'train_active_rate': tr_act,
                           'test_active_rate':  te_act,
                           'dev_train': abs(tr_act - global_rate),
                           'dev_test':  abs(te_act - global_rate),
                           'mean_dev':  (abs(tr_act - global_rate) +
                                         abs(te_act - global_rate)) / 2.0})
ret_df = pd.DataFrame(retention_rows).sort_values('mean_dev')
save_table(ret_df, 'class_retention', where='main')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
x = np.arange(len(ret_df))
w = 0.35
ax.bar(x - w/2, ret_df['train_active_rate'] * 100, w, label='train', color='#4477AA', edgecolor='black')
ax.bar(x + w/2, ret_df['test_active_rate']  * 100, w, label='test',  color='#CC6677', edgecolor='black')
ax.axhline(global_rate * 100, color='black', ls='--', lw=1.5,
           label=f'global rate ({100*global_rate:.1f}%)')
ax.set_xticks(x); ax.set_xticklabels(ret_df['config'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('active rate (%)')
ax.set_title('HIV active rate per fold (dashed = global; SHIELD should be closest)')
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.bar(x, ret_df['mean_dev'] * 100, color='#EE7733', edgecolor='black')
ax2.set_xticks(x); ax2.set_xticklabels(ret_df['config'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('mean |fold_rate - global_rate| (%)')
ax2.set_title('Class balance deviation (lower = better)')
ax2.axhline(STRAT_TOL * 100, color='red', ls=':', lw=1, label=f'STRAT_TOL={STRAT_TOL}')
ax2.legend(fontsize=8)

plt.tight_layout()
save_fig(fig, 'class_retention', where='main')
plt.show()

mean_class_retention_dev = ret_df.set_index('config')['mean_dev']


# In[34]:


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


# In[35]:


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


# ## 7. Dimension-reduction visualizations of splits

# In[36]:


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

for emb_name, emb in embeds.items():
    np.save(OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.npy', emb)
    pd.DataFrame(emb, columns=[f'{emb_name}_1', f'{emb_name}_2']).to_csv(
        OUT_INTER / 'featurization' / f'embedding_{emb_name.lower()}.csv', index=True, index_label='row_id'
    )
    print(f'Saved {emb_name} embedding: {emb.shape}')


# In[84]:


# Cell 7.2 - split-colored scatter panels
showcase = ['RANDOM', 'CLASS_STRATIFIED', 'SCAFFOLD', 'KMEANS', 'KENNARDSTONE', 'SPLIT_BASE', 'SPLIT_DATASAIL',
            'DC_BUTINA', 'DC_FINGERPRINT', 'DC_MAXMIN', 'DC_SCAFFOLD', 'DC_WEIGHT',
            'HMD-SHIELD', 'H-SHIELD', 'M-SHIELD', 'D-SHIELD', 'MD-SHIELD', 'HD-SHIELD', 'HM-SHIELD']
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


#  t-SNE colored by target class
_tsne_emb = embeds.get('tSNE')
if _tsne_emb is not None:
    fig, ax = plt.subplots(figsize=(6, 5))
    _labeled = prim_labeled
    _colors = np.where(Y_PRIMARY[_labeled] == 1, '#CC6677', '#4477AA')
    ax.scatter(_tsne_emb[_labeled & (Y_PRIMARY == 0), 0],
               _tsne_emb[_labeled & (Y_PRIMARY == 0), 1],
               s=5, c='#4477AA', alpha=0.55, label='inactive (0)', rasterized=True)
    ax.scatter(_tsne_emb[_labeled & (Y_PRIMARY == 1), 0],
               _tsne_emb[_labeled & (Y_PRIMARY == 1), 1],
               s=5, c='#CC6677', alpha=0.75, label='active (1)', rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f't-SNE colored by {TARGET_COL}', fontsize=11)
    ax.legend(loc='best', fontsize=8, markerscale=2)
    plt.tight_layout()
    save_fig(fig, 'tsne_target_class', where='main')
    plt.show()
else:
    print('t-SNE embedding not available in embeds dict.')


# ## 8. OOD Assessment Metrics

# In[38]:


# Cell 8.1 - OOD characterization metrics per split

from sklearn.neighbors  import NearestNeighbors
from scipy.spatial      import ConvexHull, Delaunay
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
import matplotlib.ticker as mticker
import warnings

# t-SNE embeddings 
_tsne_2d = embeds.get('tSNE')   # shape (n_samples, 2)

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


# In[39]:


# Cell 8.2 - export SMILES + fold assignments for every split
_all = {}
for k, v in globals().get('baseline_results', {}).items():       _all[k] = v
for k, v in globals().get('shield_results', {}).items():         _all[f'SHIELD6_{k}'] = v
for k, v in globals().get('lambda_results', {}).items():         _all[f'LAMBDA12_{k}'] = v
for k, v in globals().get('rounding_results', {}).items():       _all[f'ROUND14_{k}'] = v
for k, v in globals().get('coarsen_fold_results', {}).items():   _all[k] = v

export = pd.DataFrame({'smiles': df[SMILES_COL].values})
if 'mol_id' in df.columns:
    export.insert(0, 'mol_id', df['mol_id'].values)
for _raw in ['_split_base', '_split_datasail']:
    if _raw in df.columns:
        export[f'RAW{_raw}'] = df[_raw].values
if TARGET_COL in df.columns:
    export[TARGET_COL] = df[TARGET_COL].values

for name, fold in _all.items():
    fold = np.asarray(fold)
    if fold.shape[0] != len(df):
        print(f'  SKIP {name}: length {fold.shape[0]} != n={len(df)}'); continue
    export[name] = fold.astype(int)

_out = OUT_DIR / 'all_splits_folds.csv'
export.to_csv(_out, index=False)
print(f'Exported {export.shape[0]:,} rows x {export.shape[1]} cols -> {_out}')
print('Fold convention: 0 = train, 1 = holdout/test.')
print('Split columns:', [c for c in export.columns
                         if c not in ('mol_id', 'smiles', TARGET_COL,
                                      'RAW_split_base', 'RAW_split_datasail')])


# In[40]:


# Cell 8.3 - Export all splits to a single combined CSV

_splits_df = df.copy()

for split_name, fold_arr in all_splits.items():
    col = 'fold__' + split_name.replace('::', '__').replace(' ', '_')
    _splits_df[col] = fold_arr

_fold_cols = [c for c in _splits_df.columns if c.startswith('fold__')]
print(f'Splits embedded: {len(_fold_cols)}')
for c in _fold_cols:
    counts = _splits_df[c].value_counts().sort_index().to_dict()
    print(f'  {c:55s}  {counts}')

_out_path = OUT_INTER / 'splits' / 'all_splits_combined.csv'
_splits_df.to_csv(_out_path, index=True, index_label='row_id')
print(f'\nSaved: {_out_path}  (shape: {_splits_df.shape})')


# ## 9. SHIELD (alpha, beta, gamma) sweep, Pareto frontier, Hamming stability

# In[41]:


# Cell 9.1 - shield.sweep over alpha x beta x gamma
sweep_kwargs = dict(
    n=n_super, weights=super_weights, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hier_super, affinity=AM,
    affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE, 
    stratification_strategy=STRAT_STRATEGY,
    class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
    eta=ETA, mu=MU, nu=NU, tau=TAU,
    max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
    integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
    rounding_mode=PRIMARY_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
    stage0_samples=SHIELD_STAGE0_SAMPLES,

    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
    kernel_matrix=K_MATRIX,
    class_labels=CLASS_LABELS,
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


# ## 10. SI — confidence-gap rounding vs pipage rounding

# In[45]:


# Cell 10.1 - run HMD-SHIELD with both rounding modes
rounding_results = {}
for rmode in ['pipage', 'confidence_gap']:
    print(f'  rounding={rmode}')
    t0 = time.time()
    res = shield_split(
        n=n_super, weights=super_weights, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hier_super, affinity=AM,
        affinity_provenance='label-blind-handcrafted',
        d_mode=DIST_MODE, 
        stratification_strategy=STRAT_STRATEGY,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=rmode, milp_size_limit=SHIELD_MILP_SIZE, milp_time_limit_s=SHIELD_MILP_TIME, coarsen=False,
        stage0_samples=SHIELD_STAGE0_SAMPLES,

        sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
        kernel_matrix=K_MATRIX,
        class_labels=CLASS_LABELS,
    )
    rounding_results[rmode] = lift_super_to_atoms(res.fold_assignment.cpu().numpy().astype(int))
    print(f'    wall={time.time()-t0:.1f}s')

if len(rounding_results) == 2:
    f1, f2 = rounding_results['pipage'], rounding_results['confidence_gap']
    hamming = int((f1 != f2).sum())
    print(f'\nHamming distance pipage vs confidence_gap: {hamming} / {n} '
          f'({100 * hamming / n:.2f}%)')

cg_rep1 = rounding_results['confidence_gap']
res_cg2 = shield_split(
    n=n_super, weights=super_weights, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hier_super, affinity=AM, affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE, 
    stratification_strategy=STRAT_STRATEGY, 
    class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
    alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, mu=MU, nu=NU, tau=TAU,
    max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED + 1, device=SHIELD_DEVICE, nodes=N_JOBS,
    integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
    rounding_mode='confidence_gap', milp_size_limit=SHIELD_MILP_SIZE, milp_time_limit_s=SHIELD_MILP_TIME, coarsen=False,
    stage0_samples=SHIELD_STAGE0_SAMPLES,

    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
    kernel_matrix=K_MATRIX,
    class_labels=CLASS_LABELS,
)
cg_rep2 = lift_super_to_atoms(res_cg2.fold_assignment.cpu().numpy().astype(int))
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
