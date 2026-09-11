# ## 1. Setup, imports, configuration, output directories

# In[1]:


# Cell 1.1 - imports
import os, sys, json, time, math, hashlib, pickle, warnings, importlib, platform
import itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sps
from scipy.stats import wasserstein_distance, spearmanr, pearsonr

import sklearn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer

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


# In[2]:


# Cell 1.2 - global configuration
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'LP_PDBBind.csv')
DATASET_NAME = 'lp_pdbbind'
SMILES_COL   = 'smiles'
SEQ_COL      = 'seq'
TARGET_COL   = 'value'
HEADER_COL   = 'header'
DROP_COLS    = ['Unnamed: 0', 'date']
SPLIT_COL    = 'new_split'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 0

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.1
STRAT_TOL   = 0.2

# Protein featurization
KMER_K       = 3
KMER_NFEATS  = 16384

ECFP_RADIUS  = 2
ECFP_NBITS   = 1024

KNN_AFFINITY_K = 25

# Stratification on pKd
STRAT_STRATEGY = 'quantile'
STRAT_N_BINS   = 4

# SHIELD coefficients - main run
ALPHA, BETA, GAMMA, ETA, MU = 1.0, 1.0, 1.0, 10.0, 0.5
NU, TAU = 0.0, 0.0
DIST_MODE = 'match'
OT_REG    = 0.1

# Two-entity modes to run as separate primary configurations
TWO_ENTITY_MODES = ['warm', 'cold-both']            

# Sweep grid
SWEEP_GRID = {
    'alpha': [0.5, 1.0, 2.0, 5.0],
    'beta':  [0.5, 1.0, 2.0, 5.0],
    'gamma': [0.5, 1.0, 2.0, 5.0],
}
HAMMING_STABILITY_THRESHOLD = 0.10

# kNN purity / alpha-shape
KNN_K = [5, 10, 25]

# Solver / computational
SHIELD_MAX_ITER     = 15
SHIELD_STAGE0_SAMPLES = 100
SHIELD_SINKHORN_ITER =50
SHIELD_INIT_BACKEND = 'auto'
SHIELD_ROUNDING     = 'pipage'
SHIELD_MILP_SIZE    = 25_000
SHIELD_MILP_TIME_S  = 200.0
SHIELD_DEVICE       = 'cpu'
N_JOBS              = 36

H_DEFINITION = 'hybrid'
N_PROT_FAMILIES  = 100
N_LIG_CHEMOTYPES = 75

EVAL_REF_N_SUPER       = 8192
KNN_M_K                = 25
STRAT_N_BINS_TARGET    = 24

DATASAIL_PATH = str(_REPO_ROOT / 'splits' / 'LP_PDBBind_DataSAIL.csv')
PRECOMPUTED_SPLIT_COLS = {
    'BASE_new_split':   'new_split',
    'DATASAIL_S1_lig':  'DataSAIL S1_lig',
    'DATASAIL_S1_prot': 'DataSAIL S1_prot',
    'DATASAIL_S2':      'DataSAIL S2',
}
TYPE_COL     = 'type'
CATEGORY_COL = 'category'

PRECOMPUTED_FOLD_MAP = {'train': 0, 'val': 1, 'test': 0}

# ML evaluation
ML_RUN = True

print(f'Config loaded. MAX_ROWS={MAX_ROWS}, K={N_FOLDS}, '
      f'two-entity modes={TWO_ENTITY_MODES}')


# In[3]:


# Cell 1.3 - output directories and figure-saving utilities
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
    'rdkit':       _try('rdkit'),
    'xgboost':     _try('xgboost'),
    'umap':        _try('umap', 'umap'),
    'alphashape':  _try('alphashape'),
    'shapely':     _try('shapely'),
    'esm':         _try('esm'),
    'transformers':_try('transformers'),
    'pymetis':     _try('pymetis'),
    'kahip':       _try('kahip'),
    'gurobipy':    _try('gurobipy'),
}
for k, v in _versions.items():
    print(f'  {k:14s}  {v if v else "MISSING; required to get the same exact results"}')

with open(OUT_META / 'environment.json', 'w') as fh:
    json.dump({'python': sys.version, 'platform': platform.platform(),
               'versions': _versions,
               'shield_version': __import__('shield').__version__}, fh, indent=2)

if _OPT['rdkit'] is None:
    raise RuntimeError('RDKit is required for ligand featurization.')
else:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')

if _OPT['gurobipy'] is None:
    raise RuntimeError('gurobipy + a Gurobi license are required for SHIELD LP/MILP.')




# ## 2. Data loading and cleaning

# In[5]:


# Cell 2.1 - load LP-PDBBind (+ the 3 DataSAIL split columns), apply the cleaning protocol
from rdkit import Chem as _Chem
from rdkit import RDLogger as _RDLogger; _RDLogger.DisableLog('rdApp.*')

print(f'Loading from {DATASAIL_PATH}')
df_raw = pd.read_csv(DATASAIL_PATH)
print(f'Raw shape: {df_raw.shape}')

to_drop = [c for c in DROP_COLS if c in df_raw.columns]
df = df_raw.drop(columns=to_drop)
print(f'Dropped columns {to_drop}')

before = len(df)
df = df.dropna(subset=[SEQ_COL, TARGET_COL]).reset_index(drop=True)
print(f'Dropped {before-len(df):5d} rows: missing seq/target')
before = len(df)
df = df.dropna(subset=[SMILES_COL]).reset_index(drop=True)
print(f'Dropped {before-len(df):5d} rows: missing SMILES')
before = len(df)
_ok = df[SMILES_COL].apply(lambda s: _Chem.MolFromSmiles(s) is not None)
df = df[_ok.values].reset_index(drop=True)
print(f'Dropped {before-len(df):5d} rows: RDKit-unparseable SMILES')

n = len(df)
pKd = df[TARGET_COL].values.astype(np.float32)
print(f'\nClean working set: n={n} protein-ligand complexes')
print(f'  unique proteins (seq)  = {df[SEQ_COL].nunique():6d}  (mean rows/protein {n/df[SEQ_COL].nunique():.2f})')
print(f'  unique ligands (smiles)= {df[SMILES_COL].nunique():6d}  (mean rows/ligand  {n/df[SMILES_COL].nunique():.2f})')
print(f'  pKd range = [{pKd.min():.2f}, {pKd.max():.2f}]  mean = {pKd.mean():.2f}')
print('  per precomputed-split label counts (NaN = unassigned in that split):')
for _k, _c in PRECOMPUTED_SPLIT_COLS.items():
    print(f'    {_k:18s}: {df[_c].value_counts(dropna=False).to_dict()}')


# In[6]:


# Cell 2.2 - subsample to MAX_ROWS (keeping the train/val/test ratio)
if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = (df.groupby(SPLIT_COL, group_keys=False)
            .apply(lambda g: g.sample(n=max(2, int(round(MAX_ROWS * len(g) / len(df_raw)))),
                                      random_state=RANDOM_SEED, replace=False))
            .reset_index(drop=True))
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df):,} rows')
n = len(df)
pKd = df[TARGET_COL].values.astype(np.float32)
print(f'Working n={n}; pKd range=[{pKd.min():.2f}, {pKd.max():.2f}]')


# In[7]:


# Cell 2.3 - merge the 4 precomputed 3-fold splits (base + 3 DataSAIL) into 2 folds
precomputed_splits = {}
_merge_rows = []
for _name, _col in PRECOMPUTED_SPLIT_COLS.items():
    fold = np.array([-1 if pd.isna(v) else PRECOMPUTED_FOLD_MAP.get(str(v), -1)
                     for v in df[_col].values], dtype=int)
    precomputed_splits[_name] = fold
    n0, n1, nx = int((fold == 0).sum()), int((fold == 1).sum()), int((fold < 0).sum())
    _merge_rows.append({'split': _name, 'column': _col, 'n_train(0)': n0, 'n_test(1)': n1,
                        'n_excluded(-1)': nx, 'test_frac': round(n1 / max(n0 + n1, 1), 3)})
merge_df = pd.DataFrame(_merge_rows)
save_table(merge_df, 'precomputed_split_merge_audit', where='si')
print('3->2 merge via PRECOMPUTED_FOLD_MAP =', PRECOMPUTED_FOLD_MAP)
display(merge_df)
print(f'\nGuidance: SHIELD targets FOLD_RATIOS = {FOLD_RATIOS} (80/20). The precomputed splits')
print('keep their INTRINSIC test_frac above (the DataSAIL / LP-PDBBind design). To move a')
print('label between folds, edit ONE value in PRECOMPUTED_FOLD_MAP in Cell 1.2 — e.g. set')
print('"val": 1 to enlarge a small held-out fold toward 20%. Nothing else needs changing.')


# In[8]:


# Cell 2.4 - EDA: pKd distribution, per-type counts, sequence-length histogram
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0,0].hist(pKd, bins=40, color='#4477AA', edgecolor='black')
axes[0,0].set_xlabel('pKd'); axes[0,0].set_ylabel('count')
axes[0,0].set_title(f'pKd distribution (n={n}, mean={pKd.mean():.2f})')

t_counts = df['type'].value_counts().head(10)
axes[0,1].barh(t_counts.index[::-1], t_counts.values[::-1], color='#332288')
axes[0,1].set_xlabel('count'); axes[0,1].set_title('Top protein-function categories')

seq_len = df[SEQ_COL].str.len().values
axes[1,0].hist(seq_len, bins=40, color='#117733', edgecolor='black')
axes[1,0].set_xlabel('sequence length (residues)'); axes[1,0].set_ylabel('count')
axes[1,0].set_title(f'Sequence-length distribution (median={int(np.median(seq_len))})')

cat_counts = df['category'].value_counts()
axes[1,1].bar(cat_counts.index, cat_counts.values, color='#CC6677', edgecolor='black')
axes[1,1].set_title('PDBbind sub-set membership')
axes[1,1].set_ylabel('count')

plt.tight_layout()
save_fig(fig, 'eda_panel', where='si')
plt.show()


# In[9]:


# Cell 2.5 - protein and ligand identity counts (entity-cardinality audit)
n_unique_proteins = df[SEQ_COL].nunique()
print(f'Unique protein sequences (= cold-left entities, exact-match): {n_unique_proteins} '
      f'across {n} rows -> mean rows per protein {n / n_unique_proteins:.2f}')
print(f'Unique PDB headers: {df[HEADER_COL].nunique()}')

cluster_audit = pd.DataFrame({
    'flag': ['CL1', 'CL2', 'CL3'],
    'n_true': [int(df['CL1'].sum()), int(df['CL2'].sum()), int(df['CL3'].sum())],
    'n_false':[int((~df['CL1']).sum()), int((~df['CL2']).sum()), int((~df['CL3']).sum())],
})
save_table(cluster_audit, 'cluster_flag_audit', where='si')
display(cluster_audit)


# ## 3. Featurization

# In[10]:


PROT_SVD_DIM = int(globals().get('PROT_SVD_DIM', 64))
def kmer_corpus(seqs, k):
    return [' '.join(s[i:i+k] for i in range(len(s) - k + 1)) for s in seqs]
print(f'Protein k-mer features (k={KMER_K}, hash dim={KMER_NFEATS}) -> TruncatedSVD({PROT_SVD_DIM})...')
_hv = HashingVectorizer(n_features=KMER_NFEATS, alternate_sign=False, norm='l2',
                        analyzer='word', token_pattern=r'(?u)\b\w+\b')
X_prot_sparse = _hv.transform(kmer_corpus(df[SEQ_COL].tolist(), KMER_K))     
X_prot = TruncatedSVD(n_components=PROT_SVD_DIM, random_state=RANDOM_SEED).fit_transform(
    X_prot_sparse).astype(np.float32)
X_prot /= (np.linalg.norm(X_prot, axis=1, keepdims=True) + 1e-9)              
np.save(OUT_INTER / 'featurization' / 'X_prot.npy', X_prot)
print(f'  X_prot {X_prot.shape} (dense, low-rank); dense n x n similarity AVOIDED '
      f'({n}x{n} = {n*n/1e9:.1f}e9 entries not built)')


# In[11]:


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
LIG_SVD_DIM = int(globals().get('LIG_SVD_DIM', 64))
fps = []
for s in df[SMILES_COL]:
    m = Chem.MolFromSmiles(s)
    fps.append(AllChem.GetMorganFingerprintAsBitVect(m, ECFP_RADIUS, nBits=ECFP_NBITS))
bv = np.zeros((n, ECFP_NBITS), dtype=np.uint8)
_buf = np.zeros((ECFP_NBITS,), dtype=np.int8)
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, _buf); bv[i] = _buf.astype(np.uint8)
X_lig = TruncatedSVD(n_components=LIG_SVD_DIM, random_state=RANDOM_SEED).fit_transform(
    bv.astype(np.float32)).astype(np.float32)
X_lig /= (np.linalg.norm(X_lig, axis=1, keepdims=True) + 1e-9)
np.save(OUT_INTER / 'featurization' / 'X_lig.npy', X_lig)
print(f'Ligand ECFP bit-vectors {bv.shape}; low-rank X_lig {X_lig.shape}; dense Tanimoto AVOIDED')


# In[12]:


from sklearn.neighbors import NearestNeighbors
X_joint = np.hstack([X_prot, X_lig]).astype(np.float32)
PROT_DIM = X_prot.shape[1]
np.save(OUT_INTER / 'featurization' / 'X_joint.npy', X_joint)

def build_sparse_bilinear_M(Xj, w, k, prot_dim, device='cpu'):
    """Symmetric k-NN graph; edge weight = max(prot_sim,0)*max(lig_sim,0)*w_i*w_j.
    `w` are node masses (ones at row scale; super-node counts after coarsening -> mass-faithful M).
    Returns a torch SPARSE COO tensor of shape (m, m)."""
    Xj = np.asarray(Xj, np.float32); m = Xj.shape[0]; w = np.asarray(w, np.float64)
    kk = int(min(k, m - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1, metric='cosine').fit(Xj)
    _, idx = nn.kneighbors(Xj)
    nbr = idx[:, 1:]
    P, L = Xj[:, :prot_dim], Xj[:, prot_dim:]
    sp = np.maximum((P[:, None, :] * P[nbr]).sum(-1), 0.0)
    sl = np.maximum((L[:, None, :] * L[nbr]).sum(-1), 0.0)
    wt = (sp * sl) * (w[:, None] * w[nbr])
    src = np.repeat(np.arange(m), kk); dst = nbr.reshape(-1); val = wt.reshape(-1)
    keep = val > 0
    src, dst, val = src[keep], dst[keep], val[keep]
    ii = np.concatenate([src, dst]); jj = np.concatenate([dst, src])
    vv = np.concatenate([val, val]).astype(np.float32)
    A = torch.sparse_coo_tensor(torch.tensor(np.vstack([ii, jj]), dtype=torch.long),
                                torch.tensor(vv, dtype=torch.float32), (m, m)).coalesce()
    return A

AM = build_sparse_bilinear_M(X_joint, np.ones(n), KNN_M_K, PROT_DIM, SHIELD_DEVICE)
print(f'M (sparse bilinear k-NN, k={KNN_M_K}): {AM._nnz()} nonzeros (~{AM._nnz()/n:.0f}/row); '
      f'dense-equiv {n*n} entries NOT built')


# In[13]:


# Cell 3.4 - D term: pKd distribution shift via SMALL-support OT (D-match)
_pkd_bins = np.unique(np.quantile(pKd, np.linspace(0.0, 1.0, STRAT_N_BINS_TARGET + 1)))
_pkd_centers = (0.5 * (_pkd_bins[:-1] + _pkd_bins[1:])).astype(np.float32)
M_SUPPORT = len(_pkd_centers)
C = np.abs(pKd[:, None] - _pkd_centers[None, :]).astype(np.float32)
C /= max(C.max(), 1e-6)
_hist, _ = np.histogram(pKd, bins=_pkd_bins); _hist = _hist.astype(np.float32); _hist /= _hist.sum()
TARGET_MARG = np.tile(_hist.reshape(1, -1), (N_FOLDS, 1)).astype(np.float32)
Y_TARGET = pKd.reshape(-1, 1).astype(np.float32)
np.save(OUT_INTER / 'featurization' / 'cost_small_support.npy', C)
print(f'D-match cost C {C.shape} (n x {M_SUPPORT} pKd-bin support), target marginal {TARGET_MARG.shape}; '
      f'linear in n (dense n x n cost AVOIDED)')


# ## 4. Hierarchy construction and protein / ligand entity IDs

# In[14]:


# Cell 4.1 - Murcko scaffolds, ring systems, atom-count bins for the ligand hierarchy
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
atom_bin = pd.cut(natoms, bins=[-0.5, 14.5, 24.5, 34.5, np.inf],
                  labels=['ha<=14', 'ha15-24', 'ha25-34', 'ha>=35']).astype(str)

df['_scaffold'] = scaffolds
df['_ring']     = rings
df['_atombin']  = atom_bin
print(f'Distinct scaffolds: {df["_scaffold"].nunique()}')
print(f'Distinct rings:     {df["_ring"].nunique()}')
print(f'Atom-count bins:    {df["_atombin"].value_counts().to_dict()}')


# In[15]:


# Cell 4.2 - shared helpers for the domain H: cluster entities + build the two-axis hierarchy
from sklearn.cluster import MiniBatchKMeans
LAMBDA_PROT = float(globals().get('LAMBDA_PROT', 1.0))
LAMBDA_LIG  = float(globals().get('LAMBDA_LIG',  1.0))

def cluster_entities(X, n_clusters, seed=RANDOM_SEED):
    """KMeans labels in a low-rank embedding; sizable, non-singleton families."""
    k = int(min(max(n_clusters, 1), X.shape[0]))
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init='auto',
                           batch_size=2048).fit_predict(X).astype(np.int64)

def _empty_coo():
    return (np.array([], np.int64), np.array([], np.int64), np.array([], np.float64))

def _level(name, ids, lam):
    """One containment level: descendants = rows grouped by their family id (disjoint)."""
    desc = {}
    for i, v in enumerate(np.asarray(ids)):
        desc.setdefault(int(v), []).append(i)
    return {'name': name, 'descendants': [desc[k] for k in sorted(desc)],
            'similarity': _empty_coo(), 'lambda': float(lam)}

def build_two_axis_hierarchy(prot_fam, lig_fam, lam_prot=LAMBDA_PROT, lam_lig=LAMBDA_LIG):
    """Deepest-first per SHIELD convention: ligand chemotype (finer) then protein family.
    Containment-only (no similarity edges) -> exactly the 'keep each family intact in one
    fold' constraint that makes cold splits well-posed, and it is O(n) to build."""
    return [_level('ligand_chemotype', lig_fam, lam_lig),
            _level('protein_family',   prot_fam, lam_prot)]

def _cluster_size_report(prot_fam, lig_fam, title):
    """Print + plot family-size histograms and a feasibility check for a ~test-fold-sized holdout."""
    test_target = int(round(FOLD_RATIOS[1] * n))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (ids, lab) in zip(ax, [(prot_fam, 'protein families'), (lig_fam, 'ligand chemotypes')]):
        sizes = np.bincount(np.asarray(ids))
        a.hist(sizes, bins=40, color='#4477AA', edgecolor='black')
        a.set_title(f'{lab}: {len(sizes)} clusters\nmedian size {int(np.median(sizes))}, '
                    f'singletons {int((sizes==1).sum())}')
        a.set_xlabel('rows per cluster'); a.set_ylabel('count')
    fig.suptitle(f'H = {title}  (target test fold ~{test_target} rows)', fontweight='bold')
    plt.tight_layout(); save_fig(fig, f'H_cluster_sizes_{title}', where='si'); plt.show()
    for ids, lab in [(prot_fam, 'protein'), (lig_fam, 'ligand')]:
        sizes = np.sort(np.bincount(np.asarray(ids)))[::-1]
        k_need = int(np.searchsorted(np.cumsum(sizes), test_target) + 1)
        print(f'  {lab:8s}: {len(sizes)} clusters, largest {sizes[0]}, '
              f'~{k_need} largest clusters fill a {test_target}-row test fold, '
              f'{int((sizes==1).sum())} singletons')
    print('  TUNE N_PROT_FAMILIES / N_LIG_CHEMOTYPES in Cell 1.2: fewer clusters -> larger, '
          'less-fragmented holdouts (and more coarsening headroom); more clusters -> finer cold splits.')

print('H helpers ready: cluster_entities, build_two_axis_hierarchy, _cluster_size_report '
      f'(active H_DEFINITION = {H_DEFINITION!r}).')


# In[16]:


# Cell 4.3 - hybrid 2-level both axes: protein type -> seq sub-clusters, scaffold -> ECFP sub-clusters

HYBRID_SIZE_TOL = 0.25

def _adaptive_refine(coarse, feat, n_target):
    """Split only oversized coarse groups; sub-cluster count proportional to group size."""
    coarse = np.asarray(coarse)
    avg = max(len(coarse) / max(int(n_target), 1), 1.0)
    thresh = avg * (1.0 + HYBRID_SIZE_TOL)
    fine = np.zeros(len(coarse), np.int64); nxt = 0
    for g in np.unique(coarse):
        idx = np.where(coarse == g)[0]; s = len(idx)
        if s <= thresh:
            fine[idx] = nxt; nxt += 1
        else:
            k = max(2, int(round(s / avg)))
            sub = cluster_entities(feat[idx], k)
            fine[idx] = sub + nxt; nxt += int(sub.max()) + 1
    return fine

prot_coarse = pd.factorize(df[TYPE_COL].astype(str))[0].astype(np.int64)
n_types     = len(np.unique(prot_coarse))
prot_fine   = _adaptive_refine(prot_coarse, X_prot, N_PROT_FAMILIES)
lig_coarse = pd.factorize(df['_scaffold'].astype(str))[0].astype(np.int64)
lig_fine   = cluster_entities(X_lig, N_LIG_CHEMOTYPES)
prot_family_of, lig_family_of = prot_fine, lig_fine

hierarchy_main = [
    _level('ligand_chemotype_fine',  lig_fine,    LAMBDA_LIG),
    _level('ligand_scaffold_coarse', lig_coarse,  LAMBDA_LIG * 0.2),
    _level('protein_seq_fine',       prot_fine,   LAMBDA_PROT),
    _level('protein_type_coarse',    prot_coarse, LAMBDA_PROT * 0.2),
]
_ps = np.bincount(prot_fine); _ls = np.bincount(lig_fine)
_avg_p = n / max(N_PROT_FAMILIES, 1); _avg_l = n / max(N_LIG_CHEMOTYPES, 1)
print(f'[H option 3 = hybrid, size-adaptive] '
        f'protein: {n_types} types -> {len(_ps)} seq sub-families '
        f'(target avg {_avg_p:.0f}; actual min/med/max {_ps.min()}/{int(np.median(_ps))}/{_ps.max()}); '
        f'ligand: {len(np.unique(lig_coarse))} scaffolds -> {len(_ls)} ECFP chemotypes '
        f'(target avg {_avg_l:.0f}; actual min/med/max {_ls.min()}/{int(np.median(_ls))}/{_ls.max()}); '
        f'{len(hierarchy_main)}-level H')
_cluster_size_report(prot_family_of, lig_family_of, 'hybrid')


# In[17]:


# Cell 4.4 - finalize cold ENTITIES + pair indices (used by SHIELD cold mode)
assert ('prot_family_of' in dir()) and ('lig_family_of' in dir()) and ('hierarchy_main' in dir()), \
    ("No H option ran -- set H_DEFINITION in Cell 1.2 to exactly one of "
     "'similarity_clusters' | 'type_scaffold' | 'hybrid'.")
protein_entity_of = np.asarray(prot_family_of, np.int64)
ligand_entity_of  = np.asarray(lig_family_of,  np.int64)
pair_indices = np.column_stack([np.arange(n), np.arange(n)]).astype(np.int64)
hier_levels_obj = build_hierarchy_levels(hierarchy_main)
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)
_n_cells = len(set(zip(protein_entity_of.tolist(), ligand_entity_of.tolist())))
print(f'Active H = {H_DEFINITION!r}: {len(hierarchy_main)} levels {[L["name"] for L in hierarchy_main]}')
print(f'Cold entities: {len(np.unique(protein_entity_of))} protein families, '
      f'{len(np.unique(ligand_entity_of))} ligand chemotypes')
print(f'(family, chemotype) cells = {_n_cells}  ->  this is the FLOOR for pure-in-both super-nodes '
      f'(Section 15 coarsening cannot go below it while keeping cold-both feasible).')


# In[18]:


# Cell 4.5 - external H-first TWO-ENTITY coarsening + FIXED scale-invariant L(pi) reference

def h_first_two_entity_coarsen(prot_fam, lig_fam, feat, n_target, seed=RANDOM_SEED):
    """Super-nodes PURE in (protein family, ligand family). Cells larger than the per-super-node
    budget are split by KMeans on `feat` (keeping the cell's pf/lf). Used for the Section-15 sweep
    where SHIELD runs cold modes on super-nodes. Returns
    labels, centroids, super_weights, super_pf, super_lf, n_super, n_cells."""
    prot_fam = np.asarray(prot_fam); lig_fam = np.asarray(lig_fam); feat = np.asarray(feat, np.float32)
    m = len(prot_fam)
    cells = {}
    for i in range(m):
        cells.setdefault((int(prot_fam[i]), int(lig_fam[i])), []).append(i)
    n_cells = len(cells)
    budget = max(1, math.ceil(m / max(int(n_target), 1)))
    labels = np.full(m, -1, np.int64); nxt = 0
    proj = feat @ feat.mean(0)
    for (pf, lf), idx in cells.items():
        idx = np.asarray(idx, np.int64)
        if len(idx) <= budget:
            labels[idx] = nxt; nxt += 1
        else:
            ksub = int(math.ceil(len(idx) / budget))
            order = idx[np.argsort(proj[idx], kind='stable')]
            for chunk in np.array_split(order, ksub):
                if len(chunk):
                    labels[chunk] = nxt; nxt += 1
    n_super = int(nxt)
    w = np.bincount(labels, minlength=n_super).astype(np.float64)
    cent = np.zeros((n_super, feat.shape[1]), np.float64)
    np.add.at(cent, labels, feat.astype(np.float64))
    cent = (cent / np.maximum(w[:, None], 1.0)).astype(np.float32)
    _, _first = np.unique(labels, return_index=True)
    spf = np.asarray(prot_fam)[_first].astype(np.int64)
    slf = np.asarray(lig_fam)[_first].astype(np.int64)
    return labels, cent, w, spf, slf, n_super, n_cells

def super_bilinear(cent, prot_dim=None):
    """Dense bilinear protein x ligand similarity on super-node centroids (small: n_super x n_super)."""
    pd_ = PROT_DIM if prot_dim is None else prot_dim
    P, L = cent[:, :pd_], cent[:, pd_:]
    A = (np.maximum(P @ P.T, 0.0) * np.maximum(L @ L.T, 0.0)).astype(np.float32)
    np.fill_diagonal(A, 0.0); return A

_ref_k = int(min(EVAL_REF_N_SUPER, n))
labels_ref = cluster_entities(X_joint, _ref_k)
n_ref = int(labels_ref.max()) + 1
w_ref = np.bincount(labels_ref, minlength=n_ref).astype(np.float64)
cent_ref = np.zeros((n_ref, X_joint.shape[1]), np.float64)
np.add.at(cent_ref, labels_ref, X_joint.astype(np.float64))
cent_ref = (cent_ref / np.maximum(w_ref[:, None], 1.0)).astype(np.float32)
A_ref  = super_bilinear(cent_ref)
AM_ref = A_ref
TOTAL_SIM_REF = float(w_ref @ (A_ref @ w_ref)); TOTAL_M_REF = TOTAL_SIM_REF

def _fold_mass_to_ref(fold):
    """Reference-node mass split across folds (SOFT membership). Rows with fold<0 (excluded
    from a precomputed split) contribute NO mass. Rows sum to <= w_ref."""
    fold = np.asarray(fold).astype(int)
    M = np.zeros((n_ref, N_FOLDS), np.float64)
    keep = fold >= 0
    np.add.at(M, (labels_ref[keep], fold[keep]), 1.0)
    return M

def cross_fold_similarity(fold):
    """Scale-invariant L(pi): mass-weighted cross-fold bilinear similarity on the fixed reference.
    The total is taken over the split's KEPT rows only (excluded -1 rows contribute nothing and
    are NOT counted as leakage), so precomputed splits with many unassigned rows score correctly."""
    M = _fold_mass_to_ref(fold)
    wk = M.sum(1)
    total = float(wk @ (A_ref @ wk))
    within = sum(float(M[:, k] @ (A_ref @ M[:, k])) for k in range(N_FOLDS))
    raw = max(total - within, 0.0)
    return raw, raw / max(total, 1e-9)

print(f'FIXED L(pi) reference: {n} rows -> {n_ref} k-means super-nodes (target {EVAL_REF_N_SUPER}); '
      f'soft mass-weighted, independent of the Section-15 n_super sweep.')
print('Section-15 sweep uses h_first_two_entity_coarsen (pure-in-both) so cold modes stay feasible.')


# ## 5. Baseline splitters

# In[19]:


# Cell 5.1 - baseline split implementations
from sklearn.cluster import KMeans

def split_random(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); cut = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[perm[:cut]] = 0
    return f

def split_scaffold(seed):
    rng = np.random.default_rng(seed)
    scaffold_to_idx = defaultdict(list)
    for i, s in enumerate(df['_scaffold']):
        scaffold_to_idx[s].append(i)
    order = list(scaffold_to_idx.values()); rng.shuffle(order)
    f = np.ones(n, dtype=int); cum = 0; target_train = int(round(FOLD_RATIOS[0] * n))
    for g in order:
        if cum + len(g) <= target_train:
            for j in g: f[j] = 0
            cum += len(g)
        else: break
    return f

def split_protein_family(seed):
    """Hold out whole PROTEIN FAMILIES (cold-protein baseline; uses the H protein entities)."""
    rng = np.random.default_rng(seed)
    entity_to_idx = defaultdict(list)
    for i, e in enumerate(protein_entity_of):
        entity_to_idx[int(e)].append(i)
    order = list(entity_to_idx.values()); rng.shuffle(order)
    f = np.ones(n, dtype=int); cum = 0; target_train = int(round(FOLD_RATIOS[0] * n))
    for g in order:
        if cum + len(g) <= target_train:
            for j in g: f[j] = 0
            cum += len(g)
        else: break
    return f

def split_kmeans(seed):
    X = np.hstack([bv.astype(np.float32), X_prot]).astype(np.float32)
    km = KMeans(n_clusters=N_FOLDS, n_init='auto', random_state=seed).fit(X)
    labels = km.labels_; sizes = np.bincount(labels, minlength=N_FOLDS); order = np.argsort(-sizes)
    fold_of = np.zeros(N_FOLDS, dtype=int); fold_of[order[0]] = 0; fold_of[order[1:]] = 1
    return fold_of[labels]

def split_cluster_flag(col):
    """LP-PDBBind CL1/CL2/CL3 boolean cluster flags: larger group -> train (0)."""
    arr = df[col].astype(int).values; sizes = np.bincount(arr, minlength=2)
    return np.where(arr == np.argmax(sizes), 0, 1).astype(int)

BASELINE_SPLITTERS = {
    'RANDOM':          lambda seed=RANDOM_SEED: split_random(seed),
    'SCAFFOLD_LIGAND': lambda seed=RANDOM_SEED: split_scaffold(seed),
    'PROTEIN_FAMILY':  lambda seed=RANDOM_SEED: split_protein_family(seed),
    'KMEANS':          lambda seed=RANDOM_SEED: split_kmeans(seed),
    'CL1_FLAG':        lambda seed=RANDOM_SEED: split_cluster_flag('CL1'),
    'CL2_FLAG':        lambda seed=RANDOM_SEED: split_cluster_flag('CL2'),
    'CL3_FLAG':        lambda seed=RANDOM_SEED: split_cluster_flag('CL3'),
}


# In[20]:


# Cell 5.2 - additional splitters: DeepChem-style
# (DC_BUTINA, DC_FINGERPRINT, DC_MAXMIN, DC_SCAFFOLD, DC_WEIGHT)

import warnings as _w51b
_w51b.filterwarnings('ignore')

# greedy cluster packer (shared by DC_BUTINA, DC_SCAFFOLD) 
def _greedy_pack(labels, n_clusters):
    sizes = np.bincount(labels, minlength=n_clusters)
    order = np.argsort(-sizes); target_train = int(round(FOLD_RATIOS[0] * n))
    fold_of = np.ones(n_clusters, dtype=int); cum = 0
    for cid in order:
        if cum + sizes[cid] <= target_train: fold_of[cid] = 0; cum += sizes[cid]
        else: break
    return fold_of[labels]


# DeepChem-style splits 

def split_dc_butina(seed):
    """Sphere-exclusion (Leader) clustering on ECFP fingerprints — C++ LeaderPicker,
    distance cutoff 0.4 (Tanimoto >= 0.6). Falls back to exact Butina if unavailable."""
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
    """Fingerprint-sorted split: lexicographic ordering on packed ECFP bits."""
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
    """DeepChem scaffold split: clusters by Murcko scaffold, greedy-packs into train/test."""
    from collections import defaultdict as _dd
    grp = _dd(list)
    for i, sc in enumerate(df['_scaffold']): grp[sc].append(i)
    labels_map = {sc: cid for cid, sc in enumerate(grp)}
    labels = np.array([labels_map[sc] for sc in df['_scaffold']])
    return _greedy_pack(labels, len(grp))

def split_dc_weight(seed):
    """Molecular-weight split: heaviest ligands go to test."""
    from rdkit.Chem.Descriptors import MolWt as _MolWt
    mw = np.array([_MolWt(m) if m else 0.0 for m in mols])
    n_test = int(round(FOLD_RATIOS[1] * n))
    f = np.zeros(n, dtype=int); f[np.argsort(-mw)[:n_test]] = 1
    return f

ADDITIONAL_SPLITTERS = {
    'DC_BUTINA':      lambda seed=RANDOM_SEED: split_dc_butina(seed),
    'DC_FINGERPRINT': lambda seed=RANDOM_SEED: split_dc_fingerprint(seed),
    'DC_MAXMIN':      lambda seed=RANDOM_SEED: split_dc_maxmin(seed),
    'DC_SCAFFOLD':    lambda seed=RANDOM_SEED: split_dc_scaffold(seed),
    'DC_WEIGHT':      lambda seed=RANDOM_SEED: split_dc_weight(seed),
}
BASELINE_SPLITTERS.update(ADDITIONAL_SPLITTERS)
print(f'ADDITIONAL_SPLITTERS registered ({len(ADDITIONAL_SPLITTERS)} new); '
      f'BASELINE_SPLITTERS now has {len(BASELINE_SPLITTERS)} total.')


# In[21]:


# Cell 5.3 - run baseline splitters + register the precomputed splits (base + 3 DataSAIL)
baseline_results = {}
for name, fn in BASELINE_SPLITTERS.items():
    f = fn(); baseline_results[name] = f
    kept = f >= 0
    print(f'  {name:18s} sizes={(np.bincount(f[kept], minlength=N_FOLDS)/max(kept.sum(),1)).round(3).tolist()}')
for name, f in precomputed_splits.items():
    baseline_results[name] = f
    kept = f >= 0
    print(f'  {name:18s} sizes={(np.bincount(f[kept], minlength=N_FOLDS)/max(kept.sum(),1)).round(3).tolist()} '
          f'(excluded {int((~kept).sum())})')
for name, f in baseline_results.items():
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(
        OUT_INTER / 'splits' / f'{name}.csv', index=False)
print(f'Registered {len(baseline_results)} baseline/precomputed splits.')


# ## 6. SHIELD primary — warm + cold-left + cold-right + cold-both + ablations

# In[22]:


SHIELD_CONFIGS = {
    'HMD-SHIELD_warm': dict(
        alpha=ALPHA, beta=BETA, gamma=GAMMA,
        eta=ETA, mu=MU, mode='warm'
    ),
    'HMD-SHIELD_cold-both': dict(
        alpha=ALPHA, beta=BETA, gamma=GAMMA,
        eta=ETA, mu=MU, mode='cold-both'
    ),
}

for term, param in {'H': 'alpha', 'M': 'beta', 'D': 'gamma'}.items():
    cfg = dict(
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        eta=ETA,
        mu=MU,
        mode='warm'
    )

    cfg[param] = {'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA}[param]

    SHIELD_CONFIGS[f'{term}-SHIELD_warm'] = cfg

print('SHIELD config grid:')
for k, c in SHIELD_CONFIGS.items():
    print(f'  {k:30s}  {c}')


# In[23]:


# Cell 6.2 - run all SHIELD configurations
shield_results = {}
shield_diagnostics = {}
t_all = time.time()
CACHED_NORMALIZERS = None

Y_TARGET = pKd.reshape(-1, 1).astype(np.float32)

for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    mode = cfg['mode']
    kwargs = dict(
        n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main if cfg['alpha'] > 0 else None,
        affinity=AM if cfg['beta'] > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        cost=C if cfg['gamma'] > 0 else None,
        target_marginals=TARGET_MARG if cfg['gamma'] > 0 else None,
        y_target=Y_TARGET,
        stratification_strategy=STRAT_STRATEGY if cfg['mu'] > 0 else 'none',
        n_bins=STRAT_N_BINS,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        two_entity_mode=mode,
        pair_indices=pair_indices if mode != 'warm' else None,
        left_entity_of=protein_entity_of if mode in ('cold-left','cold-both') else None,
        right_entity_of=ligand_entity_of if mode in ('cold-right','cold-both') else None,
        pair_observation_targets=pKd.reshape(-1, 1) if mode != 'warm' else None,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'], mu=cfg['mu'], nu=NU, tau=TAU,
        epsilon_OT=OT_REG,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        milp_time_limit_s=SHIELD_MILP_TIME_S,
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


# In[24]:


# Cell 6.3 - Stage-0 normalizer table + init back-end used
norm_rows = []
for cfg_name, diag in shield_diagnostics.items():
    md_ = diag['metadata']
    s0 = md_.get('stage0_normalizers', {})
    norm_rows.append({
        'config':       cfg_name,
        'H_norm':       s0.get('H'),
        'M_norm':       s0.get('M'),
        'D_norm':       s0.get('D'),
        'strat_norm':   s0.get('strat'),
        'bal_norm':     s0.get('bal'),
        'init_backend': md_.get('init_backend'),
        'milp_invoked': md_.get('milp_polish', {}).get('invoked', False),
        'milp_reason':  md_.get('milp_polish', {}).get('trigger', ''),
    })
norm_df = pd.DataFrame(norm_rows)
save_table(norm_df, 'stage0_normalizers_and_milp', where='si')
norm_df


# ## 7. Leakage diagnostics 

# In[ ]:


# Cell 7.1 - cross-fold structural similarity L(pi) on the FIXED scale-invariant reference (Cell 4.5)
lpi_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    raw, scaled = cross_fold_similarity(fold)
    lpi_rows.append({'method': 'SHIELD' if name in shield_results else 'baseline',
                     'config': name, 'L_pi': raw, 'scaled_L_pi': scaled})
lpi_df = pd.DataFrame(lpi_rows).sort_values('scaled_L_pi')
save_table(lpi_df, 'cross_fold_similarity', where='main')
lpi_df


# In[26]:


# Cell 7.2 - L(pi) bar chart
fig, ax = plt.subplots(figsize=(12, 4))
agg = lpi_df.set_index('config')['scaled_L_pi'].sort_values()
colors = ['#888888' if c in baseline_results else '#4477AA' for c in agg.index]
ax.bar(range(len(agg)), agg.values, color=colors, edgecolor='black')
ax.axhline(agg.loc['RANDOM'] if 'RANDOM' in agg.index else 1.0,
           color='red', ls='--', lw=1, label='RANDOM baseline')
ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg.index, rotation=45, ha='right')
ax.set_ylabel('scaled L(pi)')
ax.set_title('Cross-fold pair-similarity (lower = stronger OOD)')
ax.legend()
plt.tight_layout()
save_fig(fig, 'L_pi_bar_chart', where='main')
plt.show()


# In[27]:


# Cell 7.3 - per-channel decomposition L_H (H containment), L_M (fixed ref, soft), L_W (pKd)
def per_channel_leakage(fold):
    fold = np.asarray(fold).astype(int)
    z = torch.zeros((n, N_FOLDS), dtype=torch.float32)
    z[np.arange(n), np.clip(fold, 0, N_FOLDS - 1)] = 1.0
    L_H = float(_evaluate_h_term_from_z(z, hier_levels_obj))
    M = _fold_mass_to_ref(fold); wk = M.sum(1)
    total_m = float(wk @ (AM_ref @ wk))
    within_m = sum(float(M[:, k] @ (AM_ref @ M[:, k])) for k in range(N_FOLDS))
    L_M = max(total_m - within_m, 0.0)
    tr = pKd[fold == 0]; te = pKd[fold == 1]
    L_W = float(wasserstein_distance(tr, te)) if len(tr) and len(te) else float('nan')
    return L_H, L_M, L_W

ch_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    L_H, L_M, L_W = per_channel_leakage(fold)
    ch_rows.append({'method': 'SHIELD' if name in shield_results else 'baseline',
                    'config': name, 'L_H': L_H, 'L_M': L_M, 'L_W': L_W})
channel_df = pd.DataFrame(ch_rows)
save_table(channel_df, 'channel_decomposition', where='main')
channel_df


# In[28]:


# Cell 7.4 - kNN purity in the joint (protein || ligand) embedding (uses X_joint from Cell 3.3)
def knn_purity(fold, X, k):
    fold = np.asarray(fold)
    test_idx = np.where(fold == 1)[0]
    if not len(test_idx): return float('nan')

    Xt = X[test_idx].astype(np.float32)
    X_full = X.astype(np.float32)

    d = (Xt * Xt).sum(1, keepdims=True) + (X_full * X_full).sum(1)[None, :] - 2 * Xt @ X_full.T

    nn = np.argpartition(d, kth=min(k + 1, X_full.shape[0] - 1), axis=1)[:, :k + 1]

    for i in range(nn.shape[0]):
        nn[i] = nn[i][np.argsort(d[i, nn[i]])]
    nn = nn[:, 1:k + 1]

    return float(np.mean([np.mean(fold[row] == 0) for row in nn]))

purity_rows = []
for name, fold in {**baseline_results, **shield_results}.items():
    if (np.asarray(fold) >= 0).sum() < 2:
        continue
    for k in KNN_K:
        purity_rows.append({'config': name, 'k': k, 'knn_purity': knn_purity(fold, X_joint, k)})
purity_df = pd.DataFrame(purity_rows)
save_table(purity_df, 'knn_purity', where='main')
display(purity_df)

fig, axes = plt.subplots(1, len(KNN_K), figsize=(5 * len(KNN_K), 4))
if len(KNN_K) == 1: axes = [axes]
for ax, k in zip(axes, KNN_K):
    sub = purity_df[purity_df['k'] == k].set_index('config')['knn_purity'].sort_values()
    colors = ['#888888' if c in baseline_results else '#4477AA' for c in sub.index]
    ax.bar(range(len(sub)), sub.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.index, rotation=45, ha='right')
    ax.axhline(FOLD_RATIOS[0], color='red', ls='--', lw=1, label='random baseline')
    ax.set_title(f'kNN purity (k={k})'); ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
save_fig(fig, 'knn_purity_panels', where='main')
plt.show()


# In[29]:


# Cell 7.5 - Train Potential Leakage (TPL)
from scipy.stats import mannwhitneyu

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
    tpl_val, tpl_lo, tpl_hi, (phi_tr, phi_te, h) = tpl(X_joint, fold)
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


# ## 8. Dimension-reduction visualizations of splits

# In[30]:


# Cell 8.1 - embeddings
embeds = {}
try:
    embeds['tSNE'] = TSNE(n_components=2, perplexity=30, init='pca',
                          random_state=RANDOM_SEED).fit_transform(X_joint)
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


# In[31]:


# Cell 8.2 - split-colored scatter panels
showcase = ['RANDOM', 'SCAFFOLD_LIGAND', 'PROTEIN_FAMILY', 'KMEANS', 'CL1_FLAG', 'CL2_FLAG', 'CL3_FLAG', 'DC_WEIGHT', 'DC_SCAFFOLD', 'DC_MAXMIN', 'DC_FINGERPRINT', 'DC_BUTINA',
            'BASE_new_split', 'DATASAIL_S1_lig', 'DATASAIL_S1_prot', 'DATASAIL_S2', 
            'HMD-SHIELD_warm', 'HMD-SHIELD_cold-both', 'H-SHIELD_warm', 'M-SHIELD_warm', 'D-SHIELD_warm']

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
        ax.set_title(f'{name}\nscaled L(pi)={scaled:.3f}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(showcase):]: ax.set_visible(False)
    axes[0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{emb_name} embedding colored by fold', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_splits',
             where='main' if emb_name in ('PCA','UMAP') else 'si')
    plt.show()

# Add pKd-colored panel to each embedding
import matplotlib.cm as cm
import matplotlib.colors as mcolors

for emb_name, emb in embeds.items():
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=pKd, cmap='RdYlBu_r',
        s=6, alpha=0.65,
        vmin=np.percentile(pKd, 2),
        vmax=np.percentile(pKd, 98),
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('pKd (binding affinity)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{emb_name} — binding affinity (pKd)', fontsize=10)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_pkd',
             where='main' if emb_name in ('PCA', 'UMAP') else 'si')
    plt.show()


# In[32]:


# 8.3 diagnostic only
showcase = [
    'RANDOM', 'SCAFFOLD_LIGAND', 'PROTEIN_FAMILY', 'KMEANS',
    'CL1_FLAG', 'CL2_FLAG', 'CL3_FLAG',
    'BASE_new_split', 'DATASAIL_S1_lig', 'DATASAIL_S1_prot',
    'DATASAIL_S2', 'HMD-SHIELD_warm', 'HMD-SHIELD_cold-both',
]

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


# ## 9. OOD Assessment Metrics

# In[33]:


# Cell 9.1 - OOD characterization metrics per split: NN distance stats + hull outside rate

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
    Xtr_161 = X_joint[mask_tr_161]
    Xte_161 = X_joint[mask_te_161]

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


# MASTER FIGURE 1 — Bubble map: 5 metrics encoded in one scatter

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


# MASTER FIGURE 2 — Parallel coordinates: 4 distance metrics across 12 splits

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


# In[34]:


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
