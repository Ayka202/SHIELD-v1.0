# ## 1. Setup, imports, configuration, output directories

# In[1]:


# Cell 1.1 - imports

import os, sys, json, time, math, hashlib, pickle, warnings, importlib, platform, re
import itertools
from pathlib import Path
from collections import defaultdict, Counter
import subprocess

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sps
from scipy.stats import wasserstein_distance

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, balanced_accuracy_score, accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

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
DATASET_PATH = str(_REPO_ROOT / 'benchmarks' / 'glass.json')
DATASET_NAME = 'matbench_glass'
COMP_COL     = 'composition'
TARGET_COL   = 'gfa'

# Size control
MAX_ROWS    = None
RANDOM_SEED = 42

# Splits
N_FOLDS     = 2
FOLD_RATIOS = [0.8, 0.2]
BALANCE_TOL = 0.05
STRAT_TOL   = 0.05                   

# Affinity (M term)
KNN_AFFINITY_K = 20                  

# Hierarchy lambda (single level)
H_KMEANS_K               = 60
H_CENTROID_COSINE_THRESH = 0.75
LAMBDA_MAIN = [1.0]

# Stratification on the binary target
STRAT_STRATEGY = 'quantile'
STRAT_N_BINS   = 5

# SHIELD coefficients
ALPHA, BETA, GAMMA, ETA, MU = 0.5, 0.2, 0.7, 0.5, 0.5
NU, TAU = 0.0, 0.0
DIST_MODE = 'match'
OT_REG    = 0.10
SHIELD_SINKHORN_ITER = 50

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

print(f'Config loaded. MAX_ROWS={MAX_ROWS}, K={N_FOLDS}, '
      f'lambda={LAMBDA_MAIN}, strat={STRAT_STRATEGY}, class_delta={STRAT_TOL}')


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
    'pymatgen':    _try('pymatgen'),
    'matminer':    _try('matminer'),
    'matbench':    _try('matbench'),
    'modnet':      _try('modnet'),
    'crabnet':     _try('crabnet'),
    'xgboost':     _try('xgboost'),
    'umap':        _try('umap', 'umap'),
    'alphashape':  _try('alphashape'),
    'shapely':     _try('shapely'),
    'gurobipy':    _try('gurobipy'),
}
for k, v in _versions.items():
    print(f'  {k:14s}  {v if v else "MISSING; required to get the same exact results)"}')

with open(OUT_META / 'environment.json', 'w') as fh:
    json.dump({'python': sys.version, 'platform': platform.platform(),
               'versions': _versions,
               'shield_version': __import__('shield').__version__}, fh, indent=2)

if _OPT['gurobipy'] is None:
    raise RuntimeError('gurobipy + a Gurobi license are required for SHIELD LP/MILP.')


# ## 2. Data loading and EDA

# In[5]:


# Cell 2.1 - load matbench_glass from the orient='split' JSON
print(f'Loading from {DATASET_PATH}')
with open(DATASET_PATH, 'r') as fh:
    raw = json.load(fh)
df_full = pd.DataFrame(raw['data'], columns=raw['columns'], index=raw['index'])
print(f'Loaded shape: {df_full.shape}')
print(f'Columns:     {df_full.columns.tolist()}')
print(f'Dtypes:      {df_full.dtypes.to_dict()}')
print(f'Missing:     {df_full.isna().sum().to_dict()}')
print(f'Unique compositions: {df_full[COMP_COL].nunique()} / {len(df_full)}')
print(f'Class balance:       {df_full[TARGET_COL].value_counts().to_dict()}')

if MAX_ROWS is not None and len(df_full) > MAX_ROWS:
    df = df_full.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f'Subsampled to {len(df)} rows (seed={RANDOM_SEED})')
else:
    df = df_full.reset_index(drop=True)

n = len(df)
y = df[TARGET_COL].astype(int).values
print(f'Working n={n}, positive (glass-former) rate: {y.mean()*100:.2f}%')


# In[6]:


# Cell 2.2 - composition parsing helpers (pymatgen if available, regex fallback)
if _OPT['pymatgen'] is not None:
    from pymatgen.core import Composition as _PMG_Comp
    def parse_composition(s):
        try:
            c = _PMG_Comp(s)
            return {str(el): float(v) for el, v in c.fractional_composition.as_dict().items()}
        except Exception:
            return None
else:
    _ELEM_RE = re.compile(r'([A-Z][a-z]?)(\(?[\d\.]*\)?)?')
    def parse_composition(s):
        """Naive composition parser: handles forms like 'Al10Co23B17',
        'Al(NiB)2', 'Fe60Cr20Ni20'. Falls back to flat element fractions."""
        try:
            def _expand(m):
                inner, mult = m.group(1), float(m.group(2) or 1)
                parts = re.findall(r'([A-Z][a-z]?)([\d\.]*)', inner)
                return ''.join(f'{el}{(float(cnt) if cnt else 1)*mult:g}' for el, cnt in parts)
            while '(' in s:
                s = re.sub(r'\(([A-Za-z0-9.]+)\)([\d\.]+)', _expand, s, count=1)
            parts = re.findall(r'([A-Z][a-z]?)([\d\.]*)', s)
            counts = {}
            for el, cnt in parts:
                if not el: continue
                counts[el] = counts.get(el, 0.0) + (float(cnt) if cnt else 1.0)
            total = sum(counts.values())
            if total <= 0: return None
            return {el: c / total for el, c in counts.items()}
        except Exception:
            return None

parsed = [parse_composition(s) for s in df[COMP_COL]]
valid  = np.array([p is not None for p in parsed])
if not valid.all():
    print(f'Dropping {(~valid).sum()} unparseable compositions')
    df = df[valid].reset_index(drop=True)
    y  = y[valid]
    parsed = [p for p in parsed if p is not None]
    n = len(df)

ALL_ELEMENTS = sorted(set(el for p in parsed for el in p.keys()))
print(f'Distinct elements observed: {len(ALL_ELEMENTS)}')
print(f'Example: {parsed[1]}')


# In[7]:


# Cell 2.3 - element-fraction matrix (one column per element)
def _element_matrix(parsed_list, elements):
    mat = np.zeros((len(parsed_list), len(elements)), dtype=np.float32)
    el_to_idx = {e: i for i, e in enumerate(elements)}
    for i, p in enumerate(parsed_list):
        for el, frac in p.items():
            if el in el_to_idx:
                mat[i, el_to_idx[el]] = frac
    return mat

X_frac = _element_matrix(parsed, ALL_ELEMENTS)
print(f'Element-fraction matrix: shape={X_frac.shape}, sparsity={(X_frac == 0).mean():.4f}')

n_elements_in_comp = (X_frac > 0).sum(axis=1)
print(f'Constituent count distribution: '
      f'{Counter(n_elements_in_comp.tolist())}')


# In[8]:


# Cell 2.4 - EDA panels (target balance, constituent count, element popularity)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# (a) class balance
class_counts = df[TARGET_COL].value_counts()
axes[0,0].bar(['non-glass-former', 'glass-former'],
              [class_counts.get(False, 0), class_counts.get(True, 0)],
              color=['#CC6677', '#4477AA'], edgecolor='black')
axes[0,0].set_ylabel('count')
axes[0,0].set_title(f'gfa target balance (positive rate={y.mean()*100:.1f}%)')

# (b) constituent count
cnt_counts = Counter(n_elements_in_comp.tolist())
ks = sorted(cnt_counts.keys())
axes[0,1].bar(ks, [cnt_counts[k] for k in ks], color='#332288', edgecolor='black')
axes[0,1].set_xlabel('# distinct elements per composition')
axes[0,1].set_ylabel('count')
axes[0,1].set_title('Constituent-count distribution')

# (c) top elements by row count
top_n = 30
elem_count = (X_frac > 0).sum(axis=0)
order = np.argsort(-elem_count)[:top_n]
axes[1,0].barh([ALL_ELEMENTS[i] for i in order[::-1]],
               elem_count[order[::-1]], color='#117733', edgecolor='black')
axes[1,0].set_xlabel('# rows containing element')
axes[1,0].set_title(f'Top {top_n} elements by row presence')

# (d) class balance vs constituent count
per_k = {k: (y[n_elements_in_comp == k].mean()
             if (n_elements_in_comp == k).sum() > 0 else 0.0)
         for k in ks}
axes[1,1].bar(ks, [per_k[k] for k in ks], color='#CC6677', edgecolor='black')
axes[1,1].axhline(y.mean(), color='black', ls='--', lw=1, label='pooled positive rate')
axes[1,1].set_xlabel('# distinct elements')
axes[1,1].set_ylabel('positive rate')
axes[1,1].set_title('GFA rate by constituent count')
axes[1,1].legend()

plt.tight_layout()
save_fig(fig, 'eda_panel', where='si')
plt.show()


# ## 3. Featurization — element-fraction baseline, Magpie features (optional), affinity, cost

# In[9]:


# Cell 3.1 - chemistry-system identifiers (sorted unique element set)
chem_systems = ['-'.join(sorted(p.keys())) for p in parsed]
df['_chem_system'] = chem_systems
chem_system_id = pd.factorize(chem_systems)[0].astype(np.int64)

print(f'Distinct chemical systems: {len(np.unique(chem_system_id))}')
sysctr = Counter(chem_systems)
top10 = sysctr.most_common(10)
print('Top 10 systems by row count:')
for s, c in top10:
    print(f'  {s:30s} {c:5d}')


# In[10]:


# Cell 3.2 - element family classification 
ELEMENT_FAMILY = {}
def _add(family, elems):
    for e in elems: ELEMENT_FAMILY[e] = family

_add('alkali',       ['Li','Na','K','Rb','Cs','Fr'])
_add('alkaline_earth',['Be','Mg','Ca','Sr','Ba','Ra'])
_add('transition_metal',
     ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
      'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
      'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
      'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn'])
_add('post_transition',['Al','Ga','In','Tl','Sn','Pb','Bi','Po'])
_add('metalloid',    ['B','Si','Ge','As','Sb','Te'])
_add('nonmetal',     ['H','C','N','O','P','S','Se','F','Cl','Br','I','At'])
_add('noble_gas',    ['He','Ne','Ar','Kr','Xe','Rn'])
_add('lanthanide',   ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'])
_add('actinide',     ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'])

def elements_to_family_signature(elements):
    fams = sorted(set(ELEMENT_FAMILY.get(e, 'unknown') for e in elements))
    return '|'.join(fams)

family_signature = [elements_to_family_signature(p.keys()) for p in parsed]
df['_family_sig'] = family_signature
family_sig_id = pd.factorize(family_signature)[0].astype(np.int64)
print(f'Distinct family signatures: {len(np.unique(family_sig_id))}')

# Constituent-count bin
df['_constituent_count'] = n_elements_in_comp


# In[11]:


# Cell 3.3 - Magpie features (matminer) or element-fraction fallback
CACHE_FEAT = OUT_INTER / 'featurization' / f'features_n{n}.npy'

if _OPT['matminer'] is not None and _OPT['pymatgen'] is not None:
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition as _PMGc
        if CACHE_FEAT.exists():
            X_feat = np.load(CACHE_FEAT)
            print(f'Loaded cached Magpie features from {CACHE_FEAT.name}')
        else:
            print('Computing Magpie features via matminer...')
            ep = ElementProperty.from_preset('magpie')
            comps = [_PMGc(s) for s in df[COMP_COL]]
            feat_list = []
            for c in comps:
                try:
                    feat_list.append(ep.featurize(c))
                except Exception:
                    feat_list.append([np.nan] * len(ep.feature_labels()))
            X_feat = np.asarray(feat_list, dtype=np.float64)
            # impute median per column
            for j in range(X_feat.shape[1]):
                col = X_feat[:, j]
                m = np.isfinite(col)
                if m.any():
                    X_feat[~m, j] = np.median(col[m])
                else:
                    X_feat[:, j] = 0.0
            np.save(CACHE_FEAT, X_feat)
            print(f'Magpie features: shape={X_feat.shape}')
    except Exception as exc:
        print(f'matminer Magpie featurization failed ({exc}); using element-fraction fallback')
        X_feat = X_frac.copy()
        np.save(CACHE_FEAT, X_feat)
else:
    print('matminer/pymatgen unavailable -- using element-fraction features')
    X_feat = X_frac.copy()
    np.save(CACHE_FEAT, X_feat)

# Scale features for affinity and ML
scaler = StandardScaler()
X_feat_scaled = scaler.fit_transform(X_feat).astype(np.float32)
print(f'Scaled feature matrix: shape={X_feat_scaled.shape}')


# In[12]:


# Cell 3.4 - Affinity matrices + k-means composition clustering

from sklearn.cluster import KMeans

def _knn_sparsify(A, k):
    n_ = A.shape[0]; kk = min(k, n_ - 1)
    knn_idx = np.argpartition(-A, kth=kk, axis=1)[:, :kk]
    out = np.zeros_like(A)
    rr = np.repeat(np.arange(n_), kk); cc = knn_idx.reshape(-1)
    out[rr, cc] = A[rr, cc]
    return np.maximum(out, out.T)

A_full    = cosine_similarity(X_feat_scaled).astype(np.float32)
np.fill_diagonal(A_full, 0.0)
AM_cosine = _knn_sparsify(A_full, KNN_AFFINITY_K)
np.save(OUT_INTER / 'featurization' / 'affinity_full.npy',  A_full)
np.save(OUT_INTER / 'featurization' / 'affinity_knn.npy',   AM_cosine)
print(f'Magpie cosine affinity: shape={A_full.shape}')

print(f'K-means on X_frac (k={H_KMEANS_K}, shape={X_frac.shape})...')
_km = KMeans(n_clusters=H_KMEANS_K, n_init=10, max_iter=300,
             random_state=RANDOM_SEED)
kmeans_labels    = _km.fit_predict(X_frac).astype(np.int64)
kmeans_centroids = _km.cluster_centers_.astype(np.float32)
_csizes = np.bincount(kmeans_labels, minlength=H_KMEANS_K)
print(f'  Cluster sizes: min={_csizes.min()}  mean={_csizes.mean():.1f}  '
      f'max={_csizes.max()}  singletons={(_csizes==1).sum()}')
np.save(OUT_INTER / 'featurization' / 'kmeans_labels.npy',    kmeans_labels)
np.save(OUT_INTER / 'featurization' / 'kmeans_centroids.npy', kmeans_centroids)

_xf_norms = np.linalg.norm(X_frac, axis=1, keepdims=True).clip(1e-12)
X_frac_n  = (X_frac / _xf_norms).astype(np.float32)
A_frac    = (X_frac_n @ X_frac_n.T).astype(np.float32)
np.fill_diagonal(A_frac, 0.0)

_same_cl = (kmeans_labels[:, None] == kmeans_labels[None, :])
A_frac[_same_cl] = 0.0
del _same_cl

AM = _knn_sparsify(A_frac, KNN_AFFINITY_K)
del A_frac
np.save(OUT_INTER / 'featurization' / 'affinity_xfrac_cross.npy', AM)

_sc2 = (kmeans_labels[:, None] == kmeans_labels[None, :])
_ws  = float(AM[_sc2].max())  if _sc2.any() else 0.0
_xs  = float(AM[~_sc2].mean()) if (~_sc2).any() else 0.0
del _sc2
print(f'\nCross-cluster cosine k-NN (new AM / M):')
print(f'  AM non-zero: {float((AM>0).mean())*100:.2f}%  max={float(AM.max()):.4f}')
print(f'  Within-cluster AM max (must be 0): {_ws:.6f}  '
      f'{"OK" if _ws < 1e-6 else "FAIL"}')
print(f'  Cross-cluster AM mean: {_xs:.6f}')


# In[13]:


# Cell 3.5 - cost matrix (1 - cosine), pooled target marginal for D-match
C = (1.0 - A_full).astype(np.float32)
C = np.clip((C + C.T) / 2.0, 0.0, 1.0)
np.fill_diagonal(C, 0.0)
np.save(OUT_INTER / 'featurization' / 'cost_matrix.npy', C)
print(f'Cost matrix: shape={C.shape}, mean={C.mean():.4f}')

TARGET_MARG = np.full((N_FOLDS, n), 1.0 / n, dtype=np.float32)


# ## 4. single-level k-means hierarchy with one lambda

# In[14]:


# Cell 4.1 - hierarchy builder (H term)

def build_hierarchy(df_local, lambda_per_level):
    '''Single-level k-means composition hierarchy.

    lambda_per_level = [lambda_clusters]  -- single value.
    Groups come from kmeans_labels (computed in Cell 3.4).
    Inter-group similarity uses centroid cosine in X_frac space.
    '''
    desc = [[] for _ in range(H_KMEANS_K)]
    for i, g in enumerate(kmeans_labels):
        desc[int(g)].append(i)

    _norms = np.linalg.norm(kmeans_centroids, axis=1, keepdims=True).clip(1e-12)
    _C_n   = kmeans_centroids / _norms
    C_sim  = (_C_n @ _C_n.T).clip(0.0, 1.0).astype(np.float32)
    np.fill_diagonal(C_sim, 0.0)

    ii, jj = np.where(
        (C_sim > H_CENTROID_COSINE_THRESH) &
        (np.arange(H_KMEANS_K)[:, None] < np.arange(H_KMEANS_K)[None, :])
    )
    if len(ii) == 0:
        rs = cs = np.array([], np.int64)
        vs = np.array([], np.float64)
    else:
        vals = C_sim[ii, jj].astype(np.float64)
        rs   = np.concatenate([ii, jj]).astype(np.int64)
        cs   = np.concatenate([jj, ii]).astype(np.int64)
        vs   = np.concatenate([vals, vals])

    return [
        {'name': 'kmeans_clusters', 'descendants': desc,
         'similarity': (rs, cs, vs),
         'lambda': float(lambda_per_level[0])},
    ]

hierarchy_main = build_hierarchy(df, LAMBDA_MAIN)
for L in hierarchy_main:
    _ra, _ca, _va = L['similarity']
    _n_edges = len(_va) // 2
    _vn = np.asarray(_va)
    _grp_sz = [len(d) for d in L['descendants']]
    print(f"  {L['name']:18s}: {len(L['descendants'])} groups  "
          f"{_n_edges} cosine edges (>{H_CENTROID_COSINE_THRESH})  "
          f"max={float(_vn.max()) if _n_edges else 0:.4f}  "
          f"mean={float(_vn.mean()) if _n_edges else 0:.4f}  "
          f"lambda={L['lambda']}")
    print(f"    group sizes: min={min(_grp_sz)}  mean={np.mean(_grp_sz):.1f}  "
          f"max={max(_grp_sz)}  singletons={sum(s==1 for s in _grp_sz)}")
with open(OUT_INTER / 'featurization' / 'hierarchy_main.pkl', 'wb') as fh:
    pickle.dump(hierarchy_main, fh)


# In[15]:


# Cell 4.2 - H term visuals: k-means composition clusters on t-SNE + inter-cluster centroid-cosine edges.

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.patches as mpatches

print('Computing t-SNE (PCA 50 dims, perplexity=40)...')
_n_pca = min(50, X_feat_scaled.shape[1])
X_pca  = PCA(n_components=_n_pca, random_state=RANDOM_SEED).fit_transform(X_feat_scaled)
XY     = TSNE(n_components=2, perplexity=40, n_iter=1000,
              learning_rate='auto', init='pca',
              random_state=RANDOM_SEED).fit_transform(X_pca)
print(f'  t-SNE done: {XY.shape}')

# 2. Two-panel t-SNE 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): k-means cluster ID
ax = axes[0]
sc1 = ax.scatter(XY[:, 0], XY[:, 1],
                 c=kmeans_labels,
                 cmap=plt.cm.get_cmap('gist_rainbow', H_KMEANS_K),
                 vmin=0, vmax=H_KMEANS_K - 1,
                 s=7, alpha=0.60, linewidths=0)
plt.colorbar(sc1, ax=ax, fraction=0.035, pad=0.02,
             label=f'k-means cluster ID (k={H_KMEANS_K})')
ax.set_title(f'H — k-means clusters (k={H_KMEANS_K}) on X_frac\n'
             f'GOOD: compact same-color blobs; each = one composition neighbourhood',
             fontsize=10)
ax.axis('off')
_desc_km  = hierarchy_main[0]['descendants']
_hull_n   = 0
for _gi, _desc in enumerate(_desc_km):
    if len(_desc) < 15: continue
    _pts = XY[_desc]
    try:
        _h = ConvexHull(_pts)
        _hp = np.vstack([_pts[_h.vertices], _pts[_h.vertices[0]]])
        ax.plot(_hp[:, 0], _hp[:, 1], lw=0.9, alpha=0.35, color='black')
        _hull_n += 1
    except Exception:
        pass
_szs = [len(d) for d in _desc_km]
ax.text(0.02, 0.02,
        f'{_hull_n} clusters (>=15 comps) outlined  |  '
        f'sizes: min={min(_szs)} mean={np.mean(_szs):.0f} max={max(_szs)}',
        transform=ax.transAxes, fontsize=7, va='bottom')

ax = axes[1]
ax.scatter(XY[:, 0], XY[:, 1],
           c=chem_system_id, cmap='tab20', s=7, alpha=0.55, linewidths=0)
ax.set_title('Reference: chemistry system (element-set identity)\n'
             'Compare to Panel (a): do cluster boundaries align with element sets?',
             fontsize=10)
ax.axis('off')

plt.suptitle('H term verification: k-means composition clusters on t-SNE\n'
             f'k={H_KMEANS_K} clusters on X_frac. Each cluster = composition neighbourhood.',
             fontsize=10, y=1.01)
plt.tight_layout()
save_fig(fig, 'H_tsne_kmeans_clusters', where='si')
plt.show()

# 3. Inter-cluster cosine edges on t-SNE 
_hier_obj = build_hierarchy_levels(hierarchy_main)
_L0_level = _hier_obj[0]
_rows_h   = _L0_level.similarity_rows
_cols_h   = _L0_level.similarity_cols
_vals_h   = _L0_level.similarity_vals
_k        = len(_desc_km)

# Cluster centroids in t-SNE space
_cents_tsne = np.array([XY[np.array(d)].mean(axis=0) if d else np.zeros(2)
                         for d in _desc_km])

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(XY[:, 0], XY[:, 1], c=kmeans_labels,
           cmap=plt.cm.get_cmap('gist_rainbow', H_KMEANS_K),
           vmin=0, vmax=H_KMEANS_K-1,
           s=5, alpha=0.22, linewidths=0, zorder=1)

_TOP  = min(200, len(_vals_h) // 2)
_ord  = np.argsort(-_vals_h)[:_TOP * 2]
_drw  = set()
_cmap_h = plt.get_cmap('Blues')
for _ei in _ord:
    _g, _gp = int(_rows_h[_ei]), int(_cols_h[_ei])
    if _g >= _gp or (_g, _gp) in _drw or _g >= _k or _gp >= _k: continue
    _drw.add((_g, _gp))
    if len(_drw) > _TOP: break
    _v  = float(_vals_h[_ei])
    _p1, _p2 = _cents_tsne[_g], _cents_tsne[_gp]
    ax.plot([_p1[0], _p2[0]], [_p1[1], _p2[1]], '-',
            color=_cmap_h(_v), lw=0.5 + 2.0 * _v, alpha=max(0.3, _v), zorder=2)

_sm = ScalarMappable(cmap='Blues', norm=Normalize(0, 1))
_sm.set_array([])
plt.colorbar(_sm, ax=ax, fraction=0.025, pad=0.01,
             label='Inter-cluster cosine (X_frac centroids)')
_n_edges_h = len(_vals_h) // 2
ax.set_title(f'H inter-cluster cosine edges (top {_TOP} of {_n_edges_h} total)\n'
             f'threshold={H_CENTROID_COSINE_THRESH} | '
             f'GOOD: edges connect adjacent t-SNE blobs (short-range)',
             fontsize=10)
ax.axis('off')
plt.tight_layout()
save_fig(fig, 'H_tsne_cluster_edges', where='si')
plt.show()

# 4. Summary 
_vn = np.asarray(_vals_h)
print(f'H summary:')
print(f'  Clusters: {_k}  |  Inter-cluster edges: {_n_edges_h}')
print(f'  Cosine range: [{float(_vn.min()) if _n_edges_h else 0:.3f},'
      f' {float(_vn.max()) if _n_edges_h else 0:.3f}]  '
      f'mean={float(_vn.mean()) if _n_edges_h else 0:.3f}')
print(f'  Cluster sizes: min={min(_szs)}  mean={np.mean(_szs):.1f}  max={max(_szs)}')
print(f'  Test fold can hold ~{int(n*0.2/np.mean(_szs))} complete clusters out of {_k}.')


# In[16]:


# Cell 4.3 - M term verification: cross-cluster cosine k-NN on X_frac.

# 1. Cluster-level summary
_nrm    = np.linalg.norm(kmeans_centroids, axis=1, keepdims=True).clip(1e-12)
C_sim   = (kmeans_centroids / _nrm) @ (kmeans_centroids / _nrm).T
C_sim   = C_sim.clip(0.0, 1.0).astype(np.float32)
np.fill_diagonal(C_sim, 0.0)

# Cluster centroids in t-SNE space
_cents_m = np.array([XY[kmeans_labels == _g].mean(axis=0)
                      for _g in range(H_KMEANS_K)])
_szs_m   = np.array([(kmeans_labels == _g).sum() for _g in range(H_KMEANS_K)])

_n_nonzero = int((C_sim > 0).sum()) // 2
print(f'M cross-cluster cosine (X_frac space):')
print(f'  Clusters: {H_KMEANS_K}  |  cross-cluster pairs with cosine>0: {_n_nonzero}')
print(f'  C_sim max={float(C_sim.max()):.4f}  '
      f'mean(>0)={float(C_sim[C_sim>0].mean()) if _n_nonzero else 0:.4f}')
print(f'  AM non-zero: {float((AM>0).mean())*100:.2f}%  '
      f'max={float(AM.max()):.4f}')

# 2. Two-panel edge plots 
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
_cmap_m = plt.get_cmap('YlOrRd')

for _ax_i, _n_top in enumerate([50, 150]):
    _ax = axes[_ax_i]
    # Background: all compositions
    _ax.scatter(XY[:, 0], XY[:, 1], c=kmeans_labels,
                cmap=plt.cm.get_cmap('gist_rainbow', H_KMEANS_K),
                vmin=0, vmax=H_KMEANS_K-1,
                s=4, alpha=0.18, linewidths=0, zorder=1)
    # Cluster centroids (sized by cluster size)
    _ax.scatter(_cents_m[:, 0], _cents_m[:, 1],
                c=np.arange(H_KMEANS_K),
                cmap=plt.cm.get_cmap('gist_rainbow', H_KMEANS_K),
                vmin=0, vmax=H_KMEANS_K-1,
                s=(6 + 1.5 * _szs_m).clip(6, 100),
                alpha=0.9, edgecolors='black', linewidths=0.4, zorder=3)
    # Draw top-N cluster-level M edges
    _flat = np.argsort(-C_sim.ravel())
    _drwm, _skip_m = 0, set()
    for _fi in _flat:
        _r, _cp = divmod(int(_fi), H_KMEANS_K)
        if _r >= _cp or (_r, _cp) in _skip_m: continue
        _v = float(C_sim[_r, _cp])
        if _v <= 0: break
        if _drwm >= _n_top: break
        _skip_m.add((_r, _cp))
        _p1, _p2 = _cents_m[_r], _cents_m[_cp]
        _ax.plot([_p1[0], _p2[0]], [_p1[1], _p2[1]], '-',
                 color=_cmap_m(_v), lw=0.3 + 1.7*_v, alpha=max(0.25, _v), zorder=2)
        _drwm += 1
    _ax.set_title(f'M — cross-cluster cosine edges (top {_n_top})\n'
                  f'Node=cluster centroid, size~cluster size | '
                  f'GOOD: edges connect nearby t-SNE nodes',
                  fontsize=9)
    _ax.axis('off')

plt.suptitle('M term verification: cross-cluster cosine k-NN on X_frac\n'
             'Within-cluster AM = 0 (H handles them); cross-cluster edges capture near-system leakage.',
             fontsize=10, y=1.01)
plt.tight_layout()
save_fig(fig, 'M_tsne_xfrac_cross_cluster', where='si')
plt.show()

# 3. Cluster cosine heatmap + distribution 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Heatmap of inter-cluster cosine (k x k)
_im = axes[0].imshow(C_sim, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
axes[0].set_xlabel('Cluster ID'); axes[0].set_ylabel('Cluster ID')
axes[0].set_title(f'Inter-cluster cosine (X_frac centroids, k={H_KMEANS_K})\n'
                  'Block structure = composition families (groups of related clusters)',
                  fontsize=9)
plt.colorbar(_im, ax=axes[0], fraction=0.04)

# (b) Distribution of non-zero inter-cluster cosine values
_cv = C_sim[C_sim > 0]
if len(_cv):
    axes[1].hist(_cv, bins=30, color='#4477AA', edgecolor='white', alpha=0.85)
    axes[1].axvline(H_CENTROID_COSINE_THRESH, color='#CC3311', lw=1.5, ls='--',
                    label=f'H threshold={H_CENTROID_COSINE_THRESH}')
    axes[1].set_xlabel('Inter-cluster cosine similarity (X_frac)', fontsize=9)
    axes[1].set_ylabel('# cluster pairs', fontsize=9)
    axes[1].set_title('M — inter-cluster cosine distribution\n'
                      'Pairs above H_CENTROID_COSINE_THRESH also have H edges',
                      fontsize=9)
    axes[1].legend(fontsize=8)
plt.tight_layout()
save_fig(fig, 'M_cluster_cosine_heatmap', where='si')
plt.show()

# 4. Sanity checks 
_sc3 = (kmeans_labels[:, None] == kmeans_labels[None, :])
_ws3 = float(AM[_sc3].max()) if _sc3.any() else 0.0
_xs3 = float(AM[~_sc3].mean()) if (~_sc3).any() else 0.0
del _sc3
print(f'\nM sanity check:')
print(f'  Within-cluster AM max  (must be 0): {_ws3:.8f}  '
      f'{"OK" if _ws3 < 1e-6 else "FAIL"}')
print(f'  Cross-cluster AM mean             : {_xs3:.6f}')
print(f'  AM max                            : {float(AM.max()):.6f}')
print(f'\nSummary: H and M are now BOTH defined in X_frac composition space.')
print(f'  H: {H_KMEANS_K} k-means clusters, centroid cosine similarity.')
print(f'  M: cross-cluster cosine k-NN (k={KNN_AFFINITY_K}) on X_frac.')
print(f'  Both are locally coherent (no long-range edges by construction).')


# In[17]:


# Cell 4.4 — D term (D-match) verification: OT cost structure and distributional properties on t-SNE.

# 1. Cost statistics 
_within_costs, _cross_sample = [], []
_rng_d = np.random.default_rng(RANDOM_SEED)
for _s in df['_chem_system'].unique():
    _idx = np.where(df['_chem_system'].values == _s)[0]
    if len(_idx) > 1:
        _sub = C[np.ix_(_idx, _idx)]
        _within_costs.extend(_sub[np.triu_indices(len(_idx), k=1)].tolist())
# Sample cross-system costs
_ii = _rng_d.integers(0, n, 80_000)
_jj = _rng_d.integers(0, n, 80_000)
_cross_mask = df['_chem_system'].values[_ii] != df['_chem_system'].values[_jj]
_cross_sample = C[_ii[_cross_mask], _jj[_cross_mask]].tolist()[:30_000]

print('D-match cost matrix (C = 1 - cosine on Magpie features):')
print(f'  Shape: {C.shape}  |  range: [{C.min():.4f}, {C.max():.4f}]')
if _within_costs:
    print(f'  Within-system mean cost : {np.mean(_within_costs):.4f}')
if _cross_sample:
    print(f'  Cross-system mean cost  : {np.mean(_cross_sample):.4f}')
    print(f'  Separation ratio: {np.mean(_cross_sample)/max(np.mean(_within_costs),1e-9):.2f}x')
print(f'  Target marginals: uniform 1/n = {1.0/n:.6f} per support point')
print(f'  Fold targets: {dict(enumerate(np.round(np.array(FOLD_RATIOS)*n).astype(int)))}')

# 2. Four-panel D verification figure 
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
ax.scatter(XY[y==0, 0], XY[y==0, 1], c='#CC6677', s=6, alpha=0.40,
           label=f'non-glass-former (y=0, n={int((y==0).sum())})')
ax.scatter(XY[y==1, 0], XY[y==1, 1], c='#4477AA', s=6, alpha=0.40,
           label=f'glass-former (y=1, n={int((y==1).sum())})')
ax.set_title('D — GFA target distribution in t-SNE\n'
             'D-match will ensure each fold has balanced class and feature coverage',
             fontsize=9)
ax.legend(fontsize=7, markerscale=2)
ax.axis('off')

ax = axes[0, 1]
_C_row_mean = C.mean(axis=1)
sc_cost = ax.scatter(XY[:, 0], XY[:, 1], c=_C_row_mean,
                     cmap='plasma', s=6, alpha=0.60, linewidths=0)
plt.colorbar(sc_cost, ax=ax, fraction=0.035, label='mean cost to all others')
ax.set_title('D — Mean OT transport cost per composition\n'
             'High cost = isolated in composition space (costly to cover in one fold)',
             fontsize=9)
ax.axis('off')

# Panel (c): within-system vs cross-system cost histograms
ax = axes[1, 0]
if _within_costs:
    ax.hist(_within_costs, bins=40, alpha=0.65, color='#4477AA',
            density=True, label=f'Within-system (n={len(_within_costs):,})')
if _cross_sample:
    ax.hist(_cross_sample, bins=40, alpha=0.65, color='#CC6677',
            density=True, label=f'Cross-system (sample {len(_cross_sample):,})')
ax.set_xlabel('OT cost C_{ij} = 1 - cosine(features)', fontsize=9)
ax.set_ylabel('density', fontsize=9)
ax.set_title('D — cost distribution: within- vs cross-system\n'
             'Good OT landscape: within-cost << cross-cost',
             fontsize=9)
ax.legend(fontsize=8)

# Panel (d): chemistry-system size distribution
ax = axes[1, 1]
_sys_sizes_all = np.array([(df['_chem_system'].values == s).sum()
                            for s in df['_chem_system'].unique()])
_sys_sizes_srt = np.sort(_sys_sizes_all)[::-1]
_too_large     = _sys_sizes_srt > n * FOLD_RATIOS[1]
_colors_bar    = ['#CC3311' if t else '#4477AA' for t in _too_large]
ax.bar(range(len(_sys_sizes_srt)), _sys_sizes_srt,
       color=_colors_bar, width=1.0, edgecolor='none')
ax.axhline(n * FOLD_RATIOS[1], color='black', ls='--', lw=1.5,
           label=f'20% fold cap ({int(n*FOLD_RATIOS[1])} comps)')
ax.set_xlabel('Chemistry systems (sorted by size)', fontsize=9)
ax.set_ylabel('# compositions', fontsize=9)
ax.set_title('D — System size distribution\n'
             'Red = too large to fit entirely in test fold;\n'
             'D ensures distributional coverage despite H/balance constraints',
             fontsize=9)
ax.legend(fontsize=8)
# Add summary annotation
_n_too_large = int(_too_large.sum())
ax.text(0.98, 0.98,
        f'{_n_too_large} systems too large for full test-fold assignment',
        transform=ax.transAxes, ha='right', va='top', fontsize=7,
        color='#CC3311')

plt.suptitle('D term verification: OT cost structure and distributional properties\n'
            'D-match (uniform target) ensures representative fold coverage across all folds.',
             fontsize=10, y=1.01)
plt.tight_layout()
save_fig(fig, 'D_term_cost_structure', where='si')
plt.show()

# 3. D-term configuration summary 
print(f'\nD-match configuration summary:')
print(f'  Mode: {DIST_MODE}  |  epsilon_OT (reg): {OT_REG}')
print(f'  Cost: 1-cosine on scaled Magpie features')
print(f'  Target: uniform 1/n for each of {n} support points (each fold)')
print(f'  D gradient: OT envelope theorem — pushes each composition toward')
print(f'    the fold most under-represented in its feature-space neighbourhood.')
print(f'  Role in multi-term SHIELD: distributional diversity regularizer;')
print(f'    prevents H+M from collapsing large systems into one fold.')


# ## 5. Baseline splitters — random, stratified, leave-chemistry-out, KMeans, matbench default CV

# In[18]:


# Cell 5.1 - baseline splitter implementations
from sklearn.cluster import KMeans

def split_random(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(FOLD_RATIOS[0] * n))
    f = np.ones(n, dtype=int); f[perm[:cut]] = 0
    return f

def split_class_stratified(seed):
    rng = np.random.default_rng(seed)
    f = np.ones(n, dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = int(round(FOLD_RATIOS[0] * len(idx)))
        f[idx[:cut]] = 0
    return f

def split_leave_chem_system_out(seed):
    rng = np.random.default_rng(seed)
    sys_to_idx = defaultdict(list)
    for i, s in enumerate(df['_chem_system']):
        sys_to_idx[s].append(i)
    order = list(sys_to_idx.values()); rng.shuffle(order)
    f = np.ones(n, dtype=int); cum = 0
    target_train = int(round(FOLD_RATIOS[0] * n))
    for g in order:
        if cum + len(g) <= target_train:
            for j in g: f[j] = 0
            cum += len(g)
        else:
            break
    return f

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

def split_kmeans(seed, n_clusters=20):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(X_feat_scaled)
    return _greedy_pack(km.labels_, n_clusters)


def split_matbench_default_cv(seed):
    """Matbench's default split is 5-fold CV; for a single-train/test
    comparison we take fold 0 of an n_splits=5 StratifiedKFold as test."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for _, test_idx in skf.split(np.arange(n).reshape(-1, 1), y):
        f = np.zeros(n, dtype=int); f[test_idx] = 1
        return f

BASELINE_SPLITTERS = {
    'RANDOM':                lambda seed=RANDOM_SEED: split_random(seed),
    'CLASS_STRATIFIED':      lambda seed=RANDOM_SEED: split_class_stratified(seed),
    'MATBENCH_DEFAULT_CV':   lambda seed=RANDOM_SEED: split_matbench_default_cv(seed),
    'LEAVE_CHEM_SYSTEM_OUT': lambda seed=RANDOM_SEED: split_leave_chem_system_out(seed),
    'KMEANS':                lambda seed=RANDOM_SEED: split_kmeans(seed),
}


# In[19]:


# Cell 5.2 - run all baselines
baseline_results = {}
for name, fn in BASELINE_SPLITTERS.items():
    f = fn()
    sizes = np.bincount(f, minlength=N_FOLDS) / n
    baseline_results[name] = f
    pd.DataFrame({'index': np.arange(n), 'fold': f}).to_csv(
        OUT_INTER / 'splits' / f'{name}.csv', index=False)
    print(f'  {name:22s}  sizes={sizes.round(3).tolist()}')


# ## 6. SHIELD primary configurations and Stage-0 normalizers

# In[20]:


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


# In[21]:


# Cell 6.2 - run all SHIELD configurations
shield_results = {}
shield_diagnostics = {}
t_all = time.time()
CACHED_NORMALIZERS = None


Y_TARGET = y.reshape(-1, 1).astype(np.float32)

for cfg_name, cfg in SHIELD_CONFIGS.items():
    t0 = time.time()
    kwargs = dict(
        n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
        hierarchy=hierarchy_main if cfg['alpha'] > 0 else None,
        affinity=AM if cfg['beta'] > 0 else None,
        affinity_provenance='label-blind-handcrafted' if cfg['beta'] > 0 else None,
        d_mode=DIST_MODE,
        cost=C if cfg['gamma'] > 0 else None,
        target_marginals=TARGET_MARG if cfg['gamma'] > 0 else None,
        y_target=Y_TARGET,
        class_labels=y.astype(np.int64),
        stratification_strategy=STRAT_STRATEGY if cfg['mu'] > 0 else 'none',
        n_bins=STRAT_N_BINS,
        class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
        alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'],
        eta=cfg['eta'], mu=cfg['mu'], nu=NU, tau=TAU,
        epsilon_OT=OT_REG,
        max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
        integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
        rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
        enforce_class_delta_after_round=True,
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
        print(f'  {cfg_name:18s}  wall={time.time()-t0:.1f}s  '
              f'sizes={(np.bincount(fold, minlength=N_FOLDS)/n).round(3).tolist()}')
    except Exception as exc:
        print(f'  {cfg_name:18s}  FAILED: {exc}')

print(f'Total SHIELD wall time: {time.time()-t_all:.1f}s')


# In[22]:


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
        'milp_invoked': md_.get('milp_polish', {}).get('invoked', False),
        'milp_reason':  md_.get('milp_polish', {}).get('trigger', ''),
    })
norm_df = pd.DataFrame(norm_rows)
save_table(norm_df, 'stage0_normalizers_and_milp', where='si')
norm_df


# ## 7. Leakage diagnostics

# In[23]:


# Cell 7.1 - cross-fold structural similarity L(pi)

A_pos = np.clip(A_full, 0.0, 1.0)

kappa = np.ones(A_pos.shape[0])

kappa_matrix = kappa.reshape(-1, 1) * kappa

TOTAL_SIM = float((A_pos * kappa_matrix).sum())

def cross_fold_similarity(fold, A=A_pos, kappa=kappa_matrix):
    """
    Calculate cross-fold similarity leak L(π) with cardinality weighting.

    Args:
        fold: array of fold assignments π(x) for each element
        A: similarity matrix (clipped to [0,1])
        kappa: cardinality weighting matrix κ(x) * κ(x')
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


# Cell 7.2 - L(pi) bar chart
fig, ax = plt.subplots(figsize=(11, 4))
agg = lpi_df.set_index('config')['scaled_L_pi'].sort_values()
colors = ['#888888' if c in baseline_results else '#4477AA' for c in agg.index]
ax.bar(range(len(agg)), agg.values, color=colors, edgecolor='black')
ax.axhline(agg.loc['RANDOM'] if 'RANDOM' in agg.index else 1.0,
           color='red', ls='--', lw=1, label='RANDOM baseline')
ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg.index, rotation=45, ha='right')
ax.set_ylabel('scaled L(pi)')
ax.set_title('Cross-fold composition-similarity (lower = stronger OOD)')
ax.legend()
plt.tight_layout()
save_fig(fig, 'L_pi_bar_chart', where='main')
plt.show()


# In[25]:


# Cell 7.3 - per-channel decomposition L_H, L_M, L_W
hier_levels_obj = build_hierarchy_levels(hierarchy_main)
AM_tensor = torch.from_numpy(AM).to(torch.float32)

def per_channel_leakage(fold):
    z = torch.zeros((n, N_FOLDS), dtype=torch.float32)
    z[np.arange(n), fold] = 1.0
    L_H = float(_evaluate_h_term_from_z(z, hier_levels_obj))
    L_M = float(cut_energy(z, AM_tensor))
    tr = y[fold == 0]; te = y[fold == 1]
    L_W = float(wasserstein_distance(tr.astype(float), te.astype(float))) \
          if len(tr) and len(te) else float('nan')
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


# Cell 7.4 - kNN purity in joint scaled feature space
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
                            'knn_purity': knn_purity(fold, X_feat_scaled, k)})
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


# In[27]:


# Cell 7.5 - alpha-shape IoU with improved visualization

emb2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_feat_scaled)
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

# Visualization with separate baseline methods and ALL SHIELD configs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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


# In[28]:


# Cell 7.6 - Train Potential Leakage (TPL)
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
        from sklearn.neighbors import NearestNeighbors
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
    from scipy.stats import mannwhitneyu
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
    tpl_val, tpl_lo, tpl_hi, (phi_tr, phi_te, h) = tpl(X_feat_scaled, fold)
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
ax.axhline(y=-0.1, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Sever leakage threshold')
ax.set_ylabel('TPL  [mean ± 95% CI]')
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

# In[29]:


# Cell 8.1 - embeddings
embeds = {}
try:
    embeds['tSNE'] = TSNE(n_components=2, perplexity=30, init='pca',
                          random_state=RANDOM_SEED).fit_transform(X_feat_scaled)
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


# In[88]:


# Cell 8.2 - split-colored scatter panels
showcase = ['RANDOM', 'CLASS_STRATIFIED', 'MATBENCH_DEFAULT_CV', 'LEAVE_CHEM_SYSTEM_OUT', 'KMEANS',
            'HMD-SHIELD', 'H-SHIELD', 'M-SHIELD', 'D-SHIELD', 'HM-SHIELD', 'HD-SHIELD', 'MD-SHIELD']
showcase = [s for s in showcase if s in baseline_results or s in shield_results]

CLASS_COLORS = {1: '#E66100', 0: '#5D3A9B'}

for emb_name, emb in embeds.items():
    total = len(showcase) + 1
    cols = min(4, total)
    rows = int(np.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    ax_idx = 0

    ax = axes[ax_idx]; ax_idx += 1
    for cls, label in [(1, 'glass-former (GFA=1)'), (0, 'non-glass-former (GFA=0)')]:
        mask = y == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=6,
                   c=CLASS_COLORS[cls], alpha=0.50, label=label)
    ax.set_title('GFA class labels', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='best', fontsize=7)

    for name in showcase:
        ax = axes[ax_idx]; ax_idx += 1
        fold = baseline_results.get(name, shield_results.get(name))
        ax.scatter(emb[fold == 0, 0], emb[fold == 0, 1], s=6, c='#4477AA',
                   alpha=0.55, label='train')
        ax.scatter(emb[fold == 1, 0], emb[fold == 1, 1], s=6, c='#CC6677',
                   alpha=0.55, label='test')
        scaled = lpi_df.set_index('config').loc[name, 'scaled_L_pi'] \
                 if name in lpi_df['config'].values else float('nan')
        ax.set_title(f'{name}\nscaled L(pi)={scaled:.3f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    axes[1].legend(loc='best', fontsize=7)

    for ax in axes[ax_idx:]: ax.set_visible(False)
    fig.suptitle(f'{emb_name} of Magpie features, colored by fold', fontsize=11)
    plt.tight_layout()
    save_fig(fig, f'dim_reduction_{emb_name.lower()}_splits',
             where='main' if emb_name in ('PCA','UMAP') else 'si')
    plt.show()


# ## 9. OOD Assessment Metrics

# In[31]:


# # Cell 9.1 - OOD characterization metrics per split

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
    Xtr_161 = X_feat_scaled[mask_tr_161]
    Xte_161 = X_feat_scaled[mask_te_161]

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


# In[32]:


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


# ## 10. SHIELD (alpha, beta, gamma) sweep, Pareto frontier, Hamming stability

# In[33]:


# Cell 10.1 - shield.sweep
sweep_kwargs = dict(
    n=n, K=N_FOLDS, r=np.array(FOLD_RATIOS, dtype=np.float32),
    hierarchy=hierarchy_main, affinity=AM,
    affinity_provenance='label-blind-handcrafted',
    d_mode=DIST_MODE, cost=C, target_marginals=TARGET_MARG,
    y_target=Y_TARGET, class_labels=y.astype(np.int64),
    stratification_strategy=STRAT_STRATEGY, n_bins=STRAT_N_BINS,
    class_delta=STRAT_TOL, balance_eps=BALANCE_TOL,
    eta=ETA, mu=MU, nu=NU, tau=TAU, epsilon_OT=OT_REG,
    max_iter=SHIELD_MAX_ITER, seed=RANDOM_SEED, device=SHIELD_DEVICE, nodes=N_JOBS,
    integer_z_final=True, init_backend=SHIELD_INIT_BACKEND,
    rounding_mode=SHIELD_ROUNDING, milp_size_limit=SHIELD_MILP_SIZE,
    stage0_samples=SHIELD_STAGE0_SAMPLES,
    precomputed_normalizers=CACHED_NORMALIZERS,
    sinkhorn_max_iter=SHIELD_SINKHORN_ITER,
)
print(f'Running shield.sweep over '
      f'{len(SWEEP_GRID["alpha"])*len(SWEEP_GRID["beta"])*len(SWEEP_GRID["gamma"])} '
      'configs...')
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


# In[34]:


# Cell 10.2 - sweep table + Pareto plot
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
            s=300, marker='*', color='gold', edgecolors='black', linewidths=1,
            label='recommended')
ax1.set_xlabel('M_norm'); ax1.set_ylabel('D_norm'); ax1.legend(fontsize=8)
ax1.set_title('Sweep: M vs D')

ax2.scatter(range(len(sweep_df)), sweep_df['composite_objective'],
            c=['gold' if x == sweep_res.recommended_index else '#4477AA'
               for x in sweep_df.index], s=60, edgecolors='black')
ax2.set_xlabel('config idx'); ax2.set_ylabel('composite objective')
ax2.set_title('Composite across sweep')

sns.heatmap(sweep_res.stability.pairwise_hamming, ax=ax3, cmap='magma',
            cbar_kws={'label': 'frac differing atoms'})
ax3.set_title(f'Hamming matrix ({len(sweep_res.stability.assignments)} points)')
plt.tight_layout()
save_fig(fig, 'sweep_pareto_hamming', where='si')
plt.show()