#!/usr/bin/env python3
"""
Table 4: H-ESFM True Distance-Rank Benchmark

Purpose
-------
Evaluates PPI score, initial embedding distance, final flow distance, and flow-improvement ranking on the H-ESFM true distance-rank benchmark.

Expected outputs
----------------
Writes cellB32_true_distance_rank_all_candidates.csv and cellB32_true_distance_rank_summary.csv.

Notes
-----
This file is standalone: it includes data loading, model setup/training, evaluation, and plotting required for this result. The modeling and metric code is kept identical to the implementation used for the manuscript artifact; only interactive shell/display syntax is adapted for Python execution.
"""



# ==============================================================================

# ============================================================
# CELL 1: Installs
# ============================================================
import subprocess
subprocess.run('pip install torch torchvision torchaudio --quiet', shell=True, check=True)
import subprocess
subprocess.run('pip install numpy pandas scikit-learn matplotlib seaborn tqdm --quiet', shell=True, check=True)
import subprocess
subprocess.run('pip install fair-esm --quiet', shell=True, check=True)
import subprocess
subprocess.run('pip install gdown --quiet', shell=True, check=True)
import subprocess
subprocess.run('pip install scipy --quiet', shell=True, check=True)
print('✓ Dependencies installed')


# ==============================================================================

# ============================================================
# CELL 2: Core imports
# ============================================================
import os
import json
import math
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, roc_curve,
    confusion_matrix
)

import esm

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './data'
os.makedirs(DATA_ROOT, exist_ok=True)

print(f'✓ Imports ready  |  device = {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')


# ==============================================================================

# ============================================================
# CELL 3: Download SHS27k from Zenodo
# ============================================================
ZENODO_FILES = {
    'SHS27k.actions.txt': 'https://zenodo.org/records/15694560/files/SHS27k.actions.txt?download=1',
    'SHS27k.seqs.tsv':    'https://zenodo.org/records/15694560/files/SHS27k.seqs.tsv?download=1',
}

for fname, url in ZENODO_FILES.items():
    dest = os.path.join(DATA_ROOT, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f'✓ {fname} already present ({os.path.getsize(dest)//1024} KB)')
    else:
        print(f'Downloading {fname} ...')
        subprocess.run(['wget', '-q', '--show-progress', '-O', dest, url], check=True)
        print(f'  ✓ saved ({os.path.getsize(dest)//1024} KB)')


# ==============================================================================

# ============================================================
# CELL 4: Parse SHS27k — loaders consistent with the benchmark implementation
# ============================================================
from tqdm.auto import tqdm

def load_tsv_seqs(path):
    seq_map = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                seq_map[parts[0].strip()] = parts[1].strip()
    print(f'  Loaded {len(seq_map)} sequences from {os.path.basename(path)}')
    return seq_map


def load_actions(path, score_threshold=0, max_rows=None):
    df = pd.read_csv(path, sep='\t', nrows=max_rows)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('item_id_a', 'protein1', 'id_a'):   rename[c] = 'id_a'
        elif cl in ('item_id_b', 'protein2', 'id_b'): rename[c] = 'id_b'
        elif cl == 'score':                            rename[c] = 'score'
    df = df.rename(columns=rename)
    if 'score' in df.columns and score_threshold > 0:
        df = df[df['score'] >= score_threshold]
    df = df[['id_a', 'id_b']].drop_duplicates()
    print(f'  Loaded {len(df)} action pairs from {os.path.basename(path)}')
    return df


def build_dataset(actions_df, seq_map, neg_ratio=1.0, max_pos=None, seed=42):
    rng = np.random.default_rng(seed)
    pos = []
    for _, row in actions_df.iterrows():
        a, b = row['id_a'], row['id_b']
        if a in seq_map and b in seq_map:
            pos.append((a, b, seq_map[a], seq_map[b], 1))
    if max_pos and len(pos) > max_pos:
        idx = rng.choice(len(pos), max_pos, replace=False)
        pos = [pos[i] for i in idx]
    print(f'  Positive pairs with sequences: {len(pos)}')
    pos_set = set((a, b) for a, b, *_ in pos)
    pos_set |= set((b, a) for a, b, *_ in pos)
    all_prots = list(seq_map.keys())
    n_neg = int(len(pos) * neg_ratio)
    neg = []
    attempts = 0
    while len(neg) < n_neg and attempts < n_neg * 30:
        a, b = rng.choice(all_prots, 2, replace=False)
        if (a, b) not in pos_set:
            neg.append((a, b, seq_map[a], seq_map[b], 0))
            pos_set.add((a, b))
        attempts += 1
    print(f'  Negative pairs sampled: {len(neg)}')
    rows = pos + neg
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=['id_a', 'id_b', 'seq_a', 'seq_b', 'label'])


def dfs_split(df, train_frac=0.70, val_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    adj = defaultdict(set)
    for _, row in df[df['label'] == 1].iterrows():
        adj[row['id_a']].add(row['id_b'])
        adj[row['id_b']].add(row['id_a'])
    proteins = list(set(df['id_a'].tolist() + df['id_b'].tolist()))
    rng.shuffle(proteins)
    visited, order = {}, []
    for start in proteins:
        if start not in visited:
            stack = [start]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited[node] = True
                    order.append(node)
                    nbrs = list(adj[node]); rng.shuffle(nbrs)
                    stack.extend(nbrs)
    n = len(order)
    tr_cut = int(n * train_frac)
    va_cut  = int(n * (train_frac + val_frac))
    rank = {'train': 0, 'val': 1, 'test': 2}
    split_map = {node: ('train' if i < tr_cut else 'val' if i < va_cut else 'test')
                 for i, node in enumerate(order)}
    df = df.copy()
    df['split'] = df.apply(
        lambda r: max(split_map.get(r['id_a'], 'train'),
                      split_map.get(r['id_b'], 'train'),
                      key=lambda s: rank[s]),
        axis=1
    )
    return df


print('── SHS27k ──')
shs27_seqs    = load_tsv_seqs(os.path.join(DATA_ROOT, 'SHS27k.seqs.tsv'))
shs27_actions = load_actions(os.path.join(DATA_ROOT, 'SHS27k.actions.txt'))
df_shs27      = build_dataset(shs27_actions, shs27_seqs, seed=SEED)
df_shs27      = dfs_split(df_shs27, seed=SEED)
df_shs27['dataset'] = 'SHS27k'
print(f'  Total: {len(df_shs27)}  pos={df_shs27["label"].mean():.3f}  '
      f'splits={df_shs27["split"].value_counts().to_dict()}')


# ==============================================================================

# ============================================================
# CELL 5: ESM-2 35M embeddings — verbatim from benchmark Cell 13
# ============================================================
AA_SET = set('ACDEFGHIKLMNPQRSTVWY')

def clean_seq(s):
    return ''.join(c for c in str(s).upper() if c in AA_SET)


EMBED_CACHE_35M = os.path.join(DATA_ROOT, 'shs27k_esm2_35M_embeddings.npz')

if os.path.exists(EMBED_CACHE_35M):
    print('Loading cached 35M embeddings ...')
    cache = np.load(EMBED_CACHE_35M, allow_pickle=True)
    emb_map_35m = {k: v for k, v in zip(cache['ids'], cache['embs'])}
    print(f'✓ Loaded {len(emb_map_35m)} embeddings  '
          f'dim={next(iter(emb_map_35m.values())).shape}')
else:
    print('Loading ESM-2 35M ...')
    esm_model_35m, alphabet_35m = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model_35m = esm_model_35m.eval().to(DEVICE)
    batch_converter_35m = alphabet_35m.get_batch_converter()
    print('✓ ESM-2 35M loaded  (dim=480, 12 layers)')

    @torch.no_grad()
    def get_esm35m_embedding(sequences, batch_size=32):
        all_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc='ESM-2 35M'):
            batch = sequences[i:i+batch_size]
            data  = [(f'p{j}', s[:1022]) for j, s in enumerate(batch)]
            _, _, tokens = batch_converter_35m(data)
            tokens = tokens.to(DEVICE)
            out  = esm_model_35m(tokens, repr_layers=[12], return_contacts=False)
            reps = out['representations'][12]
            for j, (_, seq) in enumerate(data):
                emb = reps[j, 1:len(seq)+1].mean(0).cpu().numpy()
                all_embs.append(emb)
        return np.stack(all_embs).astype(np.float32)

    all_seqs_35m = {}
    for _, row in df_shs27.iterrows():
        all_seqs_35m[row['id_a']] = clean_seq(row['seq_a'])
        all_seqs_35m[row['id_b']] = clean_seq(row['seq_b'])

    ids_35m  = list(all_seqs_35m.keys())
    seqs_35m = [all_seqs_35m[i] for i in ids_35m]
    print(f'Unique proteins: {len(seqs_35m)}')

    embs_35m    = get_esm35m_embedding(seqs_35m, batch_size=32)
    emb_map_35m = {pid: emb for pid, emb in zip(ids_35m, embs_35m)}
    np.savez_compressed(EMBED_CACHE_35M,
                        ids=np.array(ids_35m),
                        embs=embs_35m)
    print(f'✓ Saved to {EMBED_CACHE_35M}')
    del esm_model_35m

EMB_DIM = next(iter(emb_map_35m.values())).shape[0]
print(f'EMB_DIM = {EMB_DIM}')


# ==============================================================================

# ============================================================
# CELL 6: Score functions + pair features — verbatim benchmark Cell 14
# ============================================================
AA20      = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}
AA_PROP   = {
    'A':dict(hydro=1,polar=0,charge=0,  arom=0,small=1,mw=89.1),
    'C':dict(hydro=0,polar=1,charge=0,  arom=0,small=1,mw=121.2),
    'D':dict(hydro=0,polar=1,charge=-1, arom=0,small=1,mw=133.1),
    'E':dict(hydro=0,polar=1,charge=-1, arom=0,small=0,mw=147.1),
    'F':dict(hydro=1,polar=0,charge=0,  arom=1,small=0,mw=165.2),
    'G':dict(hydro=0,polar=0,charge=0,  arom=0,small=1,mw=75.1),
    'H':dict(hydro=0,polar=1,charge=0.5,arom=1,small=0,mw=155.2),
    'I':dict(hydro=1,polar=0,charge=0,  arom=0,small=0,mw=131.2),
    'K':dict(hydro=0,polar=1,charge=1,  arom=0,small=0,mw=146.2),
    'L':dict(hydro=1,polar=0,charge=0,  arom=0,small=0,mw=131.2),
    'M':dict(hydro=1,polar=0,charge=0,  arom=0,small=0,mw=149.2),
    'N':dict(hydro=0,polar=1,charge=0,  arom=0,small=1,mw=132.1),
    'P':dict(hydro=0,polar=0,charge=0,  arom=0,small=1,mw=115.1),
    'Q':dict(hydro=0,polar=1,charge=0,  arom=0,small=0,mw=146.2),
    'R':dict(hydro=0,polar=1,charge=1,  arom=0,small=0,mw=174.2),
    'S':dict(hydro=0,polar=1,charge=0,  arom=0,small=1,mw=105.1),
    'T':dict(hydro=0,polar=1,charge=0,  arom=0,small=1,mw=119.1),
    'V':dict(hydro=1,polar=0,charge=0,  arom=0,small=1,mw=117.1),
    'W':dict(hydro=1,polar=0,charge=0,  arom=1,small=0,mw=204.2),
    'Y':dict(hydro=1,polar=1,charge=0,  arom=1,small=0,mw=181.2),
}

def _build_mj():
    M = np.zeros((20, 20), dtype=np.float32)
    for i, a in enumerate(AA20):
        for j, b in enumerate(AA20):
            pa, pb = AA_PROP[a], AA_PROP[b]
            s = 0.0
            if pa['hydro'] and pb['hydro']:                                          s += 0.6
            if pa['arom']  and pb['arom']:                                           s += 0.4
            if pa['charge']*pb['charge'] < 0:                                        s += 0.9
            if pa['charge']*pb['charge'] > 0:                                        s -= 0.5
            if pa['polar'] and pb['polar'] and pa['charge']==0 and pb['charge']==0:  s += 0.3
            s -= 0.01*abs(pa['mw']-pb['mw'])
            M[i, j] = s
    return (M + M.T) / 2
MJ = _build_mj()


def kmer_jaccard(a, b, k=3):
    a, b = clean_seq(a), clean_seq(b)
    if len(a) < k or len(b) < k: return 0.0
    A = set(a[i:i+k] for i in range(len(a)-k+1))
    B = set(b[i:i+k] for i in range(len(b)-k+1))
    d = len(A|B); return float(len(A&B)/d) if d else 0.0

def profile_curve(seq, bins=64):
    seq = clean_seq(seq)
    if not seq: return np.zeros((bins, 6), dtype=np.float32)
    arr = np.array([[AA_PROP[c]['hydro'], AA_PROP[c]['polar'], AA_PROP[c]['charge'],
                     AA_PROP[c]['arom'],  AA_PROP[c]['small'], AA_PROP[c]['mw']/200.0]
                    for c in seq], dtype=np.float32)
    xo = np.linspace(0, 1, len(arr)); xn = np.linspace(0, 1, bins)
    return np.column_stack([np.interp(xn, xo, arr[:, j]) for j in range(6)]).astype(np.float32)

def structure_proxy_score(a, b):
    A, B = profile_curve(a).ravel(), profile_curve(b).ravel()
    return float(0.5*(np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)+1e-12)+1.0))

def contact_overlap_score(a, b):
    def est(seq):
        seq = clean_seq(seq)
        if not seq: return 0.0
        frac = sum(c in 'DEKRHNQSTYWF' for c in seq) / len(seq)
        return len(seq) * (0.20 + 0.45*frac)
    c1, c2 = est(a), est(b); mx = max(c1, c2)
    return float(min(c1,c2)/mx) if mx > 0 else 0.0

def chemical_compatibility_score(a, b, maxl=160):
    A, B = clean_seq(a)[:maxl], clean_seq(b)[:maxl]
    if not A or not B: return 0.5
    ia = [AA_TO_IDX[c] for c in A]; ib = [AA_TO_IDX[c] for c in B]
    return float(np.clip((MJ[np.ix_(ia,ib)].mean()+1.5)/3.0, 0, 1))

def four_scores(a, b):
    return np.array([kmer_jaccard(a,b), structure_proxy_score(a,b),
                     contact_overlap_score(a,b), chemical_compatibility_score(a,b)],
                    dtype=np.float32)


# ── Build pair features [ea|eb|ea-eb|ea*eb|scores] = 1924-d ──────────────────
print('Building pair features for SHS27k ...')
X_all, Z_all, y_all, pair_ids = [], [], [], []

for _, row in tqdm(df_shs27.iterrows(), total=len(df_shs27)):
    ea = emb_map_35m.get(row['id_a'])
    eb = emb_map_35m.get(row['id_b'])
    if ea is None or eb is None:
        continue
    scores = four_scores(clean_seq(row['seq_a']), clean_seq(row['seq_b']))
    feat   = np.concatenate([ea, eb, ea-eb, ea*eb, scores]).astype(np.float32)
    X_all.append(feat)
    Z_all.append(scores)
    y_all.append(int(row['label']))
    pair_ids.append((row['id_a'], row['id_b']))

X_all = np.stack(X_all); Z_all = np.stack(Z_all); y_all = np.array(y_all, dtype=np.float32)
print(f'Features: X={X_all.shape}  Z={Z_all.shape}  y={y_all.shape}')

rng    = np.random.default_rng(SEED)
idx    = np.arange(len(X_all)); rng.shuffle(idx)
n      = len(idx)
tr_end = int(n*0.70); va_end = int(n*0.80)
tr, va, te = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_all[tr]).astype(np.float32)
X_val   = scaler.transform(X_all[va]).astype(np.float32)
X_test  = scaler.transform(X_all[te]).astype(np.float32)
Z_train, Z_val, Z_test = Z_all[tr], Z_all[va], Z_all[te]
y_train, y_val, y_test = y_all[tr], y_all[va], y_all[te]

print(f'Train: {len(y_train)}  pos={y_train.mean():.3f}')
print(f'Val  : {len(y_val)}   pos={y_val.mean():.3f}')
print(f'Test : {len(y_test)}  pos={y_test.mean():.3f}')


# ==============================================================================

# ============================================================
# CELL 7: PPIProjectedNet — verbatim from benchmark Cell 14
# ============================================================

class PPIDataset(Dataset):
    def __init__(self, X, Z, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Z = torch.tensor(Z, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Z[i], self.y[i]

train_ds = PPIDataset(X_train, Z_train, y_train)
val_ds   = PPIDataset(X_val,   Z_val,   y_val)
test_ds  = PPIDataset(X_test,  Z_test,  y_test)

y_int   = y_train.astype(int)
counts  = np.bincount(y_int, minlength=2)
w       = np.array([1.0/max(counts[0],1), 1.0/max(counts[1],1)], dtype=np.float64)
sampler = WeightedRandomSampler(torch.DoubleTensor(w[y_int]), len(y_int), replacement=True)

BATCH_SIZE   = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)


class PPIProjectedNet(nn.Module):
    def __init__(self, emb_dim=480, d_model=256, n_heads=8, n_layers=4, p=0.20):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_a = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_b = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=p, batch_first=True, norm_first=True)
        self.enc_a = nn.TransformerEncoder(enc_layer, num_layers=n_layers//2)
        self.enc_b = nn.TransformerEncoder(enc_layer, num_layers=n_layers//2)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.cross_ln   = nn.LayerNorm(d_model)
        self.score_proj = nn.Linear(4, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model*3, d_model*2), nn.LayerNorm(d_model*2), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_model*2, d_model),   nn.LayerNorm(d_model),   nn.GELU(), nn.Dropout(p),
        )
        self.score_bottleneck = nn.Linear(d_model, 4)
        self.cls_linear = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Linear(4, 1))

    def forward(self, x):
        ea     = x[:, :self.emb_dim]
        eb     = x[:, self.emb_dim:self.emb_dim*2]
        scores = x[:, self.emb_dim*4:self.emb_dim*4+4]
        ha = self.enc_a(self.proj_a(ea).unsqueeze(1))
        hb = self.enc_b(self.proj_b(eb).unsqueeze(1))
        cross, _ = self.cross_attn(ha, hb, hb)
        cross     = self.cross_ln(ha + cross)
        hs        = self.score_proj(scores).unsqueeze(1)
        h = self.mlp(torch.cat([cross.squeeze(1),
                                 hb.squeeze(1),
                                 hs.squeeze(1)], dim=-1))
        z     = self.score_bottleneck(h)
        logit = self.cls_linear(z).squeeze(-1)
        return z, logit


model    = PPIProjectedNet(emb_dim=EMB_DIM, d_model=256, n_heads=8, n_layers=4, p=0.20).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'PPIProjectedNet  params={n_params:,}  emb_dim={EMB_DIM}')

pos_w        = float(y_train.sum())
neg_w        = float(len(y_train) - pos_w)
pos_weight   = torch.tensor([neg_w/pos_w], dtype=torch.float32, device=DEVICE)
LABEL_SMOOTH = 0.10

def compute_loss(z, logit, z_true, y):
    y_s   = y*(1-LABEL_SMOOTH) + 0.5*LABEL_SMOOTH
    l_cls = F.binary_cross_entropy_with_logits(logit, y_s, pos_weight=pos_weight)
    l_flow= F.mse_loss(z, z_true)
    return 1.0*l_cls + 0.5*l_flow

@torch.no_grad()
def evaluate(mdl, loader):
    mdl.eval()
    all_logits, all_y, all_zp, all_zt = [], [], [], []
    tloss = 0.0; n = 0
    for xb, zb, yb in loader:
        xb, zb, yb = xb.to(DEVICE), zb.to(DEVICE), yb.to(DEVICE)
        zp, logit = mdl(xb)
        l = compute_loss(zp, logit, zb, yb)
        tloss += l.item()*len(yb); n += len(yb)
        all_logits.append(logit.cpu()); all_y.append(yb.cpu())
        all_zp.append(zp.cpu());       all_zt.append(zb.cpu())
    logits = torch.cat(all_logits).numpy()
    y_np   = torch.cat(all_y).numpy()
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    preds  = (probs >= 0.5).astype(int)
    zp_np  = torch.cat(all_zp).numpy()
    zt_np  = torch.cat(all_zt).numpy()
    return {
        'loss'    : tloss/max(n,1),
        'auc'     : roc_auc_score(y_np, probs) if len(np.unique(y_np))>1 else float('nan'),
        'micro_f1': f1_score(y_np, preds, average='micro'),
        'macro_f1': f1_score(y_np, preds, average='macro'),
        'flow_mse': float(np.mean((zp_np-zt_np)**2)),
        'probs': probs, 'y_true': y_np, 'z_pred': zp_np, 'z_true': zt_np,
    }


WARMUP=5; T_MAX=150; PATIENCE=25
optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)

def lr_lambda(epoch):
    if epoch < WARMUP: return (epoch+1)/WARMUP
    p = (epoch-WARMUP)/max(T_MAX-WARMUP, 1)
    return 0.5*(1.0+np.cos(np.pi*p))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
best_val   = float('inf'); best_state = None; wait = 0

for epoch in range(1, T_MAX+1):
    model.train()
    run = 0.0; n = 0
    for xb, zb, yb in train_loader:
        xb, zb, yb = xb.to(DEVICE), zb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        zp, logit = model(xb)
        l = compute_loss(zp, logit, zb, yb)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run += l.item()*len(yb); n += len(yb)
    scheduler.step()
    tr_loss = run/max(n,1)
    val_m   = evaluate(model, val_loader)

    if val_m['loss'] < best_val:
        best_val   = val_m['loss']
        best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f'epoch {epoch:03d} | train={tr_loss:.4f} | val={val_m["loss"]:.4f} | '
              f'AUC={val_m["auc"]:.4f} | micro-F1={val_m["micro_f1"]:.4f} | '
              f'flow_mse={val_m["flow_mse"]:.4f}')
    if wait >= PATIENCE:
        print(f'Early stop at epoch {epoch}'); break

model.load_state_dict(best_state)
test_m = evaluate(model, test_loader)
print(f'\n── TEST RESULTS (SHS27k) ──────────────')
print(f'  AUC       : {test_m["auc"]:.4f}')
print(f'  micro-F1  : {test_m["micro_f1"]:.4f}')
print(f'  macro-F1  : {test_m["macro_f1"]:.4f}')
print(f'  flow_mse  : {test_m["flow_mse"]:.4f}')

cls_weights = model.cls_linear[2].weight.detach().cpu().numpy()[0]
cls_bias    = model.cls_linear[2].bias.detach().cpu().numpy()[0]
print(f'\n── Learned cls_linear weights ──')
for name, w in zip(['seq','struct','contact','chem'], cls_weights):
    print(f'  {name:10s}: {w:+.4f}')

torch.save({
    'model_state': best_state,
    'cls_weights': cls_weights,
    'cls_bias':    cls_bias,
    'scaler_mean': scaler.mean_.astype(np.float32),
    'scaler_std':  scaler.scale_.astype(np.float32),
    'emb_dim':     EMB_DIM,
    'test_micro_f1': test_m['micro_f1'],
}, os.path.join(DATA_ROOT, 'ppi_projected_net.pt'))
print('\n✓ Saved: ./data/ppi_projected_net.pt')


# ==============================================================================

# ============================================================
# CELL B3.1: Train / Load Receptor-Centric Discriminative H-ESFM
#
# Requires Cell A + Cell B2 already run.
#
# Uses existing:
#   df_shs27, emb_map_35m, model, scaler, four_scores, clean_seq,
#   eval_ppi_pair, compute_embedding_features, DEVICE, SEED, EMB_DIM
#
# Creates:
#   b31_flow_model
#   b31_flow_integrator
#   b31_cfg
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from collections import defaultdict
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

assert "df_shs27" in globals()
assert "emb_map_35m" in globals()
assert "model" in globals()
assert "scaler" in globals()
assert "four_scores" in globals()
assert "clean_seq" in globals()
assert "eval_ppi_pair" in globals()
assert "compute_embedding_features" in globals()
assert "DEVICE" in globals()
assert "SEED" in globals()
assert "EMB_DIM" in globals()


@dataclass
class B31FlowConfig:
    emb_dim: int = EMB_DIM
    d_model: int = 256
    dropout: float = 0.10

    n_flow_steps: int = 20
    flow_epochs: int = 80
    flow_batch_size: int = 64
    flow_lr: float = 3e-4
    weight_decay: float = 1e-2

    n_train_binders: int = 500
    n_train_decoys: int = 500
    n_negatives: int = 5

    save_path: str = "b31_receptor_discriminative_hesfm_large.pt"


b31_cfg = B31FlowConfig()


@torch.no_grad()
def b31_ppi_prob_from_embs(ea, eb, scores_vec):
    feat = np.concatenate([
        ea.astype(np.float32),
        eb.astype(np.float32),
        ea.astype(np.float32) - eb.astype(np.float32),
        ea.astype(np.float32) * eb.astype(np.float32),
        scores_vec.astype(np.float32),
    ]).astype(np.float32)

    X = scaler.transform(feat[None]).astype(np.float32)
    model.eval()
    _, logit = model(torch.tensor(X, device=DEVICE))
    return float(torch.sigmoid(logit).detach().cpu())


def b31_safe_scores(seq_a, seq_b):
    try:
        return four_scores(clean_seq(seq_a), clean_seq(seq_b)).astype(np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


def b31_prepare_flow_pairs(cfg):
    train_df = df_shs27[df_shs27["split"].isin(["train", "val"])].copy()

    pos_df = train_df[train_df["label"] == 1].sample(
        n=min(cfg.n_train_binders, int((train_df["label"] == 1).sum())),
        random_state=SEED + 31,
    )

    neg_df = train_df[train_df["label"] == 0].sample(
        n=min(cfg.n_train_decoys, int((train_df["label"] == 0).sum())),
        random_state=SEED + 32,
    )

    pairs = []

    for label, subdf in [(1, pos_df), (0, neg_df)]:
        for _, row in tqdm(
            subdf.iterrows(),
            total=len(subdf),
            desc=f"  B3.1 flow pairs label={label}",
        ):
            aid, bid = str(row["id_a"]), str(row["id_b"])
            ea = emb_map_35m.get(aid)
            eb = emb_map_35m.get(bid)

            if ea is None or eb is None:
                continue

            scores = b31_safe_scores(row["seq_a"], row["seq_b"])
            p = b31_ppi_prob_from_embs(ea, eb, scores)

            pairs.append({
                "rec_id": aid,
                "lig_id": bid,
                "ea": ea.astype(np.float32),
                "eb": eb.astype(np.float32),
                "scores": scores,
                "probability": float(np.clip(p, 0.05, 0.95)),
                "confidence": float(max(p, 1.0 - p)),
                "label": int(label),
            })

    rng = np.random.default_rng(SEED + 33)
    rng.shuffle(pairs)

    print(
        f"  ✓ B3.1 flow pairs: {len(pairs)} | "
        f"pos={sum(p['label']==1 for p in pairs)} "
        f"neg={sum(p['label']==0 for p in pairs)}"
    )

    return pairs


class B31SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class B31ProbabilityConditionedVelocityField(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        emb_dim = cfg.emb_dim
        d = cfg.d_model

        self.time_embed = nn.Sequential(
            B31SinusoidalTimeEmbed(d),
            nn.Linear(d, d),
            nn.SiLU(),
        )

        self.prob_embed = nn.Sequential(
            nn.Linear(2, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, d),
        )

        self.receptor_proj = nn.Sequential(
            nn.Linear(emb_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.ligand_proj = nn.Sequential(
            nn.Linear(emb_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.direction_proj = nn.Sequential(
            nn.Linear(emb_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.interact_proj = nn.Sequential(
            nn.Linear(emb_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.velocity_net = nn.Sequential(
            nn.Linear(d * 6, d * 4),
            nn.LayerNorm(d * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 4, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.GELU(),
        )

        self.velocity_head = nn.Linear(d, emb_dim)

        self.direction_gate = nn.Sequential(
            nn.Linear(d + 2, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, e_ligand, t, e_receptor, prob, conf):
        e_ligand = e_ligand.float()
        e_receptor = e_receptor.float()
        t = t.float()
        prob = prob.float()
        conf = conf.float()

        t_emb = self.time_embed(t)
        pc = torch.stack([prob, conf], dim=-1)
        p_emb = self.prob_embed(pc)

        r_emb = self.receptor_proj(e_receptor)
        l_emb = self.ligand_proj(e_ligand)

        direction = e_receptor - e_ligand
        dir_emb = self.direction_proj(direction)

        interact = e_receptor * e_ligand
        int_emb = self.interact_proj(interact)

        h = self.velocity_net(torch.cat([r_emb, l_emb, dir_emb, int_emb, t_emb, p_emb], dim=-1))
        base_v = self.velocity_head(h)

        gate = torch.tanh(self.direction_gate(torch.cat([t_emb, pc], dim=-1)))
        unit_dir = F.normalize(direction, dim=-1)
        attraction = 2.0 * (prob - 0.5).unsqueeze(-1)

        return base_v * gate + 0.3 * attraction * unit_dir


class B31FlowIntegrator:
    def __init__(self, flow_model, n_steps=20):
        self.model = flow_model
        self.n_steps = n_steps

    @torch.no_grad()
    def integrate(self, e_ligand, e_receptor, prob, conf):
        B = e_ligand.shape[0]
        device = e_ligand.device
        dt = 1.0 / self.n_steps

        e_t = e_ligand.clone()
        d0 = torch.norm(e_receptor - e_ligand, dim=-1)

        self.model.eval()

        for step in range(self.n_steps):
            t = torch.full((B,), step / self.n_steps, device=device)
            v = self.model(e_t, t, e_receptor, prob, conf)
            e_t = e_t + dt * v

        dT = torch.norm(e_receptor - e_t, dim=-1)

        return {
            "initial_dist": d0,
            "final_dist": dT,
            "improvement": d0 - dT,
            "final_embedding": e_t,
        }

    def integrate_train(self, e_ligand, e_receptor, prob, conf):
        B = e_ligand.shape[0]
        device = e_ligand.device
        dt = 1.0 / self.n_steps

        e_t = e_ligand.clone()
        d0 = torch.norm(e_receptor - e_ligand, dim=-1)

        for step in range(self.n_steps):
            t = torch.full((B,), step / self.n_steps, device=device)
            v = self.model(e_t, t, e_receptor, prob, conf)
            e_t = e_t + dt * v

        dT = torch.norm(e_receptor - e_t, dim=-1)

        return {
            "initial_dist": d0,
            "final_dist": dT,
            "improvement": d0 - dT,
            "final_embedding": e_t,
        }


class B31ContrastiveDataset(Dataset):
    def __init__(self, pairs, n_negatives=3):
        self.binders = [p for p in pairs if p["label"] == 1]
        self.decoys = [p for p in pairs if p["label"] == 0]
        self.n_negatives = n_negatives

        print(f"  B31ContrastiveDataset: {len(self.binders)} binders, {len(self.decoys)} decoys")

    def __len__(self):
        return len(self.binders)

    def __getitem__(self, idx):
        pos = self.binders[idx]

        neg_idx = np.random.choice(
            len(self.decoys),
            min(self.n_negatives, len(self.decoys)),
            replace=False,
        )

        negs = [self.decoys[i] for i in neg_idx]
        while len(negs) < self.n_negatives:
            negs.append(negs[-1])

        return {
            "e_receptor": pos["ea"].astype(np.float32),
            "e_pos_ligand": pos["eb"].astype(np.float32),
            "pos_prob": float(pos["probability"]),
            "pos_conf": float(pos["confidence"]),
            "e_neg_ligands": np.stack([n["eb"].astype(np.float32) for n in negs]),
            "neg_probs": np.array([float(n["probability"]) for n in negs], dtype=np.float32),
            "neg_confs": np.array([float(n["confidence"]) for n in negs], dtype=np.float32),
        }


def b31_collate(batch):
    return {
        "e_receptor": torch.tensor(np.stack([b["e_receptor"] for b in batch])),
        "e_pos_ligand": torch.tensor(np.stack([b["e_pos_ligand"] for b in batch])),
        "pos_prob": torch.tensor([b["pos_prob"] for b in batch], dtype=torch.float32),
        "pos_conf": torch.tensor([b["pos_conf"] for b in batch], dtype=torch.float32),
        "e_neg_ligands": torch.tensor(np.stack([b["e_neg_ligands"] for b in batch])),
        "neg_probs": torch.tensor(np.stack([b["neg_probs"] for b in batch])),
        "neg_confs": torch.tensor(np.stack([b["neg_confs"] for b in batch])),
    }


class B31ContrastiveRankingLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_rank=1.0, lambda_attract=0.5, lambda_repel=0.3):
        super().__init__()
        self.margin = margin
        self.lambda_rank = lambda_rank
        self.lambda_attract = lambda_attract
        self.lambda_repel = lambda_repel

    def forward(self, pos_final, neg_final, pos_impr, neg_impr, pos_prob, neg_prob):
        ranking = F.relu(pos_final.unsqueeze(1) - neg_final + self.margin).mean()
        attract = (F.relu(-pos_impr) * pos_prob).mean()
        repel = (F.relu(neg_impr) * (1.0 - neg_prob)).mean()
        total = self.lambda_rank * ranking + self.lambda_attract * attract + self.lambda_repel * repel
        return {"total": total, "ranking": ranking, "attract": attract, "repel": repel}


def train_or_load_b31_flow(cfg=b31_cfg, force_retrain=False):
    if os.path.exists(cfg.save_path) and not force_retrain:
        print(f"✓ Loading B3.1 flow model from {cfg.save_path}")
        ckpt = torch.load(cfg.save_path, map_location=DEVICE, weights_only=False)

        flow_model = B31ProbabilityConditionedVelocityField(cfg).to(DEVICE)
        flow_model.load_state_dict(ckpt["model_state"])
        flow_model.eval()

        return flow_model, B31FlowIntegrator(flow_model, cfg.n_flow_steps), ckpt.get("history", [])

    print("\n" + "=" * 90)
    print("CELL B3.1: Training receptor-centric discriminative H-ESFM")
    print("=" * 90)

    pairs = b31_prepare_flow_pairs(cfg)

    dataset = B31ContrastiveDataset(pairs, cfg.n_negatives)
    loader = DataLoader(
        dataset,
        batch_size=cfg.flow_batch_size,
        shuffle=True,
        collate_fn=b31_collate,
        num_workers=0,
    )

    flow_model = B31ProbabilityConditionedVelocityField(cfg).to(DEVICE)
    integrator = B31FlowIntegrator(flow_model, cfg.n_flow_steps)

    loss_fn = B31ContrastiveRankingLoss()

    opt = torch.optim.AdamW(
        flow_model.parameters(),
        lr=cfg.flow_lr,
        weight_decay=cfg.weight_decay,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.flow_epochs)

    history = []

    print(f"  Parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    print(f"\n  {'Epoch':>5} {'Total':>10} {'Rank':>10} {'Attr':>10} {'Repel':>10}")
    print("  " + "-" * 55)

    for epoch in range(1, cfg.flow_epochs + 1):
        flow_model.train()
        logs = defaultdict(list)

        for batch in loader:
            e_r = batch["e_receptor"].to(DEVICE)
            e_p = batch["e_pos_ligand"].to(DEVICE)
            pp = batch["pos_prob"].to(DEVICE)
            pc = batch["pos_conf"].to(DEVICE)

            e_n = batch["e_neg_ligands"].to(DEVICE)
            npb = batch["neg_probs"].to(DEVICE)
            nc = batch["neg_confs"].to(DEVICE)

            B, K, _ = e_n.shape

            pos = integrator.integrate_train(e_p, e_r, pp, pc)

            neg_final = []
            neg_impr = []

            for k in range(K):
                neg = integrator.integrate_train(e_n[:, k], e_r, npb[:, k], nc[:, k])
                neg_final.append(neg["final_dist"])
                neg_impr.append(neg["improvement"])

            neg_final = torch.stack(neg_final, dim=1)
            neg_impr = torch.stack(neg_impr, dim=1)

            loss = loss_fn(
                pos["final_dist"],
                neg_final,
                pos["improvement"],
                neg_impr,
                pp,
                npb,
            )

            opt.zero_grad()
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            opt.step()

            for k, v in loss.items():
                logs[k].append(float(v.detach().cpu()))

        sched.step()

        row = {k: float(np.mean(v)) for k, v in logs.items()}
        row["epoch"] = epoch
        history.append(row)

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.flow_epochs:
            print(
                f"  {epoch:5d} {row['total']:10.4f} {row['ranking']:10.4f} "
                f"{row['attract']:10.4f} {row['repel']:10.4f}"
            )

    torch.save({
        "model_state": flow_model.state_dict(),
        "config": cfg,
        "history": history,
    }, cfg.save_path)

    print(f"\n✓ Saved B3.1 flow model to {cfg.save_path}")

    flow_model.eval()
    return flow_model, integrator, history


# b31_flow_model, b31_flow_integrator, b31_flow_history = train_or_load_b31_flow(
#     b31_cfg,
#     force_retrain=False,
# )
b31_flow_model, b31_flow_integrator, b31_flow_history = train_or_load_b31_flow(
    b31_cfg,
    force_retrain=True,
)

print("✓ CELL B3.1 COMPLETE")


# ==============================================================================

# ============================================================
# CELL B3.2: True Distance-Rank Benchmark for B3.1 Flow
#
# This evaluates what B3.1 trained:
#   binder final distance < decoy final distance
#
# Comparison:
#   1. PPI rank: higher PPI is better
#   2. Initial-distance rank: lower initial distance is better
#   3. Final-flow-distance rank: lower final distance is better
#   4. Improvement rank: larger distance improvement is better
#
# Candidate pool:
#   all unique train/val proteins except receptor
#
# This is NOT the stochastic ToT tree. This is the true rank test
# aligned with the B3.1 rank loss.
# ============================================================

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

assert "df_shs27" in globals()
assert "b31_flow_integrator" in globals()
assert "emb_map_35m" in globals()
assert "eval_ppi_pair" in globals()
assert "DEVICE" in globals()
assert "SEED" in globals()

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

B32_N_TRIALS = 10
B32_BATCH_SIZE = 256
B32_SPLITS = ["train", "val"]
B32_SAVE = True

# Use a different seed so receptors can differ from previous cells
B32_RECEPTOR_SEED = SEED + 999


# ============================================================
# Data setup
# ============================================================

trainval_df = df_shs27[df_shs27["split"].isin(B32_SPLITS)].copy()

trainval_df["id_a"] = trainval_df["id_a"].astype(str)
trainval_df["id_b"] = trainval_df["id_b"].astype(str)

trainval_pos = trainval_df[trainval_df["label"] == 1].copy()

all_proteins_trainval = pd.concat([
    trainval_df[["id_a", "seq_a"]].rename(columns={"id_a": "id", "seq_a": "seq"}),
    trainval_df[["id_b", "seq_b"]].rename(columns={"id_b": "id", "seq_b": "seq"}),
]).drop_duplicates("id").reset_index(drop=True)

all_proteins_trainval["id"] = all_proteins_trainval["id"].astype(str)

receptor_counts = pd.concat([
    trainval_pos["id_a"],
    trainval_pos["id_b"],
]).value_counts()

valid_receptors = receptor_counts[receptor_counts >= 1].index.tolist()

rng_receptor = np.random.default_rng(B32_RECEPTOR_SEED)
selected_receptors = [
    str(rng_receptor.choice(valid_receptors))
    for _ in range(B32_N_TRIALS)
]

print("\n" + "=" * 100)
print("CELL B3.2: TRUE DISTANCE-RANK BENCHMARK")
print("=" * 100)
print(f"Split              : {B32_SPLITS}")
print(f"Unique proteins    : {len(all_proteins_trainval)}")
print(f"Receptors tested   : {B32_N_TRIALS}")
print(f"Candidate pool     : all train/val proteins except receptor")
print(f"Batch size         : {B32_BATCH_SIZE}")


# ============================================================
# Helpers
# ============================================================

def b32_get_receptor_seq(receptor_id):
    rows = trainval_df[
        (trainval_df["id_a"] == receptor_id)
        | (trainval_df["id_b"] == receptor_id)
    ]

    if len(rows) == 0:
        return None

    row = rows.iloc[0]

    if str(row["id_a"]) == receptor_id:
        return row["seq_a"]
    return row["seq_b"]


def b32_get_known_binders(receptor_id):
    ka = set(
        trainval_pos[trainval_pos["id_a"] == receptor_id]["id_b"]
        .astype(str)
        .tolist()
    )

    kb = set(
        trainval_pos[trainval_pos["id_b"] == receptor_id]["id_a"]
        .astype(str)
        .tolist()
    )

    return ka | kb


def b32_flow_distances_batched(e_receptor_np, ligand_embs_np, base_ppi_np):
    """
    Vectorized B3.1 flow integration.

    Returns:
      initial_dist, final_dist, improvement
    """

    all_initial = []
    all_final = []
    all_impr = []

    e_r_full = torch.tensor(
        e_receptor_np.astype(np.float32),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    for start in range(0, len(ligand_embs_np), B32_BATCH_SIZE):
        end = min(start + B32_BATCH_SIZE, len(ligand_embs_np))

        eb = torch.tensor(
            ligand_embs_np[start:end].astype(np.float32),
            dtype=torch.float32,
            device=DEVICE,
        )

        e_r = e_r_full.repeat(end - start, 1)

        prob = torch.tensor(
            np.clip(base_ppi_np[start:end], 0.05, 0.95),
            dtype=torch.float32,
            device=DEVICE,
        )

        conf = torch.maximum(prob, 1.0 - prob)

        with torch.no_grad():
            out = b31_flow_integrator.integrate(eb, e_r, prob, conf)

        all_initial.append(out["initial_dist"].detach().cpu().numpy())
        all_final.append(out["final_dist"].detach().cpu().numpy())
        all_impr.append(out["improvement"].detach().cpu().numpy())

    return (
        np.concatenate(all_initial),
        np.concatenate(all_final),
        np.concatenate(all_impr),
    )


def b32_eval_true_rank_for_receptor(receptor_id, trial):
    receptor_seq = b32_get_receptor_seq(receptor_id)
    known_binders = b32_get_known_binders(receptor_id)

    candidates = all_proteins_trainval[
        all_proteins_trainval["id"] != receptor_id
    ].reset_index(drop=True).copy()

    candidates["Y"] = candidates["id"].apply(
        lambda x: 1 if str(x) in known_binders else 0
    )

    ea = emb_map_35m.get(str(receptor_id))

    if ea is None or receptor_seq is None:
        return None, None

    rows = []
    ligand_embs = []
    base_ppis = []

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc=f"  Eval receptor {str(receptor_id)[-10:]}",
        leave=False,
    ):
        cid = str(row["id"])
        lseq = row["seq"]

        eb = emb_map_35m.get(cid)
        if eb is None:
            continue

        try:
            base_ppi, scores_vec, _ = eval_ppi_pair(
                receptor_id,
                receptor_seq,
                cid,
                lseq,
            )
            base_ppi = float(base_ppi)
        except Exception:
            base_ppi = 0.0

        rows.append({
            "trial": trial,
            "receptor": receptor_id,
            "ligand": cid,
            "is_known": int(cid in known_binders),
            "base_ppi": base_ppi,
        })

        ligand_embs.append(eb.astype(np.float32))
        base_ppis.append(base_ppi)

    if len(rows) == 0:
        return None, None

    ligand_embs = np.stack(ligand_embs).astype(np.float32)
    base_ppis = np.array(base_ppis, dtype=np.float32)

    initial_dist, final_dist, improvement = b32_flow_distances_batched(
        ea.astype(np.float32),
        ligand_embs,
        base_ppis,
    )

    df = pd.DataFrame(rows)

    df["initial_dist"] = initial_dist
    df["final_dist"] = final_dist
    df["improvement"] = improvement

    # Ranks:
    #   PPI: high is good
    #   distances: low is good
    #   improvement: high is good
    df["rank_ppi"] = df["base_ppi"].rank(ascending=False, method="min")
    df["rank_initial_dist"] = df["initial_dist"].rank(ascending=True, method="min")
    df["rank_final_dist"] = df["final_dist"].rank(ascending=True, method="min")
    df["rank_improvement"] = df["improvement"].rank(ascending=False, method="min")

    y = df["is_known"].astype(int).values

    if len(np.unique(y)) > 1:
        auc_ppi = roc_auc_score(y, df["base_ppi"].values)
        auc_initial = roc_auc_score(y, -df["initial_dist"].values)
        auc_final = roc_auc_score(y, -df["final_dist"].values)
        auc_impr = roc_auc_score(y, df["improvement"].values)

        ap_ppi = average_precision_score(y, df["base_ppi"].values)
        ap_initial = average_precision_score(y, -df["initial_dist"].values)
        ap_final = average_precision_score(y, -df["final_dist"].values)
        ap_impr = average_precision_score(y, df["improvement"].values)
    else:
        auc_ppi = auc_initial = auc_final = auc_impr = np.nan
        ap_ppi = ap_initial = ap_final = ap_impr = np.nan

    known = df[df["is_known"] == 1].copy()

    summary = {
        "trial": trial,
        "receptor": receptor_id,
        "n_binders_pool": int(df["is_known"].sum()),
        "pool_size": len(df),

        "best_rank_ppi": int(known["rank_ppi"].min()) if len(known) else None,
        "best_rank_initial_dist": int(known["rank_initial_dist"].min()) if len(known) else None,
        "best_rank_final_dist": int(known["rank_final_dist"].min()) if len(known) else None,
        "best_rank_improvement": int(known["rank_improvement"].min()) if len(known) else None,

        "mean_rank_ppi": float(known["rank_ppi"].mean()) if len(known) else np.nan,
        "mean_rank_initial_dist": float(known["rank_initial_dist"].mean()) if len(known) else np.nan,
        "mean_rank_final_dist": float(known["rank_final_dist"].mean()) if len(known) else np.nan,
        "mean_rank_improvement": float(known["rank_improvement"].mean()) if len(known) else np.nan,

        "auc_ppi": auc_ppi,
        "auc_initial_dist": auc_initial,
        "auc_final_dist": auc_final,
        "auc_improvement": auc_impr,

        "ap_ppi": ap_ppi,
        "ap_initial_dist": ap_initial,
        "ap_final_dist": ap_final,
        "ap_improvement": ap_impr,

        "mean_improvement_known": float(df[df["is_known"] == 1]["improvement"].mean())
            if int(df["is_known"].sum()) > 0 else np.nan,
        "mean_improvement_decoy": float(df[df["is_known"] == 0]["improvement"].mean())
            if int((df["is_known"] == 0).sum()) > 0 else np.nan,
    }

    return df, summary


# ============================================================
# Main benchmark
# ============================================================

all_tables = []
summaries = []

print("\n" + "=" * 100)
print("Running true-rank benchmark...")
print("=" * 100)

print(
    f'{"Trial":>6} {"Receptor":>18} {"N_b":>4} {"Pool":>6} '
    f'{"PPI":>8} {"InitD":>8} {"FinalD":>8} {"Improve":>8} '
    f'{"ΔInit-Final":>12}'
)
print("─" * 100)

for trial, receptor_id in enumerate(selected_receptors, start=1):
    df_rec, summary = b32_eval_true_rank_for_receptor(receptor_id, trial)

    if df_rec is None:
        continue

    all_tables.append(df_rec)
    summaries.append(summary)

    delta_init_final = (
        summary["best_rank_initial_dist"] - summary["best_rank_final_dist"]
        if summary["best_rank_initial_dist"] is not None
        and summary["best_rank_final_dist"] is not None
        else np.nan
    )

    arrow = "↑" if delta_init_final > 0 else "↓" if delta_init_final < 0 else "="

    print(
        f'{trial:>6} {receptor_id[-18:]:>18} '
        f'{summary["n_binders_pool"]:>4} {summary["pool_size"]:>6} '
        f'{summary["best_rank_ppi"]:>8} '
        f'{summary["best_rank_initial_dist"]:>8} '
        f'{summary["best_rank_final_dist"]:>8} '
        f'{summary["best_rank_improvement"]:>8} '
        f'{str(round(delta_init_final, 0) if delta_init_final == delta_init_final else "N/A"):>11}{arrow}'
    )


df_b32_true_rank = pd.concat(all_tables, ignore_index=True)
df_b32_true_summary = pd.DataFrame(summaries)

valid = df_b32_true_summary.dropna(subset=[
    "best_rank_initial_dist",
    "best_rank_final_dist",
]).copy()

print("\n" + "=" * 100)
print("SUMMARY — TRUE DISTANCE-RANK BENCHMARK")
print("=" * 100)

print(df_b32_true_summary[[
    "trial",
    "receptor",
    "n_binders_pool",
    "pool_size",
    "best_rank_ppi",
    "best_rank_initial_dist",
    "best_rank_final_dist",
    "best_rank_improvement",
    "auc_ppi",
    "auc_initial_dist",
    "auc_final_dist",
    "auc_improvement",
]].to_string(index=False))

if len(valid) > 0:
    d_best = valid["best_rank_initial_dist"] - valid["best_rank_final_dist"]
    d_mean = valid["mean_rank_initial_dist"] - valid["mean_rank_final_dist"]

    try:
        _, p_best = stats.wilcoxon(
            valid["best_rank_initial_dist"],
            valid["best_rank_final_dist"],
        )
    except Exception:
        p_best = np.nan

    try:
        _, p_mean = stats.wilcoxon(
            valid["mean_rank_initial_dist"],
            valid["mean_rank_final_dist"],
        )
    except Exception:
        p_mean = np.nan

    print("\n  Best known-binder rank, lower is better:")
    print(f"    PPI score              : {valid['best_rank_ppi'].mean():.2f}")
    print(f"    Initial distance       : {valid['best_rank_initial_dist'].mean():.2f}")
    print(f"    Final flowed distance  : {valid['best_rank_final_dist'].mean():.2f}")
    print(f"    Improvement score      : {valid['best_rank_improvement'].mean():.2f}")

    print(f"\n  Δ best rank (Initial - Final): {d_best.mean():+.2f}  p={p_best:.4f}")
    print(f"  Flow improved best rank      : {(d_best > 0).sum()} / {len(valid)}")
    print(f"  Flow tied best rank          : {(d_best == 0).sum()} / {len(valid)}")
    print(f"  Flow worsened best rank      : {(d_best < 0).sum()} / {len(valid)}")

    print("\n  Mean known-binder rank, lower is better:")
    print(f"    PPI score              : {valid['mean_rank_ppi'].mean():.2f}")
    print(f"    Initial distance       : {valid['mean_rank_initial_dist'].mean():.2f}")
    print(f"    Final flowed distance  : {valid['mean_rank_final_dist'].mean():.2f}")
    print(f"    Improvement score      : {valid['mean_rank_improvement'].mean():.2f}")

    print(f"\n  Δ mean rank (Initial - Final): {d_mean.mean():+.2f}  p={p_mean:.4f}")

    print("\n  AUC:")
    print(f"    PPI score              : {valid['auc_ppi'].mean():.4f}")
    print(f"    Initial distance       : {valid['auc_initial_dist'].mean():.4f}")
    print(f"    Final flowed distance  : {valid['auc_final_dist'].mean():.4f}")
    print(f"    Improvement            : {valid['auc_improvement'].mean():.4f}")

    print("\n  Average Precision:")
    print(f"    PPI score              : {valid['ap_ppi'].mean():.4f}")
    print(f"    Initial distance       : {valid['ap_initial_dist'].mean():.4f}")
    print(f"    Final flowed distance  : {valid['ap_final_dist'].mean():.4f}")
    print(f"    Improvement            : {valid['ap_improvement'].mean():.4f}")

    print("\n  Mean distance improvement:")
    print(f"    Known binders          : {valid['mean_improvement_known'].mean():+.4f}")
    print(f"    Decoys                 : {valid['mean_improvement_decoy'].mean():+.4f}")

if B32_SAVE:
    df_b32_true_rank.to_csv("cellB32_true_distance_rank_all_candidates.csv", index=False)
    df_b32_true_summary.to_csv("cellB32_true_distance_rank_summary.csv", index=False)

    print("\n✓ Saved:")
    print("  cellB32_true_distance_rank_all_candidates.csv")
    print("  cellB32_true_distance_rank_summary.csv")

print("\nCELL B3.2 TRUE DISTANCE-RANK BENCHMARK COMPLETE")
