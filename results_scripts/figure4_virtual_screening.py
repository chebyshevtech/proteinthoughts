#!/usr/bin/env python3
"""
Figure 4: Virtual Screening Example

Purpose
-------
Trains PPIProjectedNet on SHS27k, runs the 500-candidate virtual-screening trials, and renders the selected annotated screening figure.

Expected outputs
----------------
Writes screen_10_trials.png and selected_screen_annotated.png.

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
# CELL 10: 10 random trials — AUC and Micro-F1 across screens
# Each trial picks a fresh receptor + 500-candidate pool
# ============================================================

from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model.eval()

N_TRIALS    = 10
POOL_SIZE   = 500
THRESHOLD   = 0.5
trial_rng   = np.random.default_rng(SEED + 111)

results = []

# Pre-build protein pool once
all_proteins = pd.concat([
    df_shs27[['id_a', 'seq_a']].rename(columns={'id_a': 'id', 'seq_a': 'seq'}),
    df_shs27[['id_b', 'seq_b']].rename(columns={'id_b': 'id', 'seq_b': 'seq'})
]).drop_duplicates('id').reset_index(drop=True)

# Only consider receptors that have at least 1 known binder in test split
test_pos = df_shs27[(df_shs27['split'] == 'test') & (df_shs27['label'] == 1)]
receptor_counts = pd.concat([test_pos['id_a'], test_pos['id_b']]).value_counts()
valid_receptors = receptor_counts[receptor_counts >= 1].index.tolist()

print(f'Valid receptors to sample from: {len(valid_receptors)}')
print(f'Running {N_TRIALS} trials ...\n')
print(f'{"Trial":>6} {"Receptor":>15} {"Binders":>8} {"AUC":>8} {"Micro-F1":>10} {"Best Rank":>10}')
print('-' * 65)

for trial in range(1, N_TRIALS + 1):

    # ── Pick receptor ──────────────────────────────────────────────────────────
    receptor_id = trial_rng.choice(valid_receptors)

    # Get receptor sequence
    rec_rows = df_shs27[(df_shs27['id_a'] == receptor_id) | (df_shs27['id_b'] == receptor_id)]
    rec_row  = rec_rows.iloc[0]
    receptor_seq = rec_row['seq_a'] if rec_row['id_a'] == receptor_id else rec_row['seq_b']

    # Known binders from dataset (test split only)
    known_a = set(test_pos[test_pos['id_a'] == receptor_id]['id_b'].tolist())
    known_b = set(test_pos[test_pos['id_b'] == receptor_id]['id_a'].tolist())
    known_binders = known_a | known_b

    # ── Build candidate pool ───────────────────────────────────────────────────
    pool       = all_proteins[all_proteins['id'] != receptor_id].reset_index(drop=True)
    known_rows = pool[pool['id'].isin(known_binders)].reset_index(drop=True)
    decoy_pool = pool[~pool['id'].isin(known_binders)].reset_index(drop=True)

    n_fill   = min(POOL_SIZE - len(known_rows), len(decoy_pool))
    d_idx    = trial_rng.choice(len(decoy_pool), n_fill, replace=False)
    decoys   = decoy_pool.iloc[d_idx].reset_index(drop=True)

    candidates = pd.concat([known_rows, decoys], ignore_index=True)
    shuffle_idx = trial_rng.permutation(len(candidates))
    candidates  = candidates.iloc[shuffle_idx].reset_index(drop=True)
    candidates['DATASET_LABEL'] = candidates['id'].apply(
        lambda cid: 1 if cid in known_binders else 0
    )

    # Skip trial if only one class present (can't compute AUC)
    if candidates['DATASET_LABEL'].nunique() < 2:
        continue

    # ── Build features ─────────────────────────────────────────────────────────
    feats, score_vecs = [], []
    ea = emb_map_35m.get(receptor_id, np.zeros(EMB_DIM, dtype=np.float32))
    for _, row in candidates.iterrows():
        eb     = emb_map_35m.get(row['id'], np.zeros(EMB_DIM, dtype=np.float32))
        scores = four_scores(clean_seq(receptor_seq), clean_seq(row['seq']))
        feat   = np.concatenate([ea, eb, ea - eb, ea * eb, scores]).astype(np.float32)
        feats.append(feat)
        score_vecs.append(scores)

    X_s = scaler.transform(np.stack(feats).astype(np.float32)).astype(np.float32)

    # ── Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(torch.tensor(X_s, device=DEVICE))[1]
        probs  = torch.sigmoid(logits).cpu().numpy()

    # ── Metrics ────────────────────────────────────────────────────────────────
    y_true   = candidates['DATASET_LABEL'].values.astype(int)
    y_pred   = (probs >= THRESHOLD).astype(int)
    auc      = roc_auc_score(y_true, probs)
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    ranked   = np.argsort(probs)[::-1]
    best_rank= int(np.where(y_true[ranked] == 1)[0][0]) + 1  # 1-based

    results.append({
        'trial'      : trial,
        'receptor'   : receptor_id,
        'n_binders'  : int(y_true.sum()),
        'pool_size'  : len(candidates),
        'auc'        : auc,
        'micro_f1'   : micro_f1,
        'best_rank'  : best_rank,
    })

    print(f'{trial:>6} {receptor_id:>15} {int(y_true.sum()):>8} '
          f'{auc:>8.4f} {micro_f1:>10.4f} {best_rank:>10}')

# ── Summary ────────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)

mean_auc      = df_results['auc'].mean()
std_auc       = df_results['auc'].std()
mean_f1       = df_results['micro_f1'].mean()
std_f1        = df_results['micro_f1'].std()
mean_rank     = df_results['best_rank'].mean()

print('\n' + '='*65)
print(f'SUMMARY ACROSS {N_TRIALS} TRIALS')
print('='*65)
print(f'  AUC        : {mean_auc:.4f}  ±  {std_auc:.4f}')
print(f'  Micro-F1   : {mean_f1:.4f}  ±  {std_f1:.4f}')
print(f'  Mean best rank : {mean_rank:.1f} / {POOL_SIZE}')
print('='*65)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

trials = df_results['trial'].values

# AUC per trial
ax = axes[0]
ax.bar(trials, df_results['auc'], color='steelblue', alpha=0.8, edgecolor='black')
ax.axhline(mean_auc, color='red', lw=2, ls='--', label=f'Mean={mean_auc:.4f}')
ax.axhline(0.5, color='gray', lw=1.5, ls=':', label='Random (0.5)')
ax.set_xticks(trials)
ax.set_xlabel('Trial')
ax.set_ylabel('AUC')
ax.set_title('AUC per trial')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)

# Micro-F1 per trial
ax = axes[1]
ax.bar(trials, df_results['micro_f1'], color='darkorange', alpha=0.8, edgecolor='black')
ax.axhline(mean_f1, color='red', lw=2, ls='--', label=f'Mean={mean_f1:.4f}')
ax.set_xticks(trials)
ax.set_xlabel('Trial')
ax.set_ylabel('Micro-F1')
ax.set_title('Micro-F1 per trial')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)

# Best rank of true binder per trial
ax = axes[2]
ax.bar(trials, df_results['best_rank'], color='green', alpha=0.8, edgecolor='black')
ax.axhline(mean_rank, color='red', lw=2, ls='--', label=f'Mean={mean_rank:.1f}')
ax.axhline(1, color='green', lw=1.5, ls=':', label='Perfect (rank 1)')
ax.set_xticks(trials)
ax.set_xlabel('Trial')
ax.set_ylabel('Best true binder rank')
ax.set_title(f'Recovery rank (out of {POOL_SIZE})')
ax.legend(fontsize=9)

plt.suptitle(
    f'PPIProjectedNet — {N_TRIALS} random virtual screens\n'
    f'AUC: {mean_auc:.4f}±{std_auc:.4f}   '
    f'Micro-F1: {mean_f1:.4f}±{std_f1:.4f}   '
    f'Mean best rank: {mean_rank:.1f}/{POOL_SIZE}',
    fontsize=10
)
plt.tight_layout()
plt.savefig('screen_10_trials.png', dpi=120, bbox_inches='tight')
plt.show()
print('✓ Saved screen_10_trials.png')
print(df_results.to_string(index=False))


# ==============================================================================

# ============================================================
# CELL 14: Save the figure for a selected trial with gene annotations
# ============================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import requests

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family']    = 'DejaVu Sans'
mpl.rcParams['font.size']      = 28
mpl.rcParams['axes.titlesize'] = 52
mpl.rcParams['axes.labelsize'] = 44
mpl.rcParams['xtick.labelsize']= 40
mpl.rcParams['ytick.labelsize']= 40
mpl.rcParams['legend.fontsize']= 38

# ── 1. Advance RNG to correct trial state ─────────────────────────────────────
_rng = np.random.default_rng(SEED + 111)
TARGET_TRIAL_IDX = 6

for _t in range(TARGET_TRIAL_IDX):
    _row        = df_results.iloc[_t]
    _rec_id_tmp = _row['receptor']
    _pool_tmp   = all_proteins[all_proteins['id'] != _rec_id_tmp].reset_index(drop=True)
    _known_a_tmp = set(test_pos[test_pos['id_a'] == _rec_id_tmp]['id_b'].tolist())
    _known_b_tmp = set(test_pos[test_pos['id_b'] == _rec_id_tmp]['id_a'].tolist())
    _kb_tmp      = _known_a_tmp | _known_b_tmp
    _kr_tmp  = _pool_tmp[_pool_tmp['id'].isin(_kb_tmp)]
    _dp_tmp  = _pool_tmp[~_pool_tmp['id'].isin(_kb_tmp)]
    _nf_tmp  = min(500 - len(_kr_tmp), len(_dp_tmp))
    _rng.choice(len(_dp_tmp), _nf_tmp, replace=False)
    _rng.permutation(min(500, len(_kr_tmp) + _nf_tmp))

# ── 2. Build pool ──────────────────────────────────────────────────────────────
_row         = df_results.iloc[TARGET_TRIAL_IDX]
receptor_id  = _row['receptor']
sel_auc      = float(_row['auc'])
sel_f1       = float(_row['micro_f1'])

_rec_rows    = df_shs27[(df_shs27['id_a'] == receptor_id) |
                         (df_shs27['id_b'] == receptor_id)]
_rec_row     = _rec_rows.iloc[0]
receptor_seq = _rec_row['seq_a'] if _rec_row['id_a'] == receptor_id \
               else _rec_row['seq_b']

_known_a      = set(test_pos[test_pos['id_a'] == receptor_id]['id_b'].tolist())
_known_b      = set(test_pos[test_pos['id_b'] == receptor_id]['id_a'].tolist())
known_binders = _known_a | _known_b

pool       = all_proteins[all_proteins['id'] != receptor_id].reset_index(drop=True)
known_rows = pool[pool['id'].isin(known_binders)].reset_index(drop=True)
decoy_pool = pool[~pool['id'].isin(known_binders)].reset_index(drop=True)
n_fill     = min(500 - len(known_rows), len(decoy_pool))
d_idx      = _rng.choice(len(decoy_pool), n_fill, replace=False)
decoys     = decoy_pool.iloc[d_idx].reset_index(drop=True)

candidates = pd.concat([known_rows, decoys], ignore_index=True)
shuf       = _rng.permutation(len(candidates))
candidates = candidates.iloc[shuf].reset_index(drop=True)
candidates['DATASET_LABEL'] = candidates['id'].apply(
    lambda cid: 1 if cid in known_binders else 0
)

# ── 3. Features + inference ────────────────────────────────────────────────────
feats = []
ea    = emb_map_35m.get(receptor_id, np.zeros(EMB_DIM, dtype=np.float32))
for _, crow in candidates.iterrows():
    eb     = emb_map_35m.get(crow['id'], np.zeros(EMB_DIM, dtype=np.float32))
    scores = four_scores(clean_seq(receptor_seq), clean_seq(crow['seq']))
    feats.append(np.concatenate([ea, eb, ea - eb, ea * eb, scores]).astype(np.float32))

X_s = scaler.transform(np.stack(feats).astype(np.float32)).astype(np.float32)
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_s, device=DEVICE))[1]
    probs  = torch.sigmoid(logits).cpu().numpy()

df_t = pd.DataFrame({
    'CANDIDATE_ID' : candidates['id'].values,
    'DATASET_LABEL': candidates['DATASET_LABEL'].values,
    'PROB_BINDING' : probs,
})
df_t = df_t.sort_values('PROB_BINDING', ascending=False).reset_index(drop=True)
df_t.index      = df_t.index + 1
df_t.index.name = 'RANK'

n_true           = int(df_t['DATASET_LABEL'].sum())
ranks_of_binders = df_t[df_t['DATASET_LABEL'] == 1].index.tolist()
best_rank        = min(ranks_of_binders)

# ── 4. Resolve gene names ONLY for true binders (plotted annotations) ──────────
def resolve_gene_name(ensp):
    ensp = ensp.replace('9606.', '')
    # Strategy 1: Ensembl xrefs
    try:
        r = requests.get(
            f'https://rest.ensembl.org/xrefs/id/{ensp}',
            headers={'Content-Type': 'application/json'}, timeout=10
        )
        for x in r.json():
            if x.get('dbname') in ('HGNC', 'EntrezGene', 'Uniprot_gn'):
                return x.get('display_id')
    except Exception:
        pass
    # Strategy 2: MyGene.info
    try:
        r = requests.get(
            f'https://mygene.info/v3/query?q=ensembl.protein:{ensp}&species=human&fields=symbol',
            timeout=10
        )
        hits = r.json().get('hits', [])
        if hits:
            return hits[0].get('symbol', ensp)
    except Exception:
        pass
    return ensp   # fallback to raw ID

# Only the true binder rows will be annotated on the plot
binder_ids = df_t[df_t['DATASET_LABEL'] == 1]['CANDIDATE_ID'].tolist()
print(f'Resolving {len(binder_ids)} gene name(s) ...')
gene_map = {raw_id: resolve_gene_name(raw_id) for raw_id in binder_ids}
print('  ' + ', '.join(f'{k} → {v}' for k, v in gene_map.items()))

receptor_gene = resolve_gene_name(receptor_id)
print(f'  {receptor_id} → {receptor_gene}  (receptor)')

df_t['GENE'] = df_t['CANDIDATE_ID'].map(gene_map).fillna('')
df_annotated = df_t.copy()

# ── 5. Plot ────────────────────────────────────────────────────────────────────
BLUE = '#4878CF'
RED  = '#D65F5F'
GREY = '#888888'

FS_TITLE    = 52
FS_LABEL    = 44
FS_TICK     = 40
FS_LEGEND   = 38
FS_ANNOT    = 36
FS_SUPTITLE = 38

fig, axes = plt.subplots(1, 2, figsize=(36, 14))

# Left — histogram
ax = axes[0]
decoy_probs  = df_annotated[df_annotated['DATASET_LABEL'] == 0]['PROB_BINDING'].values
binder_probs = df_annotated[df_annotated['DATASET_LABEL'] == 1]['PROB_BINDING'].values
ax.hist(decoy_probs,  bins=30, color=BLUE, alpha=0.75,
        label=f'Non-binders  (n={len(decoy_probs)})',
        edgecolor='white', linewidth=0.4)
ax.hist(binder_probs, bins=max(len(binder_probs), 5), color=RED, alpha=0.90,
        label=f'True binders  (n={len(binder_probs)})',
        edgecolor='white', linewidth=0.4)
ax.axvline(0.5, color=GREY, lw=3.0, ls='--', label='Decision threshold (0.5)')
ax.set_xlabel('Predicted binding probability', fontsize=FS_LABEL)
ax.set_ylabel('Candidate count',               fontsize=FS_LABEL)
ax.set_title('(a)  Binding score distribution', fontsize=FS_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right — ranked curve
ax = axes[1]
mask0 = df_annotated['DATASET_LABEL'] == 0
mask1 = df_annotated['DATASET_LABEL'] == 1
ax.scatter(df_annotated[mask0].index, df_annotated[mask0]['PROB_BINDING'],
           color=BLUE, s=25, alpha=0.45, label='Non-binder (label=0)')
ax.scatter(df_annotated[mask1].index, df_annotated[mask1]['PROB_BINDING'],
           color=RED, s=600, zorder=5, marker='*', label='True binder (label=1)')

# Annotations — staggered BELOW the star, no overlap
for i, r in enumerate(ranks_of_binders):
    _y    = float(df_annotated.loc[r, 'PROB_BINDING'])
    _gene = df_annotated.loc[r, 'GENE'] or str(df_annotated.loc[r, 'CANDIDATE_ID'])

    # Place text below and to the right, staggered per binder
    _x_offset = min(r + 60 + i * 80, 420)
    _y_offset = _y - 0.18 - i * 0.12   # negative = below the point

    ax.annotate(f'{_gene}  (rank {r})',
                xy=(r, _y),
                xytext=(_x_offset, _y_offset),
                fontsize=FS_ANNOT, color=RED, fontweight='bold',
                va='top',
                arrowprops=dict(arrowstyle='->', color=RED, lw=2.5,
                                connectionstyle='arc3,rad=-0.1'))

ax.axhline(0.5, color=GREY, lw=3.0, ls='--', label='Threshold (0.5)')
ax.set_xlabel('Candidate rank  (1 = highest predicted probability)', fontsize=FS_LABEL)
ax.set_ylabel('Predicted binding probability',                        fontsize=FS_LABEL)
ax.set_title('(b)  Ranked candidate curve', fontsize=FS_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Shorter two-line suptitle
fig.suptitle(
    f'Receptor: {receptor_gene}  |  AUC = {sel_auc:.4f}  |  Micro-F1 = {sel_f1:.4f}\n'
    f'Best recovery rank = {best_rank} / 500  |  '
    f'{n_true} known binder(s) + {500 - n_true} decoys',
    fontsize=FS_SUPTITLE, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('selected_screen_annotated.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Saved selected_screen_annotated.png')
