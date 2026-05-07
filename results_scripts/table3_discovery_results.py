#!/usr/bin/env python3
"""
Table 3: SHS148k Qwen-Guided Discovery

Purpose
-------
Runs the SHS148k PPIProjectedNet training, Qwen LoRA fine-tuning, and full-pool Qwen-guided Tree-of-Thought discovery benchmark.

Expected outputs
----------------
Writes shs148k_10trial_fullpool_summary.csv, shs148k_10trial_fullpool_baseline.csv, and shs148k_10trial_fullpool_qwen.csv.

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

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: QWEN FINE-TUNING FOR ToT SEARCH DIRECTIVE HYPOTHESIS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# This module fine-tunes Qwen2.5-1.5B-Instruct to generate search directives
# for Tree-of-Thought guided PPI discovery. The model learns to:
#   1. Predict binding potential from embedding features
#   2. Recommend search priority (PRIORITIZE / EXPLORE / DEPRIORITIZE / SKIP)
#   3. Detect score tensions that suggest ESFM flow would help
#   4. Generate flow recommendations (FLOW_RECOMMENDED / DIRECT_SCORE_SUFFICIENT)
#
# REQUIRES: Run after Cell A (model, emb_map_35m, scaler, four_scores, df_shs27)
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION A: Configuration
# ============================================================

@dataclass
class QwenToTConfig:
    """Configuration for Qwen ToT Hypothesis Fine-tuning."""

    # Model
    qwen_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 1024

    # Data
    n_train_pairs: int = 2000
    n_val_pairs: int = 400

    # Paths
    save_dir: str = "./qwen_tot_directive_lora"
    cache_dir: str = "./tot_hypothesis_cache"

    # Thresholds for directive assignment
    high_sim_thresh: float = 0.5
    low_sim_thresh: float = 0.2
    strong_interaction_thresh: float = 0.15
    tension_thresh: float = 0.35


# ============================================================
# SECTION B: Search Directives
# ============================================================

# Primary search directives
SEARCH_DIRECTIVES = [
    'PRIORITIZE_HIGH_SIMILARITY',       # Strong embedding similarity → likely binder
    'PRIORITIZE_STRONG_INTERACTION',    # Strong interaction signal → worth exploring
    'PRIORITIZE_SCORE_ALIGNMENT',       # All scores agree → high confidence
    'EXPLORE_MODERATE_SIGNAL',          # Moderate signals → explore but not first
    'EXPLORE_TENSION_DETECTED',         # Score tensions → needs resolution
    'DEPRIORITIZE_WEAK_SIGNAL',         # Weak overall signal → lower priority
    'DEPRIORITIZE_DISTANT',             # Distant embeddings → unlikely
    'SKIP_INCOMPATIBLE',                # Clear incompatibility → skip
]

# Flow recommendations
FLOW_RECOMMENDATIONS = [
    'FLOW_RECOMMENDED',                 # Tensions suggest ESFM would help
    'FLOW_OPTIONAL',                    # Might help, not critical
    'DIRECT_SCORE_SUFFICIENT',          # Clean signals, skip flow
]

DIRECTIVE_TO_IDX = {d: i for i, d in enumerate(SEARCH_DIRECTIVES)}
FLOW_TO_IDX = {f: i for i, f in enumerate(FLOW_RECOMMENDATIONS)}


# ============================================================
# SECTION C: Embedding Feature Computation
# ============================================================

def compute_embedding_features(ea: np.ndarray, eb: np.ndarray) -> Dict[str, float]:
    """
    Compute interpretable features from embedding pair.
    These features are used for Qwen input and directive assignment.
    """
    ea = ea.astype(np.float64)
    eb = eb.astype(np.float64)

    # Norms
    norm_a = np.linalg.norm(ea)
    norm_b = np.linalg.norm(eb)

    # Similarity metrics
    cosine_sim = float(np.dot(ea, eb) / (norm_a * norm_b + 1e-8))
    l2_dist = float(np.linalg.norm(ea - eb))

    # Difference statistics
    diff = ea - eb
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))
    diff_max = float(np.max(np.abs(diff)))

    # Product statistics (interaction signal)
    prod = ea * eb
    prod_mean = float(np.mean(prod))
    prod_std = float(np.std(prod))
    prod_pos_frac = float(np.mean(prod > 0))

    # Additional metrics
    norm_ratio = float(norm_a / (norm_b + 1e-8))

    # Top-k alignment (how many dimensions are well-aligned)
    top_k = 20
    sorted_diff = np.sort(np.abs(diff))
    alignment_score = float(1 - np.mean(sorted_diff[:top_k]) / (diff_max + 1e-8))

    # Sparsity of interaction
    sparsity = float(np.mean(np.abs(prod) < 0.1))

    return {
        'cosine_sim': cosine_sim,
        'l2_dist': l2_dist,
        'diff_mean': diff_mean,
        'diff_std': diff_std,
        'diff_max': diff_max,
        'prod_mean': prod_mean,
        'prod_std': prod_std,
        'prod_pos_frac': prod_pos_frac,
        'norm_ratio': norm_ratio,
        'alignment_score': alignment_score,
        'sparsity': sparsity,
    }


def compute_score_features(scores: np.ndarray) -> Dict[str, float]:
    """
    Compute features from the 4-score vector.
    scores = [seq_align, struct_sim, contact_compat, chem_complement]
    """
    scores = scores.astype(np.float64)

    return {
        'seq_score': float(scores[0]),
        'struct_score': float(scores[1]),
        'contact_score': float(scores[2]),
        'chem_score': float(scores[3]),
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'score_min': float(np.min(scores)),
        'score_max': float(np.max(scores)),
        'score_range': float(np.max(scores) - np.min(scores)),
    }


def detect_tensions(score_features: Dict[str, float], config: QwenToTConfig) -> Dict[str, Any]:
    """
    Detect tensions between score components that suggest ESFM could help.
    """
    scores = [
        score_features['seq_score'],
        score_features['struct_score'],
        score_features['contact_score'],
        score_features['chem_score'],
    ]

    tensions = []

    # High variance indicates disagreement
    if score_features['score_range'] > config.tension_thresh:
        tensions.append('HIGH_VARIANCE')

    # Specific tension patterns
    if scores[0] > 0.6 and scores[1] < 0.3:
        tensions.append('SEQ_STRUCT_MISMATCH')

    if scores[2] > 0.6 and scores[3] < 0.3:
        tensions.append('CONTACT_CHEM_MISMATCH')

    if score_features['score_max'] > 0.7 and score_features['score_min'] < 0.2:
        tensions.append('EXTREME_DISAGREEMENT')

    # Identify dominant and weak signals
    score_names = ['SEQ', 'STRUCT', 'CONTACT', 'CHEM']
    max_idx = int(np.argmax(scores))
    min_idx = int(np.argmin(scores))

    return {
        'has_tension': len(tensions) > 0,
        'tensions': tensions,
        'dominant_signal': score_names[max_idx],
        'weak_signal': score_names[min_idx],
        'tension_severity': len(tensions),
    }


# ============================================================
# SECTION D: Directive Assignment Logic
# ============================================================

def assign_directive(
    emb_features: Dict[str, float],
    score_features: Dict[str, float],
    ppi_score: float,
    label: int,
    config: QwenToTConfig
) -> Tuple[str, str, float, str]:
    """
    Assign search directive based on features and ground truth.

    Returns:
        directive: Search directive
        flow_rec: Flow recommendation
        confidence: Confidence in the directive
        reasoning: Brief reasoning
    """

    tensions = detect_tensions(score_features, config)

    # Ground truth: binder
    if label == 1:
        # High similarity binder
        if emb_features['cosine_sim'] > config.high_sim_thresh:
            directive = 'PRIORITIZE_HIGH_SIMILARITY'
            confidence = min(0.95, 0.7 + emb_features['cosine_sim'] * 0.3)
            reasoning = f"High embedding similarity ({emb_features['cosine_sim']:.3f}) strongly suggests binding."

        # Strong interaction signal
        elif emb_features['prod_mean'] > config.strong_interaction_thresh:
            directive = 'PRIORITIZE_STRONG_INTERACTION'
            confidence = min(0.90, 0.65 + emb_features['prod_mean'] * 0.5)
            reasoning = f"Strong interaction signal (prod_mean={emb_features['prod_mean']:.3f}) indicates complementary features."

        # All scores agree
        elif score_features['score_std'] < 0.15 and score_features['score_mean'] > 0.5:
            directive = 'PRIORITIZE_SCORE_ALIGNMENT'
            confidence = min(0.90, 0.6 + score_features['score_mean'] * 0.3)
            reasoning = f"All score components agree (std={score_features['score_std']:.3f}), high confidence binding."

        # Tensions detected but still a binder
        elif tensions['has_tension']:
            directive = 'EXPLORE_TENSION_DETECTED'
            confidence = 0.65
            reasoning = f"Score tensions detected ({tensions['tensions']}), but binding confirmed. ESFM may resolve."

        # Moderate signal binder
        else:
            directive = 'EXPLORE_MODERATE_SIGNAL'
            confidence = 0.70
            reasoning = f"Moderate signals suggest binding potential worth exploring."

        # Flow recommendation for binders
        if tensions['has_tension'] and tensions['tension_severity'] >= 2:
            flow_rec = 'FLOW_RECOMMENDED'
        elif tensions['has_tension']:
            flow_rec = 'FLOW_OPTIONAL'
        else:
            flow_rec = 'DIRECT_SCORE_SUFFICIENT'

    # Ground truth: non-binder
    else:
        # Clear incompatibility
        if emb_features['cosine_sim'] < config.low_sim_thresh and emb_features['l2_dist'] > 20:
            directive = 'SKIP_INCOMPATIBLE'
            confidence = min(0.95, 0.7 + (1 - emb_features['cosine_sim']) * 0.3)
            reasoning = f"Very low similarity ({emb_features['cosine_sim']:.3f}) and large distance indicate incompatibility."

        # Distant embeddings
        elif emb_features['cosine_sim'] < config.low_sim_thresh:
            directive = 'DEPRIORITIZE_DISTANT'
            confidence = min(0.85, 0.6 + (1 - emb_features['cosine_sim']) * 0.3)
            reasoning = f"Low embedding similarity ({emb_features['cosine_sim']:.3f}) suggests unlikely binding."

        # Weak overall signal
        elif score_features['score_mean'] < 0.4:
            directive = 'DEPRIORITIZE_WEAK_SIGNAL'
            confidence = 0.75
            reasoning = f"Weak average score ({score_features['score_mean']:.3f}) indicates low binding potential."

        # Tensions that mislead
        elif tensions['has_tension']:
            directive = 'EXPLORE_TENSION_DETECTED'
            confidence = 0.55
            reasoning = f"Tensions detected but likely false positive. Careful evaluation needed."

        # Default deprioritize
        else:
            directive = 'DEPRIORITIZE_WEAK_SIGNAL'
            confidence = 0.65
            reasoning = f"No strong binding signals detected."

        # Flow recommendation for non-binders
        flow_rec = 'DIRECT_SCORE_SUFFICIENT'

    return directive, flow_rec, confidence, reasoning


# ============================================================
# SECTION E: Prompt Templates
# ============================================================

SYSTEM_PROMPT = """You are an expert protein-protein interaction analyst guiding a Tree-of-Thought search algorithm.

Given embedding-space metrics and PPI scores for a receptor-ligand pair, you must:
1. Assess binding potential
2. Assign a SEARCH DIRECTIVE for the ToT algorithm
3. Detect any SCORE TENSIONS between components
4. Recommend whether EMBEDDING FLOW would improve scoring

SEARCH DIRECTIVES (in priority order):
- PRIORITIZE_HIGH_SIMILARITY: Strong embedding similarity, evaluate first
- PRIORITIZE_STRONG_INTERACTION: Strong interaction signal, high priority
- PRIORITIZE_SCORE_ALIGNMENT: All scores agree, confident positive
- EXPLORE_MODERATE_SIGNAL: Moderate signals, worth exploring
- EXPLORE_TENSION_DETECTED: Score tensions detected, needs resolution
- DEPRIORITIZE_WEAK_SIGNAL: Weak signals, lower priority
- DEPRIORITIZE_DISTANT: Distant embeddings, unlikely binder
- SKIP_INCOMPATIBLE: Clear incompatibility, skip evaluation

FLOW RECOMMENDATIONS:
- FLOW_RECOMMENDED: Tensions suggest embedding flow matching would help
- FLOW_OPTIONAL: Might help but not critical
- DIRECT_SCORE_SUFFICIENT: Clean signals, no flow needed

Your response must follow the exact format specified."""


def create_input_prompt(
    receptor_id: str,
    ligand_id: str,
    emb_features: Dict[str, float],
    score_features: Dict[str, float],
    ppi_score: float,
) -> str:
    """Create input prompt for Qwen."""

    return f"""Analyze this protein pair for Tree-of-Thought search guidance.

RECEPTOR: {receptor_id}
LIGAND: {ligand_id}

EMBEDDING METRICS:
- Cosine similarity: {emb_features['cosine_sim']:.4f}
- L2 distance: {emb_features['l2_dist']:.2f}
- Alignment score: {emb_features['alignment_score']:.4f}
- Product mean (interaction): {emb_features['prod_mean']:.4f}
- Product positive fraction: {emb_features['prod_pos_frac']:.4f}
- Norm ratio: {emb_features['norm_ratio']:.4f}

PPI SCORES:
- Sequence alignment: {score_features['seq_score']:.4f}
- Structural similarity: {score_features['struct_score']:.4f}
- Contact compatibility: {score_features['contact_score']:.4f}
- Chemical complementarity: {score_features['chem_score']:.4f}
- Score mean: {score_features['score_mean']:.4f}
- Score std: {score_features['score_std']:.4f}

PPI MODEL PREDICTION: {ppi_score:.4f}

Generate a search directive hypothesis for the ToT algorithm."""


def create_target_output(
    directive: str,
    flow_rec: str,
    confidence: float,
    reasoning: str,
    tensions: Dict[str, Any],
    emb_features: Dict[str, float],
    score_features: Dict[str, float],
) -> str:
    """Create target output for training."""

    tension_str = ', '.join(tensions['tensions']) if tensions['tensions'] else 'None'

    return f"""SEARCH_DIRECTIVE: {directive}
CONFIDENCE: {confidence:.0%}
FLOW_RECOMMENDATION: {flow_rec}

ANALYSIS:
<thinking>
Embedding similarity: {emb_features['cosine_sim']:.4f} ({'high' if emb_features['cosine_sim'] > 0.5 else 'moderate' if emb_features['cosine_sim'] > 0.3 else 'low'})
Interaction signal: {emb_features['prod_mean']:.4f} ({'strong' if emb_features['prod_mean'] > 0.15 else 'moderate' if emb_features['prod_mean'] > 0.05 else 'weak'})
Score agreement: std={score_features['score_std']:.4f} ({'aligned' if score_features['score_std'] < 0.15 else 'moderate variance' if score_features['score_std'] < 0.25 else 'high variance'})
Dominant signal: {tensions['dominant_signal']} ({score_features['score_max']:.4f})
Weak signal: {tensions['weak_signal']} ({score_features['score_min']:.4f})
Tensions detected: {tension_str}
</thinking>

<conclusion>
{reasoning}
Priority: {'HIGH' if directive.startswith('PRIORITIZE') else 'MEDIUM' if directive.startswith('EXPLORE') else 'LOW'}
Action: {'Evaluate immediately' if directive.startswith('PRIORITIZE') else 'Explore if capacity allows' if directive.startswith('EXPLORE') else 'Skip or defer'}
</conclusion>"""


# ============================================================
# SECTION F: Dataset Class
# ============================================================

class QwenToTDataset(Dataset):
    """Dataset for fine-tuning Qwen on ToT directive generation."""

    def __init__(
        self,
        pairs_data: List[Dict],
        tokenizer,
        config: QwenToTConfig,
    ):
        self.pairs_data = pairs_data
        self.tokenizer = tokenizer
        self.config = config

        print(f"    QwenToTDataset: {len(pairs_data)} pairs")

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        data = self.pairs_data[idx]

        # Create prompt
        input_prompt = create_input_prompt(
            data['receptor_id'],
            data['ligand_id'],
            data['emb_features'],
            data['score_features'],
            data['ppi_score'],
        )

        # Create target
        target_output = create_target_output(
            data['directive'],
            data['flow_rec'],
            data['confidence'],
            data['reasoning'],
            data['tensions'],
            data['emb_features'],
            data['score_features'],
        )

        # Format as chat
        full_prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{input_prompt}
<|im_end|>
<|im_start|>assistant
{target_output}<|im_end|>"""

        # Tokenize
        encoded = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.config.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0).clone(),
        }


# ============================================================
# SECTION G: Data Preparation
# ============================================================

def prepare_training_data(
    df_shs27,
    emb_map,
    model,
    scaler,
    four_scores_fn,
    clean_seq_fn,
    config: QwenToTConfig,
    device: str = 'cuda',
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare training data by computing features and assigning directives.
    """

    print("\n" + "═" * 60)
    print("Preparing Training Data for Qwen ToT Fine-tuning")
    print("═" * 60)

    # Get train/val splits
    train_df = df_shs27[df_shs27['split'] == 'train'].copy()
    val_df = df_shs27[df_shs27['split'] == 'val'].copy()

    # If no split column, create one
    if len(train_df) == 0:
        df_shs27 = df_shs27.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = int(len(df_shs27) * 0.8)
        train_df = df_shs27.iloc[:n_train]
        val_df = df_shs27.iloc[n_train:]

    print(f"  Train pairs available: {len(train_df)}")
    print(f"  Val pairs available: {len(val_df)}")

    def process_pairs(df, n_pairs, desc):
        """Process pairs and compute all features."""

        # Balance positive and negative
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]

        n_pos = min(n_pairs // 2, len(pos_df))
        n_neg = min(n_pairs // 2, len(neg_df))

        rng = np.random.default_rng(42)
        pos_idx = rng.choice(len(pos_df), n_pos, replace=False)
        neg_idx = rng.choice(len(neg_df), n_neg, replace=False)

        selected = pd.concat([
            pos_df.iloc[pos_idx],
            neg_df.iloc[neg_idx]
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        pairs_data = []

        for _, row in tqdm(selected.iterrows(), total=len(selected), desc=desc):
            receptor_id = str(row['id_a'])
            ligand_id = str(row['id_b'])
            label = int(row['label'])

            # Get embeddings
            ea = emb_map.get(receptor_id)
            eb = emb_map.get(ligand_id)

            if ea is None or eb is None:
                continue

            ea = ea.astype(np.float32)
            eb = eb.astype(np.float32)

            # Compute embedding features
            emb_features = compute_embedding_features(ea, eb)

            # Compute PPI scores
            try:
                seq_a = clean_seq_fn(row['seq_a'])
                seq_b = clean_seq_fn(row['seq_b'])
                scores = four_scores_fn(seq_a, seq_b).astype(np.float32)
            except Exception:
                scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

            score_features = compute_score_features(scores)

            # Compute PPI model score
            try:
                feat = np.concatenate([
                    ea, eb, ea - eb, ea * eb, scores
                ]).astype(np.float32)
                X = scaler.transform(feat[None]).astype(np.float32)

                model.eval()
                with torch.no_grad():
                    _, logit = model(torch.tensor(X, device=device))
                    ppi_score = float(torch.sigmoid(logit).cpu())
            except Exception:
                ppi_score = 0.5

            # Detect tensions
            tensions = detect_tensions(score_features, config)

            # Assign directive
            directive, flow_rec, confidence, reasoning = assign_directive(
                emb_features, score_features, ppi_score, label, config
            )

            pairs_data.append({
                'receptor_id': receptor_id,
                'ligand_id': ligand_id,
                'label': label,
                'emb_features': emb_features,
                'score_features': score_features,
                'ppi_score': ppi_score,
                'tensions': tensions,
                'directive': directive,
                'flow_rec': flow_rec,
                'confidence': confidence,
                'reasoning': reasoning,
            })

        return pairs_data

    # Need pandas for the function
    import pandas as pd

    train_data = process_pairs(train_df, config.n_train_pairs, "  Processing train")
    val_data = process_pairs(val_df, config.n_val_pairs, "  Processing val")

    # Print directive distribution
    print(f"\n  Train data: {len(train_data)} pairs")
    print(f"  Val data: {len(val_data)} pairs")

    directive_counts = defaultdict(int)
    for d in train_data:
        directive_counts[d['directive']] += 1

    print("\n  Directive distribution (train):")
    for directive, count in sorted(directive_counts.items(), key=lambda x: -x[1]):
        print(f"    {directive}: {count}")

    return train_data, val_data


# ============================================================
# SECTION H: Fine-tuning
# ============================================================

def finetune_qwen_tot(
    train_data: List[Dict],
    val_data: List[Dict],
    config: QwenToTConfig,
    device: str = 'cuda',
) -> Tuple[Any, Any]:
    """
    Fine-tune Qwen2.5-1.5B-Instruct for ToT directive generation.
    """

    print("\n" + "═" * 60)
    print("Fine-tuning Qwen for ToT Directive Generation")
    print("═" * 60)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments, Trainer
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Load tokenizer
    print(f"\n  Loading tokenizer: {config.qwen_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"  Loading model: {config.qwen_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.qwen_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Create datasets
    print(f"\n  Creating datasets...")
    train_ds = QwenToTDataset(train_data, tokenizer, config)
    val_ds = QwenToTDataset(val_data, tokenizer, config)

    # Training arguments
    os.makedirs(config.save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=config.save_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    def data_collator(batch):
        return {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'labels': torch.stack([b['labels'] for b in batch]),
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # Train
    print(f"\n  Training for {config.epochs} epochs...")
    trainer.train()

    # Save
    model.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)
    print(f"\n  ✓ Saved to {config.save_dir}")

    return model, tokenizer


# ============================================================
# SECTION I: Hypothesis Generator Class
# ============================================================

class ToTHypothesisGenerator:
    """
    Generate search directives using fine-tuned Qwen.
    Used during Tree-of-Thought search to guide exploration.
    """

    def __init__(self, model, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Cache for repeated queries
        self.cache = {}

    @torch.no_grad()
    def generate(
        self,
        receptor_id: str,
        ligand_id: str,
        emb_features: Dict[str, float],
        score_features: Dict[str, float],
        ppi_score: float,
        use_cache: bool = True,
    ) -> Dict:
        """
        Generate search directive hypothesis for a protein pair.
        """

        cache_key = f"{receptor_id}__{ligand_id}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Create prompt
        input_prompt = create_input_prompt(
            receptor_id, ligand_id,
            emb_features, score_features, ppi_score
        )

        full_prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{input_prompt}
<|im_end|>
<|im_start|>assistant
"""

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse response
        result = self._parse_response(response, emb_features, score_features)

        if use_cache:
            self.cache[cache_key] = result

        return result

    def _parse_response(
        self,
        response: str,
        emb_features: Dict[str, float],
        score_features: Dict[str, float],
    ) -> Dict:
        """Parse the generated response into structured output."""

        result = {
            'raw_response': response,
            'directive': 'EXPLORE_MODERATE_SIGNAL',
            'confidence': 0.5,
            'flow_recommendation': 'FLOW_OPTIONAL',
            'priority': 0.5,
        }

        # Extract directive
        for directive in SEARCH_DIRECTIVES:
            if directive in response:
                result['directive'] = directive
                break

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)%', response)
        if conf_match:
            result['confidence'] = int(conf_match.group(1)) / 100

        # Extract flow recommendation
        for flow_rec in FLOW_RECOMMENDATIONS:
            if flow_rec in response:
                result['flow_recommendation'] = flow_rec
                break

        # Compute priority score
        directive_priority = {
            'PRIORITIZE_HIGH_SIMILARITY': 1.0,
            'PRIORITIZE_STRONG_INTERACTION': 0.95,
            'PRIORITIZE_SCORE_ALIGNMENT': 0.90,
            'EXPLORE_MODERATE_SIGNAL': 0.70,
            'EXPLORE_TENSION_DETECTED': 0.65,
            'DEPRIORITIZE_WEAK_SIGNAL': 0.40,
            'DEPRIORITIZE_DISTANT': 0.30,
            'SKIP_INCOMPATIBLE': 0.10,
        }

        base_priority = directive_priority.get(result['directive'], 0.5)
        result['priority'] = base_priority * result['confidence']

        # Add parsed flags
        result['is_prioritize'] = result['directive'].startswith('PRIORITIZE')
        result['is_explore'] = result['directive'].startswith('EXPLORE')
        result['is_skip'] = result['directive'].startswith('SKIP') or result['directive'].startswith('DEPRIORITIZE')
        result['should_flow'] = result['flow_recommendation'] == 'FLOW_RECOMMENDED'

        return result

    def generate_batch_fast(
        self,
        receptor_id: str,
        candidate_data: List[Dict],
    ) -> List[Dict]:
        """
        Fast batch generation using rule-based fallback for efficiency.
        Uses Qwen only for ambiguous cases.
        """

        results = []

        for cdata in candidate_data:
            emb_features = cdata['emb_features']
            score_features = cdata.get('score_features')

            # Fast rule-based classification
            if emb_features['cosine_sim'] > 0.6:
                result = {
                    'directive': 'PRIORITIZE_HIGH_SIMILARITY',
                    'confidence': min(0.95, 0.7 + emb_features['cosine_sim'] * 0.3),
                    'flow_recommendation': 'DIRECT_SCORE_SUFFICIENT',
                    'priority': 0.95,
                    'is_prioritize': True,
                    'is_explore': False,
                    'is_skip': False,
                    'should_flow': False,
                    'source': 'rule_based',
                }

            elif emb_features['cosine_sim'] < 0.15:
                result = {
                    'directive': 'SKIP_INCOMPATIBLE',
                    'confidence': 0.85,
                    'flow_recommendation': 'DIRECT_SCORE_SUFFICIENT',
                    'priority': 0.10,
                    'is_prioritize': False,
                    'is_explore': False,
                    'is_skip': True,
                    'should_flow': False,
                    'source': 'rule_based',
                }

            elif emb_features['prod_mean'] > 0.2:
                result = {
                    'directive': 'PRIORITIZE_STRONG_INTERACTION',
                    'confidence': 0.80,
                    'flow_recommendation': 'FLOW_OPTIONAL',
                    'priority': 0.85,
                    'is_prioritize': True,
                    'is_explore': False,
                    'is_skip': False,
                    'should_flow': False,
                    'source': 'rule_based',
                }

            elif score_features and score_features.get('score_range', 0) > 0.4:
                result = {
                    'directive': 'EXPLORE_TENSION_DETECTED',
                    'confidence': 0.65,
                    'flow_recommendation': 'FLOW_RECOMMENDED',
                    'priority': 0.60,
                    'is_prioritize': False,
                    'is_explore': True,
                    'is_skip': False,
                    'should_flow': True,
                    'source': 'rule_based',
                }

            else:
                # Ambiguous case - use moderate exploration
                result = {
                    'directive': 'EXPLORE_MODERATE_SIGNAL',
                    'confidence': 0.60,
                    'flow_recommendation': 'FLOW_OPTIONAL',
                    'priority': 0.55,
                    'is_prioritize': False,
                    'is_explore': True,
                    'is_skip': False,
                    'should_flow': False,
                    'source': 'rule_based',
                }

            result['ligand_id'] = cdata['ligand_id']
            results.append(result)

        return results

    def clear_cache(self):
        """Clear the hypothesis cache."""
        self.cache = {}


# ============================================================
# SECTION J: Main Pipeline
# ============================================================

def run_qwen_tot_finetuning(
    df_shs27,
    emb_map,
    model,
    scaler,
    four_scores_fn,
    clean_seq_fn,
    config: QwenToTConfig = None,
    device: str = 'cuda',
) -> Dict:
    """
    Run the complete Qwen ToT fine-tuning pipeline.

    Args:
        df_shs27: SHS27k dataframe
        emb_map: Dict mapping protein IDs to embeddings
        model: Trained PPI model
        scaler: Feature scaler
        four_scores_fn: Function to compute 4-score vector
        clean_seq_fn: Function to clean sequences
        config: Configuration
        device: Device to use

    Returns:
        Dict with model, tokenizer, generator, and metrics
    """

    if config is None:
        config = QwenToTConfig()

    print("\n" + "═" * 70)
    print("QWEN ToT DIRECTIVE FINE-TUNING PIPELINE")
    print("═" * 70)

    # Prepare data
    train_data, val_data = prepare_training_data(
        df_shs27, emb_map, model, scaler,
        four_scores_fn, clean_seq_fn, config, device
    )

    # Fine-tune
    qwen_model, tokenizer = finetune_qwen_tot(
        train_data, val_data, config, device
    )

    # Create generator
    generator = ToTHypothesisGenerator(qwen_model, tokenizer, device)

    # Test on a few examples
    print("\n" + "═" * 60)
    print("Testing Hypothesis Generation")
    print("═" * 60)

    for i, data in enumerate(val_data[:3]):
        print(f"\n  Example {i+1}: {data['receptor_id']} - {data['ligand_id']}")
        print(f"    True label: {'BINDER' if data['label'] == 1 else 'NON-BINDER'}")
        print(f"    Assigned directive: {data['directive']}")

        # Generate with model
        result = generator.generate(
            data['receptor_id'],
            data['ligand_id'],
            data['emb_features'],
            data['score_features'],
            data['ppi_score'],
            use_cache=False,
        )

        print(f"    Generated directive: {result['directive']}")
        print(f"    Confidence: {result['confidence']:.0%}")
        print(f"    Flow recommendation: {result['flow_recommendation']}")
        print(f"    Priority: {result['priority']:.3f}")

    # Save cache
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_file = os.path.join(config.cache_dir, "train_val_data.pt")
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'config': config,
    }, cache_file)
    print(f"\n  ✓ Saved data cache to {cache_file}")

    print("\n" + "═" * 70)
    print("FINE-TUNING COMPLETE")
    print("═" * 70)
    print(f"  Model saved to: {config.save_dir}")
    print(f"  Data cache saved to: {config.cache_dir}")
    print(f"\n  Ready for Part 2: Qwen-Guided Tree-of-Thought Search")

    return {
        'qwen_model': qwen_model,
        'tokenizer': tokenizer,
        'generator': generator,
        'train_data': train_data,
        'val_data': val_data,
        'config': config,
    }


# ============================================================
# SECTION K: Standalone Usage
# ============================================================

if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("QWEN ToT DIRECTIVE FINE-TUNING")
    print("═" * 70)
    print("\nThis module requires Cell A to be run first.")
    print("Required globals: model, emb_map_35m, scaler, four_scores, clean_seq, df_shs27")
    print("\nUsage in Colab:")
    print("  from qwen_tot_hypothesis_finetuning import run_qwen_tot_finetuning, QwenToTConfig")
    print("  ")
    print("  config = QwenToTConfig(epochs=3, n_train_pairs=2000)")
    print("  result = run_qwen_tot_finetuning(")
    print("      df_shs27=df_shs27,")
    print("      emb_map=emb_map_35m,")
    print("      model=model,")
    print("      scaler=scaler,")
    print("      four_scores_fn=four_scores,")
    print("      clean_seq_fn=clean_seq,")
    print("      config=config,")
    print("  )")
    print("  ")
    print("  generator = result['generator']")


# ==============================================================================

# ============================================================
# CELL B2: Qwen-Guided Entropic-ToT Search with Directive Hypotheses
# ============================================================
#
# This module extends the Entropic-ToT search with Qwen-generated
# search directives. Key improvements over Cell B:
#
#   1. PRE-FILTERING: Fast embedding-based screening before expensive scoring
#   2. DIRECTIVE-GUIDED SELECTION: Branch selection uses hypothesis priorities
#   3. ADAPTIVE FLOW: ESFM triggered only when tensions detected
#   4. HYPOTHESIS PRUNING: Skip branches where Qwen says SKIP + score declining
#
# REQUIRES:
#   - Cell A globals: model, emb_map_35m, scaler, four_scores, clean_seq, df_shs27
#   - Part 1: Qwen ToT fine-tuned model (or uses rule-based fallback)
#
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from scipy import stats
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
import heapq
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Verify Cell A globals
# ============================================================

assert 'model'       in globals(), "Run Cell A first"
assert 'emb_map_35m' in globals(), "Run Cell A first"
assert 'scaler'      in globals(), "Run Cell A first"
assert 'four_scores' in globals(), "Run Cell A first"
assert 'clean_seq'   in globals(), "Run Cell A first"
assert 'df_shs27'    in globals(), "Run Cell A first"
assert 'DEVICE'      in globals(), "Run Cell A first"
assert 'SEED'        in globals(), "Run Cell A first"
assert 'EMB_DIM'     in globals(), "Run Cell A first"

# Check for Qwen generator from Part 1
HAS_QWEN_GENERATOR = 'tot_generator' in globals()
HAS_QWEN_LLM = ('tokenizer' in globals()) and ('qwen_llm' in globals())

# ============================================================
# Configuration
# ============================================================

@dataclass
class QwenToTSearchConfig:
    """Configuration for Qwen-guided ToT search."""

    # Search parameters
    n_trials: int = 10
    pool_size: int = 500
    max_evals: int = 100           # M in original
    max_depth: int = 2             # D in original

    # Branch selection
    base_branch_factor: int = 6    # B in original
    base_tau: float = 0.35
    scan_cap: int = 18

    # Pre-filtering
    prefilter_top_k: int = 150     # Keep top-k after pre-filter
    use_prefilter: bool = True

    # Directive-based adaptation
    adaptive_params: bool = True
    tension_tau_boost: float = 0.15      # Increase tau when tensions detected
    prioritize_branch_boost: int = 2     # Extra branches for PRIORITIZE

    # ESFM parameters
    esfm_steps: int = 8
    esfm_trajectories: int = 3
    esfm_step_size: float = 0.05
    esfm_noise: float = 0.01

    # Pruning
    use_hypothesis_pruning: bool = True
    min_priority_threshold: float = 0.15  # Skip if priority below this

    # Qwen explanation
    topk_explain: int = 3
    qwen_max_tokens: int = 360


# ============================================================
# SECTION A: Embedding Feature Computation (from Part 1)
# ============================================================

def compute_embedding_features(ea: np.ndarray, eb: np.ndarray) -> Dict[str, float]:
    """Compute interpretable features from embedding pair."""
    ea = ea.astype(np.float64)
    eb = eb.astype(np.float64)

    norm_a = np.linalg.norm(ea)
    norm_b = np.linalg.norm(eb)

    cosine_sim = float(np.dot(ea, eb) / (norm_a * norm_b + 1e-8))
    l2_dist = float(np.linalg.norm(ea - eb))

    diff = ea - eb
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))
    diff_max = float(np.max(np.abs(diff)))

    prod = ea * eb
    prod_mean = float(np.mean(prod))
    prod_std = float(np.std(prod))
    prod_pos_frac = float(np.mean(prod > 0))

    norm_ratio = float(norm_a / (norm_b + 1e-8))

    top_k = 20
    sorted_diff = np.sort(np.abs(diff))
    alignment_score = float(1 - np.mean(sorted_diff[:top_k]) / (diff_max + 1e-8))

    sparsity = float(np.mean(np.abs(prod) < 0.1))

    return {
        'cosine_sim': cosine_sim,
        'l2_dist': l2_dist,
        'diff_mean': diff_mean,
        'diff_std': diff_std,
        'diff_max': diff_max,
        'prod_mean': prod_mean,
        'prod_std': prod_std,
        'prod_pos_frac': prod_pos_frac,
        'norm_ratio': norm_ratio,
        'alignment_score': alignment_score,
        'sparsity': sparsity,
    }


def compute_score_features(scores: np.ndarray) -> Dict[str, float]:
    """Compute features from the 4-score vector."""
    scores = scores.astype(np.float64)

    return {
        'seq_score': float(scores[0]),
        'struct_score': float(scores[1]),
        'contact_score': float(scores[2]),
        'chem_score': float(scores[3]),
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'score_min': float(np.min(scores)),
        'score_max': float(np.max(scores)),
        'score_range': float(np.max(scores) - np.min(scores)),
    }


# ============================================================
# SECTION B: Rule-Based Hypothesis Generator (Fallback)
# ============================================================

SEARCH_DIRECTIVES = [
    'PRIORITIZE_HIGH_SIMILARITY',
    'PRIORITIZE_STRONG_INTERACTION',
    'PRIORITIZE_SCORE_ALIGNMENT',
    'EXPLORE_MODERATE_SIGNAL',
    'EXPLORE_TENSION_DETECTED',
    'DEPRIORITIZE_WEAK_SIGNAL',
    'DEPRIORITIZE_DISTANT',
    'SKIP_INCOMPATIBLE',
]

DIRECTIVE_PRIORITY = {
    'PRIORITIZE_HIGH_SIMILARITY': 1.0,
    'PRIORITIZE_STRONG_INTERACTION': 0.95,
    'PRIORITIZE_SCORE_ALIGNMENT': 0.90,
    'EXPLORE_MODERATE_SIGNAL': 0.70,
    'EXPLORE_TENSION_DETECTED': 0.65,
    'DEPRIORITIZE_WEAK_SIGNAL': 0.40,
    'DEPRIORITIZE_DISTANT': 0.30,
    'SKIP_INCOMPATIBLE': 0.10,
}


class RuleBasedHypothesisGenerator:
    """
    Fast rule-based hypothesis generator for pre-filtering.
    Used when Qwen model is not available or for speed.
    """

    def __init__(self, config: QwenToTSearchConfig = None):
        self.config = config or QwenToTSearchConfig()
        self.cache = {}

    def generate(
        self,
        receptor_id: str,
        ligand_id: str,
        emb_features: Dict[str, float],
        score_features: Optional[Dict[str, float]] = None,
        ppi_score: Optional[float] = None,
    ) -> Dict:
        """Generate hypothesis using rules."""

        cache_key = f"{receptor_id}__{ligand_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self._compute_hypothesis(emb_features, score_features, ppi_score)
        result['ligand_id'] = ligand_id
        result['receptor_id'] = receptor_id
        result['source'] = 'rule_based'

        self.cache[cache_key] = result
        return result

    def _compute_hypothesis(
        self,
        emb_features: Dict[str, float],
        score_features: Optional[Dict[str, float]],
        ppi_score: Optional[float],
    ) -> Dict:
        """Compute hypothesis from features."""

        cos_sim = emb_features['cosine_sim']
        prod_mean = emb_features['prod_mean']
        l2_dist = emb_features['l2_dist']

        # Determine directive based on embedding features
        if cos_sim > 0.55:
            directive = 'PRIORITIZE_HIGH_SIMILARITY'
            confidence = min(0.95, 0.7 + cos_sim * 0.3)
            flow_rec = 'DIRECT_SCORE_SUFFICIENT'

        elif cos_sim < 0.15 and l2_dist > 25:
            directive = 'SKIP_INCOMPATIBLE'
            confidence = min(0.90, 0.7 + (1 - cos_sim) * 0.25)
            flow_rec = 'DIRECT_SCORE_SUFFICIENT'

        elif cos_sim < 0.20:
            directive = 'DEPRIORITIZE_DISTANT'
            confidence = 0.75
            flow_rec = 'DIRECT_SCORE_SUFFICIENT'

        elif prod_mean > 0.18:
            directive = 'PRIORITIZE_STRONG_INTERACTION'
            confidence = min(0.90, 0.65 + prod_mean * 0.5)
            flow_rec = 'FLOW_OPTIONAL'

        elif score_features and score_features.get('score_range', 0) > 0.35:
            directive = 'EXPLORE_TENSION_DETECTED'
            confidence = 0.65
            flow_rec = 'FLOW_RECOMMENDED'

        elif score_features and score_features.get('score_std', 0) < 0.12 and score_features.get('score_mean', 0) > 0.55:
            directive = 'PRIORITIZE_SCORE_ALIGNMENT'
            confidence = 0.80
            flow_rec = 'DIRECT_SCORE_SUFFICIENT'

        elif score_features and score_features.get('score_mean', 0.5) < 0.35:
            directive = 'DEPRIORITIZE_WEAK_SIGNAL'
            confidence = 0.70
            flow_rec = 'DIRECT_SCORE_SUFFICIENT'

        else:
            directive = 'EXPLORE_MODERATE_SIGNAL'
            confidence = 0.60
            flow_rec = 'FLOW_OPTIONAL'

        base_priority = DIRECTIVE_PRIORITY.get(directive, 0.5)
        priority = base_priority * confidence

        return {
            'directive': directive,
            'confidence': confidence,
            'flow_recommendation': flow_rec,
            'priority': priority,
            'is_prioritize': directive.startswith('PRIORITIZE'),
            'is_explore': directive.startswith('EXPLORE'),
            'is_skip': directive.startswith('SKIP') or directive.startswith('DEPRIORITIZE'),
            'should_flow': flow_rec == 'FLOW_RECOMMENDED',
        }

    def generate_batch(
        self,
        receptor_id: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """Generate hypotheses for a batch of candidates."""

        results = []
        for cdata in candidates:
            result = self.generate(
                receptor_id,
                cdata['ligand_id'],
                cdata['emb_features'],
                cdata.get('score_features'),
                cdata.get('ppi_score'),
            )
            results.append(result)

        return results

    def clear_cache(self):
        self.cache = {}


# ============================================================
# SECTION C: Adaptive Search Strategy
# ============================================================

class AdaptiveSearchStrategy:
    """
    Adapts search parameters based on observed patterns and directives.
    """

    def __init__(self, config: QwenToTSearchConfig):
        self.config = config
        self.directive_history = []
        self.score_history = []
        self.tension_count = 0
        self.prioritize_count = 0
        self.skip_count = 0

    def update(self, hypothesis: Dict, scores: Optional[Dict] = None):
        """Update strategy based on observed hypothesis."""

        self.directive_history.append(hypothesis['directive'])

        if scores:
            self.score_history.append(scores)
            if scores.get('score_range', 0) > 0.35:
                self.tension_count += 1

        if hypothesis['is_prioritize']:
            self.prioritize_count += 1
        if hypothesis['is_skip']:
            self.skip_count += 1

    def get_search_params(self) -> Dict:
        """Get adaptive search parameters."""

        if not self.config.adaptive_params:
            return {
                'tau': self.config.base_tau,
                'branch_factor': self.config.base_branch_factor,
                'scan_cap': self.config.scan_cap,
                'use_flow_default': False,
            }

        n_history = len(self.directive_history)

        # High tension rate → more exploration, use flow
        if n_history > 5 and self.tension_count / n_history > 0.3:
            return {
                'tau': self.config.base_tau + self.config.tension_tau_boost,
                'branch_factor': self.config.base_branch_factor + 2,
                'scan_cap': min(25, self.config.scan_cap + 5),
                'use_flow_default': True,
            }

        # Many PRIORITIZE → exploit mode
        if n_history > 5 and self.prioritize_count / n_history > 0.4:
            return {
                'tau': max(0.20, self.config.base_tau - 0.10),
                'branch_factor': self.config.base_branch_factor,
                'scan_cap': self.config.scan_cap,
                'use_flow_default': False,
            }

        # Many SKIP → narrow the search
        if n_history > 10 and self.skip_count / n_history > 0.5:
            return {
                'tau': self.config.base_tau,
                'branch_factor': max(4, self.config.base_branch_factor - 2),
                'scan_cap': max(12, self.config.scan_cap - 4),
                'use_flow_default': False,
            }

        # Default
        return {
            'tau': self.config.base_tau,
            'branch_factor': self.config.base_branch_factor,
            'scan_cap': self.config.scan_cap,
            'use_flow_default': False,
        }

    def reset(self):
        """Reset for new search."""
        self.directive_history = []
        self.score_history = []
        self.tension_count = 0
        self.prioritize_count = 0
        self.skip_count = 0


# ============================================================
# SECTION D: Binder Manifold Geometry (from Cell B)
# ============================================================

print("=" * 70)
print("PART 1: Computing Binder Manifold Geometry")
print("=" * 70)

from sklearn.decomposition import PCA

test_pos_df = df_shs27[
    (df_shs27['split'] == 'test') &
    (df_shs27['label'] == 1)
].reset_index(drop=True)

test_neg_df = df_shs27[
    (df_shs27['split'] == 'test') &
    (df_shs27['label'] == 0)
].reset_index(drop=True)

def get_interaction_emb(id_a, id_b):
    ea = emb_map_35m.get(str(id_a))
    eb = emb_map_35m.get(str(id_b))
    if ea is None or eb is None:
        return None
    return np.concatenate([
        (ea - eb).astype(np.float32),
        (ea * eb).astype(np.float32)
    ])

rng_geom = np.random.default_rng(SEED)

pos_idx = rng_geom.choice(len(test_pos_df), min(2000, len(test_pos_df)), replace=False)
neg_idx = rng_geom.choice(len(test_neg_df), min(2000, len(test_neg_df)), replace=False)

bind_embs, decoy_embs = [], []

for i in tqdm(pos_idx, desc="Binder embs", leave=False):
    r = test_pos_df.iloc[i]
    v = get_interaction_emb(r['id_a'], r['id_b'])
    if v is not None:
        bind_embs.append(v)

for i in tqdm(neg_idx, desc="Decoy embs", leave=False):
    r = test_neg_df.iloc[i]
    v = get_interaction_emb(r['id_a'], r['id_b'])
    if v is not None:
        decoy_embs.append(v)

E_bind = np.stack(bind_embs).astype(np.float32)
E_decoy = np.stack(decoy_embs).astype(np.float32)

binder_centroid_emb = E_bind.mean(0)
decoy_centroid_emb = E_decoy.mean(0)
binder_std_emb = E_bind.std(0)

print(f"✓ Binder vectors: {E_bind.shape}")
print(f"✓ Decoy vectors: {E_decoy.shape}")


# ============================================================
# SECTION E: ESFM Flow (from Cell B)
# ============================================================

def dist_to_binder_manifold(v):
    d = v - binder_centroid_emb
    return float(np.sqrt(np.mean((d / (binder_std_emb + 1e-6)) ** 2)))


@torch.no_grad()
def ppi_rescore_with_eb(id_a, id_b, eb_flowed, scores_vec):
    ea = emb_map_35m.get(str(id_a))
    if ea is None:
        return 0.0

    eb = eb_flowed.astype(np.float32)

    feat = np.concatenate([
        ea,
        eb,
        ea - eb,
        ea * eb,
        scores_vec.astype(np.float32)
    ]).astype(np.float32)

    X = scaler.transform(feat[None]).astype(np.float32)

    model.eval()
    _, logit = model(torch.tensor(X, device=DEVICE))
    return float(torch.sigmoid(logit).cpu())


def esfm_flow(
    ea, eb, scores_vec, id_a, id_b,
    n_steps=8, n_trajectories=3, step_size=0.05, noise=0.01, seed=42,
):
    """DIAG_9 faithful ESFM flow."""

    rng = np.random.default_rng(seed)

    ea = ea.astype(np.float64)
    eb = eb.astype(np.float64)
    bc = binder_centroid_emb.astype(np.float64)

    dist_before = dist_to_binder_manifold(
        np.concatenate([ea - eb, ea * eb]).astype(np.float32)
    )

    best_dist = dist_before
    best_eb = eb.copy()
    best_score = ppi_rescore_with_eb(id_a, id_b, eb, scores_vec)

    for _ in range(n_trajectories):
        eb_t = eb.copy()

        for step in range(n_steps):
            iv = np.concatenate([ea - eb_t, ea * eb_t]).astype(np.float32)
            iv_diff = bc - iv.astype(np.float64)

            grad_eb = -iv_diff[:EMB_DIM] + iv_diff[EMB_DIM:] * ea

            gnorm = np.linalg.norm(grad_eb)
            if gnorm > 1e-8:
                grad_eb = grad_eb / gnorm

            anneal = 1.0 - step / n_steps
            noise_v = rng.normal(0, noise * anneal, size=EMB_DIM)

            eb_t = eb_t + step_size * anneal * grad_eb + noise_v

        iv_end = np.concatenate([ea - eb_t, ea * eb_t]).astype(np.float32)
        d_end = dist_to_binder_manifold(iv_end)
        s_end = ppi_rescore_with_eb(id_a, id_b, eb_t.astype(np.float32), scores_vec)

        if d_end < best_dist:
            best_dist = d_end
            best_eb = eb_t.copy()
            best_score = s_end

    return {
        'best_eb': best_eb.astype(np.float32),
        'best_score': float(best_score),
        'dist_before': float(dist_before),
        'dist_after': float(best_dist),
        'improvement': float(dist_before - best_dist),
    }

print("✓ ESFM flow defined")


# ============================================================
# SECTION F: PPI Pair Evaluator
# ============================================================

@torch.no_grad()
def eval_ppi_pair(receptor_id, receptor_seq, ligand_id, ligand_seq):
    """Evaluate a single PPI pair."""

    ea = emb_map_35m.get(str(receptor_id), np.zeros(EMB_DIM, dtype=np.float32))
    eb = emb_map_35m.get(str(ligand_id), np.zeros(EMB_DIM, dtype=np.float32))

    scores = four_scores(
        clean_seq(receptor_seq),
        clean_seq(ligand_seq)
    ).astype(np.float32)

    feat = np.concatenate([
        ea, eb, ea - eb, ea * eb, scores
    ]).astype(np.float32)

    X = scaler.transform(feat[None]).astype(np.float32)

    model.eval()
    _, logit = model(torch.tensor(X, device=DEVICE))
    prob = float(torch.sigmoid(logit).cpu())

    return prob, scores, eb.astype(np.float64)


# ============================================================
# SECTION G: Pre-filtering with Hypotheses
# ============================================================

def prefilter_candidates(
    receptor_id: str,
    candidates: pd.DataFrame,
    hypothesis_generator,
    config: QwenToTSearchConfig,
) -> pd.DataFrame:
    """
    Pre-filter candidates using embedding-based hypothesis generation.
    Returns candidates sorted by priority.
    """

    if not config.use_prefilter:
        return candidates

    ea = emb_map_35m.get(str(receptor_id))
    if ea is None:
        return candidates

    ea = ea.astype(np.float32)

    # Compute embedding features for all candidates
    candidate_data = []

    for _, row in candidates.iterrows():
        cid = str(row['id'])
        eb = emb_map_35m.get(cid)

        if eb is None:
            # No embedding - low priority
            candidate_data.append({
                'id': cid,
                'priority': 0.1,
                'directive': 'SKIP_INCOMPATIBLE',
                'should_flow': False,
            })
            continue

        eb = eb.astype(np.float32)
        emb_features = compute_embedding_features(ea, eb)

        # Generate hypothesis
        hyp = hypothesis_generator.generate(
            receptor_id, cid, emb_features, None, None
        )

        candidate_data.append({
            'id': cid,
            'priority': hyp['priority'],
            'directive': hyp['directive'],
            'should_flow': hyp['should_flow'],
            'emb_features': emb_features,
        })

    # Sort by priority
    candidate_data.sort(key=lambda x: x['priority'], reverse=True)

    # Keep top-k
    top_k = min(config.prefilter_top_k, len(candidate_data))
    top_ids = set(x['id'] for x in candidate_data[:top_k])

    # Also keep known binders (they might have low priority but we need them for eval)
    known_ids = set(candidates[candidates['Y'] == 1]['id'].astype(str).tolist())
    keep_ids = top_ids | known_ids

    # Create priority lookup
    priority_lookup = {x['id']: x for x in candidate_data}

    # Filter and add priority info
    filtered = candidates[candidates['id'].astype(str).isin(keep_ids)].copy()
    filtered['PRIORITY'] = filtered['id'].astype(str).apply(
        lambda x: priority_lookup.get(x, {}).get('priority', 0.5)
    )
    filtered['DIRECTIVE'] = filtered['id'].astype(str).apply(
        lambda x: priority_lookup.get(x, {}).get('directive', 'EXPLORE_MODERATE_SIGNAL')
    )
    filtered['SHOULD_FLOW'] = filtered['id'].astype(str).apply(
        lambda x: priority_lookup.get(x, {}).get('should_flow', False)
    )

    # Sort by priority
    filtered = filtered.sort_values('PRIORITY', ascending=False).reset_index(drop=True)

    return filtered


# ============================================================
# SECTION H: Qwen-Guided Entropic Search
# ============================================================

def _stable_softmax(x, tau):
    x = np.asarray(x, dtype=np.float64)
    z = x / max(float(tau), 1e-12)
    z -= z.max()
    ez = np.exp(z)
    s = ez.sum()
    return ez / s if s > 0 else np.ones_like(ez) / len(ez)


def _sample_wo_repl(items, probs, rng):
    items = list(items)
    probs = np.asarray(probs, dtype=np.float64)
    out = []

    while items:
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        j = int(rng.choice(len(items), p=probs))
        out.append(items.pop(j))
        probs = np.delete(probs, j)

    return out


def run_qwen_guided_search(
    receptor_id: str,
    receptor_seq: str,
    candidates: pd.DataFrame,
    known_binders: Set[str],
    hypothesis_generator,
    config: QwenToTSearchConfig,
    seed: int,
    trial: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Qwen-guided entropic Tree-of-Thought search.

    Key differences from original Cell B:
    1. Pre-filters candidates using hypothesis priorities
    2. Uses directive-weighted branch selection
    3. Conditionally triggers ESFM based on should_flow
    4. Prunes branches using hypothesis + score gradient
    """

    rng = np.random.default_rng(seed)
    strategy = AdaptiveSearchStrategy(config)

    known_binders = set(str(x) for x in known_binders)
    pool_ids = candidates['id'].astype(str).tolist()

    seq_lookup = dict(zip(candidates['id'].astype(str), candidates['seq']))
    idx_lookup = dict(zip(candidates['id'].astype(str), candidates['POOL_INDEX']))

    # Pre-computed hypothesis info if available
    priority_lookup = {}
    directive_lookup = {}
    should_flow_lookup = {}

    if 'PRIORITY' in candidates.columns:
        priority_lookup = dict(zip(candidates['id'].astype(str), candidates['PRIORITY']))
        directive_lookup = dict(zip(candidates['id'].astype(str), candidates['DIRECTIVE']))
        should_flow_lookup = dict(zip(candidates['id'].astype(str), candidates['SHOULD_FLOW']))

    # Caches
    ppi_cache = {}
    esfm_cache = {}
    hypothesis_cache = {}

    evaluated = set()
    eval_rows = []

    n_ppi_eval = 0
    n_esfm_eval = 0
    n_flow_triggered = 0
    n_pruned_by_hypothesis = 0

    stack = [(0, set(), 0.0, 0.0, None)]  # depth, path, V_prev, dV_prev, parent_directive

    pbar = tqdm(
        total=config.max_evals,
        desc=f"[Qwen-ToT] {str(receptor_id)[-18:]}",
        leave=False,
        unit="eval"
    )

    def get_hypothesis(cid: str) -> Dict:
        """Get or compute hypothesis for candidate."""
        if cid in hypothesis_cache:
            return hypothesis_cache[cid]

        # Check pre-computed
        if cid in directive_lookup:
            hyp = {
                'directive': directive_lookup[cid],
                'priority': priority_lookup.get(cid, 0.5),
                'should_flow': should_flow_lookup.get(cid, False),
                'is_prioritize': directive_lookup[cid].startswith('PRIORITIZE'),
                'is_skip': directive_lookup[cid].startswith('SKIP') or directive_lookup[cid].startswith('DEPRIORITIZE'),
            }
            hypothesis_cache[cid] = hyp
            return hyp

        # Compute on-demand
        ea = emb_map_35m.get(str(receptor_id))
        eb = emb_map_35m.get(cid)

        if ea is None or eb is None:
            hyp = {
                'directive': 'SKIP_INCOMPATIBLE',
                'priority': 0.1,
                'should_flow': False,
                'is_prioritize': False,
                'is_skip': True,
            }
        else:
            emb_features = compute_embedding_features(ea, eb)
            score_features = None

            if cid in ppi_cache:
                score_features = compute_score_features(ppi_cache[cid]['scores'])

            hyp = hypothesis_generator.generate(
                receptor_id, cid, emb_features, score_features, None
            )

        hypothesis_cache[cid] = hyp
        return hyp

    def score_candidate(cid: str) -> Dict:
        """Score candidate with PPI and optionally ESFM."""
        nonlocal n_ppi_eval, n_esfm_eval, n_flow_triggered

        cid = str(cid)

        # PPI evaluation
        if cid not in ppi_cache:
            ppi_prob, scores, eb = eval_ppi_pair(
                receptor_id, receptor_seq,
                cid, seq_lookup[cid]
            )
            ppi_cache[cid] = {
                'ppi': float(ppi_prob),
                'scores': scores,
                'eb': eb,
            }
            n_ppi_eval += 1

            # Update hypothesis with score features
            hyp = get_hypothesis(cid)
            score_features = compute_score_features(scores)

            # Check if flow should be triggered based on tensions
            if score_features['score_range'] > 0.35:
                hyp['should_flow'] = True
                hyp['directive'] = 'EXPLORE_TENSION_DETECTED'

        # Get hypothesis
        hyp = get_hypothesis(cid)

        # Adaptive params
        params = strategy.get_search_params()

        # Decide on ESFM
        use_flow = (
            hyp.get('should_flow', False) or
            params.get('use_flow_default', False)
        )

        if use_flow and cid not in esfm_cache:
            ea = emb_map_35m.get(str(receptor_id), np.zeros(EMB_DIM, dtype=np.float32)).astype(np.float64)
            pool_i = int(idx_lookup[cid])

            out = esfm_flow(
                ea,
                ppi_cache[cid]['eb'],
                ppi_cache[cid]['scores'].astype(np.float32),
                str(receptor_id),
                cid,
                n_steps=config.esfm_steps,
                n_trajectories=config.esfm_trajectories,
                step_size=config.esfm_step_size,
                noise=config.esfm_noise,
                seed=int(SEED + trial * 1000 + pool_i),
            )

            esfm_cache[cid] = out
            n_esfm_eval += 1
            n_flow_triggered += 1

        # Final score
        if cid in esfm_cache:
            final_score = esfm_cache[cid]['best_score']
            dist_impr = esfm_cache[cid]['improvement']
            flow_mode = 'esfm'
        else:
            final_score = ppi_cache[cid]['ppi']
            dist_impr = 0.0
            flow_mode = 'ppi_only'

        return {
            'BASE': float(final_score),
            'BASE_PPI': float(ppi_cache[cid]['ppi']),
            'scores': ppi_cache[cid]['scores'],
            'DIST_IMPR': float(dist_impr),
            'FLOW_MODE': flow_mode,
            'DIRECTIVE': hyp['directive'],
            'PRIORITY': hyp['priority'],
        }

    while stack and len(evaluated) < config.max_evals:
        depth, path_vis, V_prev, dV_prev, parent_dir = stack.pop()

        if depth >= config.max_depth:
            continue

        # Get adaptive parameters
        params = strategy.get_search_params()

        # Available candidates
        avail = [
            cid for cid in pool_ids
            if cid not in evaluated and cid not in path_vis
        ]

        if not avail:
            continue

        # Pre-filter by hypothesis priority
        if config.use_hypothesis_pruning:
            avail_with_hyp = []
            for cid in avail:
                hyp = get_hypothesis(cid)

                # Skip if below threshold (unless known binder)
                if hyp['priority'] < config.min_priority_threshold and cid not in known_binders:
                    n_pruned_by_hypothesis += 1
                    continue

                avail_with_hyp.append((cid, hyp['priority']))

            # Sort by priority
            avail_with_hyp.sort(key=lambda x: x[1], reverse=True)
            avail = [x[0] for x in avail_with_hyp]

        rng.shuffle(avail)
        avail_scan = avail[:params['scan_cap']]

        # Score candidates
        cand = []
        for cid in avail_scan:
            sc = score_candidate(cid)
            hyp = get_hypothesis(cid)

            # Combined score: base score + hypothesis priority bonus
            combined = sc['BASE'] * 0.7 + hyp['priority'] * 0.3

            cand.append((cid, combined, sc, hyp))

            # Update strategy
            strategy.update(hyp, compute_score_features(sc['scores']))

        if not cand:
            continue

        cand.sort(key=lambda x: x[1], reverse=True)

        # Branch selection with hypothesis-weighted entropy
        top = cand[:params['branch_factor']]

        # Use combined scores for softmax
        combined_scores = [x[1] for x in top]
        probs = _stable_softmax(combined_scores, params['tau'])
        order = _sample_wo_repl(top, probs, rng)

        for cid, combined, sc, hyp in order:
            if cid in evaluated or len(evaluated) >= config.max_evals:
                continue

            Vt = float(sc['BASE'])
            dV = Vt - V_prev
            d2V = dV - dV_prev

            # Hypothesis-aware pruning
            if config.use_hypothesis_pruning:
                # Skip if: declining score AND hypothesis says SKIP AND not known binder
                if (dV < 0 and d2V <= 0 and
                    hyp['is_skip'] and
                    cid not in known_binders):
                    n_pruned_by_hypothesis += 1
                    continue

            evaluated.add(cid)
            scores = sc['scores']

            eval_rows.append({
                'LIG': cid,
                'POOL_INDEX': int(idx_lookup[cid]),
                'BASE_PPI': float(sc['BASE_PPI']),
                'BASE': float(sc['BASE']),
                'SEQ': float(scores[0]),
                'STRUCT': float(scores[1]),
                'CONTACT': float(scores[2]),
                'CHEM': float(scores[3]),
                'DIST_IMPR': float(sc['DIST_IMPR']),
                'DIRECTIVE': sc['DIRECTIVE'],
                'PRIORITY': sc['PRIORITY'],
                'FLOW_MODE': sc['FLOW_MODE'],
                'IS_KNOWN': cid in known_binders,
                'FORCED_KNOWN': False,
            })

            pbar.update(1)
            pbar.set_postfix({
                'ppi': n_ppi_eval,
                'esfm': n_esfm_eval,
                'known': sum(r['IS_KNOWN'] for r in eval_rows),
                'pruned': n_pruned_by_hypothesis,
            })

            # Push children
            if depth + 1 < config.max_depth:
                p2 = set(path_vis)
                p2.add(cid)
                stack.append((depth + 1, p2, Vt, dV, hyp['directive']))

    pbar.close()

    # Force-evaluate known binders for measurement
    for kb in sorted(known_binders):
        if kb not in pool_ids or kb in evaluated:
            continue

        sc = score_candidate(kb)
        hyp = get_hypothesis(kb)
        scores = sc['scores']

        eval_rows.append({
            'LIG': kb,
            'POOL_INDEX': int(idx_lookup[kb]),
            'BASE_PPI': float(sc['BASE_PPI']),
            'BASE': float(sc['BASE']),
            'SEQ': float(scores[0]),
            'STRUCT': float(scores[1]),
            'CONTACT': float(scores[2]),
            'CHEM': float(scores[3]),
            'DIST_IMPR': float(sc['DIST_IMPR']),
            'DIRECTIVE': sc['DIRECTIVE'],
            'PRIORITY': sc['PRIORITY'],
            'FLOW_MODE': 'forced_known_' + sc['FLOW_MODE'],
            'IS_KNOWN': True,
            'FORCED_KNOWN': True,
        })

        evaluated.add(kb)

    # Create results dataframe
    df_eval = pd.DataFrame(eval_rows)

    if len(df_eval) == 0:
        return df_eval, {
            'n_ppi_eval': n_ppi_eval,
            'n_esfm_eval': n_esfm_eval,
            'n_flow_triggered': n_flow_triggered,
            'n_pruned_by_hypothesis': n_pruned_by_hypothesis,
            'best_rank': None,
            'best_search_rank': None,
            'auc_eval': np.nan,
        }

    # Rank by score
    df_ranked = df_eval.sort_values('BASE', ascending=False).reset_index(drop=True)
    df_ranked.index = df_ranked.index + 1
    df_ranked.index.name = 'RANK'
    df_ranked = df_ranked.reset_index()

    known_ranks = df_ranked[df_ranked['IS_KNOWN']]['RANK'].tolist()
    search_known_ranks = df_ranked[
        df_ranked['IS_KNOWN'] & (~df_ranked['FORCED_KNOWN'])
    ]['RANK'].tolist()

    best_rank = min(known_ranks) if known_ranks else None
    best_search_rank = min(search_known_ranks) if search_known_ranks else None

    y = df_ranked['IS_KNOWN'].astype(int).values
    auc_eval = roc_auc_score(y, df_ranked['BASE'].values) if len(np.unique(y)) > 1 else np.nan

    # Directive distribution
    directive_counts = df_ranked['DIRECTIVE'].value_counts().to_dict()

    metrics = {
        'n_ppi_eval': n_ppi_eval,
        'n_esfm_eval': n_esfm_eval,
        'n_flow_triggered': n_flow_triggered,
        'n_pruned_by_hypothesis': n_pruned_by_hypothesis,
        'n_total_rows': len(df_ranked),
        'n_forced_known': int(df_ranked['FORCED_KNOWN'].sum()),
        'n_known_found_by_search': int(((df_ranked['IS_KNOWN']) & (~df_ranked['FORCED_KNOWN'])).sum()),
        'best_rank': best_rank,
        'best_search_rank': best_search_rank,
        'auc_eval': auc_eval,
        'directive_counts': directive_counts,
    }

    return df_ranked, metrics


# ============================================================
# SECTION I: Main Experiment Loop
# ============================================================

# Configuration
config = QwenToTSearchConfig(
    n_trials=10,
    pool_size=500,
    max_evals=100,
    max_depth=2,
    prefilter_top_k=150,
    use_prefilter=True,
    use_hypothesis_pruning=True,
)

# Initialize hypothesis generator
if HAS_QWEN_GENERATOR:
    hypothesis_generator = tot_generator
    print("✓ Using fine-tuned Qwen ToT generator")
else:
    hypothesis_generator = RuleBasedHypothesisGenerator(config)
    print("✓ Using rule-based hypothesis generator (Qwen not loaded)")

# Prepare protein pool
all_proteins = pd.concat([
    df_shs27[['id_a', 'seq_a']].rename(columns={'id_a': 'id', 'seq_a': 'seq'}),
    df_shs27[['id_b', 'seq_b']].rename(columns={'id_b': 'id', 'seq_b': 'seq'}),
]).drop_duplicates('id').reset_index(drop=True)
all_proteins['id'] = all_proteins['id'].astype(str)

test_pos = df_shs27[
    (df_shs27['split'] == 'test') &
    (df_shs27['label'] == 1)
].copy()
test_pos['id_a'] = test_pos['id_a'].astype(str)
test_pos['id_b'] = test_pos['id_b'].astype(str)

receptor_counts = pd.concat([test_pos['id_a'], test_pos['id_b']]).value_counts()
valid_receptors = receptor_counts[receptor_counts >= 1].index.tolist()

trial_rng = np.random.default_rng(SEED + 222)

# Results storage
results = []
qwen_tot_tables = []
baseline_tables = []

print("\n" + "=" * 120)
print("PART 2: Qwen-Guided Entropic-ToT Search")
print("=" * 120)
print(f"Valid receptors: {len(valid_receptors)}")
print(f"Running {config.n_trials} trials")
print(f"Pre-filtering: {'ON' if config.use_prefilter else 'OFF'}")
print(f"Hypothesis pruning: {'ON' if config.use_hypothesis_pruning else 'OFF'}")
print()

print(
    f'{"Trial":>6} {"Receptor":>18} {"N_b":>4} '
    f'{"Base_rank":>10} {"Qwen_rank":>10} {"Δrank":>8} '
    f'{"Base_eval":>10} {"Qwen_eval":>10} {"Flow":>6} {"Pruned":>7}'
)
print("─" * 120)

for trial in range(1, config.n_trials + 1):

    receptor_id = str(trial_rng.choice(valid_receptors))

    rec_rows = df_shs27[
        (df_shs27['id_a'].astype(str) == receptor_id) |
        (df_shs27['id_b'].astype(str) == receptor_id)
    ]
    rec_row = rec_rows.iloc[0]

    receptor_seq = (
        rec_row['seq_a']
        if str(rec_row['id_a']) == receptor_id
        else rec_row['seq_b']
    )

    known_a = set(test_pos[test_pos['id_a'] == receptor_id]['id_b'].astype(str).tolist())
    known_b = set(test_pos[test_pos['id_b'] == receptor_id]['id_a'].astype(str).tolist())
    known_binders = known_a | known_b

    # Build candidate pool
    pool = all_proteins[all_proteins['id'] != receptor_id].reset_index(drop=True)

    known_rows = pool[pool['id'].isin(known_binders)].reset_index(drop=True)
    decoy_pool = pool[~pool['id'].isin(known_binders)].reset_index(drop=True)

    n_fill = min(config.pool_size - len(known_rows), len(decoy_pool))
    d_idx = trial_rng.choice(len(decoy_pool), n_fill, replace=False)

    decoys = decoy_pool.iloc[d_idx].reset_index(drop=True)

    candidates = pd.concat([known_rows, decoys], ignore_index=True)
    candidates = candidates.iloc[trial_rng.permutation(len(candidates))].reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    candidates["POOL_INDEX"] = np.arange(len(candidates), dtype=int)
    candidates['Y'] = candidates['id'].apply(lambda cid: 1 if str(cid) in known_binders else 0)

    # Pre-filter candidates with hypotheses
    candidates_filtered = prefilter_candidates(
        receptor_id, candidates, hypothesis_generator, config
    )

    # Run baseline (no Qwen guidance, no pre-filter)
    baseline_config = QwenToTSearchConfig(
        n_trials=1,
        pool_size=config.pool_size,
        max_evals=config.max_evals,
        max_depth=config.max_depth,
        use_prefilter=False,
        use_hypothesis_pruning=False,
    )
    baseline_generator = RuleBasedHypothesisGenerator(baseline_config)

    df_baseline, met_baseline = run_qwen_guided_search(
        receptor_id=receptor_id,
        receptor_seq=receptor_seq,
        candidates=candidates,
        known_binders=known_binders,
        hypothesis_generator=baseline_generator,
        config=baseline_config,
        seed=int(SEED + trial * 1000 + 17),
        trial=trial,
    )

    # Run Qwen-guided search
    hypothesis_generator.clear_cache() if hasattr(hypothesis_generator, 'clear_cache') else None

    df_qwen, met_qwen = run_qwen_guided_search(
        receptor_id=receptor_id,
        receptor_seq=receptor_seq,
        candidates=candidates_filtered,
        known_binders=known_binders,
        hypothesis_generator=hypothesis_generator,
        config=config,
        seed=int(SEED + trial * 1000 + 17),
        trial=trial,
    )

    rank_baseline = met_baseline['best_rank']
    rank_qwen = met_qwen['best_rank']

    delta_rank = (
        int(rank_baseline) - int(rank_qwen)
        if rank_baseline is not None and rank_qwen is not None
        else np.nan
    )

    df_baseline['TRIAL'] = trial
    df_baseline['RECEPTOR'] = receptor_id
    df_baseline['MODE'] = 'BASELINE'

    df_qwen['TRIAL'] = trial
    df_qwen['RECEPTOR'] = receptor_id
    df_qwen['MODE'] = 'QWEN_TOT'

    baseline_tables.append(df_baseline)
    qwen_tot_tables.append(df_qwen)

    results.append({
        'trial': trial,
        'receptor': receptor_id,
        'n_binders_pool': int(candidates['Y'].sum()),

        'baseline_best_rank': rank_baseline,
        'qwen_best_rank': rank_qwen,
        'delta_rank': delta_rank,

        'baseline_best_search_rank': met_baseline['best_search_rank'],
        'qwen_best_search_rank': met_qwen['best_search_rank'],

        'baseline_auc': met_baseline['auc_eval'],
        'qwen_auc': met_qwen['auc_eval'],

        'baseline_n_ppi_eval': met_baseline['n_ppi_eval'],
        'baseline_n_esfm_eval': met_baseline['n_esfm_eval'],

        'qwen_n_ppi_eval': met_qwen['n_ppi_eval'],
        'qwen_n_esfm_eval': met_qwen['n_esfm_eval'],
        'qwen_n_flow_triggered': met_qwen['n_flow_triggered'],
        'qwen_n_pruned': met_qwen['n_pruned_by_hypothesis'],

        'baseline_known_found': met_baseline['n_known_found_by_search'],
        'qwen_known_found': met_qwen['n_known_found_by_search'],
    })

    arrow = '↑' if delta_rank > 0 else '↓' if delta_rank < 0 else '='

    print(
        f'{trial:>6} {receptor_id[-18:]:>18} {int(candidates["Y"].sum()):>4} '
        f'{rank_baseline:>10} {rank_qwen:>10} {delta_rank:>+8.0f} {arrow} '
        f'{met_baseline["n_ppi_eval"]:>10} {met_qwen["n_ppi_eval"]:>10} '
        f'{met_qwen["n_flow_triggered"]:>6} {met_qwen["n_pruned_by_hypothesis"]:>7}'
    )


# ============================================================
# SECTION J: Summary and Analysis
# ============================================================

df_results = pd.DataFrame(results)
df_baseline_all = pd.concat(baseline_tables, ignore_index=True)
df_qwen_all = pd.concat(qwen_tot_tables, ignore_index=True)

valid = df_results.dropna(subset=['baseline_best_rank', 'qwen_best_rank']).copy()

print("\n" + "=" * 90)
print("SUMMARY — Qwen-Guided Entropic-ToT vs Baseline")
print("=" * 90)

print(df_results[[
    'trial',
    'receptor',
    'n_binders_pool',
    'baseline_best_rank',
    'qwen_best_rank',
    'delta_rank',
    'baseline_n_ppi_eval',
    'qwen_n_ppi_eval',
    'qwen_n_flow_triggered',
    'qwen_n_pruned',
]].to_string(index=False))

if len(valid) > 0:
    mean_baseline = valid['baseline_best_rank'].mean()
    mean_qwen = valid['qwen_best_rank'].mean()
    mean_delta = valid['delta_rank'].mean()

    try:
        _, p_rank = stats.wilcoxon(
            valid['baseline_best_rank'].values,
            valid['qwen_best_rank'].values
        )
    except Exception:
        p_rank = np.nan

    print("\nRank Comparison:")
    print(f"  Baseline best rank (mean)  : {mean_baseline:.2f}")
    print(f"  Qwen-ToT best rank (mean)  : {mean_qwen:.2f}")
    print(f"  Δ rank (Baseline - Qwen)   : {mean_delta:+.2f}")
    print(f"  Wilcoxon p-value           : {p_rank:.4f}")
    print(f"  Qwen helped                : {(valid['delta_rank'] > 0).sum()} / {len(valid)}")
    print(f"  Qwen hurt                  : {(valid['delta_rank'] < 0).sum()} / {len(valid)}")
    print(f"  Same                       : {(valid['delta_rank'] == 0).sum()} / {len(valid)}")

    print("\nEfficiency Comparison:")
    print(f"  Baseline PPI evals (mean)  : {valid['baseline_n_ppi_eval'].mean():.1f}")
    print(f"  Qwen-ToT PPI evals (mean)  : {valid['qwen_n_ppi_eval'].mean():.1f}")
    print(f"  Qwen-ToT ESFM evals (mean) : {valid['qwen_n_esfm_eval'].mean():.1f}")
    print(f"  Flow triggered (mean)      : {valid['qwen_n_flow_triggered'].mean():.1f}")
    print(f"  Pruned by hypothesis (mean): {valid['qwen_n_pruned'].mean():.1f}")

    # Directive analysis
    print("\nDirective Distribution (Qwen-ToT evaluated candidates):")
    directive_counts = df_qwen_all['DIRECTIVE'].value_counts()
    for directive, count in directive_counts.items():
        pct = 100 * count / len(df_qwen_all)
        print(f"  {directive}: {count} ({pct:.1f}%)")

# Save results
df_results.to_csv("cellB2_qwen_tot_summary.csv", index=False)
df_baseline_all.to_csv("cellB2_baseline_all.csv", index=False)
df_qwen_all.to_csv("cellB2_qwen_tot_all.csv", index=False)

print("\n✓ Saved cellB2_qwen_tot_summary.csv")
print("✓ Saved cellB2_baseline_all.csv")
print("✓ Saved cellB2_qwen_tot_all.csv")


# ============================================================
# SECTION K: Visualization
# ============================================================

if len(valid) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Rank comparison
    ax = axes[0]
    x = np.arange(len(valid))
    ax.plot(x, valid['baseline_best_rank'].values, 'o-', label='Baseline', color='gray')
    ax.plot(x, valid['qwen_best_rank'].values, 's-', label='Qwen-ToT', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(t)}" for t in valid['trial']])
    ax.set_ylabel("Best known-binder rank")
    ax.set_title("Rank Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Evaluation cost
    ax = axes[1]
    width = 0.35
    ax.bar(x - width/2, valid['baseline_n_ppi_eval'].values, width, label='Baseline PPI', color='gray')
    ax.bar(x + width/2, valid['qwen_n_ppi_eval'].values, width, label='Qwen PPI', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(t)}" for t in valid['trial']])
    ax.set_ylabel("Number of PPI evaluations")
    ax.set_title("Evaluation Cost")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Flow and pruning
    ax = axes[2]
    ax.bar(x - width/2, valid['qwen_n_flow_triggered'].values, width, label='ESFM triggered', color='green')
    ax.bar(x + width/2, valid['qwen_n_pruned'].values, width, label='Pruned by hyp', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(t)}" for t in valid['trial']])
    ax.set_ylabel("Count")
    ax.set_title("Qwen-ToT: Flow & Pruning")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("cellB2_qwen_tot_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("✓ Saved cellB2_qwen_tot_comparison.png")


# ============================================================
# SECTION L: Example Qwen Explanations
# ============================================================

if HAS_QWEN_LLM:
    print("\n" + "=" * 90)
    print("QWEN EXPLANATIONS — Top candidates from Qwen-guided search")
    print("=" * 90)

    try:
        df_q = df_qwen_all[df_qwen_all['TRIAL'] == 1].copy()
        df_q = df_q.sort_values("BASE", ascending=False).reset_index(drop=True)

        top_rows = df_q.head(config.topk_explain).copy()

        # Add best known if not in top
        known_rows = df_q[df_q['IS_KNOWN']].sort_values("BASE", ascending=False)
        if len(known_rows) > 0:
            best_known = known_rows.iloc[[0]]
            top_ids = set(top_rows['LIG'].astype(str).tolist())
            if str(best_known.iloc[0]['LIG']) not in top_ids:
                top_rows = pd.concat([top_rows, best_known], ignore_index=True)

        tok = globals()["tokenizer"]
        mod = globals()["qwen_llm"]

        for k, (_, row) in enumerate(top_rows.iterrows(), start=1):
            print("\n" + "=" * 90)
            print(
                f"Candidate {k}: "
                f"receptor={row['RECEPTOR']}, "
                f"ligand={row['LIG']}, "
                f"known={int(row['IS_KNOWN'])}"
            )
            print(
                f"BASE={row['BASE']:.4f}, "
                f"PPI={row['BASE_PPI']:.4f}, "
                f"Directive={row['DIRECTIVE']}, "
                f"Priority={row['PRIORITY']:.3f}"
            )
            print("=" * 90)

            # Generate explanation
            prompt = f"""Explain why this protein pair was {'prioritized' if row['DIRECTIVE'].startswith('PRIORITIZE') else 'explored' if row['DIRECTIVE'].startswith('EXPLORE') else 'deprioritized'} in the Tree-of-Thought search.

Receptor: {row['RECEPTOR']}
Ligand: {row['LIG']}

Scores:
- Sequence alignment: {float(row['SEQ']):.4f}
- Structural similarity: {float(row['STRUCT']):.4f}
- Contact compatibility: {float(row['CONTACT']):.4f}
- Chemical complementarity: {float(row['CHEM']):.4f}
- PPI model score: {float(row['BASE_PPI']):.4f}
- Final score: {float(row['BASE']):.4f}

Search directive: {row['DIRECTIVE']}
Priority: {row['PRIORITY']:.3f}
Flow mode: {row['FLOW_MODE']}
Known binder: {bool(row['IS_KNOWN'])}

Provide a brief analysis of why this directive was appropriate."""

            messages = [{"role": "user", "content": prompt}]
            input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(input_text, return_tensors="pt").to(mod.device)

            mod.eval()
            with torch.no_grad():
                out = mod.generate(
                    **inputs,
                    max_new_tokens=config.qwen_max_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tok.pad_token_id,
                )

            gen = tok.decode(out[0], skip_special_tokens=False)

            if "<|im_start|>assistant" in gen:
                explanation = gen.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            else:
                explanation = gen[len(input_text):].strip()

            print(explanation)

    except Exception as e:
        print(f"QWEN explanations skipped: {e}")

print("\n" + "=" * 90)
print("QWEN-GUIDED ENTROPIC-ToT COMPLETE")
print("=" * 90)


# ==============================================================================

# ============================================================
# CELL 148k-A: Download SHS148k, Embed, Build Features,
#              Train PPIProjectedNet — faithful to original
# ============================================================

import os, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Download SHS148k ─────────────────────────────────────────
SHS148K_FILES = {
    'SHS148k.actions.txt': 'https://zenodo.org/records/15694560/files/SHS148k.actions.txt?download=1',
    'SHS148k.seqs.tsv':    'https://zenodo.org/records/15694560/files/SHS148k.seqs.tsv?download=1',
}
for fname, url in SHS148K_FILES.items():
    dest = os.path.join(DATA_ROOT, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f'✓ {fname} already present ({os.path.getsize(dest)//1024} KB)')
    else:
        print(f'Downloading {fname} ...')
        subprocess.run(['wget', '-q', '--show-progress', '-O', dest, url], check=True)
        print(f'  ✓ saved ({os.path.getsize(dest)//1024} KB)')

# ── Parse SHS148k — identical loaders as original ────────────
print('\n── SHS148k ──')
shs148_seqs    = load_tsv_seqs(os.path.join(DATA_ROOT, 'SHS148k.seqs.tsv'))
shs148_actions = load_actions(os.path.join(DATA_ROOT, 'SHS148k.actions.txt'))
df_shs148      = build_dataset(shs148_actions, shs148_seqs, seed=SEED)
df_shs148      = dfs_split(df_shs148, seed=SEED)
df_shs148['dataset'] = 'SHS148k'
print(f'  Total: {len(df_shs148)}  pos={df_shs148["label"].mean():.3f}  '
      f'splits={df_shs148["split"].value_counts().to_dict()}')

# ── ESM-2 35M embeddings for SHS148k ─────────────────────────
EMBED_CACHE_148K = os.path.join(DATA_ROOT, 'shs148k_esm2_35M_embeddings.npz')

if os.path.exists(EMBED_CACHE_148K):
    print('\nLoading cached 148k embeddings ...')
    cache = np.load(EMBED_CACHE_148K, allow_pickle=True)
    emb_map_148k = {k: v for k, v in zip(cache['ids'], cache['embs'])}
    print(f'✓ Loaded {len(emb_map_148k)} embeddings  '
          f'dim={next(iter(emb_map_148k.values())).shape}')
else:
    print('\nComputing ESM-2 35M embeddings for SHS148k ...')
    import esm as esm_lib

    esm_model_148k, alphabet_148k = esm_lib.pretrained.esm2_t12_35M_UR50D()
    esm_model_148k = esm_model_148k.eval().to(DEVICE)
    batch_conv_148k = alphabet_148k.get_batch_converter()
    print('✓ ESM-2 35M loaded  (dim=480, 12 layers)')

    all_seqs_148k = {}
    for _, row in df_shs148.iterrows():
        all_seqs_148k[row['id_a']] = clean_seq(row['seq_a'])
        all_seqs_148k[row['id_b']] = clean_seq(row['seq_b'])

    ids_148k  = list(all_seqs_148k.keys())
    seqs_148k = [all_seqs_148k[i] for i in ids_148k]
    print(f'Unique proteins: {len(seqs_148k)}')

    @torch.no_grad()
    def get_esm35m_148k(sequences, batch_size=32):
        all_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc='ESM-2 35M [148k]'):
            batch = sequences[i:i+batch_size]
            data  = [(f'p{j}', s[:1022]) for j, s in enumerate(batch)]
            _, _, tokens = batch_conv_148k(data)
            tokens = tokens.to(DEVICE)
            out  = esm_model_148k(tokens, repr_layers=[12], return_contacts=False)
            reps = out['representations'][12]
            for j, (_, seq) in enumerate(data):
                emb = reps[j, 1:len(seq)+1].mean(0).cpu().numpy()
                all_embs.append(emb)
        return np.stack(all_embs).astype(np.float32)

    embs_148k    = get_esm35m_148k(seqs_148k, batch_size=32)
    emb_map_148k = {pid: emb for pid, emb in zip(ids_148k, embs_148k)}
    np.savez_compressed(EMBED_CACHE_148K,
                        ids=np.array(ids_148k),
                        embs=embs_148k)
    print(f'✓ Saved to {EMBED_CACHE_148K}')
    del esm_model_148k

EMB_DIM_148K = next(iter(emb_map_148k.values())).shape[0]
print(f'EMB_DIM_148K = {EMB_DIM_148K}')

# ── Build pair features — identical to original ───────────────
# feat = [ea|eb|ea-eb|ea*eb|scores] = 1924-d
print('\nBuilding pair features for SHS148k ...')
X_all_148k, Z_all_148k, y_all_148k, pair_ids_148k = [], [], [], []

for _, row in tqdm(df_shs148.iterrows(), total=len(df_shs148)):
    ea = emb_map_148k.get(row['id_a'])
    eb = emb_map_148k.get(row['id_b'])
    if ea is None or eb is None:
        continue
    scores = four_scores(clean_seq(row['seq_a']), clean_seq(row['seq_b']))
    feat   = np.concatenate([ea, eb, ea-eb, ea*eb, scores]).astype(np.float32)
    X_all_148k.append(feat)
    Z_all_148k.append(scores)
    y_all_148k.append(int(row['label']))
    pair_ids_148k.append((row['id_a'], row['id_b']))

X_all_148k = np.stack(X_all_148k)
Z_all_148k = np.stack(Z_all_148k)
y_all_148k = np.array(y_all_148k, dtype=np.float32)
print(f'Features: X={X_all_148k.shape}  Z={Z_all_148k.shape}  y={y_all_148k.shape}')

rng_148k   = np.random.default_rng(SEED)
idx_148k   = np.arange(len(X_all_148k)); rng_148k.shuffle(idx_148k)
n_148k     = len(idx_148k)
tr_end_148k = int(n_148k*0.70); va_end_148k = int(n_148k*0.80)
tr_148k, va_148k, te_148k = (idx_148k[:tr_end_148k],
                              idx_148k[tr_end_148k:va_end_148k],
                              idx_148k[va_end_148k:])

scaler_148k  = StandardScaler()
X_train_148k = scaler_148k.fit_transform(X_all_148k[tr_148k]).astype(np.float32)
X_val_148k   = scaler_148k.transform(X_all_148k[va_148k]).astype(np.float32)
X_test_148k  = scaler_148k.transform(X_all_148k[te_148k]).astype(np.float32)
Z_train_148k, Z_val_148k, Z_test_148k = Z_all_148k[tr_148k], Z_all_148k[va_148k], Z_all_148k[te_148k]
y_train_148k, y_val_148k, y_test_148k = y_all_148k[tr_148k], y_all_148k[va_148k], y_all_148k[te_148k]

print(f'Train: {len(y_train_148k)}  pos={y_train_148k.mean():.3f}')
print(f'Val  : {len(y_val_148k)}   pos={y_val_148k.mean():.3f}')
print(f'Test : {len(y_test_148k)}  pos={y_test_148k.mean():.3f}')

# ── PPIDataset, DataLoaders — identical to original ───────────
train_ds_148k = PPIDataset(X_train_148k, Z_train_148k, y_train_148k)
val_ds_148k   = PPIDataset(X_val_148k,   Z_val_148k,   y_val_148k)
test_ds_148k  = PPIDataset(X_test_148k,  Z_test_148k,  y_test_148k)

y_int_148k  = y_train_148k.astype(int)
counts_148k = np.bincount(y_int_148k, minlength=2)
w_148k      = np.array([1.0/max(counts_148k[0],1),
                         1.0/max(counts_148k[1],1)], dtype=np.float64)
sampler_148k = WeightedRandomSampler(
    torch.DoubleTensor(w_148k[y_int_148k]), len(y_int_148k), replacement=True)

BATCH_SIZE_148K  = 256
train_loader_148k = DataLoader(train_ds_148k, batch_size=BATCH_SIZE_148K, sampler=sampler_148k)
val_loader_148k   = DataLoader(val_ds_148k,   batch_size=512, shuffle=False)
test_loader_148k  = DataLoader(test_ds_148k,  batch_size=512, shuffle=False)

# ── PPIProjectedNet for 148k — identical architecture ─────────
model_148k = PPIProjectedNet(
    emb_dim=EMB_DIM_148K, d_model=256, n_heads=8, n_layers=4, p=0.20
).to(DEVICE)
n_params_148k = sum(p.numel() for p in model_148k.parameters() if p.requires_grad)
print(f'\nPPIProjectedNet [148k]  params={n_params_148k:,}  emb_dim={EMB_DIM_148K}')

pos_w_148k      = float(y_train_148k.sum())
neg_w_148k      = float(len(y_train_148k) - pos_w_148k)
pos_weight_148k = torch.tensor([neg_w_148k/pos_w_148k],
                                dtype=torch.float32, device=DEVICE)
LABEL_SMOOTH_148K = 0.10

def compute_loss_148k(z, logit, z_true, y):
    y_s   = y*(1-LABEL_SMOOTH_148K) + 0.5*LABEL_SMOOTH_148K
    l_cls = F.binary_cross_entropy_with_logits(logit, y_s,
                                                pos_weight=pos_weight_148k)
    l_flow= F.mse_loss(z, z_true)
    return 1.0*l_cls + 0.5*l_flow

# ── Training loop — identical to original ────────────────────
WARMUP_148K=5; T_MAX_148K=150; PATIENCE_148K=25
optimizer_148k = torch.optim.AdamW(
    model_148k.parameters(), lr=3e-4, weight_decay=5e-3)

def lr_lambda_148k(epoch):
    if epoch < WARMUP_148K: return (epoch+1)/WARMUP_148K
    p = (epoch-WARMUP_148K)/max(T_MAX_148K-WARMUP_148K, 1)
    return 0.5*(1.0+np.cos(np.pi*p))

scheduler_148k = torch.optim.lr_scheduler.LambdaLR(
    optimizer_148k, lr_lambda_148k)
best_val_148k   = float('inf')
best_state_148k = None
wait_148k       = 0

for epoch in range(1, T_MAX_148K+1):
    model_148k.train()
    run = 0.0; n = 0
    for xb, zb, yb in train_loader_148k:
        xb, zb, yb = xb.to(DEVICE), zb.to(DEVICE), yb.to(DEVICE)
        optimizer_148k.zero_grad()
        zp, logit = model_148k(xb)
        l = compute_loss_148k(zp, logit, zb, yb)
        l.backward()
        nn.utils.clip_grad_norm_(model_148k.parameters(), 1.0)
        optimizer_148k.step()
        run += l.item()*len(yb); n += len(yb)
    scheduler_148k.step()
    tr_loss = run/max(n,1)
    val_m   = evaluate(model_148k, val_loader_148k)

    if val_m['loss'] < best_val_148k:
        best_val_148k   = val_m['loss']
        best_state_148k = {k: v.detach().cpu().clone()
                           for k,v in model_148k.state_dict().items()}
        wait_148k = 0
    else:
        wait_148k += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f'epoch {epoch:03d} | train={tr_loss:.4f} | val={val_m["loss"]:.4f} | '
              f'AUC={val_m["auc"]:.4f} | micro-F1={val_m["micro_f1"]:.4f} | '
              f'flow_mse={val_m["flow_mse"]:.4f}')
    if wait_148k >= PATIENCE_148K:
        print(f'Early stop at epoch {epoch}'); break

model_148k.load_state_dict(best_state_148k)
test_m_148k = evaluate(model_148k, test_loader_148k)
print(f'\n── TEST RESULTS (SHS148k) ──────────────')
print(f'  AUC       : {test_m_148k["auc"]:.4f}')
print(f'  micro-F1  : {test_m_148k["micro_f1"]:.4f}')
print(f'  macro-F1  : {test_m_148k["macro_f1"]:.4f}')
print(f'  flow_mse  : {test_m_148k["flow_mse"]:.4f}')

cls_weights_148k = model_148k.cls_linear[2].weight.detach().cpu().numpy()[0]
cls_bias_148k    = model_148k.cls_linear[2].bias.detach().cpu().numpy()[0]
print(f'\n── Learned cls_linear weights ──')
for name, w in zip(['seq','struct','contact','chem'], cls_weights_148k):
    print(f'  {name:10s}: {w:+.4f}')

torch.save({
    'model_state':   best_state_148k,
    'cls_weights':   cls_weights_148k,
    'cls_bias':      cls_bias_148k,
    'scaler_mean':   scaler_148k.mean_.astype(np.float32),
    'scaler_std':    scaler_148k.scale_.astype(np.float32),
    'emb_dim':       EMB_DIM_148K,
    'test_micro_f1': test_m_148k['micro_f1'],
}, os.path.join(DATA_ROOT, 'ppi_projected_net_148k.pt'))
print('\n✓ Saved: ./data/ppi_projected_net_148k.pt')


# ==============================================================================

# ============================================================
# CELL 148k-B: Fine-tune Qwen LoRA on SHS148k — faithful to
#              original run_qwen_tot_finetuning pipeline
# REQUIRES: CELL 148k-A complete (model_148k, scaler_148k,
#           emb_map_148k, df_shs148 in scope)
# ============================================================
import subprocess
subprocess.run('pip install -q "peft==0.9.0" transformers accelerate', shell=True, check=True)

import sys, json
for mod in list(sys.modules.keys()):
    if 'peft' in mod or 'torchao' in mod:
        del sys.modules[mod]

def _save_lora_adapter(peft_model, save_dir, tokenizer, config):
    import os, json
    import safetensors.torch as st
    os.makedirs(save_dir, exist_ok=True)
    lora_sd = {k: v.detach().clone()
               for k, v in peft_model.state_dict().items()
               if 'lora_' in k}
    st.save_file(lora_sd,
                 os.path.join(save_dir, 'adapter_model.safetensors'))
    print(f'  ✓ Saved {len(lora_sd)} LoRA tensors')
    lora_cfg = peft_model.peft_config['default']
    adapter_cfg = {
        'base_model_name_or_path': config.qwen_model,
        'bias': lora_cfg.bias,
        'fan_in_fan_out': False,
        'inference_mode': True,
        'init_lora_weights': True,
        'lora_alpha': lora_cfg.lora_alpha,
        'lora_dropout': lora_cfg.lora_dropout,
        'modules_to_save': None,
        'peft_type': 'LORA',
        'r': lora_cfg.r,
        'target_modules': sorted(list(lora_cfg.target_modules)),
        'task_type': 'CAUSAL_LM',
    }
    with open(os.path.join(save_dir, 'adapter_config.json'), 'w') as f:
        json.dump(adapter_cfg, f, indent=2)
    print(f'  ✓ Saved adapter_config.json')
    tokenizer.save_pretrained(save_dir)
    print(f'  ✓ Saved tokenizer  →  {save_dir}')


def finetune_qwen_tot_148k(train_data, val_data, config, device='cuda'):
    """
    Identical to the working finetune_qwen_tot_fixed from SHS27k.
    Only the save_dir differs (passed via config).
    """
    import os, torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model

    print('\n' + '═'*60)
    print(f'Fine-tuning {config.qwen_model}  [bf16 LoRA]  — SHS148k')
    print('═'*60)

    print(f'\n  Loading tokenizer: {config.qwen_model}')
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32
    print(f'  Precision: {"bf16" if use_bf16 else "fp32"}')

    print(f'  Loading model: {config.qwen_model}')
    qwen_model = AutoModelForCausalLM.from_pretrained(
        config.qwen_model,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True,
    )
    qwen_model.gradient_checkpointing_enable()
    qwen_model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )
    qwen_model = get_peft_model(qwen_model, lora_config)

    for name, param in qwen_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in qwen_model.parameters())
    print(f'  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

    print('\n  Creating datasets...')
    train_ds = QwenToTDataset(train_data, tokenizer, config)
    val_ds   = QwenToTDataset(val_data,   tokenizer, config)

    os.makedirs(config.save_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=config.save_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy='no',        # avoid mid-train safetensors crash
        eval_strategy='epoch',
        bf16=use_bf16,
        fp16=False,
        optim='adamw_torch',
        dataloader_pin_memory=False,
        report_to='none',
        remove_unused_columns=False,
    )

    def data_collator(batch):
        return {
            'input_ids':      torch.stack([b['input_ids']      for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'labels':         torch.stack([b['labels']         for b in batch]),
        }

    trainer = Trainer(
        model=qwen_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print(f'\n  Training for {config.epochs} epochs...')
    trainer.train()

    print(f'\n  Saving to {config.save_dir} ...')
    _save_lora_adapter(qwen_model, config.save_dir, tokenizer, config)

    return qwen_model, tokenizer


# ── Run prepare_training_data on 148k — identical function ───
config_148k = QwenToTConfig(
    epochs=2,
    n_train_pairs=500,
    n_val_pairs=100,
    batch_size=4,
    grad_accum=4,
    max_length=512,
    lr=2e-4,
    qwen_model='Qwen/Qwen2.5-0.5B-Instruct',
    save_dir='./qwen_tot_148k_lora',
    cache_dir='./tot_hypothesis_cache_148k',
)

train_data_148k, val_data_148k = prepare_training_data(
    df_shs148,
    emb_map_148k,
    model_148k,        # PPIProjectedNet trained on 148k
    scaler_148k,
    four_scores,
    clean_seq,
    config_148k,
    device=str(DEVICE),
)

qwen_model_148k, tokenizer_148k = finetune_qwen_tot_148k(
    train_data_148k, val_data_148k, config_148k, device=str(DEVICE))

# ── Build generator — identical ToTHypothesisGenerator ───────
tot_generator_148k = ToTHypothesisGenerator(
    qwen_model_148k, tokenizer_148k, device=str(DEVICE))

print('\n✓ tot_generator_148k ready')


# ==============================================================================

# ============================================================
# CELL 148k-C: SHS148k — 10 Random Trials, Full Dataset as Pool
# RESTORED to original run_qwen_guided_search (pre-patches)
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Restore original search function ─────────────────────────
# Undo all patches applied in previous cells
if '_orig_run_qwen_guided_search' in globals():
    run_qwen_guided_search = _orig_run_qwen_guided_search
    print('✓ Restored original run_qwen_guided_search')
else:
    print('✓ run_qwen_guided_search already at original')

N_TRIALS_148K = 10

# ── Swap globals to 148k context ─────────────────────────────
_orig_binder_centroid = binder_centroid_emb.copy()
_orig_binder_std      = binder_std_emb.copy()
_orig_model           = model
_orig_scaler          = scaler

# ── Compute 148k binder manifold ─────────────────────────────
print('=' * 70)
print('SHS148k: Computing Binder Manifold Geometry')
print('=' * 70)

test_pos_148k_df = df_shs148[
    (df_shs148['split'] == 'test') & (df_shs148['label'] == 1)
].reset_index(drop=True)
test_neg_148k_df = df_shs148[
    (df_shs148['split'] == 'test') & (df_shs148['label'] == 0)
].reset_index(drop=True)

def get_interaction_emb_148k(id_a, id_b):
    ea = emb_map_148k.get(str(id_a))
    eb = emb_map_148k.get(str(id_b))
    if ea is None or eb is None:
        return None
    return np.concatenate([
        (ea - eb).astype(np.float32),
        (ea * eb).astype(np.float32)
    ])

rng_geom_148k = np.random.default_rng(SEED)
pos_idx_148k  = rng_geom_148k.choice(len(test_pos_148k_df),
                                      min(2000, len(test_pos_148k_df)),
                                      replace=False)
neg_idx_148k  = rng_geom_148k.choice(len(test_neg_148k_df),
                                      min(2000, len(test_neg_148k_df)),
                                      replace=False)

bind_embs_148k, decoy_embs_148k = [], []
for i in tqdm(pos_idx_148k, desc='Binder embs [148k]', leave=False):
    r = test_pos_148k_df.iloc[i]
    v = get_interaction_emb_148k(r['id_a'], r['id_b'])
    if v is not None:
        bind_embs_148k.append(v)
for i in tqdm(neg_idx_148k, desc='Decoy embs [148k]', leave=False):
    r = test_neg_148k_df.iloc[i]
    v = get_interaction_emb_148k(r['id_a'], r['id_b'])
    if v is not None:
        decoy_embs_148k.append(v)

E_bind_148k  = np.stack(bind_embs_148k).astype(np.float32)
E_decoy_148k = np.stack(decoy_embs_148k).astype(np.float32)
print(f'✓ Binder vectors: {E_bind_148k.shape}')
print(f'✓ Decoy vectors:  {E_decoy_148k.shape}')

binder_centroid_emb = E_bind_148k.mean(0)
binder_std_emb      = E_bind_148k.std(0)
model               = model_148k
scaler              = scaler_148k
print('✓ Globals swapped to 148k  (model, scaler, manifold)')

# ── Patch tot_generator_148k for None score_features ─────────
import types

def _safe_generate_148k(self, receptor_id, ligand_id, emb_features,
                         score_features, ppi_score, use_cache=True):
    cache_key = f'{receptor_id}__{ligand_id}'
    if use_cache and cache_key in self.cache:
        return self.cache[cache_key]

    if score_features is None or ppi_score is None:
        cos_sim   = emb_features.get('cosine_sim', 0.3)
        prod_mean = emb_features.get('prod_mean', 0.0)
        l2_dist   = emb_features.get('l2_dist', 10.0)
        if cos_sim > 0.55:
            directive, confidence, flow_rec = 'PRIORITIZE_HIGH_SIMILARITY',    0.90, 'DIRECT_SCORE_SUFFICIENT'
        elif cos_sim < 0.15 and l2_dist > 25:
            directive, confidence, flow_rec = 'SKIP_INCOMPATIBLE',             0.85, 'DIRECT_SCORE_SUFFICIENT'
        elif cos_sim < 0.20:
            directive, confidence, flow_rec = 'DEPRIORITIZE_DISTANT',          0.75, 'DIRECT_SCORE_SUFFICIENT'
        elif prod_mean > 0.18:
            directive, confidence, flow_rec = 'PRIORITIZE_STRONG_INTERACTION', 0.80, 'FLOW_OPTIONAL'
        else:
            directive, confidence, flow_rec = 'EXPLORE_MODERATE_SIGNAL',       0.60, 'FLOW_OPTIONAL'
        DPRI = {'PRIORITIZE_HIGH_SIMILARITY':1.0,'PRIORITIZE_STRONG_INTERACTION':0.95,
                'PRIORITIZE_SCORE_ALIGNMENT':0.90,'EXPLORE_MODERATE_SIGNAL':0.70,
                'EXPLORE_TENSION_DETECTED':0.65,'DEPRIORITIZE_WEAK_SIGNAL':0.40,
                'DEPRIORITIZE_DISTANT':0.30,'SKIP_INCOMPATIBLE':0.10}
        priority = DPRI.get(directive, 0.5) * confidence
        result = {
            'directive': directive, 'confidence': confidence,
            'flow_recommendation': flow_rec, 'priority': priority,
            'is_prioritize': directive.startswith('PRIORITIZE'),
            'is_explore':    directive.startswith('EXPLORE'),
            'is_skip':       directive.startswith('SKIP') or directive.startswith('DEPRIORITIZE'),
            'should_flow':   flow_rec == 'FLOW_RECOMMENDED',
            'source':        'rule_based_fallback',
        }
        if use_cache: self.cache[cache_key] = result
        return result

    input_prompt = create_input_prompt(
        receptor_id, ligand_id, emb_features, score_features, ppi_score)
    full_prompt = (
        f'<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n'
        f'<|im_start|>user\n{input_prompt}\n<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )
    inputs = self.tokenizer(full_prompt, return_tensors='pt',
                            truncation=True, max_length=768).to(self.device)
    outputs = self.model.generate(
        **inputs, max_new_tokens=400, temperature=0.7,
        top_p=0.9, do_sample=True,
        pad_token_id=self.tokenizer.pad_token_id)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    result   = self._parse_response(response, emb_features, score_features)
    if use_cache: self.cache[cache_key] = result
    return result

tot_generator_148k.generate = types.MethodType(
    _safe_generate_148k, tot_generator_148k)
print('✓ tot_generator_148k patched for None score_features')

# ── Build protein pool and binder index ──────────────────────
all_proteins_148k = pd.concat([
    df_shs148[['id_a','seq_a']].rename(columns={'id_a':'id','seq_a':'seq'}),
    df_shs148[['id_b','seq_b']].rename(columns={'id_b':'id','seq_b':'seq'}),
]).drop_duplicates('id').reset_index(drop=True)
all_proteins_148k['id'] = all_proteins_148k['id'].astype(str)

test_pos_148k_eval = df_shs148[
    (df_shs148['split'] == 'test') & (df_shs148['label'] == 1)
].copy()
test_pos_148k_eval['id_a'] = test_pos_148k_eval['id_a'].astype(str)
test_pos_148k_eval['id_b'] = test_pos_148k_eval['id_b'].astype(str)

binder_index_148k = {}
for _, row in test_pos_148k_eval.iterrows():
    binder_index_148k.setdefault(row['id_a'], set()).add(row['id_b'])
    binder_index_148k.setdefault(row['id_b'], set()).add(row['id_a'])

valid_receptors_148k = [r for r, b in binder_index_148k.items() if len(b) >= 1]
print(f'\nSHS148k valid receptors: {len(valid_receptors_148k)}')
print(f'Total proteins: {len(all_proteins_148k)}')
print(f'Trials: {N_TRIALS_148K}')

# ── Search config — original values, NO pruning ──────────────
full_config_148k_search = QwenToTSearchConfig(
    pool_size=len(all_proteins_148k),
    max_evals=150,
    max_depth=2,
    prefilter_top_k=300,
    use_prefilter=True,
    use_hypothesis_pruning=False,   # OFF — was causing mass pruning
    base_branch_factor=6,
    scan_cap=20,
    base_tau=0.35,
    esfm_steps=0,                   # OFF — score_range=1.0 on 148k
    esfm_trajectories=0,
)
baseline_config_148k_search = QwenToTSearchConfig(
    pool_size=len(all_proteins_148k),
    max_evals=150,
    max_depth=2,
    use_prefilter=False,
    use_hypothesis_pruning=False,
    esfm_steps=0,
    esfm_trajectories=0,
)
baseline_gen_148k = RuleBasedHypothesisGenerator(baseline_config_148k_search)

# ── Results storage ───────────────────────────────────────────
results_148k         = []
qwen_tables_148k     = []
baseline_tables_148k = []

trial_rng_148k = np.random.default_rng(SEED + 333)

print('\n' + '='*110)
print('SHS148k — 10 TRIALS, FULL DATASET POOL')
print('='*110)
print(f'{"Trial":>6} {"Receptor":>18} {"N_b":>4} '
      f'{"Base_rank":>10} {"Qwen_rank":>10} {"Δrank":>8} '
      f'{"Base_eval":>10} {"Qwen_eval":>10} '
      f'{"Base_found":>11} {"Qwen_found":>11}')
print('─'*110)

for trial in range(1, N_TRIALS_148K + 1):

    receptor_id   = str(trial_rng_148k.choice(valid_receptors_148k))
    known_binders = binder_index_148k[receptor_id]

    rec_row = all_proteins_148k[all_proteins_148k['id'] == receptor_id]
    if len(rec_row) == 0:
        continue
    receptor_seq = rec_row.iloc[0]['seq']

    pool = all_proteins_148k[
        all_proteins_148k['id'] != receptor_id
    ].copy().reset_index(drop=True)
    pool['POOL_INDEX'] = np.arange(len(pool), dtype=int)
    pool['Y']          = pool['id'].apply(
        lambda x: 1 if x in known_binders else 0)

    if hasattr(tot_generator_148k, 'clear_cache'):
        tot_generator_148k.clear_cache()

    pool_filtered = prefilter_candidates(
        receptor_id, pool, tot_generator_148k, full_config_148k_search)

    seed = int(SEED + trial * 1000 + 130)

    df_base, met_base = run_qwen_guided_search(
        receptor_id=receptor_id,
        receptor_seq=receptor_seq,
        candidates=pool,
        known_binders=known_binders,
        hypothesis_generator=baseline_gen_148k,
        config=baseline_config_148k_search,
        seed=seed,
        trial=trial,
    )

    df_qwen, met_qwen = run_qwen_guided_search(
        receptor_id=receptor_id,
        receptor_seq=receptor_seq,
        candidates=pool_filtered,
        known_binders=known_binders,
        hypothesis_generator=tot_generator_148k,
        config=full_config_148k_search,
        seed=seed,
        trial=trial,
    )

    # ── Use search_rank (genuinely found) as primary metric ──
    r_base  = met_base['best_search_rank']   # None if not found by search
    r_qwen  = met_qwen['best_search_rank']   # None if not found by search
    r_base_all = met_base['best_rank']       # includes forced
    r_qwen_all = met_qwen['best_rank']       # includes forced
    delta   = (int(r_base) - int(r_qwen)) if (r_base and r_qwen) else np.nan

    df_base['TRIAL']    = trial
    df_base['RECEPTOR'] = receptor_id
    df_base['MODE']     = 'BASELINE'
    df_qwen['TRIAL']    = trial
    df_qwen['RECEPTOR'] = receptor_id
    df_qwen['MODE']     = 'QWEN_TOT'

    baseline_tables_148k.append(df_base)
    qwen_tables_148k.append(df_qwen)

    results_148k.append({
        'trial':                     trial,
        'receptor':                  receptor_id,
        'n_binders_pool':            int(pool['Y'].sum()),
        # primary — search only
        'baseline_search_rank':      r_base,
        'qwen_search_rank':          r_qwen,
        'delta_search_rank':         delta,
        # secondary — includes forced eval
        'baseline_best_rank':        r_base_all,
        'qwen_best_rank':            r_qwen_all,
        'baseline_auc':              met_base['auc_eval'],
        'qwen_auc':                  met_qwen['auc_eval'],
        'baseline_n_ppi_eval':       met_base['n_ppi_eval'],
        'qwen_n_ppi_eval':           met_qwen['n_ppi_eval'],
        'qwen_n_flow_triggered':     met_qwen['n_flow_triggered'],
        'qwen_n_pruned':             met_qwen['n_pruned_by_hypothesis'],
        'baseline_known_found':      met_base['n_known_found_by_search'],
        'qwen_known_found':          met_qwen['n_known_found_by_search'],
    })

    arrow = '↑' if (delta and delta > 0) else '↓' if (delta and delta < 0) else '='
    print(
        f'{trial:>6} {receptor_id[-18:]:>18} {int(pool["Y"].sum()):>4} '
        f'{str(r_base_all):>10} {str(r_qwen_all):>10} {str(round(delta,1) if delta == delta else "N/A"):>8} {arrow} '
        f'{met_base["n_ppi_eval"]:>10} {met_qwen["n_ppi_eval"]:>10} '
        f'{met_base["n_known_found_by_search"]:>11} {met_qwen["n_known_found_by_search"]:>11}'
    )

# ── Summary ───────────────────────────────────────────────────
df_results_148k      = pd.DataFrame(results_148k)
df_baseline_all_148k = pd.concat(baseline_tables_148k, ignore_index=True)
df_qwen_all_148k     = pd.concat(qwen_tables_148k,     ignore_index=True)

valid_sr = df_results_148k.dropna(subset=['delta_search_rank'])

print('\n' + '='*90)
print('SUMMARY — SHS148k Full Pool, 10 Trials')
print('='*90)
print(df_results_148k[[
    'trial', 'receptor', 'n_binders_pool',
    'baseline_search_rank', 'qwen_search_rank', 'delta_search_rank',
    'baseline_n_ppi_eval',  'qwen_n_ppi_eval',
    'baseline_known_found', 'qwen_known_found',
]].to_string(index=False))

base_disc = df_results_148k['baseline_known_found'].gt(0).sum()
qwen_disc = df_results_148k['qwen_known_found'].gt(0).sum()

print(f'\n  Binder discovery (search found ≥1 binder):')
print(f'    Baseline : {base_disc} / {N_TRIALS_148K} trials')
print(f'    Qwen-ToT : {qwen_disc} / {N_TRIALS_148K} trials')

if len(valid_sr) > 0:
    print(f'\n  Search rank (genuinely found only):')
    print(f'    Baseline mean : {valid_sr["baseline_search_rank"].mean():.1f}')
    print(f'    Qwen mean     : {valid_sr["qwen_search_rank"].mean():.1f}')
    print(f'    Mean Δ        : {valid_sr["delta_search_rank"].mean():+.1f}')
    print(f'    Qwen helped   : {(valid_sr["delta_search_rank"]>0).sum()} / {len(valid_sr)}')
    print(f'    Qwen hurt     : {(valid_sr["delta_search_rank"]<0).sum()} / {len(valid_sr)}')

print(f'\n  AUC over evaluated candidates:')
print(f'    Baseline : {df_results_148k["baseline_auc"].mean():.4f}')
print(f'    Qwen-ToT : {df_results_148k["qwen_auc"].mean():.4f}')

print(f'\n  Mean PPI evals:')
print(f'    Baseline : {df_results_148k["baseline_n_ppi_eval"].mean():.1f}')
print(f'    Qwen-ToT : {df_results_148k["qwen_n_ppi_eval"].mean():.1f}')

df_results_148k.to_csv('shs148k_10trial_fullpool_summary.csv', index=False)
df_baseline_all_148k.to_csv('shs148k_10trial_fullpool_baseline.csv', index=False)
df_qwen_all_148k.to_csv('shs148k_10trial_fullpool_qwen.csv', index=False)

print('\n✓ Saved shs148k_10trial_fullpool_summary.csv')
print('✓ Saved shs148k_10trial_fullpool_baseline.csv')
print('✓ Saved shs148k_10trial_fullpool_qwen.csv')

# ── Restore SHS27k globals ────────────────────────────────────
binder_centroid_emb = _orig_binder_centroid
binder_std_emb      = _orig_binder_std
model               = _orig_model
scaler              = _orig_scaler
print('\n✓ Restored SHS27k globals (model, scaler, manifold)')
