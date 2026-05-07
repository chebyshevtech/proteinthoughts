#!/usr/bin/env python3
"""
Antibody-Antigen Score Distributions and Tension Map

Purpose
-------
Computes antibody-antigen score panels and plots the four-score distribution and structure/contact tension map.

Expected outputs
----------------
Writes ant_boxplot_scores.png and ant_tensionmap.png.

Notes
-----
This file is standalone: it includes data loading, model setup/training, evaluation, and plotting required for this result. The modeling and metric code is kept identical to the implementation used for the manuscript artifact; only interactive shell/display syntax is adapted for Python execution.
"""



# ==============================================================================

# ============================================================================
# CELL 1: seq_ppi (Sun et al.) + build training table (pairs + seqs)
# - clones repo
# - loads Supp-AB.tsv (pairs + labels)
# - auto-picks best id->sequence dictionary by coverage
# - outputs: ppi_dataset (DataFrame) with columns: protein1, protein2, label, seq1, seq2
# ============================================================================

import subprocess
subprocess.run('pip -q install pandas numpy tqdm', shell=True, check=True)

import os, subprocess, glob, re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_URL = "https://github.com/muhaochen/seq_ppi.git"
REPO_DIR = "seq_ppi"
PAIR_PATH = "seq_ppi/sun/preprocessed/Supp-AB.tsv"

if not os.path.isdir(REPO_DIR):
    print("Cloning seq_ppi repository...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
else:
    print("seq_ppi already exists, skipping clone.")

assert os.path.isfile(PAIR_PATH), f"Missing {PAIR_PATH}. Run `find seq_ppi/sun -maxdepth 3 -type f` to inspect."

# --- Load pairs (has header) ---
df = pd.read_csv(PAIR_PATH, sep="\t", header=0, dtype=str)
df.columns = [c.lower().strip() for c in df.columns]
if not (("v1" in df.columns) and ("v2" in df.columns) and ("label" in df.columns)):
    raise RuntimeError(f"Unexpected columns in Supp-AB.tsv: {df.columns.tolist()}")

pairs_df = pd.DataFrame({
    "protein1": df["v1"].astype(str).str.strip(),
    "protein2": df["v2"].astype(str).str.strip(),
    "label":   df["label"].astype(int),
})
pairs_df = pairs_df[pairs_df["protein1"] != pairs_df["protein2"]].drop_duplicates()
pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Pairs:", pairs_df.shape, "Pos rate:", pairs_df["label"].mean())
ids_needed = set(pairs_df["protein1"]) | set(pairs_df["protein2"])
print("Unique proteins needed:", len(ids_needed))

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO\-]+$", re.I)

def load_id2seq_from_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, dtype=str).dropna(how="all")
    if df.shape[1] < 2:
        return {}
    ids = df.iloc[:, 0].astype(str).str.strip()

    best_j, best_score = None, -1
    for j in range(1, df.shape[1]):
        col = df.iloc[:, j].astype(str).str.strip()
        ok = col.map(lambda s: (len(s) >= 30) and (AA_RE.match(s) is not None)).sum()
        if ok > best_score:
            best_score, best_j = ok, j

    if best_j is None or best_score < 100:
        return {}
    seqs = df.iloc[:, best_j].astype(str).str.strip()
    return dict(zip(ids, seqs))

def coverage_score(d):
    if not d: return (0, 0)
    hit = sum((x in d) for x in ids_needed)
    sample = list(d.values())[:2000]
    aa_ok = sum((len(s) >= 30) and (AA_RE.match(s) is not None) for s in sample)
    return (hit, aa_ok)

dict_cands = sorted(set(
    glob.glob("seq_ppi/sun/preprocessed/*.tsv")
    + glob.glob("seq_ppi/yeast/preprocessed/*dictionary*.tsv")
    + glob.glob("seq_ppi/multi_species/preprocessed/*dictionary*.tsv")
))

best_d, best_path, best_stats = None, None, (-1, -1)
print("\nScanning dictionary candidates:", len(dict_cands))
for p in dict_cands:
    d = load_id2seq_from_tsv(p)
    hit, aa_ok = coverage_score(d)
    if (hit, aa_ok) > best_stats:
        best_d, best_path, best_stats = d, p, (hit, aa_ok)

if best_d is None or best_stats[0] <= 0:
    raise RuntimeError("No dictionary matched your pair IDs (coverage=0).")

print("\nChosen sequence source:", best_path)
print("Coverage:", best_stats[0], "/", len(ids_needed), "| AA-like(sample):", best_stats[1])

pairs_df["seq1"] = pairs_df["protein1"].map(best_d)
pairs_df["seq2"] = pairs_df["protein2"].map(best_d)

before = len(pairs_df)
pairs_df = pairs_df.dropna(subset=["seq1","seq2"]).reset_index(drop=True)
print("Dropped missing sequences:", before - len(pairs_df), "Remaining:", len(pairs_df))

bad = (~pairs_df["seq1"].str.match(AA_RE)) | (~pairs_df["seq2"].str.match(AA_RE))
print("Non-AA-like pairs:", bad.sum())
pairs_df = pairs_df[~bad].reset_index(drop=True)

MAX_LEN = 1022
pairs_df = pairs_df[
    (pairs_df["seq1"].str.len().between(30, MAX_LEN)) &
    (pairs_df["seq2"].str.len().between(30, MAX_LEN))
].reset_index(drop=True)

print("After length filter:", len(pairs_df), "Pos rate:", pairs_df["label"].mean())

ppi_dataset = pairs_df
ppi_dataset.head()


# ==============================================================================

# ============================================================================
# CELL 2: Deterministic "baseline" 4-scores from sequences + train small MLP
# Model design you asked for:
#   - inputs: cheap pair features from sequences
#   - penultimate layer: 4 numbers (interpretable as the 4 scores)
#   - last layer: probability of binding (sigmoid)
# Training data:
#   - y_score = deterministic 4-score vector computed from (seq1, seq2)
#   - y_bind  = dataset label
# Loss:
#   BCE(bind) + alpha * MSE(scores)
# ============================================================================

import subprocess
subprocess.run('pip -q install torch scikit-learn matplotlib', shell=True, check=True)

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# -----------------------------
# Fast deterministic 4-score functions (sequence-only)
# -----------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a:i for i,a in enumerate(AA)}

# simple property table for "chemistry"
# (hydrophobicity-ish, charge-ish, polarity-ish) — coarse but deterministic
PROP = {
    "A": ( 1.8,  0.0, 0.0), "C": ( 2.5,  0.0, 0.2), "D": (-3.5, -1.0, 1.0), "E": (-3.5, -1.0, 1.0),
    "F": ( 2.8,  0.0, 0.1), "G": (-0.4,  0.0, 0.0), "H": (-3.2,  0.5, 0.7), "I": ( 4.5,  0.0, 0.0),
    "K": (-3.9,  1.0, 1.0), "L": ( 3.8,  0.0, 0.0), "M": ( 1.9,  0.0, 0.1), "N": (-3.5,  0.0, 1.0),
    "P": (-1.6,  0.0, 0.3), "Q": (-3.5,  0.0, 1.0), "R": (-4.5,  1.0, 1.0), "S": (-0.8,  0.0, 1.0),
    "T": (-0.7,  0.0, 0.8), "V": ( 4.2,  0.0, 0.0), "W": (-0.9,  0.0, 0.3), "Y": (-1.3,  0.0, 0.4),
}
def _clean(seq: str) -> str:
    seq = (seq or "").upper()
    return "".join([c for c in seq if c in AA])

def aa_freq(seq: str) -> np.ndarray:
    seq = _clean(seq)
    x = np.zeros(20, dtype=np.float32)
    if len(seq) == 0:
        return x
    for c in seq:
        x[AA2I[c]] += 1.0
    x /= float(len(seq))
    return x

def kmer_set(seq: str, k=3):
    seq = _clean(seq)
    if len(seq) < k:
        return set()
    return {seq[i:i+k] for i in range(len(seq)-k+1)}

def sequence_similarity_score(seqA: str, seqB: str, k=3) -> float:
    # fast proxy (Jaccard of k-mers)
    A = kmer_set(seqA, k=k)
    B = kmer_set(seqB, k=k)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return float(inter / union) if union else 0.0

def _pseudo_coords(seq: str) -> np.ndarray:
    # deterministic pseudo-structure from sequence (seeded by hash)
    seq = _clean(seq)
    if len(seq) == 0:
        return np.zeros((0,3), dtype=np.float64)
    seed = (hash(seq) % (2**32 - 1))
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, size=(len(seq), 3))
    coords = np.cumsum(steps, axis=0)
    coords -= coords.mean(axis=0, keepdims=True)
    return coords.astype(np.float64)

def _kabsch_rmsd(P, Q):
    n = min(len(P), len(Q))
    if n < 5:
        return None
    P = P[:n].copy()
    Q = Q[:n].copy()
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Q2 = (Q - Q.mean(0)) @ R
    P2 = (P - P.mean(0))
    rmsd = np.sqrt(np.mean(np.sum((P2 - Q2)**2, axis=1)))
    return float(rmsd)

def structure_rmsd_score(seqA: str, seqB: str, sigma=5.0) -> float:
    # mimic exp(-RMSD/sigma) with pseudo coords
    A = _pseudo_coords(seqA)
    B = _pseudo_coords(seqB)
    rmsd = _kabsch_rmsd(A, B)
    if rmsd is None:
        return 0.0
    return float(np.exp(-rmsd / float(sigma)))

def contact_overlap_score(seqA: str, seqB: str, cutoff=4.5) -> float:
    # pseudo-contact count ratio from pseudo coords
    A = _pseudo_coords(seqA)
    B = _pseudo_coords(seqB)
    if len(A) < 5 or len(B) < 5:
        return 0.0
    # count close pairs within each chain (cheap O(n^2) but we truncate)
    def count_contacts(X, m=220):
        X = X[:min(len(X), m)]
        D2 = np.sum((X[:,None,:] - X[None,:,:])**2, axis=-1)
        # ignore diagonal
        cnt = int(np.sum((D2 < cutoff**2)) - len(X))
        return max(cnt, 0)
    cA = count_contacts(A)
    cB = count_contacts(B)
    mn, mx = min(cA, cB), max(cA, cB)
    return 0.0 if mx == 0 else float(mn / mx)

def chemical_compatibility_score(seqA: str, seqB: str) -> float:
    # mean cosine-ish compatibility between averaged property vectors
    A = _clean(seqA); B = _clean(seqB)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    def avg_prop(S):
        v = np.zeros(3, dtype=np.float64)
        for c in S:
            v += np.array(PROP[c], dtype=np.float64)
        v /= float(len(S))
        return v
    vA = avg_prop(A)
    vB = avg_prop(B)
    num = float(np.dot(vA, vB))
    den = float(np.linalg.norm(vA) * np.linalg.norm(vB) + 1e-12)
    cos = num / den
    # map [-1,1] -> [0,1]
    return float(0.5 * (cos + 1.0))

def combined_baseline_score_from_seqs(seqA, seqB, weights=None):
    if weights is None:
        weights = {'sequence': 0.25, 'structure': 0.25, 'contact': 0.25, 'chemistry': 0.25}
    s_seq = sequence_similarity_score(seqA, seqB)
    s_str = structure_rmsd_score(seqA, seqB)
    s_con = contact_overlap_score(seqA, seqB)
    s_che = chemical_compatibility_score(seqA, seqB)
    scores = {"sequence": s_seq, "structure": s_str, "contact": s_con, "chemistry": s_che}
    base = sum(weights[k] * scores[k] for k in weights)
    return float(base), scores

# -----------------------------
# Build training matrix
# -----------------------------
# For Colab speed: start with a subset; increase N_TRAIN if you want.
N_TRAIN = min(30000, len(ppi_dataset))
data = ppi_dataset.iloc[:N_TRAIN].copy()

X_list = []
Yscore_list = []
Ybind_list = []

for s1, s2, y in tqdm(zip(data["seq1"], data["seq2"], data["label"]), total=len(data)):
    # input features (cheap)
    f1 = aa_freq(s1)
    f2 = aa_freq(s2)
    prod = f1 * f2
    diff = np.abs(f1 - f2)
    x = np.concatenate([f1, f2, prod, diff], axis=0)  # 80-dim
    base, comps = combined_baseline_score_from_seqs(s1, s2)
    yscore = np.array([comps["sequence"], comps["structure"], comps["contact"], comps["chemistry"]], dtype=np.float32)

    X_list.append(x.astype(np.float32))
    Yscore_list.append(yscore)
    Ybind_list.append(int(y))

X = np.stack(X_list, axis=0)
Yscore = np.stack(Yscore_list, axis=0)
Ybind = np.array(Ybind_list, dtype=np.float32)

print("X:", X.shape, "Yscore:", Yscore.shape, "Ybind:", Ybind.shape, "pos rate:", Ybind.mean())

Xtr, Xte, Yscore_tr, Yscore_te, Ybind_tr, Ybind_te = train_test_split(
    X, Yscore, Ybind, test_size=0.2, random_state=42, stratify=Ybind
)

# -----------------------------
# Model: penultimate 4-d layer = predicted 4 scores, last = bind prob
# -----------------------------
class ScoreThenBindMLP(nn.Module):
    def __init__(self, in_dim=80, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(hid, 4)     # <-- penultimate "scores"
        self.bind_head  = nn.Linear(4, 1)       # <-- probability from scores

    def forward(self, x):
        h = self.net(x)
        s = torch.sigmoid(self.score_head(h))   # keep scores in [0,1]
        p = torch.sigmoid(self.bind_head(s)).squeeze(-1)
        return s, p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScoreThenBindMLP(in_dim=X.shape[1], hid=128).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
bce = nn.BCELoss()
mse = nn.MSELoss()

alpha = 0.5  # weight on score imitation

# data loaders
def make_loader(X, Ys, Yb, bs=512, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Ys), torch.from_numpy(Yb)
    )
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

tr_loader = make_loader(Xtr, Yscore_tr, Ybind_tr, bs=512, shuffle=True)
te_loader = make_loader(Xte, Yscore_te, Ybind_te, bs=1024, shuffle=False)

# -----------------------------
# Train
# -----------------------------
for epoch in range(1, 11):
    model.train()
    tot = 0.0
    for xb, ysb, ybb in tr_loader:
        xb = xb.to(device)
        ysb = ysb.to(device)
        ybb = ybb.to(device)

        pred_s, pred_p = model(xb)
        loss = bce(pred_p, ybb) + alpha * mse(pred_s, ysb)

        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * xb.size(0)

    # eval
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, ysb, ybb in te_loader:
            xb = xb.to(device)
            pred_s, pred_p = model(xb)
            ps.append(pred_p.detach().cpu().numpy())
            ys.append(ybb.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)

    auc = roc_auc_score(ys, ps)
    ap  = average_precision_score(ys, ps)
    print(f"epoch {epoch:02d} | loss={tot/len(Xtr):.4f} | AUC={auc:.4f} | AP={ap:.4f}")

# Save a compact training artifact
torch.save({"model_state": model.state_dict()}, "mlp_score_then_bind.pt")
print("[saved] mlp_score_then_bind.pt")


# ==============================================================================

# ============================================================================
# CELL A: Make pseudo-structure reproducible + helper functions for new pairs
# Paste this AFTER your current score functions / model training cells
# ============================================================================

import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _stable_seed_from_seq(seq: str) -> int:
    seq = _clean(seq)
    h = hashlib.md5(seq.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)

def _pseudo_coords(seq: str) -> np.ndarray:
    # reproducible pseudo-structure from sequence
    seq = _clean(seq)
    if len(seq) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rng = np.random.default_rng(_stable_seed_from_seq(seq))
    steps = rng.normal(0.0, 1.0, size=(len(seq), 3))
    coords = np.cumsum(steps, axis=0)
    coords -= coords.mean(axis=0, keepdims=True)
    return coords.astype(np.float64)

def pair_feature_vector(seq1: str, seq2: str) -> np.ndarray:
    f1 = aa_freq(seq1)
    f2 = aa_freq(seq2)
    prod = f1 * f2
    diff = np.abs(f1 - f2)
    return np.concatenate([f1, f2, prod, diff], axis=0).astype(np.float32)

def deterministic_score_vector(seq1: str, seq2: str) -> np.ndarray:
    return np.array([
        sequence_similarity_score(seq1, seq2),
        structure_rmsd_score(seq1, seq2),
        contact_overlap_score(seq1, seq2),
        chemical_compatibility_score(seq1, seq2),
    ], dtype=np.float32)

def predict_scores_and_bind(seq1: str, seq2: str):
    x = pair_feature_vector(seq1, seq2)[None, :]
    x = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        pred_s, pred_p = model(x)
    return pred_s.cpu().numpy()[0], float(pred_p.cpu().numpy()[0])

score_names = ["SEQ", "STRUCT", "CONTACT", "CHEM"]


# ==============================================================================

# ============================================================================
# CELL B1: Robust antibody-antigen extraction using RCSB polymer entity metadata
# Replaces the old Cell B
# ============================================================================

import subprocess
subprocess.run('pip -q install requests', shell=True, check=True)

import requests
import pandas as pd
import numpy as np

PDB_IDS = [
    "4M5Z",
    "1N8Z",
    "2NY7",
    "1HZH",
    "3GBN",
    "2VIS",
    "3BN9",
    "4FQY",
]

AB_KEYWORDS = [
    "antibody", "immunoglobulin", "fab", "heavy chain", "light chain",
    "variable domain", "vh", "vl", "scfv", "nanobody", "fv fragment"
]

BAD_AG_KEYWORDS = [
    "dna", "rna"
]

def rcsb_get_entry_data(pdb_id: str):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def rcsb_get_polymer_entity(pdb_id: str, entity_id: str):
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def is_protein_entity(ent_json) -> bool:
    t = ent_json.get("entity_poly", {}).get("type", "")
    return str(t).lower() == "polypeptide(l)"

def entity_description(ent_json) -> str:
    x = ent_json.get("rcsb_polymer_entity", {})
    parts = []
    for k in ["pdbx_description"]:
        v = x.get(k)
        if v:
            parts.append(str(v))
    for item in ent_json.get("rcsb_polymer_entity_name_com", []) or []:
        if isinstance(item, dict):
            nm = item.get("name")
            if nm:
                parts.append(str(nm))
    return " | ".join(parts).lower()

def entity_sequence(ent_json) -> str:
    seq = ent_json.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
    seq = "".join(str(seq).split()).upper()
    return _clean(seq)

def entity_chain_ids(ent_json):
    ids = ent_json.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
    if ids is None:
        ids = []
    return [str(x) for x in ids]

def looks_antibody_by_description(desc: str) -> bool:
    return any(k in desc for k in AB_KEYWORDS)

def looks_bad_antigen(desc: str) -> bool:
    return any(k in desc for k in BAD_AG_KEYWORDS)

def fetch_antibody_antigen_pair(pdb_id: str):
    entry = rcsb_get_entry_data(pdb_id)
    entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    if not entity_ids:
        return None

    ents = []
    for eid in entity_ids:
        ej = rcsb_get_polymer_entity(pdb_id, str(eid))
        if not is_protein_entity(ej):
            continue

        desc = entity_description(ej)
        seq = entity_sequence(ej)
        chains = entity_chain_ids(ej)

        if len(seq) < 20:
            continue

        ents.append({
            "entity_id": str(eid),
            "desc": desc,
            "seq": seq,
            "chains": chains,
            "is_ab": looks_antibody_by_description(desc),
            "is_bad_ag": looks_bad_antigen(desc),
            "len": len(seq),
        })

    if len(ents) == 0:
        return None

    ab_ents = [e for e in ents if e["is_ab"]]
    ag_ents = [e for e in ents if (not e["is_ab"]) and (not e["is_bad_ag"])]

    if len(ab_ents) == 0 or len(ag_ents) == 0:
        return None

    # Prefer shorter antibody entities (VH/VL/Fab chains usually not huge)
    ab = sorted(ab_ents, key=lambda e: (e["len"], e["entity_id"]))[0]
    # Prefer longest non-antibody protein as antigen
    ag = sorted(ag_ents, key=lambda e: (-e["len"], e["entity_id"]))[0]

    return {
        "pdb": pdb_id,
        "antibody_entity": ab["entity_id"],
        "antibody_desc": ab["desc"],
        "antibody_chain": ",".join(ab["chains"]),
        "antibody_seq": ab["seq"],
        "antigen_entity": ag["entity_id"],
        "antigen_desc": ag["desc"],
        "antigen_chain": ",".join(ag["chains"]),
        "antigen_seq": ag["seq"],
    }

pairs = []
for pdb_id in PDB_IDS:
    try:
        pair = fetch_antibody_antigen_pair(pdb_id)
        if pair is None:
            print(f"[skip] {pdb_id}: no antibody/antigen split found from metadata")
        else:
            print(f"[ok]   {pdb_id}: AB={pair['antibody_chain']} | AG={pair['antigen_chain']}")
            pairs.append(pair)
    except Exception as e:
        print(f"[skip] {pdb_id}: {e}")

ab_pairs = pd.DataFrame(pairs)
print("\nRecovered pairs:", len(ab_pairs))
ab_pairs[["pdb","antibody_chain","antigen_chain","antibody_desc","antigen_desc"]]


# ==============================================================================

# ============================================================================
# CELL B2: Score the recovered antibody-antigen pairs
# ============================================================================

if len(ab_pairs) == 0:
    raise ValueError("Recovered pairs = 0. Try adding more PDB IDs or inspect metadata.")

rows = []
for _, r in ab_pairs.iterrows():
    ag_seq = r["antigen_seq"]
    ab_seq = r["antibody_seq"]

    det = deterministic_score_vector(ag_seq, ab_seq)
    pred_s, p_bind = predict_scores_and_bind(ag_seq, ab_seq)

    rows.append({
        "pdb": r["pdb"],
        "antigen_chain": r["antigen_chain"],
        "antibody_chain": r["antibody_chain"],
        "len_antigen": len(ag_seq),
        "len_antibody": len(ab_seq),
        "SEQ_det": float(det[0]),
        "STRUCT_det": float(det[1]),
        "CONTACT_det": float(det[2]),
        "CHEM_det": float(det[3]),
        "SEQ_pred": float(pred_s[0]),
        "STRUCT_pred": float(pred_s[1]),
        "CONTACT_pred": float(pred_s[2]),
        "CHEM_pred": float(pred_s[3]),
        "p_bind": float(p_bind),
    })

ab_panel = pd.DataFrame(rows)
ab_panel["tension_score"] = ab_panel["CONTACT_det"] - ab_panel["STRUCT_det"]
ab_panel.sort_values("tension_score", ascending=False).reset_index(drop=True)


# ==============================================================================

# ============================================================================
# CELL C: Plot antibody-antigen deterministic score distributions
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np

cols = ["SEQ_det", "STRUCT_det", "CONTACT_det", "CHEM_det"]
titles = ["SEQ", "STRUCT", "CONTACT", "CHEM"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax, c, t in zip(axes.ravel(), cols, titles):
    vals = ab_panel[c].values
    bins = min(8, max(3, len(vals)))
    ax.hist(vals, bins=bins, alpha=0.85)
    ax.axvline(vals.mean(), linestyle="--", linewidth=2, label=f"mean={vals.mean():.3f}")
    ax.set_title(f"{t} distribution")
    ax.set_xlabel(t)
    ax.set_ylabel("count")
    ax.legend()

plt.tight_layout()
plt.show()


plot_data = [ab_panel[c].values for c in cols]
plt.boxplot(plot_data, tick_labels=titles, showmeans=True)
plt.savefig('ant_boxplot_scores.png', dpi=300)

rng = np.random.default_rng(0)
for i, c in enumerate(cols, start=1):
    y = ab_panel[c].values
    x = rng.normal(i, 0.04, size=len(y))
    plt.plot(x, y, "o", alpha=0.7)

plt.ylim(0, 1)
plt.ylabel("score")
plt.title("Antibody-antigen 4-score summary")
plt.show()


# ==============================================================================

# ============================================================================
# CELL D: Check the antibody-style tension pattern directly
# ============================================================================


plt.scatter(ab_panel["STRUCT_det"], ab_panel["CONTACT_det"], s=80)

for _, r in ab_panel.iterrows():
    plt.annotate(
        r["pdb"],
        (r["STRUCT_det"], r["CONTACT_det"]),
        fontsize=9,
        xytext=(4, 4),
        textcoords="offset points"
    )

plt.xlabel("STRUCT_det")
plt.ylabel("CONTACT_det")
plt.title("Antibody-antigen tension map")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.25)
plt.savefig('ant_tensionmap.png', dpi=300)
plt.show()

ab_panel[[
    "pdb", "antigen_chain", "antibody_chain",
    "SEQ_det", "STRUCT_det", "CONTACT_det", "CHEM_det",
    "tension_score", "p_bind"
]].sort_values("tension_score", ascending=False).reset_index(drop=True)
