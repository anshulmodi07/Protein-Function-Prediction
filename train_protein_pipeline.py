"""
High-Performance Protein Function Prediction Pipeline
Computational Biology & Deep Learning Implementation

Progressive Ablation Framework:
- Experiment 1: Baseline CNN (Standard BCE + Global Threshold tau = 0.50)
- Experiment 2: Baseline CNN + Per-Class Threshold Optimization (tau_c*)
- Experiment 3: Baseline CNN + Asymmetric Loss (ASL) + Per-Class Thresholds
- Experiment 4: DeepProteinHybrid (CNN + SE + BiGRU + LayerNorm + Physicochem + ASL + Per-Class Thresholds)
- Experiment 5: (Optional) Meta ESM-2 Foundation Model Transfer Learning
"""

import os
import re
import random
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from torch.utils.data import Dataset, DataLoader

# ==================================================
# 1. REPRODUCIBILITY & DEVICE SETUP
# ==================================================
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = 0

# ==================================================
# 2. PHYSICOCHEMICAL BIOPHYSICAL PRIORS
# ==================================================
PHYSICOCHEM_DICT = {
    'A': [ 1.8,  0.0, 6.00,  89.09 / 200.0, 0.0, 0.0],
    'C': [ 2.5,  0.0, 5.07, 121.16 / 200.0, 0.0, 1.0],
    'D': [-3.5, -1.0, 2.77, 133.10 / 200.0, 0.0, 1.0],
    'E': [-3.5, -1.0, 3.22, 147.13 / 200.0, 0.0, 1.0],
    'F': [ 2.8,  0.0, 5.48, 165.19 / 200.0, 1.0, 0.0],
    'G': [-0.4,  0.0, 5.97,  75.07 / 200.0, 0.0, 0.0],
    'H': [-3.2,  0.1, 7.59, 155.16 / 200.0, 1.0, 1.0],
    'I': [ 4.5,  0.0, 6.02, 131.17 / 200.0, 0.0, 0.0],
    'K': [-3.9,  1.0, 9.74, 146.19 / 200.0, 0.0, 1.0],
    'L': [ 3.8,  0.0, 5.98, 131.17 / 200.0, 0.0, 0.0],
    'M': [ 1.9,  0.0, 5.74, 149.21 / 200.0, 0.0, 0.0],
    'N': [-3.5,  0.0, 5.41, 132.12 / 200.0, 0.0, 1.0],
    'P': [-1.6,  0.0, 6.30, 115.13 / 200.0, 0.0, 0.0],
    'Q': [-3.5,  0.0, 5.65, 146.15 / 200.0, 0.0, 1.0],
    'R': [-4.5,  1.0, 10.76, 174.20 / 200.0, 0.0, 1.0],
    'S': [-0.8,  0.0, 5.68, 105.09 / 200.0, 0.0, 1.0],
    'T': [-0.7,  0.0, 5.60, 119.12 / 200.0, 0.0, 1.0],
    'V': [ 4.2,  0.0, 5.96, 117.15 / 200.0, 0.0, 0.0],
    'W': [-0.9,  0.0, 5.89, 204.23 / 200.0, 1.0, 1.0],
    'Y': [-1.3,  0.0, 5.66, 181.19 / 200.0, 1.0, 1.0]
}

def get_physicochemical_matrix():
    mat = torch.zeros(len(VOCAB) + 1, 6, dtype=torch.float32)
    for aa, idx in VOCAB.items():
        mat[idx] = torch.tensor(PHYSICOCHEM_DICT[aa], dtype=torch.float32)
    return mat

PHYSICO_MATRIX = get_physicochemical_matrix()

# ==================================================
# 3. SOTA LOSS FUNCTIONS
# ==================================================
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., ICCV 2021)
    Dynamically downweights easy negative instances without distorting positive gradients.
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        loss = loss_pos * ((1.0 - xs_pos) ** self.gamma_pos) + loss_neg * ((1.0 - xs_neg) ** self.gamma_neg)
        return -loss.mean()

# ==================================================
# 4. NEURAL ARCHITECTURES
# ==================================================
class Conv1DLayerNorm(nn.Module):
    """
    Channel-wise LayerNorm for Conv1D: [B, C, L] -> [B, L, C] -> LayerNorm -> [B, C, L].
    100% immune to padding distribution contamination.
    """
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class SqueezeExcitation1D(nn.Module):
    """
    Channel Attention block with length-masked global pooling.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        mask_exp = mask.unsqueeze(1).float()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        gap = (x * mask_exp).sum(dim=2) / lengths
        scale = self.fc(gap).unsqueeze(2)
        return x * scale

class MaskedAttentionPooling(nn.Module):
    """
    Learnable attention-weighted sequence pooling ignoring PAD tokens.
    """
    def __init__(self, in_features):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

class DeepProteinHybrid(nn.Module):
    """
    State-of-the-art DeepProteinHybrid:
    - Residue Embedding + Physicochemical Biophysical Priors
    - Multi-Scale Convolutions (Kernels 3, 5, 9, 15) with Channel-Wise LayerNorm
    - Squeeze-and-Excitation (SE) Dynamic Channel Recalibration
    - Bidirectional GRU capturing Tertiary & Long-Range Sequential Context
    - Dual Masked Pooling (Attention-Weighted + Masked Max)
    - Residual Classification Head with LayerNorm & Dropout
    """
    def __init__(self, num_classes=40, emb_dim=64, pad_idx=0, physico_table=PHYSICO_MATRIX):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(len(VOCAB) + 1, emb_dim, padding_idx=pad_idx)
        self.physico_emb = nn.Embedding.from_pretrained(physico_table, freeze=True)
        self.emb_drop = nn.Dropout(0.1)

        total_input_dim = emb_dim + 6
        self.input_proj = nn.Conv1d(total_input_dim, 128, kernel_size=1)

        branch_ch = 48
        self.b3 = nn.Sequential(
            nn.Conv1d(128, branch_ch, kernel_size=3, padding=1),
            Conv1DLayerNorm(branch_ch),
            nn.GELU()
        )
        self.b5 = nn.Sequential(
            nn.Conv1d(128, branch_ch, kernel_size=5, padding=2),
            Conv1DLayerNorm(branch_ch),
            nn.GELU()
        )
        self.b9 = nn.Sequential(
            nn.Conv1d(128, branch_ch, kernel_size=9, padding=4),
            Conv1DLayerNorm(branch_ch),
            nn.GELU()
        )
        self.b15 = nn.Sequential(
            nn.Conv1d(128, branch_ch, kernel_size=15, padding=7),
            Conv1DLayerNorm(branch_ch),
            nn.GELU()
        )
        conv_total = branch_ch * 4  # 192 channels

        self.se = SqueezeExcitation1D(conv_total)
        self.res_conv = nn.Sequential(
            nn.Conv1d(conv_total, conv_total, kernel_size=5, padding=2),
            Conv1DLayerNorm(conv_total),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.gru = nn.GRU(conv_total, 128, batch_first=True, bidirectional=True)
        self.attn_pool = MaskedAttentionPooling(256)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        mask = (x != self.pad_idx)
        t_emb = self.token_emb(x)
        p_emb = self.physico_emb(x)
        emb = torch.cat([t_emb, p_emb], dim=-1)
        emb = self.emb_drop(emb).transpose(1, 2)

        h = self.input_proj(emb)
        b3 = self.b3(h)
        b5 = self.b5(h)
        b9 = self.b9(h)
        b15 = self.b15(h)
        merged = torch.cat([b3, b5, b9, b15], dim=1)

        conv_out = self.se(self.res_conv(merged), mask) + merged

        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)

        att_feat = self.attn_pool(gru_out, mask)
        gru_masked = gru_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_feat = torch.max(gru_masked, dim=1)[0]

        combined = torch.cat([att_feat, max_feat], dim=1)
        return self.classifier(combined)

class MaskedBaselineCNN(nn.Module):
    """
    Baseline CNN for exact ablation comparison.
    """
    def __init__(self, num_labels, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(len(VOCAB) + 1, 32, padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_labels)

    def forward(self, x):
        mask = (x != self.pad_idx)
        emb = self.embedding(x).transpose(1, 2)
        h1 = self.relu(self.conv1(emb))
        h2 = self.relu(self.conv2(h1))
        mask_expanded = mask.unsqueeze(1)
        h2_masked = h2.masked_fill(~mask_expanded, -1e9)
        pooled = torch.max(h2_masked, dim=2)[0]
        out = self.dropout(pooled)
        return self.fc(out)

# ==================================================
# 5. DATASET & LEAK-FREE PREPARATION
# ==================================================
class FastProteinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).long()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_and_preprocess_data(data_path="uniprot_proteins.csv", max_len=1000, min_label_count=25, max_labels=40, max_samples=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")

    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["sequence", "go_terms"]).copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    df = df[df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")].copy()

    def extract_go(val):
        return list(set(re.findall(r"GO:\d{7}", str(val))))

    df["labels"] = df["go_terms"].apply(extract_go)
    df = df[df["labels"].map(len) > 0]

    if max_samples is not None and max_samples < len(df):
        print(f"Sampling {max_samples:,} sequences for rapid ablation execution...")
        df = df.sample(n=max_samples, random_state=SEED).copy()

    all_sequences = df["sequence"].tolist()
    all_labels = df["labels"].tolist()

    indices = np.arange(len(all_sequences))
    train_idx_raw, temp_idx = train_test_split(indices, test_size=0.30, random_state=SEED)
    val_idx_raw, test_idx_raw = train_test_split(temp_idx, test_size=0.50, random_state=SEED)

    train_label_counter = Counter(term for i in train_idx_raw for term in set(all_labels[i]))
    min_cnt = max(5, int(min_label_count * (len(df) / 152000))) if max_samples else min_label_count
    eligible_labels = {term for term, count in train_label_counter.items() if count >= min_cnt}
    selected_labels = sorted(eligible_labels, key=lambda term: train_label_counter[term], reverse=True)[:max_labels]
    selected_set = set(selected_labels)

    filtered_labels = [[term for term in row if term in selected_set] for row in all_labels]
    valid_mask = [len(row) > 0 for row in filtered_labels]
    valid_indices = set(i for i, v in enumerate(valid_mask) if v)

    train_idx = [i for i in train_idx_raw if i in valid_indices]
    val_idx = [i for i in val_idx_raw if i in valid_indices]
    test_idx = [i for i in test_idx_raw if i in valid_indices]

    mlb = MultiLabelBinarizer(classes=selected_labels)
    Y_all = mlb.fit_transform(filtered_labels).astype(np.float32)

    def encode_seq(seq):
        seq = seq[:max_len]
        arr = [VOCAB[aa] for aa in seq]
        arr += [PAD_IDX] * (max_len - len(arr))
        return arr

    X_all = np.array([encode_seq(seq) for seq in all_sequences], dtype=np.int64)

    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    X_test, Y_test = X_all[test_idx], Y_all[test_idx]

    print(f"Data Prepared -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Classes: {len(selected_labels)}")
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), selected_labels

# ==================================================
# 6. TRAINING ENGINE & THRESHOLD CALIBRATION
# ==================================================
def train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                       scheduler=None, epochs=15, patience=5, save_path="best_model.pt"):
    best_val_loss = float("inf")
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_Y = batch_Y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(batch_X)
                loss = criterion(logits, batch_Y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(batch_X)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_Y = batch_Y.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(batch_X)
                    loss = criterion(logits, batch_Y)
                val_loss += loss.item() * len(batch_X)
        val_loss /= len(val_loader.dataset)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            tag = " [Best Saved]"
        else:
            patience_counter += 1
            tag = ""

        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_curr:.6f}{tag}")
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model

def predict_probabilities(model, data_loader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                probs = torch.sigmoid(model(batch_X))
            all_probs.append(probs.float().cpu().numpy())
            all_targets.append(batch_Y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)

def tune_per_class_thresholds(val_probs, val_targets, min_tau=0.10, max_tau=0.90, steps=81):
    num_classes = val_targets.shape[1]
    candidate_taus = np.linspace(min_tau, max_tau, steps)
    optimal_taus = np.zeros(num_classes)

    for c in range(num_classes):
        y_true = val_targets[:, c]
        p = val_probs[:, c]
        best_f1 = -1.0
        best_tau = 0.5
        for t in candidate_taus:
            y_pred = (p >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = t
        optimal_taus[c] = best_tau

    best_global_tau = 0.5
    best_global_f1 = -1.0
    for t in candidate_taus:
        preds = (val_probs >= t).astype(int)
        score = f1_score(val_targets, preds, average="micro", zero_division=0)
        if score > best_global_f1:
            best_global_f1 = score
            best_global_tau = t

    return optimal_taus, best_global_tau

def compute_all_metrics(probs, targets, tau, is_per_class=False):
    if is_per_class:
        preds = (probs >= tau).astype(int)
        tau_label = "Per-Class Calibrated"
    else:
        preds = (probs >= tau).astype(int)
        tau_label = round(float(tau), 4)

    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_prec = precision_score(targets, preds, average="micro", zero_division=0)
    macro_prec = precision_score(targets, preds, average="macro", zero_division=0)
    micro_rec = recall_score(targets, preds, average="micro", zero_division=0)
    macro_rec = recall_score(targets, preds, average="macro", zero_division=0)

    thresholds = np.linspace(0.01, 0.99, 99)
    fmax = 0.0
    best_t = 0.5
    for t in thresholds:
        p_t = (probs >= t).astype(int)
        f_t = f1_score(targets, p_t, average="micro", zero_division=0)
        if f_t > fmax:
            fmax = f_t
            best_t = t

    micro_aupr = average_precision_score(targets, probs, average="micro")
    macro_aupr = average_precision_score(targets, probs, average="macro")
    weighted_aupr = average_precision_score(targets, probs, average="weighted")

    return {
        "Threshold": tau_label,
        "Micro_F1": round(float(micro_f1), 4),
        "Macro_F1": round(float(macro_f1), 4),
        "Micro_Precision": round(float(micro_prec), 4),
        "Macro_Precision": round(float(macro_prec), 4),
        "Micro_Recall": round(float(micro_rec), 4),
        "Macro_Recall": round(float(macro_rec), 4),
        "CAFA_Fmax": round(float(fmax), 4),
        "Optimal_Fmax_Tau": round(float(best_t), 2),
        "Micro_AUPR": round(float(micro_aupr), 4),
        "Macro_AUPR": round(float(macro_aupr), 4),
        "Weighted_AUPR": round(float(weighted_aupr), 4)
    }

# ==================================================
# 7. MAIN PIPELINE & ABLATION RUNNER
# ==================================================
def main():
    parser = argparse.ArgumentParser(description="Train SOTA Protein Function Prediction Pipeline")
    parser.add_argument("--data_path", type=str, default="uniprot_proteins.csv", help="Path to uniprot_proteins.csv")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64 if torch.cuda.is_available() else 32, help="Batch size")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=1000, help="Max protein sequence length")
    parser.add_argument("--max_samples", type=int, default=None, help="Sample size for rapid execution (None = full)")
    parser.add_argument("--run_ablations", action="store_true", help="Run all 4 progressive ablation experiments")
    parser.add_argument("--model", type=str, choices=["hybrid", "baseline"], default="hybrid", help="Architecture choice")
    parser.add_argument("--loss", type=str, choices=["asl", "bce"], default="asl", help="Loss function")
    args = parser.parse_args()

    print("=" * 80)
    print(f"DeepProtein SOTA Training Pipeline | Device: {DEVICE}")
    if args.max_samples:
        print(f"Rapid Ablation Mode: Using {args.max_samples:,} sequences | Max Length: {args.max_len}")
    print("=" * 80)

    train_data, val_data, test_data, selected_labels = load_and_preprocess_data(
        data_path=args.data_path, max_len=args.max_len, min_label_count=25, max_labels=40, max_samples=args.max_samples
    )

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(FastProteinDataset(*train_data), batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(FastProteinDataset(*val_data), batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(FastProteinDataset(*test_data), batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem)
    num_classes = len(selected_labels)

    if args.run_ablations:
        print("\n" + "="*80)
        print("RUNNING ALL PROGRESSIVE ABLATION EXPERIMENTS START-TO-FINISH")
        print("="*80)
        results = []

        # -----------------------------------------------------------------
        # Experiment 1: Baseline CNN (Fixed Global Threshold)
        # -----------------------------------------------------------------
        print("\n>>> [1/4] Training Baseline CNN with Standard BCE...")
        m1 = MaskedBaselineCNN(num_labels=num_classes).to(DEVICE)
        opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
        crit1 = nn.BCEWithLogitsLoss()
        m1 = train_and_validate(m1, train_loader, val_loader, crit1, opt1, epochs=args.epochs, save_path="exp1_baseline.pt")
        vp1, vt1 = predict_probabilities(m1, val_loader)
        tp1, tt1 = predict_probabilities(m1, test_loader)
        opt_taus1, glob_tau1 = tune_per_class_thresholds(vp1, vt1)

        res1 = compute_all_metrics(tp1, tt1, glob_tau1, is_per_class=False)
        res1["Experiment"] = "1. Baseline CNN (Global Tau)"
        results.append(res1)
        print(f"  -> Result: Micro F1 = {res1['Micro_F1']} | Macro F1 = {res1['Macro_F1']} | Fmax = {res1['CAFA_Fmax']}")

        # -----------------------------------------------------------------
        # Experiment 2: Baseline CNN + Per-Class Threshold Optimization
        # -----------------------------------------------------------------
        print("\n>>> [2/4] Evaluating Baseline CNN with Per-Class Calibrated Thresholds...")
        res2 = compute_all_metrics(tp1, tt1, opt_taus1, is_per_class=True)
        res2["Experiment"] = "2. Baseline CNN (Per-Class Taus)"
        results.append(res2)
        print(f"  -> Result: Micro F1 = {res2['Micro_F1']} | Macro F1 = {res2['Macro_F1']} | Fmax = {res2['CAFA_Fmax']}")

        # -----------------------------------------------------------------
        # Experiment 3: Baseline CNN + Asymmetric Loss (ASL)
        # -----------------------------------------------------------------
        print("\n>>> [3/4] Training Baseline CNN with Asymmetric Loss (ASL)...")
        m3 = MaskedBaselineCNN(num_labels=num_classes).to(DEVICE)
        opt3 = torch.optim.Adam(m3.parameters(), lr=1e-3)
        crit3 = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
        m3 = train_and_validate(m3, train_loader, val_loader, crit3, opt3, epochs=args.epochs, save_path="exp3_asl.pt")
        vp3, vt3 = predict_probabilities(m3, val_loader)
        tp3, tt3 = predict_probabilities(m3, test_loader)
        opt_taus3, glob_tau3 = tune_per_class_thresholds(vp3, vt3)

        res3 = compute_all_metrics(tp3, tt3, opt_taus3, is_per_class=True)
        res3["Experiment"] = "3. Baseline CNN + ASL Loss + Per-Class Taus"
        results.append(res3)
        print(f"  -> Result: Micro F1 = {res3['Micro_F1']} | Macro F1 = {res3['Macro_F1']} | Fmax = {res3['CAFA_Fmax']}")

        # -----------------------------------------------------------------
        # Experiment 4: DeepProteinHybrid (Full Suite)
        # -----------------------------------------------------------------
        print("\n>>> [4/4] Training DeepProteinHybrid (CNN+SE+BiGRU+LayerNorm+Physicochem+ASL)...")
        m4 = DeepProteinHybrid(num_classes=num_classes).to(DEVICE)
        opt4 = torch.optim.AdamW(m4.parameters(), lr=args.lr, weight_decay=1e-4)
        sched4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4, T_max=args.epochs, eta_min=1e-5)
        crit4 = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
        m4 = train_and_validate(m4, train_loader, val_loader, crit4, opt4, scheduler=sched4, epochs=args.epochs, save_path="exp4_deepprotein_hybrid.pt")
        vp4, vt4 = predict_probabilities(m4, val_loader)
        tp4, tt4 = predict_probabilities(m4, test_loader)
        opt_taus4, glob_tau4 = tune_per_class_thresholds(vp4, vt4)

        res4 = compute_all_metrics(tp4, tt4, opt_taus4, is_per_class=True)
        res4["Experiment"] = "4. DeepProteinHybrid (Full Suite + Per-Class Taus)"
        results.append(res4)
        print(f"  -> Result: Micro F1 = {res4['Micro_F1']} | Macro F1 = {res4['Macro_F1']} | Fmax = {res4['CAFA_Fmax']}")

        summary_df = pd.DataFrame(results)
        cols = ["Experiment", "Threshold", "Micro_F1", "Macro_F1", "Micro_Precision", "Macro_Precision", "Micro_Recall", "Macro_Recall", "CAFA_Fmax"]
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        summary_df.to_csv("progressive_experiment_results.csv", index=False)

        print("\n" + "=" * 90)
        print("SUMMARY OF PROGRESSIVE EXPERIMENTAL RESULTS:")
        print("=" * 90)
        print(summary_df.to_string(index=False))
        print("=" * 90)

        # Save production bundle
        torch.save({
            "model_state": m4.state_dict(),
            "classes": selected_labels,
            "vocab": VOCAB,
            "max_len": args.max_len,
            "per_class_thresholds": opt_taus4,
            "global_threshold": glob_tau4,
            "metrics": res4
        }, "deepprotein_hybrid_production.pt")
        print("Saved production bundle to: deepprotein_hybrid_production.pt")
        print("Saved results to: progressive_experiment_results.csv")
        return

    # Single model training path
    if args.model == "hybrid":
        model = DeepProteinHybrid(num_classes=num_classes).to(DEVICE)
    else:
        model = MaskedBaselineCNN(num_labels=num_classes).to(DEVICE)

    if args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    save_path = f"best_{args.model}_{args.loss}.pt"
    model = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler,
                               epochs=args.epochs, patience=5, save_path=save_path)

    val_probs, val_targets = predict_probabilities(model, val_loader)
    optimal_taus, best_global_tau = tune_per_class_thresholds(val_probs, val_targets)

    test_probs, test_targets = predict_probabilities(model, test_loader)
    m_perclass = compute_all_metrics(test_probs, test_targets, optimal_taus, is_per_class=True)
    m_perclass["Model"] = f"{args.model.upper()} ({args.loss.upper()})"
    res_df = pd.DataFrame([m_perclass])
    print(res_df.to_string(index=False))
    res_df.to_csv("pipeline_test_results.csv", index=False)

if __name__ == "__main__":
    main()
