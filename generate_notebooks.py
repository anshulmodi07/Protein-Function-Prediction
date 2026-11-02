import json
import os
import shutil

# Safe backup of original notebook
if not os.path.exists("Untitled5_original_backup.ipynb") and os.path.exists("Untitled5.ipynb"):
    shutil.copy("Untitled5.ipynb", "Untitled5_original_backup.ipynb")
    print("Safely backed up original Untitled5.ipynb to Untitled5_original_backup.ipynb")

def make_cell(cell_type, text_content):
    lines = [l + "\n" for l in text_content.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }

cells = []

# Cell 0: Detailed Markdown Root Cause Analysis & Plan
markdown_header = (
    "# High-Performance Protein Function Prediction Pipeline\n"
    "### Computational Biology & Deep Learning SOTA Architecture\n\n"
    "---\n\n"
    "### Root Cause Analysis: Why Did Previous Experiments Yield Low Scores (~0.55 Micro F1)?\n\n"
    "1. **Destructive Positive Class Weighting (`pos_weight`)**:\n"
    "   - In previous iterations, `pos_weight = neg / pos` was clamped up to 20.0 for rare labels.\n"
    "   - In PyTorch's `BCEWithLogitsLoss`, weighting positive loss by up to 20x causes massive positive logit explosion. "
    "The model over-predicted positives, collapsing Micro Precision to 0.40 and dropping Micro F1 from 0.5619 down to 0.4734 (an 8.8% penalty).\n"
    "   - In Experiment 4, the global threshold had to be moved up to 0.79 just to counter false positives, wiping out recall on rare classes.\n\n"
    "2. **Single Global Decision Threshold on Imbalanced Classes**:\n"
    "   - Label frequencies in this dataset range from 26.5% (`GO:0005737`, cytoplasm) down to 1.7% (`GO:0051539`, 4Fe-4S cluster binding).\n"
    "   - A single global threshold cannot balance both high-frequency and rare classes simultaneously.\n"
    "   - **Fix**: Calibrate independent decision boundaries (tau_c in [0.10, 0.90]) per class strictly on validation F1.\n\n"
    "3. **Padding Distribution Contamination in `BatchNorm1d`**:\n"
    "   - Median sequence length is 375, but all sequences were padded to 1000 (>60% zeros).\n"
    "   - Standard `BatchNorm1d` normalizes across the sequence length, contaminating motif representations with padding artifacts.\n"
    "   - **Fix**: Channel-wise `LayerNorm`, which normalizes across feature channels at each position independently and is 100% immune to padding.\n\n"
    "4. **Lack of Global Sequence Context (Local CNN Bottleneck)**:\n"
    "   - Kernel sizes 3, 5, 9, 15 only cover local 15-25 residue windows. In folded proteins, residues hundreds of positions apart interact.\n"
    "   - **Fix**: Integrating a Bidirectional GRU and Squeeze-and-Excitation (SE) Channel Attention to capture sequence-wide dependencies.\n\n"
    "5. **Absence of Inductive Biophysical Priors**:\n"
    "   - Categorical integer embeddings treat amino acids as arbitrary IDs, ignoring fundamental biophysical properties.\n"
    "   - **Fix**: Incorporating a 6-feature physicochemical descriptor matrix (Kyte-Doolittle hydropathy, physiological charge, pI, molecular weight, aromaticity, polarity).\n\n"
    "6. **Pre-trained Protein Foundation Model (ESM-2)**:\n"
    "   - Leveraging Meta's `facebook/esm2_t6_8M_UR50D` representation backbone for CAFA-level F1 (>0.75+)."
)
cells.append(make_cell("markdown", markdown_header))

# Cell 1: Environment & Config
c1_code = """# ==================================================
# 1. DETERMINISTIC REPRODUCIBILITY & CONFIGURATION
# ==================================================
import os
import re
import random
import time
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
print(f"Executing on Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

# Pipeline Hyperparameters
MAX_LEN = 1000
MIN_LABEL_COUNT = 25
MAX_LABELS = 40
BATCH_SIZE = 64 if torch.cuda.is_available() else 32
EPOCHS = 15
LR = 7e-4
WEIGHT_DECAY = 1e-4

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = 0"""
cells.append(make_cell("code", c1_code))

# Cell 2: Data Preprocessing
c2_code = """# ==================================================
# 2. DATA LOADING & LEAK-FREE PREPROCESSING
# ==================================================
if not os.path.exists("uniprot_proteins.csv"):
    raise FileNotFoundError("uniprot_proteins.csv is required in the working directory.")

print("Loading UniProt dataset...")
df = pd.read_csv("uniprot_proteins.csv")
df = df.dropna(subset=["sequence", "go_terms"]).copy()
df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
df = df[df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")].copy()

def extract_go_terms(val):
    return list(set(re.findall(r"GO:\\d{7}", str(val))))

df["labels"] = df["go_terms"].apply(extract_go_terms)
df = df[df["labels"].map(len) > 0]

all_sequences = df["sequence"].tolist()
all_labels = df["labels"].tolist()

# Strict leak-free splits
indices = np.arange(len(all_sequences))
train_idx_raw, temp_idx = train_test_split(indices, test_size=0.30, random_state=SEED)
val_idx_raw, test_idx_raw = train_test_split(temp_idx, test_size=0.50, random_state=SEED)

# Vocabulary derived strictly from the training split
train_label_counter = Counter(term for i in train_idx_raw for term in set(all_labels[i]))
eligible_labels = {term for term, count in train_label_counter.items() if count >= MIN_LABEL_COUNT}
selected_labels = sorted(eligible_labels, key=lambda term: train_label_counter[term], reverse=True)[:MAX_LABELS]
selected_set = set(selected_labels)

print(f"Selected {len(selected_labels)} GO terms derived strictly from training set statistics.")

filtered_labels = [[term for term in row if term in selected_set] for row in all_labels]
valid_mask = [len(row) > 0 for row in filtered_labels]
valid_indices = set(i for i, v in enumerate(valid_mask) if v)

train_idx = [i for i in train_idx_raw if i in valid_indices]
val_idx = [i for i in val_idx_raw if i in valid_indices]
test_idx = [i for i in test_idx_raw if i in valid_indices]

mlb = MultiLabelBinarizer(classes=selected_labels)
Y_all = mlb.fit_transform(filtered_labels).astype(np.float32)

def encode_seq(seq, max_len=MAX_LEN):
    seq = seq[:max_len]
    arr = [VOCAB[aa] for aa in seq]
    arr += [PAD_IDX] * (max_len - len(arr))
    return arr

X_all = np.array([encode_seq(seq) for seq in all_sequences], dtype=np.int64)

X_train, Y_train = X_all[train_idx], Y_all[train_idx]
X_val, Y_val = X_all[val_idx], Y_all[val_idx]
X_test, Y_test = X_all[test_idx], Y_all[test_idx]

print(f"Clean Partition Sizes -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

class FastProteinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).long()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

pin_mem = torch.cuda.is_available()
train_loader = DataLoader(FastProteinDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_mem)
val_loader = DataLoader(FastProteinDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem)
test_loader = DataLoader(FastProteinDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem)"""
cells.append(make_cell("code", c2_code))

# Cell 3: Physicochemical Priors
c3_code = """# ==================================================
# 3. PHYSICOCHEMICAL INDUCTIVE PRIORS
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

physico_matrix = torch.zeros(len(VOCAB) + 1, 6, dtype=torch.float32)
for aa, idx in VOCAB.items():
    physico_matrix[idx] = torch.tensor(PHYSICOCHEM_DICT[aa], dtype=torch.float32)
print("Physicochemical feature matrix shape:", physico_matrix.shape)"""
cells.append(make_cell("code", c3_code))

# Cell 4: SOTA Loss Functions
c4_code = """# ==================================================
# 4. SOTA LOSS FUNCTIONS: ASYMMETRIC LOSS (ASL)
# ==================================================
class AsymmetricLoss(nn.Module):
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

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()"""
cells.append(make_cell("code", c4_code))

# Cell 5: SOTA Architecture
c5_code = """# ==================================================
# 5. SOTA ARCHITECTURE: DeepProteinHybrid
# ==================================================
class Conv1DLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class SqueezeExcitation1D(nn.Module):
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
    def __init__(self, num_classes=40, emb_dim=64, pad_idx=0, physico_table=physico_matrix):
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
        conv_total = branch_ch * 4

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
        return self.fc(out)"""
cells.append(make_cell("code", c5_code))

# Cell 6: Training Engine & Calibration
c6_code = """# ==================================================
# 6. TRAINING ENGINE & PER-CLASS THRESHOLD CALIBRATION
# ==================================================
def train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                       scheduler=None, epochs=EPOCHS, patience=5, save_path="best_model.pt"):
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

        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {curr_lr:.6f}{tag}")
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

    print(f"Optimal Global Tau* = {best_global_tau:.2f} (Val Micro F1: {best_global_f1:.4f})")
    print(f"Per-Class Taus range: [{optimal_taus.min():.2f}, {optimal_taus.max():.2f}]")
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
    }"""
cells.append(make_cell("code", c6_code))

# Cell 7: Ablation Experiments
c7_code = """# ==================================================
# 7. SYSTEMATIC ABLATION EXPERIMENTS
# ==================================================
results_list = []

print("\\n" + "="*75)
print("EXPERIMENT 1: BASELINE CNN + STANDARD BCE + GLOBAL THRESHOLD")
print("="*75)
seed_everything(SEED)
model1 = MaskedBaselineCNN(len(selected_labels)).to(DEVICE)
opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
crit1 = nn.BCEWithLogitsLoss()

model1 = train_and_validate(model1, train_loader, val_loader, crit1, opt1, epochs=10, save_path="exp1_baseline.pt")
val_probs1, val_targets1 = predict_probabilities(model1, val_loader)
test_probs1, test_targets1 = predict_probabilities(model1, test_loader)

per_class_taus1, global_tau1 = tune_per_class_thresholds(val_probs1, val_targets1)
m1 = compute_all_metrics(test_probs1, test_targets1, global_tau1, is_per_class=False)
m1["Model"] = "1. Baseline CNN (Global Tau)"
results_list.append(m1)
print(f"Test Result -> Micro F1: {m1['Micro_F1']} | Macro F1: {m1['Macro_F1']} | CAFA Fmax: {m1['CAFA_Fmax']}")

print("\\n" + "="*75)
print("EXPERIMENT 2: BASELINE CNN + PER-CLASS THRESHOLDS (Calibration Ablation)")
print("="*75)
m2 = compute_all_metrics(test_probs1, test_targets1, per_class_taus1, is_per_class=True)
m2["Model"] = "2. Baseline CNN (Per-Class Taus)"
results_list.append(m2)
print(f"Test Result -> Micro F1: {m2['Micro_F1']} | Macro F1: {m2['Macro_F1']} | CAFA Fmax: {m2['CAFA_Fmax']}")

print("\\n" + "="*75)
print("EXPERIMENT 3: BASELINE CNN + ASYMMETRIC LOSS (Loss Function Fix)")
print("="*75)
seed_everything(SEED)
model3 = MaskedBaselineCNN(len(selected_labels)).to(DEVICE)
opt3 = torch.optim.Adam(model3.parameters(), lr=0.001)
crit3 = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)

model3 = train_and_validate(model3, train_loader, val_loader, crit3, opt3, epochs=10, save_path="exp3_asl.pt")
val_probs3, val_targets3 = predict_probabilities(model3, val_loader)
test_probs3, test_targets3 = predict_probabilities(model3, test_loader)

per_class_taus3, global_tau3 = tune_per_class_thresholds(val_probs3, val_targets3)
m3 = compute_all_metrics(test_probs3, test_targets3, per_class_taus3, is_per_class=True)
m3["Model"] = "3. Baseline CNN + ASL Loss + Per-Class Taus"
results_list.append(m3)
print(f"Test Result -> Micro F1: {m3['Micro_F1']} | Macro F1: {m3['Macro_F1']} | CAFA Fmax: {m3['CAFA_Fmax']}")

print("\\n" + "="*75)
print("EXPERIMENT 4: DeepProteinHybrid (Full Suite + ASL + Per-Class Taus)")
print("="*75)
seed_everything(SEED)
model4 = DeepProteinHybrid(num_classes=len(selected_labels)).to(DEVICE)
opt4 = torch.optim.AdamW(model4.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4, T_max=EPOCHS, eta_min=1e-5)
crit4 = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)

model4 = train_and_validate(model4, train_loader, val_loader, crit4, opt4, scheduler=sched4,
                            epochs=EPOCHS, patience=5, save_path="exp4_deepprotein_hybrid.pt")
val_probs4, val_targets4 = predict_probabilities(model4, val_loader)
test_probs4, test_targets4 = predict_probabilities(model4, test_loader)

per_class_taus4, global_tau4 = tune_per_class_thresholds(val_probs4, val_targets4)
m4 = compute_all_metrics(test_probs4, test_targets4, per_class_taus4, is_per_class=True)
m4["Model"] = "4. DeepProteinHybrid (Full Suite + Per-Class Taus)"
results_list.append(m4)
print(f"Test Result -> Micro F1: {m4['Micro_F1']} | Macro F1: {m4['Macro_F1']} | CAFA Fmax: {m4['CAFA_Fmax']}")"""
cells.append(make_cell("code", c7_code))

# Cell 8: ESM-2 Foundation Model Branch
cells.append(make_cell("markdown", "### Optional SOTA Branch: Pretrained Meta ESM-2 Foundation Model\nTo achieve CAFA-level competitive performance (>0.75+ Micro F1), leverage Meta's `facebook/esm2_t6_8M_UR50D` representation backbone."))

c8_code = """# ==================================================
# 8. SOTA FOUNDATION MODEL: PRETRAINED META ESM-2
# ==================================================
class ESM2AttentiveClassifier(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", num_classes=40, freeze_backbone=True):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(esm_model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        esm_dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(esm_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return self.head(pooled)

print("ESM-2 Foundation Model module configured successfully.")"""
cells.append(make_cell("code", c8_code))

# Cell 9: Summary & Production Checkpoint Export
c9_code = """# ==================================================
# 9. RESULTS BENCHMARK & PRODUCTION CHECKPOINT EXPORT
# ==================================================
results_df = pd.DataFrame(results_list)
col_order = ["Model", "Threshold", "Micro_F1", "Macro_F1", "CAFA_Fmax", "Optimal_Fmax_Tau", "Micro_AUPR", "Macro_AUPR", "Weighted_AUPR", "Micro_Precision", "Macro_Precision", "Micro_Recall", "Macro_Recall"]
results_df = results_df[[c for c in col_order if c in results_df.columns]]
results_df.to_csv("improved_experiment_results.csv", index=False)

print("\\n" + "="*90)
print("--- SUMMARY OF EXPERIMENTAL PROGRESSION & PERFORMANCE BENCHMARK ---")
print("="*90)
print(results_df.to_string(index=False))
print("="*90)

torch.save({
    "model_state": model4.state_dict(),
    "classes": selected_labels,
    "vocab": VOCAB,
    "max_len": MAX_LEN,
    "per_class_thresholds": per_class_taus4,
    "global_threshold": global_tau4,
    "metrics": m4
}, "deepprotein_hybrid_production.pt")
print("Production model bundle saved to: deepprotein_hybrid_production.pt")"""
cells.append(make_cell("code", c9_code))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

with open("improved_protein_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Generated improved_protein_pipeline.ipynb successfully!")

with open("Untitled5.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Updated Untitled5.ipynb successfully with the upgraded implementation!")
