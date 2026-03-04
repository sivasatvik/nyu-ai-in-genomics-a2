# =============================================================================
# Assignment 2: Single-Cell RNA-seq Analysis — Braun et al. ccRCC Atlas
# =============================================================================
# This file implements the complete pipeline described in the assignment:
#   Part 1: Data Understanding and Classical Baselines
#     1.1  Data understanding and preprocessing
#     1.2  Semi-supervised label masking (30 %)
#     1.3  PCA + kNN baseline + UMAPs
#     1.4  Deep MLP classifier baseline
#   Part 2: Unsupervised Autoencoders and scVI/scANVI Pipeline
#     2.1  Deep Autoencoder baseline
#     2.2  scVI pre-training
#     2.3  scANVI training and annotation
#     2.4  Orthogonal projection on test.h5ad
#     2.5  Interpretability via Captum Integrated Gradients
# =============================================================================

import pickle
import random
import sys
import warnings

# Force line-buffered stdout so print statements appear immediately in sbatch /
# Singularity logs rather than being flushed only at script exit.
sys.stdout.reconfigure(line_buffering=True)

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scvi
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, frameon=False)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_PATH = Path("braun_dataset.h5ad")
TEST_PATH = Path("test.h5ad")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Download datasets from Google Drive (skipped if files already exist)
# --------------------------------------------------------------------------- #
import gdown  # noqa: E402  (imported here to keep library block together)

_GDRIVE_URLS = {
    DATA_PATH: "https://drive.google.com/uc?id=1ZZWbVq-qwGUr76WSecPUrKiyLJP4VVtz",
    TEST_PATH: "https://drive.google.com/uc?id=1uZkdqE5df-0Mul8VFahCyQoDjXJ-0IFP",
}

for _path, _url in _GDRIVE_URLS.items():
    if not _path.exists():
        print(f"Downloading {_path.name} from Google Drive …")
        gdown.download(_url, str(_path), quiet=False, fuzzy=False)
        print(f"  Saved to {_path}")
    else:
        print(f"{_path.name} already present — skipping download.")

# =============================================================================
# PART 1 — Data Understanding and Classical Baselines
# =============================================================================

# ---------------------------------------------------------------------------
# 1.1  Load and understand the data
# ---------------------------------------------------------------------------

print("=" * 70)
print("1.1  Data Understanding and Preprocessing")
print("=" * 70)

braun_dataset = sc.read_h5ad(DATA_PATH)
print(braun_dataset)
print("\n.obs columns:", braun_dataset.obs.columns.tolist())
print("\nFirst few rows of .obs:")
print(braun_dataset.obs.head().to_string())

# --- Optionally subsample to ≤20 000 cells for speed ---
MAX_CELLS = 20_000
if braun_dataset.n_obs > MAX_CELLS:
    sc.pp.subsample(braun_dataset, n_obs=MAX_CELLS, random_state=SEED)
    print(f"\nSubsampled to {braun_dataset.n_obs} cells.")

# Commentary --------------------------------------------------------------- #
# Cell-type annotation is central to interpreting the tumour micro-
# environment (TME) because single cells do not carry visible morphological
# markers after dissociation.  In ccRCC, correctly distinguishing tumour
# cells from tumour-infiltrating lymphocytes (TILs) or tumour-associated
# macrophages (TAMs) is essential: TAMs can adopt either an anti-tumour (M1)
# or pro-tumour (M2) phenotype, and the balance of these populations correlates
# with patient prognosis.  Without accurate annotation we cannot ask which
# populations are enriched in resistant vs. sensitive tumours.
# -------------------------------------------------------------------------- #

# Compute basic QC metrics if not already present
if "n_genes_by_counts" not in braun_dataset.obs.columns:
    sc.pp.calculate_qc_metrics(
        braun_dataset, percent_top=None, log1p=False, inplace=True
    )

# ---- Summary table --------------------------------------------------------
print("\n--- Per-batch summary (mean ± std) ---")
batch_summary = (
    braun_dataset.obs.groupby("batch")[["n_genes_by_counts", "total_counts"]]
    .agg(["mean", "median", "std"])
    .round(1)
)
print(batch_summary.to_string())

print("\n--- Per-celltype summary (mean ± std) ---")
ct_summary = (
    braun_dataset.obs.groupby("celltype")[["n_genes_by_counts", "total_counts"]]
    .agg(["mean", "median"])
    .round(1)
)
print(ct_summary.to_string())

# ---- Diagnostic plots ----------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ["n_genes_by_counts", "total_counts"]):
    sc.pl.violin(braun_dataset, metric, jitter=False, show=False, ax=ax)
    ax.set_title(f"Dataset-wide — {metric}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_dataset_wide.png", dpi=100)
plt.close()

# Per-batch QC violin
sc.pl.violin(
    braun_dataset,
    "n_genes_by_counts",
    groupby="batch",
    rotation=45,
    show=False,
)
plt.gcf().set_size_inches(16, 6)
plt.title("Genes expressed per batch")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_batch_genes.png", dpi=100)
plt.close()

sc.pl.violin(
    braun_dataset,
    "total_counts",
    groupby="batch",
    rotation=45,
    show=False,
)
plt.gcf().set_size_inches(16, 6)
plt.title("Total counts per batch")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_batch_counts.png", dpi=100)
plt.close()

# Per-celltype QC violin
sc.pl.violin(
    braun_dataset,
    "n_genes_by_counts",
    groupby="celltype",
    rotation=90,
    show=False,
)
plt.gcf().set_size_inches(22, 12)
plt.title("Genes expressed per cell type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_celltype_genes.png", dpi=100)
plt.close()

sc.pl.violin(
    braun_dataset,
    "total_counts",
    groupby="celltype",
    rotation=90,
    show=False,
)
plt.gcf().set_size_inches(22, 12)
plt.title("Total counts per cell type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_celltype_counts.png", dpi=100)
plt.close()

print("Saved QC plots.")

# =============================================================================
# 1.2  Semi-supervised Label Masking (30 %)
# =============================================================================

print("\n" + "=" * 70)
print("1.2  Semi-supervised Labels — Masking 30 % of Labels")
print("=" * 70)

# Store ground-truth labels
braun_dataset.obs["celltype_ground_truth"] = braun_dataset.obs["celltype"].copy()

# Random mask: 30 % of cells become "Unknown"
rng = np.random.default_rng(SEED)
n_cells = braun_dataset.n_obs
mask_idx = rng.choice(n_cells, size=int(0.30 * n_cells), replace=False)
masked_labels = braun_dataset.obs["celltype"].values.astype(object).copy()
masked_labels[mask_idx] = "Unknown"
braun_dataset.obs["celltype_masked"] = masked_labels

n_labeled = (braun_dataset.obs["celltype_masked"] != "Unknown").sum()
n_unlabeled = (braun_dataset.obs["celltype_masked"] == "Unknown").sum()
print(f"Labeled cells  : {n_labeled}")
print(f"Unlabeled cells: {n_unlabeled}")

# Cross-tab: cell type × batch (labeled cells only)
labeled_obs = braun_dataset.obs[braun_dataset.obs["celltype_masked"] != "Unknown"]
ct_batch_table = pd.crosstab(
    labeled_obs["celltype_ground_truth"], labeled_obs["batch"]
)
print("\n--- Cell type × batch counts (labeled cells) ---")
print(ct_batch_table.to_string())

# Proportions
ct_batch_prop = ct_batch_table.div(ct_batch_table.sum(axis=0), axis=1).round(3)
print("\n--- Cell type × batch proportions (per batch) ---")
print(ct_batch_prop.to_string())

absent = (ct_batch_table == 0).any(axis=1)
if absent.any():
    print("\nCell types absent from ≥1 batch:", ct_batch_table.index[absent].tolist())
else:
    print("\nAll cell types are present in every batch.")

# =============================================================================
# 1.3  PCA + kNN Baseline and UMAPs
# =============================================================================

print("\n" + "=" * 70)
print("1.3  PCA + kNN Baseline and UMAPs")
print("=" * 70)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

# Preprocessing for classical baseline
adata_cl = braun_dataset.copy()

# Normalize + log-transform
sc.pp.normalize_total(adata_cl, target_sum=1e4)
sc.pp.log1p(adata_cl)

# Highly variable genes
sc.pp.highly_variable_genes(adata_cl, n_top_genes=2000, batch_key="batch")
adata_cl = adata_cl[:, adata_cl.var["highly_variable"]].copy()

# Scale
sc.pp.scale(adata_cl, max_value=10)

# PCA
sc.tl.pca(adata_cl, svd_solver="arpack", random_state=SEED)

# Split labeled / unlabeled
labeled_mask = adata_cl.obs["celltype_masked"] != "Unknown"
adata_labeled = adata_cl[labeled_mask].copy()
adata_unlabeled = adata_cl[~labeled_mask].copy()

X_train = adata_labeled.obsm["X_pca"]
y_train = adata_labeled.obs["celltype_ground_truth"].values

X_test = adata_unlabeled.obsm["X_pca"]
y_test = adata_unlabeled.obs["celltype_ground_truth"].values

# kNN classifier
knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average="weighted", zero_division=0)
print(f"PCA+kNN  Accuracy: {acc_knn:.4f}  |  Weighted F1: {f1_knn:.4f}")

# Confusion matrix (scanpy + seaborn)
_obs_knn = pd.DataFrame({"true": y_test, "pred": y_pred_knn})
cmtx_knn = sc.metrics.confusion_matrix("true", "pred", _obs_knn)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cmtx_knn, annot=False, cmap="Blues", linewidths=0.3, ax=ax)
ax.set_title("PCA+kNN Confusion Matrix (unlabeled cells)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_knn_confusion_matrix.png", dpi=100)
plt.close()

# ROC curves (macro OvR)
classes_list = list(knn.classes_)
y_test_bin = label_binarize(y_test, classes=classes_list)
fig, ax = plt.subplots(figsize=(16, 12))
for i, cls in enumerate(classes_list):
    if y_test_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_knn[:, i])
    auc_val = roc_auc_score(y_test_bin[:, i], y_prob_knn[:, i])
    ax.plot(fpr, tpr, lw=1, label=f"{cls} (AUC={auc_val:.2f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("PCA+kNN ROC Curves (unlabeled cells)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_knn_roc.png", dpi=100)
plt.close()

# UMAP
sc.pp.neighbors(adata_cl, n_pcs=30, random_state=SEED)
sc.tl.umap(adata_cl, random_state=SEED)

# Store predictions back
adata_cl.obs["knn_predictions"] = "Unknown"
adata_cl.obs.loc[~labeled_mask, "knn_predictions"] = y_pred_knn

# UMAPs
for color_key, fname, title in [
    ("celltype_ground_truth", "1_3_umap_groundtruth.png", "Ground-truth labels"),
    ("knn_predictions", "1_3_umap_knn_pred.png", "PCA+kNN predictions"),
    ("batch", "1_3_umap_batch.png", "Batch"),
    ("n_genes_by_counts", "1_3_umap_ngenes.png", "n_genes_by_counts"),
    ("total_counts", "1_3_umap_counts.png", "total_counts"),
]:
    sc.pl.umap(adata_cl, color=color_key, title=title, show=False)
    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close()

print("Saved PCA+kNN plots.")

# =============================================================================
# 1.4  Deep MLP Classifier Baseline
# =============================================================================

print("\n" + "=" * 70)
print("1.4  Deep MLP Classifier Baseline")
print("=" * 70)

from torch.utils.data import DataLoader, TensorDataset

N_HVG = 1000
_adata_hvg = adata_cl.copy()
sc.pp.highly_variable_genes(_adata_hvg, n_top_genes=N_HVG, batch_key="batch")
adata_mlp = adata_cl[:, _adata_hvg.var["highly_variable"]].copy()
del _adata_hvg

X_mlp_labeled = adata_mlp[labeled_mask].X
X_mlp_unlabeled = adata_mlp[~labeled_mask].X

# Convert sparse → dense if needed
if hasattr(X_mlp_labeled, "toarray"):
    X_mlp_labeled = X_mlp_labeled.toarray()
    X_mlp_unlabeled = X_mlp_unlabeled.toarray()

le = LabelEncoder()
y_mlp_labeled = le.fit_transform(
    adata_mlp[labeled_mask].obs["celltype_ground_truth"].values
)
y_mlp_unlabeled = le.transform(
    adata_mlp[~labeled_mask].obs["celltype_ground_truth"].values
)
n_classes = len(le.classes_)

# Train / val split from labeled set
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_mlp_labeled, y_mlp_labeled, test_size=0.20, random_state=SEED, stratify=y_mlp_labeled
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def to_tensor(X, y):
    return (
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device),
    )


X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t = torch.tensor(X_mlp_unlabeled, dtype=torch.float32).to(device)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True
)


class MLP(nn.Module):
    """3-hidden-layer MLP: input→512→256→128→n_classes."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


mlp = MLP(N_HVG, n_classes).to(device)
criterion = nn.CrossEntropyLoss()

_mlp_ckpt = CKPT_DIR / "mlp.pt"

if _mlp_ckpt.exists():
    print(f"  Loading MLP from checkpoint: {_mlp_ckpt}")
    mlp.load_state_dict(torch.load(_mlp_ckpt, map_location=device))
    mlp.eval()
    train_losses, val_losses = [], []
else:
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    N_EPOCHS = 50
    train_losses, val_losses = [], []

    for epoch in range(N_EPOCHS):
        mlp.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(mlp(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        train_losses.append(epoch_loss / len(y_tr))

        mlp.eval()
        with torch.no_grad():
            val_loss = criterion(mlp(X_val_t), y_val_t).item()
        val_losses.append(val_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} — train_loss={train_losses[-1]:.4f}  val_loss={val_loss:.4f}")

    torch.save(mlp.state_dict(), _mlp_ckpt)
    print(f"  Saved MLP checkpoint → {_mlp_ckpt}")

# Loss curves
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses, label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("MLP Training/Validation Loss")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_4_mlp_loss_curves.png", dpi=100)
plt.close()

# Predict on masked cells
mlp.eval()
with torch.no_grad():
    logits_test = mlp(X_test_t).cpu().numpy()
y_pred_mlp = le.inverse_transform(np.argmax(logits_test, axis=1))
y_prob_mlp = torch.softmax(torch.tensor(logits_test), dim=1).numpy()

y_test_str = le.inverse_transform(y_mlp_unlabeled)
acc_mlp = accuracy_score(y_test_str, y_pred_mlp)
f1_mlp = f1_score(y_test_str, y_pred_mlp, average="weighted", zero_division=0)
print(f"MLP  Accuracy: {acc_mlp:.4f}  |  Weighted F1: {f1_mlp:.4f}")

# Confusion matrix (scanpy + seaborn)
_obs_mlp = pd.DataFrame({"true": y_test_str, "pred": y_pred_mlp})
cmtx_mlp = sc.metrics.confusion_matrix("true", "pred", _obs_mlp)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cmtx_mlp, annot=False, cmap="Blues", linewidths=0.3, ax=ax)
ax.set_title("MLP Confusion Matrix (unlabeled cells)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_4_mlp_confusion_matrix.png", dpi=100)
plt.close()

# ROC curves
y_test_bin_mlp = label_binarize(y_test_str, classes=list(le.classes_))
fig, ax = plt.subplots(figsize=(16, 12))
for i, cls in enumerate(le.classes_):
    if y_test_bin_mlp[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_test_bin_mlp[:, i], y_prob_mlp[:, i])
    auc_val = roc_auc_score(y_test_bin_mlp[:, i], y_prob_mlp[:, i])
    ax.plot(fpr, tpr, lw=1, label=f"{cls} (AUC={auc_val:.2f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("MLP ROC Curves (unlabeled cells)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_4_mlp_roc.png", dpi=100)
plt.close()

print("\n--- Architecture summary ---")
print(mlp)
print(f"\nPCA+kNN  Acc={acc_knn:.4f}  F1={f1_knn:.4f}")
print(f"MLP      Acc={acc_mlp:.4f}  F1={f1_mlp:.4f}")

# =============================================================================
# PART 2 — Deep Autoencoders and scVI/scANVI
# =============================================================================

# ---------------------------------------------------------------------------
# 2.1  Deep Autoencoder Baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2.1  Deep Autoencoder Baseline")
print("=" * 70)

LATENT_DIM = 32
AE_HIDDEN = 256
AE_INPUT = N_HVG


class Autoencoder(nn.Module):
    """Deterministic autoencoder: input→256→latent→256→input."""

    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# Use ALL cells (unsupervised)
X_ae_all = adata_mlp.X
if hasattr(X_ae_all, "toarray"):
    X_ae_all = X_ae_all.toarray()
X_ae_t = torch.tensor(X_ae_all, dtype=torch.float32).to(device)

ae_loader = DataLoader(
    TensorDataset(X_ae_t),
    batch_size=512,
    shuffle=True,
)

ae_model = Autoencoder(AE_INPUT, AE_HIDDEN, LATENT_DIM).to(device)

_ae_ckpt     = CKPT_DIR / "autoencoder.pt"
_ae_latent   = CKPT_DIR / "ae_latent.npy"

if _ae_ckpt.exists() and _ae_latent.exists():
    print(f"  Loading Autoencoder from checkpoint: {_ae_ckpt}")
    ae_model.load_state_dict(torch.load(_ae_ckpt, map_location=device))
    ae_model.eval()
    ae_losses = []
    Z_all = np.load(_ae_latent)
else:
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    ae_criterion = nn.MSELoss()

    AE_EPOCHS = 50
    ae_losses = []
    for epoch in range(AE_EPOCHS):
        ae_model.train()
        ep_loss = 0.0
        for (Xb,) in ae_loader:
            ae_optimizer.zero_grad()
            recon, _ = ae_model(Xb)
            loss = ae_criterion(recon, Xb)
            loss.backward()
            ae_optimizer.step()
            ep_loss += loss.item() * len(Xb)
        ae_losses.append(ep_loss / len(X_ae_all))
        if (epoch + 1) % 10 == 0:
            print(f"  AE Epoch {epoch+1:3d}/{AE_EPOCHS} — loss={ae_losses[-1]:.6f}")

    # Extract latent embeddings
    ae_model.eval()
    with torch.no_grad():
        _, Z_all = ae_model(X_ae_t)
    Z_all = Z_all.cpu().numpy()

    torch.save(ae_model.state_dict(), _ae_ckpt)
    np.save(_ae_latent, Z_all)
    print(f"  Saved AE checkpoint → {_ae_ckpt}")
    print(f"  Saved AE latent     → {_ae_latent}")

# Loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ae_losses, label="Reconstruction Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Autoencoder Reconstruction Loss")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_1_ae_loss.png", dpi=100)
plt.close()

# Store in AnnData for UMAP
adata_ae = adata_mlp.copy()
adata_ae.obsm["X_ae"] = Z_all

sc.pp.neighbors(adata_ae, use_rep="X_ae", random_state=SEED)
sc.tl.umap(adata_ae, random_state=SEED)

for color_key, fname, title in [
    ("celltype_ground_truth", "2_1_umap_ae_celltype.png", "AE latent — cell type"),
    ("batch", "2_1_umap_ae_batch.png", "AE latent — batch"),
]:
    sc.pl.umap(adata_ae, color=color_key, title=title, show=False)
    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close()

# kNN on AE latent for the masked cells
Z_labeled = Z_all[labeled_mask]
Z_unlabeled = Z_all[~labeled_mask]

knn_ae = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
knn_ae.fit(
    Z_labeled,
    adata_mlp[labeled_mask].obs["celltype_ground_truth"].values,
)
y_pred_ae = knn_ae.predict(Z_unlabeled)
y_test_ae = adata_mlp[~labeled_mask].obs["celltype_ground_truth"].values

acc_ae = accuracy_score(y_test_ae, y_pred_ae)
f1_ae = f1_score(y_test_ae, y_pred_ae, average="weighted", zero_division=0)
print(f"AE+kNN  Accuracy: {acc_ae:.4f}  |  Weighted F1: {f1_ae:.4f}")
print("Saved AE plots.")

# ---------------------------------------------------------------------------
# 2.2  scVI Pre-training
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2.2  scVI Pre-training")
print("=" * 70)

# scVI works on raw (integer) counts — use the original braun_dataset
adata_scvi = braun_dataset.copy()

# Store the masked labels as the scANVI label key (Unknown = unlabeled)
adata_scvi.obs["celltype_scvi"] = adata_scvi.obs["celltype_masked"].copy()

scvi.model.SCVI.setup_anndata(
    adata_scvi,
    layer=None,          # use adata.X (raw counts)
    batch_key="batch",
    labels_key="celltype_scvi",
)

_scvi_ckpt = CKPT_DIR / "scvi_model"

if _scvi_ckpt.exists():
    print(f"  Loading scVI model from checkpoint: {_scvi_ckpt}")
    vae = scvi.model.SCVI.load(str(_scvi_ckpt), adata=adata_scvi)
else:
    vae = scvi.model.SCVI(
        adata_scvi,
        n_latent=20,
        n_hidden=128,
        n_layers=2,
        gene_likelihood="nb",
    )
    vae.train(
        max_epochs=100,
        train_size=0.9,
        validation_size=0.1,
        early_stopping=True,
        plan_kwargs={"lr": 1e-3},
    )
    vae.save(str(_scvi_ckpt), overwrite=True)
    print(f"  Saved scVI checkpoint → {_scvi_ckpt}")

# Extract scVI latent
Z_scvi = vae.get_latent_representation()
adata_scvi.obsm["X_scVI"] = Z_scvi

sc.pp.neighbors(adata_scvi, use_rep="X_scVI", random_state=SEED)
sc.tl.umap(adata_scvi, random_state=SEED)

for color_key, fname, title in [
    ("batch", "2_2_umap_scvi_batch.png", "scVI latent — batch"),
    ("celltype_ground_truth", "2_2_umap_scvi_celltype.png", "scVI latent — cell type"),
]:
    sc.pl.umap(adata_scvi, color=color_key, title=title, show=False)
    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close()

print("Saved scVI UMAP plots.")

# ---------------------------------------------------------------------------
# 2.3  scANVI Training and Annotation
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2.3  scANVI Training and Annotation")
print("=" * 70)

# Hyperparameters:
#  - n_latent=20, n_hidden=128, n_layers=2 (inherited from scVI)
#  - max_epochs=20 for fine-tuning (scANVI tutorial default)
#  - unlabeled_category="Unknown" matches the mask applied in 1.2

_scanvi_ckpt = CKPT_DIR / "scanvi_model"

if _scanvi_ckpt.exists():
    print(f"  Loading scANVI model from checkpoint: {_scanvi_ckpt}")
    scanvi_model = scvi.model.SCANVI.load(str(_scanvi_ckpt), adata=adata_scvi)
else:
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category="Unknown",
        labels_key="celltype_scvi",
    )
    scanvi_model.train(
        max_epochs=20,
        n_samples_per_label=100,
        train_size=0.9,
        validation_size=0.1,
    )
    scanvi_model.save(str(_scanvi_ckpt), overwrite=True)
    print(f"  Saved scANVI checkpoint → {_scanvi_ckpt}")

# Predictions and probabilities
scanvi_preds = scanvi_model.predict(soft=False)
scanvi_probs = scanvi_model.predict(soft=True)  # DataFrame, rows=cells, cols=classes

adata_scvi.obs["scanvi_predictions"] = scanvi_preds
adata_scvi.obs["scanvi_predictions"] = adata_scvi.obs["scanvi_predictions"].astype(str)

# Recompute neighbors/UMAP on scANVI latent
Z_scanvi = scanvi_model.get_latent_representation()
adata_scvi.obsm["X_scANVI"] = Z_scanvi

sc.pp.neighbors(adata_scvi, use_rep="X_scANVI", random_state=SEED)
sc.tl.umap(adata_scvi, random_state=SEED)

for color_key, fname, title in [
    ("scanvi_predictions", "2_3_umap_scanvi_pred.png", "scANVI predictions"),
    ("celltype_ground_truth", "2_3_umap_scanvi_truth.png", "scANVI — ground truth"),
    ("batch", "2_3_umap_scanvi_batch.png", "scANVI — batch"),
]:
    sc.pl.umap(adata_scvi, color=color_key, title=title, show=False)
    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close()

# Prediction-confidence violin — choose a cell type with enough cells
# Use max probability as confidence score for each cell
max_prob = scanvi_probs.max(axis=1).values
adata_scvi.obs["scanvi_confidence"] = max_prob

sc.pl.violin(
    adata_scvi,
    "scanvi_confidence",
    groupby="scanvi_predictions",
    rotation=90,
    show=False,
)
plt.gcf().set_size_inches(16, 12)
plt.title("scANVI prediction confidence per predicted cell type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_3_scanvi_confidence_violin.png", dpi=100)
plt.close()

# Evaluate on the masked (unlabeled) cells
unlabeled_idx = adata_scvi.obs["celltype_masked"] == "Unknown"
y_true_scanvi = adata_scvi.obs.loc[unlabeled_idx, "celltype_ground_truth"].values
y_pred_scanvi = adata_scvi.obs.loc[unlabeled_idx, "scanvi_predictions"].values

acc_scanvi = accuracy_score(y_true_scanvi, y_pred_scanvi)
f1_scanvi = f1_score(y_true_scanvi, y_pred_scanvi, average="weighted", zero_division=0)
print(f"scANVI  Accuracy: {acc_scanvi:.4f}  |  Weighted F1: {f1_scanvi:.4f}")

# scANVI ROC curves (required by 2.3 spec)
classes_scanvi = scanvi_probs.columns.tolist()
y_true_scanvi_bin = label_binarize(y_true_scanvi, classes=classes_scanvi)
prob_matrix = scanvi_probs.loc[unlabeled_idx].values
fig, ax = plt.subplots(figsize=(16, 12))
for i, cls in enumerate(classes_scanvi):
    if i >= y_true_scanvi_bin.shape[1]:
        continue
    if y_true_scanvi_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_true_scanvi_bin[:, i], prob_matrix[:, i])
    auc_val = roc_auc_score(y_true_scanvi_bin[:, i], prob_matrix[:, i])
    ax.plot(fpr, tpr, lw=1, label=f"{cls} (AUC={auc_val:.2f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("scANVI ROC Curves (unlabeled cells)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_3_scanvi_roc.png", dpi=100)
plt.close()

print("Saved scANVI plots.")

# =============================================================================
# 2.4  Orthogonal Projection on test.h5ad
# =============================================================================

print("\n" + "=" * 70)
print("2.4  Orthogonal Projection — test.h5ad")
print("=" * 70)

if TEST_PATH.exists():
    adata_test = sc.read_h5ad(TEST_PATH)
    print(f"Test dataset: {adata_test}")

    # --- PCA+kNN on test set ---
    adata_test_cl = adata_test.copy()
    sc.pp.normalize_total(adata_test_cl, target_sum=1e4)
    sc.pp.log1p(adata_test_cl)

    # Common genes between test and training HVG subset
    common_genes_cl = adata_test_cl.var_names.intersection(adata_cl.var_names)
    adata_test_cl = adata_test_cl[:, common_genes_cl].copy()

    # Rebuild training data on common genes with fresh scale+PCA so that
    # the embedding space is consistent between train and test projections.
    # (adata_cl.X was scaled on 2000 genes; we need per-gene stats specifically
    # for common_genes_cl to correctly centre/scale the test data.)
    _tr_log = braun_dataset.copy()
    sc.pp.normalize_total(_tr_log, target_sum=1e4)
    sc.pp.log1p(_tr_log)
    _tr_log = _tr_log[:, common_genes_cl].copy()
    sc.pp.scale(_tr_log, max_value=10)          # stores mean/std in .var
    sc.tl.pca(_tr_log, svd_solver="arpack", random_state=SEED)

    # Project test data using TRAINING mean, std, and PCA components
    pca_components = _tr_log.varm["PCs"]        # (n_common_genes, n_pcs)
    train_mean = _tr_log.var["mean"].values
    train_std  = _tr_log.var["std"].values

    X_te = adata_test_cl.X
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()
    X_te_scaled = (X_te - train_mean) / np.where(train_std == 0, 1.0, train_std)
    np.clip(X_te_scaled, -10, 10, out=X_te_scaled)
    X_test_proj = X_te_scaled @ pca_components

    # Fit kNN on labeled training cells in the consistent common-gene PCA space
    knn_test = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    knn_test.fit(
        _tr_log[labeled_mask].obsm["X_pca"],
        _tr_log[labeled_mask].obs["celltype_ground_truth"].values,
    )
    y_pred_knn_test = knn_test.predict(X_test_proj)
    del _tr_log

    # --- MLP on test set ---
    # Apply the same normalization and log-transform as training before selecting HVGs
    adata_test_mlp = adata_test.copy()
    sc.pp.normalize_total(adata_test_mlp, target_sum=1e4)
    sc.pp.log1p(adata_test_mlp)
    sc.pp.scale(adata_test_mlp, max_value=10)

    # Select exactly the same HVGs (by name) as the training MLP data, in the same order
    train_hvg_genes = adata_mlp.var_names.tolist()
    test_gene_set = set(adata_test_mlp.var_names.tolist())
    genes_in_test = [g for g in train_hvg_genes if g in test_gene_set]
    genes_missing = [g for g in train_hvg_genes if g not in test_gene_set]

    X_test_mlp_sub = adata_test_mlp[:, genes_in_test].X
    if hasattr(X_test_mlp_sub, "toarray"):
        X_test_mlp_sub = X_test_mlp_sub.toarray()

    # Reconstruct full feature matrix (N_HVG columns) with zeros for missing genes
    X_test_mlp = np.zeros((adata_test_mlp.n_obs, N_HVG), dtype=np.float32)
    present_positions = [train_hvg_genes.index(g) for g in genes_in_test]
    X_test_mlp[:, present_positions] = X_test_mlp_sub

    mlp.eval()
    with torch.no_grad():
        logits_proj = mlp(
            torch.tensor(X_test_mlp, dtype=torch.float32).to(device)
        ).cpu().numpy()
    y_pred_mlp_test = le.inverse_transform(np.argmax(logits_proj, axis=1))

    # --- AE+kNN on test set ---
    X_ae_test = X_test_mlp  # same feature set
    ae_model.eval()
    with torch.no_grad():
        _, Z_test_ae = ae_model(
            torch.tensor(X_ae_test, dtype=torch.float32).to(device)
        )
    Z_test_ae = Z_test_ae.cpu().numpy()
    y_pred_ae_test = knn_ae.predict(Z_test_ae)

    # --- scANVI on test set ---
    adata_test_scanvi = adata_test.copy()
    if "batch" not in adata_test_scanvi.obs.columns:
        adata_test_scanvi.obs["batch"] = "test"
    adata_test_scanvi.obs["celltype_scvi"] = "Unknown"
    # prepare_query_anndata aligns genes; load_query_data creates a properly
    # registered query model instance that handles unseen batch categories
    # without triggering the "extend_categories" validation error.
    scvi.model.SCANVI.prepare_query_anndata(adata_test_scanvi, scanvi_model)
    _scanvi_query = scvi.model.SCANVI.load_query_data(
        adata_test_scanvi, scanvi_model, freeze_dropout=True
    )
    scanvi_preds_test = _scanvi_query.predict()
    y_pred_scanvi_test = np.array(scanvi_preds_test)

    # Compile results if ground truth is available
    if "celltype" in adata_test.obs.columns:
        y_true_test = adata_test.obs["celltype"].values

        def _metrics(y_true, y_pred, name):
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            print(f"  {name:<20s}  Acc={acc:.4f}  F1={f1:.4f}")

        print("\nTest-set metrics:")
        _metrics(y_true_test, y_pred_knn_test, "PCA+kNN")
        _metrics(y_true_test, y_pred_mlp_test, "MLP")
        _metrics(y_true_test, y_pred_ae_test, "AE+kNN")
        _metrics(y_true_test, y_pred_scanvi_test, "scANVI")
    else:
        print("No ground-truth labels in test.h5ad; skipping metric computation.")
        print(f"  PCA+kNN  predictions: {pd.Series(y_pred_knn_test).value_counts().to_dict()}")
        print(f"  MLP      predictions: {pd.Series(y_pred_mlp_test).value_counts().to_dict()}")
        print(f"  AE+kNN   predictions: {pd.Series(y_pred_ae_test).value_counts().to_dict()}")
        print(f"  scANVI   predictions: {pd.Series(y_pred_scanvi_test).value_counts().to_dict()}")

    # Summary table
    results_df = pd.DataFrame(
        {
            "Model": ["PCA+kNN", "MLP", "AE+kNN", "scANVI"],
            "Train Acc": [acc_knn, acc_mlp, acc_ae, acc_scanvi],
            "Train F1": [f1_knn, f1_mlp, f1_ae, f1_scanvi],
        }
    )
    print("\n--- In-distribution performance summary ---")
    print(results_df.to_string(index=False))
else:
    print(f"test.h5ad not found at {TEST_PATH}; skipping orthogonal projection.")
    results_df = pd.DataFrame(
        {
            "Model": ["PCA+kNN", "MLP", "AE+kNN", "scANVI"],
            "Masked-set Acc": [acc_knn, acc_mlp, acc_ae, acc_scanvi],
            "Masked-set F1": [f1_knn, f1_mlp, f1_ae, f1_scanvi],
        }
    )
    print(results_df.to_string(index=False))

# =============================================================================
# 2.5  Interpretability — Captum Integrated Gradients
# =============================================================================

print("\n" + "=" * 70)
print("2.5  Interpretability — Captum Integrated Gradients")
print("=" * 70)

try:
    from captum.attr import IntegratedGradients

    # We use the MLP trained in Part 1.4 for attribution.
    # Gene names come from the HVG subset stored in adata_mlp.
    gene_names = adata_mlp.var_names.tolist()

    # Pick 3 cell types present in the unlabeled test set
    unique_ct = list(np.unique(y_test_str))
    ct_for_interp = unique_ct[:3]

    ig = IntegratedGradients(mlp)
    mlp.eval()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, ct in zip(axes, ct_for_interp):
        target_idx = int(np.where(le.classes_ == ct)[0][0])

        # Collect cells of this type from unlabeled set
        ct_mask = y_test_str == ct
        X_ct = torch.tensor(X_mlp_unlabeled[ct_mask], dtype=torch.float32).to(device)
        if len(X_ct) == 0:
            ax.set_visible(False)
            continue
        # Use mean as baseline
        baseline = torch.zeros_like(X_ct[:1])
        attrs, _ = ig.attribute(
            X_ct[:min(50, len(X_ct))],  # limit for speed; safe for rare cell types
            baselines=baseline,
            target=target_idx,
            return_convergence_delta=True,
        )
        mean_attr = attrs.detach().cpu().numpy().mean(axis=0)

        top10_idx = np.argsort(np.abs(mean_attr))[-10:][::-1]
        top10_genes = [gene_names[i] if i < len(gene_names) else f"gene_{i}" for i in top10_idx]
        top10_vals = mean_attr[top10_idx]

        ax.barh(top10_genes[::-1], top10_vals[::-1], color="steelblue")
        ax.set_xlabel("Attribution score")
        ax.set_title(f"Top-10 genes — {ct}")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_5_ig_top10_genes.png", dpi=100)
    plt.close()
    print("Saved Integrated Gradients plot.")

except ImportError:
    print("captum not installed.  Skipping interpretability section.")
    print("Install with:  pip install captum")

# =============================================================================
# Final summary
# =============================================================================

print("\n" + "=" * 70)
print("Pipeline complete.  Generated output files:")
print("=" * 70)
import glob as _glob

for f in sorted(_glob.glob(str(FIGURES_DIR / "*.png"))):
    print(f"  {f}")
