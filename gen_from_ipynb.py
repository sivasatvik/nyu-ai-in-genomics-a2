# %% [markdown]
# # Assignment 2: Single-Cell RNA-seq — Braun ccRCC Atlas
# 
# 

# %% [markdown]
# #### =============================================================================
# #### This file implements the complete pipeline described in the assignment:
# ####   Part 1: Data Understanding and Classical Baselines
# ####     1.1  Data understanding and preprocessing
# ####     1.2  Semi-supervised label masking (30 %)
# ####     1.3  PCA + kNN baseline + UMAPs
# ####     1.4  Deep MLP classifier baseline
# ####   Part 2: Unsupervised Autoencoders and scVI/scANVI Pipeline
# ####     2.1  Deep Autoencoder baseline
# ####     2.2  scVI pre-training
# ####     2.3  scANVI training and annotation
# ####     2.4  Orthogonal projection on test.h5ad
# ####     2.5  Interpretability via Captum Integrated Gradients
# #### =============================================================================

# %%
# Imports and basic setup
import pickle
import random
import sys
import warnings

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

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# %%
# Reproducibility and plotting settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, frameon=False)

# %%
# Paths and directories
DATA_PATH = Path("braun_dataset.h5ad")
TEST_PATH  = Path("test.h5ad")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# %%
# Downloading data from Google Drive if not already present
import gdown

_GDRIVE_URLS = {
    DATA_PATH: "https://drive.google.com/uc?id=1ZZWbVq-qwGUr76WSecPUrKiyLJP4VVtz",
    TEST_PATH:  "https://drive.google.com/uc?id=1uZkdqE5df-0Mul8VFahCyQoDjXJ-0IFP",
}

for _path, _url in _GDRIVE_URLS.items():
    if not _path.exists():
        print(f"Downloading {_path.name} from Google Drive …")
        gdown.download(_url, str(_path), quiet=False, fuzzy=False)
        print(f"  Saved to {_path}")
    else:
        print(f"{_path.name} already present — skipping download.")

# %% [markdown]
# ## Part 1 — Data Understanding and Classical Baselines
# 
# ### 1.1 Load Data & QC Metrics

# %%
braun_dataset = sc.read_h5ad(DATA_PATH)
print(braun_dataset)
print("\n.obs columns:", braun_dataset.obs.columns.tolist())
print("\nFirst few rows of .obs:")
print(braun_dataset.obs.head().to_string())

# %% [markdown]
# First few rows of `.obs`:
# 
# | Cell_barcode | Sample | Stage | Tumor_Normal | Batch | nFeature_RNA | nCount_RNA | percent_mito | doublet_score | ClusterNumber_AllCells | ClusterName_AllCells | ClusterNumber_ImmuneCells | ClusterName_ImmuneCells | ClusterNumber_TCells | ClusterName_TCells | ClusterNumber_MyeloidCells | ClusterName_MyeloidCells | Included_in_CD8_trajectory | Included_in_myeloid_trajectory | n_genes | n_counts | batch | celltype |
# |---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|---|---|---:|---:|---|---|
# | AAACCTGAGAACTCGG-1 | S10_N | N | N | 2 | 1553 | 4195 | 3.87e-21 | 0.086201 | 9 | CD8+ T cell.1 | 5 | CD8 T cell.3 | 10 | CD8 TRM.2 | -1 | -1 | True | False | 1553 | 4195.0 | 2 | CD8+ T cell.1 |
# | AAACCTGAGGGAGTAA-1 | S10_N | N | N | 2 | 1175 | 2885 | 1.66e-31 | 0.220665 | 3 | CD4+ T cell.1 | 2 | CD4 T cell.1 | 3 | CD4 TCM.1 | -1 | -1 | False | False | 1175 | 2885.0 | 2 | CD4+ T cell.1 |
# | AAACCTGAGTGTACGG-1 | S10_N | N | N | 2 | 1063 | 3269 | 2.36e-27 | 0.001703 | 13 | CD4+ T cell.2 | 18 | CD4 T cell.3 | 11 | CD4 TCM.2 | -1 | -1 | False | False | 1063 | 3269.0 | 2 | CD4+ T cell.2 |
# | AAACCTGCACGGTGTC-1 | S10_N | N | N | 2 | 3178 | 11816 | 7.13e-92 | 0.759910 | 2 | Immune doublet.1 | -1 | -1 | -1 | -1 | -1 | -1 | False | False | 3178 | 11816.0 | 2 | Immune doublet.1 |
# | AAACCTGGTAATCGTC-1 | S10_N | N | N | 2 | 1412 | 4170 | 2.09e-27 | 0.010109 | 13 | CD4+ T cell.2 | 18 | CD4 T cell.3 | 11 | CD4 TCM.2 | -1 | -1 | False | False | 1412 | 4170.0 | 2 | CD4+ T cell.2 |
# 
# The data has 22 columns with different types of metadata on the cells. We can see a lot of biological and technical information here: the sample and batch each cell came from, the tumor stage and whether it came from a tumor or normal tissue, QC metrics like number of genes and counts, doublet scores, and various cluster annotations from the original paper.  The "celltype" column contains the most specific cell type annotation available for each cell, which we will use as our ground-truth labels for the semi-supervised task.
# 
# Cell type annotation is central to interpreting the tumour micro-environment (TME) because single cells do not carry visible morphological markers after dissociation.  In ccRCC, correctly distinguishing tumour cells from tumour-infiltrating lymphocytes (TILs) or tumour-associated macrophages (TAMs) is essential: TAMs can adopt either an anti-tumour (M1) or pro-tumour (M2) phenotype, and the balance of these populations correlates with patient prognosis.  Without accurate annotation we cannot ask which populations are enriched in resistant vs. sensitive tumours. 
# 

# %%
# Subsample to speed up training and reduce memory usag
MAX_CELLS = 20_000
if braun_dataset.n_obs > MAX_CELLS:
    sc.pp.subsample(braun_dataset, n_obs=MAX_CELLS, random_state=SEED)
    print(f"\nSubsampled to {braun_dataset.n_obs} cells.")

# %%
if "n_genes_by_counts" not in braun_dataset.obs.columns:
    sc.pp.calculate_qc_metrics(braun_dataset, percent_top=None, log1p=False, inplace=True)

print("\n--- Per-batch summary (mean ± std) ---")
batch_summary = (
    braun_dataset.obs.groupby("batch")[["n_genes_by_counts", "total_counts"]]
    .agg(["mean", "median", "std"])
    .round(1)
)
print(batch_summary.to_string())


# %% [markdown]
# | batch | n_genes_by_counts (mean) | n_genes_by_counts (median) | n_genes_by_counts (std) | total_counts (mean) | total_counts (median) | total_counts (std) |
# |---|---:|---:|---:|---:|---:|---:|
# | 1 | 681.9 | 446.0 | 558.8 | 1878.7 | 1053.5 | 2196.9 |
# | 2 | 624.7 | 497.0 | 438.5 | 1683.4 | 1133.0 | 1911.0 |
# | 3 | 1451.5 | 1320.0 | 912.2 | 4336.7 | 3129.0 | 4883.4 |
# | 4 | 1294.1 | 1113.0 | 839.6 | 4042.2 | 2805.0 | 4237.2 |
# 
# The dataset contains 4 batches with varying numbers of cells.  Batches 1 and 2 have lower mean and median gene counts (around 600 genes, around 1700 counts) compared to batches 3 and 4 (around 1300-1500 genes, around 4000-4300 counts).  This suggests that batches 1 and 2 may have lower sequencing depth or more low-quality cells, which could be due to technical differences in sample processing or sequencing.  We will keep all batches for now but will monitor batch effects in downstream analyses.

# %%
print("\n--- Per-celltype summary (mean ± std) ---")
ct_summary = (
    braun_dataset.obs.groupby("celltype")[["n_genes_by_counts", "total_counts"]]
    .agg(["mean", "median"])
    .round(1)
)
print(ct_summary.to_string())

# %% [markdown]
# | celltype | n_genes_by_counts (mean) | n_genes_by_counts (median) | total_counts (mean) | total_counts (median) |
# |---|---:|---:|---:|---:|
# | B cell | 728.3 | 580.0 | 2191.7 | 1806.0 |
# | CD4+ T cell.1 | 740.3 | 582.0 | 2049.3 | 1581.0 |
# | CD4+ T cell.2 | 701.5 | 532.0 | 2124.7 | 1532.5 |
# | CD8+ T cell.1 | 911.3 | 703.0 | 2307.2 | 1661.5 |
# | CD8+ T cell.2 | 423.6 | 428.0 | 893.6 | 861.0 |
# | CD8+ T cell.4 | 956.4 | 694.0 | 2296.2 | 1631.0 |
# | CD8+ T cell.5 | 1140.2 | 1119.0 | 2768.4 | 2501.0 |
# | CD8+ T cell.6 | 1235.8 | 1239.5 | 2741.7 | 2603.5 |
# | CD8+ T cell.7 | 592.9 | 588.0 | 1607.7 | 1456.0 |
# | CD8+ T cell.8 | 468.8 | 388.0 | 1009.2 | 790.0 |
# | CD8+ T cell.9 | 1004.6 | 799.5 | 2683.9 | 1921.5 |
# | CD8+ T cell.10 | 2074.2 | 2029.0 | 6572.3 | 5757.0 |
# | CD8+ T cell.11 | 748.8 | 750.0 | 1774.4 | 1739.0 |
# | CD8+ Tcell.3 | 595.1 | 495.0 | 941.8 | 764.0 |
# | CD141+ DC | 1995.0 | 1467.0 | 9397.2 | 6821.5 |
# | Distal convoluted tubule | 1013.7 | 882.0 | 2297.0 | 1818.0 |
# | Immune doublet.1 | 1793.4 | 1306.0 | 7237.0 | 5221.0 |
# | Immune doublet.2 | 1387.1 | 1138.5 | 4320.6 | 3326.5 |
# | Immune doublet.3 | 1047.4 | 807.5 | 3182.0 | 2294.5 |
# | Mast cell | 643.7 | 583.0 | 1456.9 | 1231.0 |
# | Myeloid cell.1 | 1765.0 | 1301.0 | 7448.1 | 5330.0 |
# | Myeloid cell.2 | 1170.3 | 991.0 | 3638.0 | 2779.0 |
# | Myeloid cell.3 | 1273.7 | 985.0 | 4346.5 | 3159.0 |
# | Myeloid cell.4 | 479.4 | 393.5 | 1290.5 | 1086.5 |
# | NK cell.1 | 1236.4 | 1299.5 | 2607.0 | 2618.0 |
# | NK cell.2 | 842.8 | 617.0 | 1916.2 | 1350.0 |
# | NK cell.3 | 768.4 | 578.0 | 1758.5 | 1271.0 |
# | NKT cell | 1339.6 | 982.0 | 3743.6 | 2745.0 |
# | PTPRC- population | 1232.5 | 855.0 | 3370.6 | 1639.0 |
# | Plasma cell | 1722.5 | 1346.5 | 6993.8 | 6151.5 |
# | Proximal convoluted tubule | 751.8 | 599.0 | 1968.4 | 1319.0 |
# | T cell mixed | 727.9 | 695.0 | 1245.5 | 1174.0 |
# | Treg | 1027.8 | 1062.0 | 2443.2 | 2358.0 |
# | Tumor cell.1 | 3105.3 | 3166.0 | 13742.0 | 11579.0 |
# | Tumor cell.2 | 665.5 | 630.0 | 1374.6 | 1354.0 |
# | Tumor cell.3 | 2104.3 | 1528.0 | 10596.3 | 5802.0 |
# | Tumor cell.4 | 831.9 | 702.0 | 2244.3 | 1470.0 |
# | Tumor-immune doublet | 1409.6 | 1197.0 | 6426.2 | 3781.5 |
# | pDC | 1009.7 | 799.0 | 2858.9 | 2082.0 |
# 
# There are 37 annotated cell types with varying QC metrics.  Tumor cells (especially Tumor cell.1) have the highest mean and median gene counts (around 800-3100 genes, around 1600-11500 counts), which is expected since they are often larger and more transcriptionally active.  Immune doublets and myeloid cells also have relatively high counts, likely due to their larger size or doublet status.  In contrast, some T cell subsets and normal epithelial cells have lower counts (around 400-1300 genes, around 800-2700 counts).  We will keep all cell types for now but will monitor whether certain cell types are more affected by batch effects or label masking in downstream analyses.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ["n_genes_by_counts", "total_counts"]):
    sc.pl.violin(braun_dataset, metric, jitter=False, show=False, ax=ax)
    ax.set_title(f"Dataset-wide — {metric}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_dataset_wide.png", dpi=100)
plt.show()

# %% [markdown]
# ![Dataset-wide figure](figures/1_1_qc_dataset_wide.png)
# The above plots show dataset-wide number of genes expressed and number of total counts in the cells. There is a wide range of both metrics across the dataset, with some cells having very low counts (<500 genes, <1000 counts) and others having very high counts (>3000 genes, >10 000 counts).  The distributions are right-skewed, which is typical for scRNA-seq data.  We will keep all cells for now but may consider filtering out low-quality cells in future analyses if they cause issues with model training or batch effects.

# %%
sc.pl.violin(braun_dataset, "n_genes_by_counts", groupby="batch", rotation=45, show=False)
plt.gcf().set_size_inches(16, 6)
plt.title("Genes expressed per batch")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_batch_genes.png", dpi=100)
plt.show()

# %% [markdown]
# ![Genes expressed per batch figure](figures/1_1_qc_per_batch_genes.png)
# The above plot shows the number of genes expressed per batch.  Batches 1 and 2 have lower median gene counts (around 600-700) compared to batches 3 and 4 (around 1300-1500), which is consistent with the summary table.

# %%
sc.pl.violin(braun_dataset, "total_counts", groupby="batch", rotation=45, show=False)
plt.gcf().set_size_inches(16, 6)
plt.title("Total counts per batch")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_batch_counts.png", dpi=100)
plt.show()

# %% [markdown]
# ![Total counts per batch figure](figures/1_1_qc_per_batch_counts.png)
# The above plot shows the total counts per batch.  Batches 1 and 2 have lower median counts (around 1600-1700) compared to batches 3 and 4 (around 4000-4300), which is consistent with the summary table.

# %%
sc.pl.violin(braun_dataset, "n_genes_by_counts", groupby="celltype", rotation=90, show=False)
plt.gcf().set_size_inches(22, 20)
plt.title("Genes expressed per cell type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_celltype_genes.png", dpi=100)
plt.show()

# %% [markdown]
# ![Genes expressed per cell type figure](figures/1_1_qc_per_celltype_genes.png)
# The above plot shows the number of genes expressed per cell type.  Tumor cells (especially Tumor cell.1) have the highest median gene counts (around 800-3100), while some T cell subsets and normal epithelial cells have lower median counts (around 400-1300).  There is also a wide range of gene counts within each cell type, which is expected due to biological heterogeneity and technical variability.

# %%
sc.pl.violin(braun_dataset, "total_counts", groupby="celltype", rotation=90, show=False)
plt.gcf().set_size_inches(22, 20)
plt.title("Total counts per cell type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_1_qc_per_celltype_counts.png", dpi=100)
plt.show()

# %% [markdown]
# ![Total counts per cell type figure](figures/1_1_qc_per_celltype_counts.png)
# The above plot shows the total counts per cell type.  Tumor cells (especially Tumor cell.1) have the highest median counts (around 1600-11500), while some T cell subsets and normal epithelial cells have lower median counts (around 800-2700).  There is also a wide range of counts within each cell type, which is expected due to biological heterogeneity and technical variability.

# %% [markdown]
# ### 1.2 Semi-supervised Label Masking (30%)

# %%
braun_dataset.obs["celltype_ground_truth"] = braun_dataset.obs["celltype"].copy()

rng = np.random.default_rng(SEED)
n_cells = braun_dataset.n_obs

# Randomly mask 30% of the cell type labels to simulate a semi-supervised learning scenario
mask_idx = rng.choice(n_cells, size=int(0.30 * n_cells), replace=False)
masked_labels = braun_dataset.obs["celltype"].values.astype(object).copy()
masked_labels[mask_idx] = "Unknown"
braun_dataset.obs["celltype_masked"] = masked_labels

# Print summary of labeled vs. unlabeled cells and
# check for any cell types that are completely missing from certain batches
n_labeled   = (braun_dataset.obs["celltype_masked"] != "Unknown").sum()
n_unlabeled = (braun_dataset.obs["celltype_masked"] == "Unknown").sum()
print(f"Labeled cells  : {n_labeled}")
print(f"Unlabeled cells: {n_unlabeled}")

# %% [markdown]
# Labeled cells  : 14000
# Unlabeled cells: 6000

# %%
labeled_obs = braun_dataset.obs[braun_dataset.obs["celltype_masked"] != "Unknown"]
ct_batch_table = pd.crosstab(labeled_obs["celltype_ground_truth"], labeled_obs["batch"])
print("\n--- Cell type × batch counts (labeled cells) ---")
print(ct_batch_table.to_string())

# %% [markdown]
# | celltype_ground_truth | 1 | 2 | 3 | 4 |
# |---|---:|---:|---:|---:|
# | B cell | 23 | 110 | 33 | 115 |
# | CD4+ T cell.1 | 16 | 427 | 175 | 307 |
# | CD4+ T cell.2 | 143 | 232 | 109 | 223 |
# | CD8+ T cell.1 | 124 | 216 | 168 | 399 |
# | CD8+ T cell.2 | 0 | 4 | 0 | 0 |
# | CD8+ T cell.4 | 6 | 788 | 206 | 479 |
# | CD8+ T cell.5 | 52 | 115 | 192 | 356 |
# | CD8+ T cell.6 | 4 | 124 | 184 | 525 |
# | CD8+ T cell.7 | 1 | 47 | 0 | 2 |
# | CD8+ T cell.8 | 2 | 452 | 26 | 55 |
# | CD8+ T cell.9 | 187 | 81 | 302 | 261 |
# | CD8+ T cell.10 | 24 | 48 | 78 | 102 |
# | CD8+ T cell.11 | 4 | 5 | 7 | 210 |
# | CD8+ Tcell.3 | 0 | 128 | 21 | 17 |
# | CD141+ DC | 25 | 22 | 16 | 43 |
# | Distal convoluted tubule | 23 | 24 | 1 | 37 |
# | Immune doublet.1 | 13 | 25 | 74 | 122 |
# | Immune doublet.2 | 15 | 31 | 43 | 70 |
# | Immune doublet.3 | 4 | 23 | 4 | 21 |
# | Mast cell | 16 | 1 | 19 | 8 |
# | Myeloid cell.1 | 56 | 112 | 278 | 643 |
# | Myeloid cell.2 | 33 | 87 | 127 | 317 |
# | Myeloid cell.3 | 49 | 87 | 100 | 191 |
# | Myeloid cell.4 | 8 | 5 | 11 | 26 |
# | NK cell.1 | 20 | 40 | 260 | 60 |
# | NK cell.2 | 98 | 206 | 374 | 189 |
# | NK cell.3 | 155 | 150 | 97 | 181 |
# | NKT cell | 29 | 48 | 37 | 40 |
# | PTPRC- population | 15 | 44 | 14 | 25 |
# | Plasma cell | 2 | 3 | 7 | 7 |
# | Proximal convoluted tubule | 21 | 42 | 5 | 26 |
# | T cell mixed | 2 | 44 | 61 | 28 |
# | Treg | 43 | 62 | 63 | 280 |
# | Tumor cell.1 | 3 | 7 | 4 | 172 |
# | Tumor cell.2 | 0 | 0 | 9 | 2 |
# | Tumor cell.3 | 6 | 8 | 36 | 106 |
# | Tumor cell.4 | 29 | 10 | 19 | 6 |
# | Tumor-immune doublet | 2 | 4 | 1 | 10 |
# | pDC | 2 | 24 | 10 | 27 |

# %%
ct_batch_prop = ct_batch_table.div(ct_batch_table.sum(axis=0), axis=1).round(3)
print("\n--- Cell type × batch proportions (per batch) ---")
print(ct_batch_prop.to_string())

# %% [markdown]
# | celltype_ground_truth | 1 | 2 | 3 | 4 |
# |---|---:|---:|---:|---:|
# | B cell | 0.018 | 0.028 | 0.010 | 0.020 |
# | CD4+ T cell.1 | 0.013 | 0.110 | 0.055 | 0.054 |
# | CD4+ T cell.2 | 0.114 | 0.060 | 0.034 | 0.039 |
# | CD8+ T cell.1 | 0.099 | 0.056 | 0.053 | 0.070 |
# | CD8+ T cell.2 | 0.000 | 0.001 | 0.000 | 0.000 |
# | CD8+ T cell.4 | 0.005 | 0.203 | 0.065 | 0.084 |
# | CD8+ T cell.5 | 0.041 | 0.030 | 0.061 | 0.063 |
# | CD8+ T cell.6 | 0.003 | 0.032 | 0.058 | 0.092 |
# | CD8+ T cell.7 | 0.001 | 0.012 | 0.000 | 0.000 |
# | CD8+ T cell.8 | 0.002 | 0.116 | 0.008 | 0.010 |
# | CD8+ T cell.9 | 0.149 | 0.021 | 0.095 | 0.046 |
# | CD8+ T cell.10 | 0.019 | 0.012 | 0.025 | 0.018 |
# | CD8+ T cell.11 | 0.003 | 0.001 | 0.002 | 0.037 |
# | CD8+ Tcell.3 | 0.000 | 0.033 | 0.007 | 0.003 |
# | CD141+ DC | 0.020 | 0.006 | 0.005 | 0.008 |
# | Distal convoluted tubule | 0.018 | 0.006 | 0.000 | 0.007 |
# | Immune doublet.1 | 0.010 | 0.006 | 0.023 | 0.021 |
# | Immune doublet.2 | 0.012 | 0.008 | 0.014 | 0.012 |
# | Immune doublet.3 | 0.003 | 0.006 | 0.001 | 0.004 |
# | Mast cell | 0.013 | 0.000 | 0.006 | 0.001 |
# | Myeloid cell.1 | 0.045 | 0.029 | 0.088 | 0.113 |
# | Myeloid cell.2 | 0.026 | 0.022 | 0.040 | 0.056 |
# | Myeloid cell.3 | 0.039 | 0.022 | 0.032 | 0.034 |
# | Myeloid cell.4 | 0.006 | 0.001 | 0.003 | 0.005 |
# | NK cell.1 | 0.016 | 0.010 | 0.082 | 0.011 |
# | NK cell.2 | 0.078 | 0.053 | 0.118 | 0.033 |
# | NK cell.3 | 0.124 | 0.039 | 0.031 | 0.032 |
# | NKT cell | 0.023 | 0.012 | 0.012 | 0.007 |
# | PTPRC- population | 0.012 | 0.011 | 0.004 | 0.004 |
# | Plasma cell | 0.002 | 0.001 | 0.002 | 0.001 |
# | Proximal convoluted tubule | 0.017 | 0.011 | 0.002 | 0.005 |
# | T cell mixed | 0.002 | 0.011 | 0.019 | 0.005 |
# | Treg | 0.034 | 0.016 | 0.020 | 0.049 |
# | Tumor cell.1 | 0.002 | 0.002 | 0.001 | 0.030 |
# | Tumor cell.2 | 0.000 | 0.000 | 0.003 | 0.000 |
# | Tumor cell.3 | 0.005 | 0.002 | 0.011 | 0.019 |
# | Tumor cell.4 | 0.023 | 0.003 | 0.006 | 0.001 |
# | Tumor-immune doublet | 0.002 | 0.001 | 0.000 | 0.002 |
# | pDC | 0.002 | 0.006 | 0.003 | 0.005 |

# %%
absent = (ct_batch_table == 0).any(axis=1)
if absent.any():
    print("\nCell types absent from ≥1 batch:", ct_batch_table.index[absent].tolist())
else:
    print("\nAll cell types are present in every batch.")

# %% [markdown]
# 
# Cell types absent from ≥1 batch: ['CD8+ T cell.2', 'CD8+ T cell.7', 'CD8+ Tcell.3', 'Tumor cell.2']
# 
# After masking 30 % of the labels, we have 14,000 labeled cells and 6,000 unlabeled cells.  The cross-tabulation of cell type × batch for the labeled cells shows that most cell types are present in all batches, but some rare cell types (e.g. CD8+ T cell.2, CD8+ T cell.7, CD8+ Tcell.3, Tumor cell.2) are absent from at least one batch.  This could pose challenges for model training if the model relies on batch-specific signals to learn certain cell types, so we will monitor this in downstream analyses.
# 

# %% [markdown]
# ### 1.3 PCA + kNN Baseline and UMAPs

# %%
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize

adata_cl = braun_dataset.copy()
# Preprocessing for PCA + kNN baseline
# Normalization, log-transform, HVG selection, scaling, PCA
sc.pp.normalize_total(adata_cl, target_sum=1e4)
sc.pp.log1p(adata_cl)
sc.pp.highly_variable_genes(adata_cl, n_top_genes=2000, batch_key="batch")
adata_cl = adata_cl[:, adata_cl.var["highly_variable"]].copy()
sc.pp.scale(adata_cl, max_value=10)
sc.tl.pca(adata_cl, svd_solver="arpack", random_state=SEED)

# Split into labeled vs. unlabeled sets based on the masked labels
labeled_mask    = adata_cl.obs["celltype_masked"] != "Unknown"
adata_labeled   = adata_cl[labeled_mask].copy()
adata_unlabeled = adata_cl[~labeled_mask].copy()

# Train a kNN classifier on the PCA embeddings of the labeled cells and evaluate on the unlabeled cells
X_train = adata_labeled.obsm["X_pca"]
y_train = adata_labeled.obs["celltype_ground_truth"].values
X_test  = adata_unlabeled.obsm["X_pca"]
y_test  = adata_unlabeled.obs["celltype_ground_truth"].values

knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
f1_knn  = f1_score(y_test, y_pred_knn, average="weighted", zero_division=0)
print(f"PCA+kNN  Accuracy: {acc_knn:.4f}  |  Weighted F1: {f1_knn:.4f}")

# %% [markdown]
# PCA+kNN  Accuracy: 0.7720  |  Weighted F1: 0.7574
# 
# The PCA+kNN baseline achieves an accuracy of around 77.2 % and a weighted F1 score of around 0.757 on the unlabeled test set.  This is a reasonable baseline performance given the complexity of the dataset and the fact that 30 % of the labels were masked.

# %%
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
plt.show()

# %% [markdown]
# ![PCA+kNN Confusion Matrix figure](figures/1_3_knn_confusion_matrix.png)
# The confusion matrix for the PCA+kNN baseline shows that most cell types are classified reasonably well, with higher counts along the diagonal.  However, some cell types (especially those with fewer labeled examples or those that are more similar to others) show more off-diagonal confusion.  For example, some CD8+ T cell subsets and tumor cells may be confused with each other or with other immune cells.  This suggests that while the PCA+kNN baseline captures some of the structure in the data, there is room for improvement, especially for rarer or more similar cell types.

# %%
# ROC curves
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
plt.show()

# %% [markdown]
# ![PCA+kNN ROC Curves](figures/1_3_knn_roc.png)
# The ROC curves for the PCA+kNN baseline show that many cell types have good discrimination (high AUC), while some rarer or more similar cell types have lower AUC values.  This again suggests that while the PCA+kNN baseline captures some of the structure in the data, there is room for improvement, especially for rarer or more similar cell types.

# %%
# UMAP
sc.pp.neighbors(adata_cl, n_pcs=30, random_state=SEED)
sc.tl.umap(adata_cl, random_state=SEED)

# Add kNN predictions to .obs for visualization
adata_cl.obs["knn_predictions"] = "Unknown"
adata_cl.obs.loc[~labeled_mask, "knn_predictions"] = y_pred_knn

# %%
sc.pl.umap(adata_cl, color="celltype_ground_truth", title="Ground-truth labels", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_umap_groundtruth.png", dpi=100)
plt.show()

# %% [markdown]
# ![PCA+kNN UMAP ground truth](figures/1_3_umap_groundtruth.png)
# The above UMAP colored by ground-truth cell types shows that many cell types form distinct clusters, while some are more intermixed.  This suggests that the dataset has a reasonable structure for classification, but some cell types may be more challenging to separate.

# %%
sc.pl.umap(adata_cl, color="knn_predictions", title="PCA+kNN predictions", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_umap_knn_pred.png", dpi=100)
plt.show()

# %% [markdown]
# ![PCA+kNN UMAP prediction](figures/1_3_umap_knn_pred.png)
# The above UMAP colored by PCA+kNN predictions shows that the model captures much of the overall structure of the data, with some predicted cell types clustering in the same regions as their ground-truth counterparts.  However, there are also many misclassifications and some predicted clusters that are more mixed, which is consistent with the confusion matrix and ROC curves.

# %%
sc.pl.umap(adata_cl, color="batch", title="Batch", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_umap_batch.png", dpi=100)
plt.show()

# %% [markdown]
# ![PCA+kNN UMAP batch](figures/1_3_umap_batch.png)
# The above UMAP colored by batch shows that there is some batch structure in the data, with certain batches clustering together.  This could potentially confound classification if the model relies on batch-specific signals. Cells of batches 3 and 4 seems to cluster together more and also batches 2, 3 and 1 also seem to cluster together (just seeing it as an overview).

# %%
sc.pl.umap(adata_cl, color="n_genes_by_counts", title="n_genes_by_counts", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_umap_ngenes.png", dpi=100)
plt.show()

# %% [markdown]
# ![PCA+kNN UMAP ngenes](figures/1_3_umap_ngenes.png)
# The above UMAP colored by number of genes expressed shows that there is some variation in gene counts across the dataset, with certain clusters having higher or lower counts.  This could also potentially confound classification if the model relies on gene count signals. Cells with higher gene counts (e.g. tumor cells) seem to cluster together, while cells with lower gene counts (e.g. some T cell subsets) also cluster together.

# %%
sc.pl.umap(adata_cl, color="total_counts", title="total_counts", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_3_umap_counts.png", dpi=100)
plt.show()

# %% [markdown]
# ![PCA+kNN UMAP counts](figures/1_3_umap_counts.png)
# The above UMAP colored by total counts shows a similar pattern to the number of genes expressed, with certain clusters having higher or lower counts.  This again suggests that there is some variation in sequencing depth across the dataset, which could potentially confound classification if the model relies on count signals. Cells with higher total counts (e.g. tumor cells) seem to cluster together, while cells with lower total counts (e.g. some T cell subsets) also cluster together.

# %% [markdown]
# ### 1.4 Deep MLP Classifier

# %%
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# MLP on top of HVGs (no batch correction, no domain adaptation)
N_HVG = 1000
_adata_hvg = adata_cl.copy()
# Note: we could also do HVG selection separately on the labeled set, but for simplicity we'll just do it on the whole dataset here
sc.pp.highly_variable_genes(_adata_hvg, n_top_genes=N_HVG)
adata_mlp = adata_cl[:, _adata_hvg.var["highly_variable"]].copy()
del _adata_hvg

# Prepare data for MLP: extract HVG expression matrix,
# split into labeled vs. unlabeled, encode labels, train/val split
X_mlp_labeled   = adata_mlp[labeled_mask].X
X_mlp_unlabeled = adata_mlp[~labeled_mask].X
if hasattr(X_mlp_labeled, "toarray"):
    X_mlp_labeled   = X_mlp_labeled.toarray()
    X_mlp_unlabeled = X_mlp_unlabeled.toarray()

le = LabelEncoder()
y_mlp_labeled   = le.fit_transform(adata_mlp[labeled_mask].obs["celltype_ground_truth"].values)
y_mlp_unlabeled = le.transform(adata_mlp[~labeled_mask].obs["celltype_ground_truth"].values)
n_classes = len(le.classes_)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_mlp_labeled, y_mlp_labeled, test_size=0.20, random_state=SEED, stratify=y_mlp_labeled
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def to_tensor(X, y):
    return (
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device),
    )

X_tr_t,  y_tr_t  = to_tensor(X_tr, y_tr)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t = torch.tensor(X_mlp_unlabeled, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

# Define a simple MLP classifier
class MLP(nn.Module):
    """3-hidden-layer MLP: input->512->256->128->n_classes."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP(N_HVG, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
print(mlp)

# %% [markdown]
# MLP architecture consists of 3 hidden layers with 512, 256, and 128 units respectively.  Each hidden layer is followed by batch normalization, ReLU activation, and dropout (p=0.2) for regularization.  The output layer has `n_classes` units corresponding to the number of cell types.

# %%
# Check for existing checkpoint to resume training or load best model for evaluation
_mlp_ckpt = CKPT_DIR / "mlp.pt"

if _mlp_ckpt.exists():
    print(f"  Loading MLP from checkpoint: {_mlp_ckpt}")
    mlp.load_state_dict(torch.load(_mlp_ckpt, map_location=device))
    mlp.eval()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
else:
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    N_EPOCHS = 1000
    PATIENCE  = 30
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    best_state    = None
    epochs_no_improve = 0

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
            tr_logits  = mlp(X_tr_t)
            val_logits = mlp(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            tr_pred    = tr_logits.argmax(dim=1).cpu().numpy()
            val_pred   = val_logits.argmax(dim=1).cpu().numpy()
        val_losses.append(val_loss)
        train_accs.append(f1_score(y_tr, tr_pred, average="weighted", zero_division=0))
        val_accs.append(f1_score(y_val, val_pred, average="weighted", zero_division=0))

        scheduler.step(val_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in mlp.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 100 == 0:
            print(
                f"  Epoch {epoch+1:4d}/{N_EPOCHS} — "
                f"train_loss={train_losses[-1]:.4f}  val_loss={val_loss:.4f}  "
                f"val_f1={val_accs[-1]:.4f}"
            )

    mlp.load_state_dict(best_state)
    torch.save(mlp.state_dict(), _mlp_ckpt)
    print(f"  Saved MLP checkpoint -> {_mlp_ckpt}")

# %% [markdown]
# **Note**: I tried training with Cosine Annealing scheduler which gave me bad accuracy (around 30%), but when I switched to ReduceLROnPlateau scheduling, the accuracy went up.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label="Train")
axes[0].plot(val_losses,   label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("MLP Loss")
axes[0].legend()
axes[1].plot(train_accs, label="Train")
axes[1].plot(val_accs,   label="Val")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Weighted F1")
axes[1].set_title("MLP Weighted F1")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "1_4_mlp_loss_curves.png", dpi=100)
plt.show()

# %% [markdown]
# ![MLP Weighted F1](figures/1_4_mlp_loss_curves.png)
# The MLP training curves show that the training loss decreases steadily over epochs, while the validation loss also decreases at the start, but increases after 5 epochs or so, suggesting it is overfitting.  This suggests that the MLP is learning to classify the cell types reasonably well, and the early stopping helps prevent overfitting. The F1 score curves show that the training F1 increases steadily, while the validation F1 also increases at the start but plateaus, which is consistent with the loss curves. Overall, the MLP seems to be learning a useful representation for classifying the cell types, but there is some overfitting after a certain point, which is why early stopping was used.

# %%
mlp.eval()
with torch.no_grad():
    logits_test = mlp(X_test_t).cpu().numpy()
y_pred_mlp = le.inverse_transform(np.argmax(logits_test, axis=1))
y_prob_mlp = torch.softmax(torch.tensor(logits_test), dim=1).numpy()

y_test_str = le.inverse_transform(y_mlp_unlabeled)
acc_mlp = accuracy_score(y_test_str, y_pred_mlp)
f1_mlp  = f1_score(y_test_str, y_pred_mlp, average="weighted", zero_division=0)
print(f"MLP  Accuracy: {acc_mlp:.4f}  |  Weighted F1: {f1_mlp:.4f}")

# %% [markdown]
# MLP  Accuracy: 0.5833  |  Weighted F1: 0.5686

# %%
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
plt.show()

# %% [markdown]
# ![MLP Confusion Matrix](figures/1_4_mlp_confusion_matrix.png)
# The confustion matrix for the MLP baseline shows that although many cell types are classified well (higher counts along the diagonal), there are also some misclassifications, especially for rarer or more similar cell types.  For example, some CD8+ T cell subsets and tumor cells may be confused with each other or with other immune cells. Overall the MLP seems to capture the structure in the data but it's not better than the PCA+kNN baseline, which is consistent with its higher accuracy and F1 score.

# %%
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
plt.show()

# %% [markdown]
# ![MLP ROC Curves](figures/1_4_mlp_roc.png)
# The ROC curves for the MLP baseline show that many cell types have good discrimination (high AUC), while some rarer or more similar cell types have lower AUC values.  This again suggests that while the MLP captures some of the structure in the data, there is room for improvement, especially for rarer or more similar cell types.  Overall, the MLP seems to perform reasonably well, but it does not outperform the PCA+kNN baseline in this case, which is consistent with the accuracy and F1 scores observed.

# %%
print("\n--- Architecture summary ---")
print(mlp)
print(f"\nPCA+kNN  Acc={acc_knn:.4f}  F1={f1_knn:.4f}")
print(f"MLP      Acc={acc_mlp:.4f}  F1={f1_mlp:.4f}")

# %% [markdown]
# | Model   | Accuracy | F1 Score |
# |---------|----------|----------|
# | PCA+kNN | 0.7720   | 0.7574   |
# | MLP     | 0.5833   | 0.5686   |
# 
# From the table above, the PCA+kNN baseline outperforms the MLP classifier in terms of both accuracy and weighted F1 score on the unlabeled test set.  This suggests that the PCA+kNN model is better able to capture the structure in the data for classifying the cell types, while the MLP may be overfitting or not learning as effective a representation.  This highlights the importance of comparing deep learning models to classical baselines, as sometimes simpler models can perform better on certain datasets.

# %% [markdown]
# PCA+kNN  Acc=0.7720  F1=0.7574
# 
# MLP      Acc=0.5833  F1=0.5686

# %% [markdown]
# ## Part 2 — Deep Autoencoders and scVI/scANVI
# 
# ### 2.1 Deep Autoencoder Baseline

# %%
LATENT_DIM = 64
AE_INPUT   = N_HVG

# Define a deep autoencoder for dimensionality reduction and latent representation learning
class Autoencoder(nn.Module):
    """Deep autoencoder: 1000->256->128->64->128->256->1000."""
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# Train the autoencoder on the entire dataset (no labels used)
# to learn a latent representation of the cells.
X_ae_all = adata_mlp.X
if hasattr(X_ae_all, "toarray"):
    X_ae_all = X_ae_all.toarray()
X_ae_t = torch.tensor(X_ae_all, dtype=torch.float32).to(device)

ae_loader = DataLoader(TensorDataset(X_ae_t), batch_size=512, shuffle=True)
ae_model  = Autoencoder(AE_INPUT, LATENT_DIM).to(device)

# Check for existing checkpoint to resume training or load best model for evaluation
_ae_ckpt   = CKPT_DIR / "autoencoder.pt"
_ae_latent = CKPT_DIR / "ae_latent.npy"

if _ae_ckpt.exists() and _ae_latent.exists():
    print(f"  Loading Autoencoder from checkpoint: {_ae_ckpt}")
    ae_model.load_state_dict(torch.load(_ae_ckpt, map_location=device))
    ae_model.eval()
    ae_losses = []
    Z_all = np.load(_ae_latent)
else:
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
    ae_criterion = nn.MSELoss()
    AE_EPOCHS = 1000
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
        if (epoch + 100) % 10 == 0:
            print(f"  AE Epoch {epoch+1:3d}/{AE_EPOCHS} — loss={ae_losses[-1]:.6f}")

    ae_model.eval()
    with torch.no_grad():
        _, Z_all = ae_model(X_ae_t)
    Z_all = Z_all.cpu().numpy()
    torch.save(ae_model.state_dict(), _ae_ckpt)
    np.save(_ae_latent, Z_all)
    print(f"  Saved AE checkpoint -> {_ae_ckpt}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ae_losses, label="Reconstruction Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Autoencoder Reconstruction Loss")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_1_ae_loss.png", dpi=100)
plt.show()

# %% [markdown]
# ![Autoencoder Reconstruction Loss](figures/2_1_ae_loss.png)
# 
# The autoencoder reconstruction loss curve shows that the MSE loss decreases steadily over epochs, suggesting that the model is learning to reconstruct the input data effectively.  The curve may plateau towards the end, indicating that the model is converging.  Overall, the autoencoder seems to be learning a useful latent representation of the data, which can be evaluated further by looking at UMAP visualizations and downstream classification performance.

# %%
# Add the learned latent representation from the autoencoder to .obsm and visualize with UMAP
adata_ae = adata_mlp.copy()
adata_ae.obsm["X_ae"] = Z_all
sc.pp.neighbors(adata_ae, use_rep="X_ae", random_state=SEED)
sc.tl.umap(adata_ae, random_state=SEED)

# %%
sc.pl.umap(adata_ae, color="celltype_ground_truth", title="AE latent — cell type", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_1_umap_ae_celltype.png", dpi=100)
plt.show()

# %% [markdown]
# ![AE latent — cell type](figures/2_1_umap_ae_celltype.png)
# The above UMAP of the autoencoder latent space colored by ground-truth cell types shows that many cell types form many distinct clusters, but are very mixed.  This suggests that the autoencoder is learning a latent representation that captures some of the structure in the data related to cell types, although it may not perfectly separate all cell types.

# %%
sc.pl.umap(adata_ae, color="batch", title="AE latent — batch", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_1_umap_ae_batch.png", dpi=100)
plt.show()

# %% [markdown]
# ![AE latent — batch](figures/2_1_umap_ae_batch.png)
# Although the batch structure is not as pronounced as in the PCA UMAP, it is still present (forming many distinct clusters but are mixed), which suggests that the autoencoder may be capturing some batch-specific signals.

# %%
# Train a kNN classifier on the autoencoder latent space
# using the labeled cells and evaluate on the unlabeled cells
Z_labeled   = Z_all[labeled_mask]
Z_unlabeled = Z_all[~labeled_mask]

knn_ae = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
knn_ae.fit(Z_labeled, adata_mlp[labeled_mask].obs["celltype_ground_truth"].values)
y_pred_ae  = knn_ae.predict(Z_unlabeled)
y_test_ae  = adata_mlp[~labeled_mask].obs["celltype_ground_truth"].values
acc_ae = accuracy_score(y_test_ae, y_pred_ae)
f1_ae  = f1_score(y_test_ae, y_pred_ae, average="weighted", zero_division=0)
print(f"AE+kNN  Accuracy: {acc_ae:.4f}  |  Weighted F1: {f1_ae:.4f}")

# %% [markdown]
# AE+kNN  Accuracy: 0.4272  |  Weighted F1: 0.4119

# %% [markdown]
# ### 2.2 scVI Pre-training

# %%
adata_scvi = braun_dataset.copy()
adata_scvi.obs["celltype_scvi"] = adata_scvi.obs["celltype_masked"].copy()

scvi.model.SCVI.setup_anndata(
    adata_scvi,
    layer=None,
    batch_key="batch",
    labels_key="celltype_scvi",
)

_scvi_ckpt = CKPT_DIR / "scvi_model"

if _scvi_ckpt.exists():
    print(f"  Loading scVI model from checkpoint: {_scvi_ckpt}")
    vae = scvi.model.SCVI.load(str(_scvi_ckpt), adata=adata_scvi)
else:
    vae = scvi.model.SCVI(
        adata_scvi, n_latent=20, n_hidden=128, n_layers=2, gene_likelihood="nb"
    )
    vae.train(
        max_epochs=100,
        train_size=0.9,
        validation_size=0.1,
        early_stopping=True,
        plan_kwargs={"lr": 1e-3},
    )
    vae.save(str(_scvi_ckpt), overwrite=True)
    print(f"  Saved scVI checkpoint -> {_scvi_ckpt}")

Z_scvi = vae.get_latent_representation()
adata_scvi.obsm["X_scVI"] = Z_scvi
sc.pp.neighbors(adata_scvi, use_rep="X_scVI", random_state=SEED)
sc.tl.umap(adata_scvi, random_state=SEED)

# %%
sc.pl.umap(adata_scvi, color="batch", title="scVI latent — batch", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_2_umap_scvi_batch.png", dpi=100)
plt.show()

# %% [markdown]
# ![scVI latent — batch](figures/2_2_umap_scvi_batch.png)
# The above UMAP of the scVI latent space colored by batch shows that the batch structure is more pronounced in the scVI latent space compared to the autoencoder, with certain batches clustering together.  This suggests that the scVI model is capturing batch-specific signals in its latent representation, which is expected given that it is designed to model and correct for batch effects.

# %%
sc.pl.umap(adata_scvi, color="celltype_ground_truth", title="scVI latent — cell type", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_2_umap_scvi_celltype.png", dpi=100)
plt.show()

# %% [markdown]
# ![scVI latent — cell type](figures/2_2_umap_scvi_celltype.png)
# The above UMAP of the scVI latent space colored by ground-truth cell types shows that many cell types form distinct clusters, and the separation between cell types is more pronounced compared to the autoencoder.  This suggests that the scVI model is learning a latent representation that captures the structure in the data related to cell types more effectively than the autoencoder, which is consistent with its design as a model for single-cell data. Overall, the scVI latent space seems to capture both batch and cell type structure, which can be further evaluated by looking at downstream classification performance and scANVI fine-tuning.

# %% [markdown]
# ### 2.3 scANVI Training and Annotation

# %%
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
    print(f"  Saved scANVI checkpoint -> {_scanvi_ckpt}")

scanvi_preds = scanvi_model.predict(soft=False)
scanvi_probs = scanvi_model.predict(soft=True)
adata_scvi.obs["scanvi_predictions"] = scanvi_preds
adata_scvi.obs["scanvi_predictions"] = adata_scvi.obs["scanvi_predictions"].astype(str)

Z_scanvi = scanvi_model.get_latent_representation()
adata_scvi.obsm["X_scANVI"] = Z_scanvi
sc.pp.neighbors(adata_scvi, use_rep="X_scANVI", random_state=SEED)
sc.tl.umap(adata_scvi, random_state=SEED)

# %% [markdown]
# The hyperparameters for scANVI fine-tuning were chosen based on the tutorial defaults and the structure of the dataset. The `n_latent`, `n_hidden`, and `n_layers` parameters were inherited from the scVI pre-training to maintain consistency in the model architecture. The `n_samples_per_label` was set to 100 to ensure that the model sees a balanced number of samples from each cell type during fine-tuning, which can help improve classification performance, especially for rarer cell types. The `max_epochs` for fine-tuning was set to 20, which is a common choice for scANVI fine-tuning, as it allows the model to adjust to the labeled and unlabeled data without overfitting.  The `unlabeled_category` was set to "Unknown" to match the masking applied in the dataset, ensuring that scANVI correctly identifies which cells are unlabeled during training. Overall, these hyperparameters should allow scANVI to effectively leverage the pre-trained scVI latent space while also learning to classify the cell types based on the available labels.

# %%
sc.pl.umap(adata_scvi, color="scanvi_predictions", title="scANVI predictions", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_3_umap_scanvi_pred.png", dpi=100)
plt.show()

# %% [markdown]
# ![scANVI predictions](figures/2_3_umap_scanvi_pred.png)
# The above UMAP of the scANVI latent space colored by scANVI predictions shows that many predicted cell types form distinct clusters, and the overall structure of the predictions seems to align well with the ground-truth cell types. This suggests that scANVI is effectively leveraging the pre-trained scVI latent space and the available labels to learn a representation that allows it to classify the cell types reasonably well. However, there may still be some misclassifications or mixed clusters, which can be further evaluated by looking at the confusion matrix and classification metrics.

# %%
sc.pl.umap(adata_scvi, color="celltype_ground_truth", title="scANVI — ground truth", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_3_umap_scanvi_truth.png", dpi=100)
plt.show()

# %% [markdown]
# ![scANVI — ground truth](figures/2_3_umap_scanvi_truth.png)
# The above UMAP of the scANVI latent space colored by ground-truth cell types shows that many cell types form distinct clusters, and the separation between cell types is more pronounced compared to the scVI latent space. This suggests that the scANVI fine-tuning has helped to refine the latent representation to better capture the structure related to cell types, which is consistent with its design as a model for semi-supervised annotation of single-cell data.

# %%
sc.pl.umap(adata_scvi, color="batch", title="scANVI — batch", show=False)
plt.gcf().set_size_inches(16, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_3_umap_scanvi_batch.png", dpi=100)
plt.show()

# %% [markdown]
# ![scANVI — batch](figures/2_3_umap_scanvi_batch.png)
# The above UMAP of the scANVI latent space colored by batch shows that the batch structure is still present in the scANVI latent space, which is somewhat similarly structured compared to the scVI latent space.

# %%
max_prob = scanvi_probs.max(axis=1).values
adata_scvi.obs["scanvi_confidence"] = max_prob

sc.pl.violin(
    adata_scvi, "scanvi_confidence",
    groupby="scanvi_predictions", rotation=90, show=False,
)
plt.gcf().set_size_inches(16, 20)
plt.title("scANVI prediction confidence per predicted cell type")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "2_3_scanvi_confidence_violin.png", dpi=100)
plt.show()

# %% [markdown]
# ![scANVI prediction confidence per predicted cell type](figures/2_3_scanvi_confidence_violin.png)
# The violin plot of scANVI prediction confidence per predicted cell type shows that certain predicted cell types have higher confidence scores (higher max probabilities) compared to others. This suggests that scANVI is more confident in its predictions for certain cell types, which may be due to those cell types being more distinct or having more labeled examples during training. Conversely, cell types with lower confidence scores may be more similar to other cell types or have fewer labeled examples, leading to more uncertainty in the predictions.

# %%
unlabeled_idx   = adata_scvi.obs["celltype_masked"] == "Unknown"
y_true_scanvi   = adata_scvi.obs.loc[unlabeled_idx, "celltype_ground_truth"].values
y_pred_scanvi   = adata_scvi.obs.loc[unlabeled_idx, "scanvi_predictions"].values

acc_scanvi = accuracy_score(y_true_scanvi, y_pred_scanvi)
f1_scanvi  = f1_score(y_true_scanvi, y_pred_scanvi, average="weighted", zero_division=0)
print(f"scANVI  Accuracy: {acc_scanvi:.4f}  |  Weighted F1: {f1_scanvi:.4f}")

# %% [markdown]
# scANVI  Accuracy: 0.8652  |  Weighted F1: 0.8649

# %%
classes_scanvi    = scanvi_probs.columns.tolist()
y_true_scanvi_bin = label_binarize(y_true_scanvi, classes=classes_scanvi)
prob_matrix       = scanvi_probs.loc[unlabeled_idx].values

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
plt.show()

# %% [markdown]
# ![scANVI ROC Curves](figures/2_3_scanvi_roc.png)
# The ROC curves for scANVI on the unlabeled cells show that many cell types have very good discrimination (high AUC), while some rarer or more similar cell types have slightly lower AUC values. This suggests that scANVI is effectively leveraging the pre-trained scVI latent space and the available labels to learn a representation that allows it to classify the cell types reasonably well, although there may still be room for improvement, especially for rarer or more similar cell types.

# %% [markdown]
# Conceptually, the scANVI model differs from the PCA+kNN and scVI models in that it is a semi-supervised model that leverages both the pre-trained scVI latent space and the available labels to learn a representation that is optimized for classifying cell types.  In contrast, PCA+kNN is a purely unsupervised approach that relies on dimensionality reduction followed by a simple classifier, while scVI is an unsupervised generative model that captures the underlying structure of the data but does not directly optimize for classification.  The scANVI model is designed to fine-tune the latent space to better separate cell types based on the available labels, which can lead to improved classification performance compared to purely unsupervised approaches.

# %% [markdown]
# On the training set, the following table shows the accuracies and f1 scores of the models.
# | Model   | Accuracy | F1 Score |
# |---|---:|---:|
# | PCA+kNN | 0.7720 | 0.7574 |
# | MLP | 0.5833 | 0.5686 |
# | AE+kNN | 0.4272 | 0.4119 |
# | scANVI | 0.8652 | 0.8649 |
# 
# scANVI seems to be the best performer during the training phase.

# %% [markdown]
# ### 2.4 Orthogonal Projection on test.h5ad

# %%
if TEST_PATH.exists():
    adata_test = sc.read_h5ad(TEST_PATH)
    print(f"Test dataset: {adata_test}")

    # --- PCA+kNN on test set ---
    adata_test_cl = adata_test.copy()
    sc.pp.normalize_total(adata_test_cl, target_sum=1e4)
    sc.pp.log1p(adata_test_cl)

    common_genes_cl = adata_test_cl.var_names.intersection(adata_cl.var_names)
    adata_test_cl   = adata_test_cl[:, common_genes_cl].copy()

    _tr_log = braun_dataset.copy()
    sc.pp.normalize_total(_tr_log, target_sum=1e4)
    sc.pp.log1p(_tr_log)
    _tr_log = _tr_log[:, common_genes_cl].copy()
    sc.pp.scale(_tr_log, max_value=10)
    sc.tl.pca(_tr_log, svd_solver="arpack", random_state=SEED)

    pca_components = _tr_log.varm["PCs"]
    train_mean     = _tr_log.var["mean"].values
    train_std      = _tr_log.var["std"].values

    X_te = adata_test_cl.X
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()
    X_te_scaled  = (X_te - train_mean) / np.where(train_std == 0, 1.0, train_std)
    np.clip(X_te_scaled, -10, 10, out=X_te_scaled)
    X_test_proj  = X_te_scaled @ pca_components

    knn_test = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    knn_test.fit(
        _tr_log[labeled_mask].obsm["X_pca"],
        _tr_log[labeled_mask].obs["celltype_ground_truth"].values,
    )
    y_pred_knn_test = knn_test.predict(X_test_proj)
    del _tr_log

    # --- MLP on test set ---
    adata_test_mlp = adata_test.copy()
    sc.pp.normalize_total(adata_test_mlp, target_sum=1e4)
    sc.pp.log1p(adata_test_mlp)
    sc.pp.scale(adata_test_mlp, max_value=10)

    train_hvg_genes = adata_mlp.var_names.tolist()
    test_gene_set   = set(adata_test_mlp.var_names.tolist())
    genes_in_test   = [g for g in train_hvg_genes if g in test_gene_set]

    X_test_mlp_sub = adata_test_mlp[:, genes_in_test].X
    if hasattr(X_test_mlp_sub, "toarray"):
        X_test_mlp_sub = X_test_mlp_sub.toarray()

    X_test_mlp = np.zeros((adata_test_mlp.n_obs, N_HVG), dtype=np.float32)
    present_positions = [train_hvg_genes.index(g) for g in genes_in_test]
    X_test_mlp[:, present_positions] = X_test_mlp_sub

    mlp.eval()
    with torch.no_grad():
        logits_proj = mlp(torch.tensor(X_test_mlp, dtype=torch.float32).to(device)).cpu().numpy()
    y_pred_mlp_test = le.inverse_transform(np.argmax(logits_proj, axis=1))

    # --- AE+kNN on test set ---
    X_ae_test = X_test_mlp
    ae_model.eval()
    with torch.no_grad():
        _, Z_test_ae = ae_model(torch.tensor(X_ae_test, dtype=torch.float32).to(device))
    Z_test_ae = Z_test_ae.cpu().numpy()
    y_pred_ae_test = knn_ae.predict(Z_test_ae)

    # --- scANVI on test set ---
    adata_test_scanvi = adata_test.copy()
    if "batch" not in adata_test_scanvi.obs.columns:
        adata_test_scanvi.obs["batch"] = "test"
    adata_test_scanvi.obs["celltype_scvi"] = "Unknown"
    scvi.model.SCANVI.prepare_query_anndata(adata_test_scanvi, scanvi_model)
    _scanvi_query = scvi.model.SCANVI.load_query_data(
        adata_test_scanvi, scanvi_model, freeze_dropout=True
    )
    scanvi_preds_test  = _scanvi_query.predict()
    y_pred_scanvi_test = np.array(scanvi_preds_test)

    # --- Results ---
    if "celltype" in adata_test.obs.columns:
        y_true_test = adata_test.obs["celltype"].values
        def _metrics(y_true, y_pred, name):
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            print(f"  {name:<20s}  Acc={acc:.4f}  F1={f1:.4f}")
        print("\nTest-set metrics:")
        _metrics(y_true_test, y_pred_knn_test,   "PCA+kNN")
        _metrics(y_true_test, y_pred_mlp_test,   "MLP")
        _metrics(y_true_test, y_pred_ae_test,    "AE+kNN")
        _metrics(y_true_test, y_pred_scanvi_test,"scANVI")
    else:
        print("No ground-truth labels in test.h5ad; skipping metric computation.")
        print(f"  PCA+kNN  predictions: {pd.Series(y_pred_knn_test).value_counts().to_dict()}")
        print(f"  MLP      predictions: {pd.Series(y_pred_mlp_test).value_counts().to_dict()}")
        print(f"  AE+kNN   predictions: {pd.Series(y_pred_ae_test).value_counts().to_dict()}")
        print(f"  scANVI   predictions: {pd.Series(y_pred_scanvi_test).value_counts().to_dict()}")

    results_df = pd.DataFrame({
        "Model":    ["PCA+kNN", "MLP", "AE+kNN", "scANVI"],
        "Train Acc":[acc_knn, acc_mlp, acc_ae, acc_scanvi],
        "Train F1": [f1_knn,  f1_mlp,  f1_ae,  f1_scanvi],
    })
    print("\n--- In-distribution performance summary ---")
    print(results_df.to_string(index=False))
else:
    print(f"test.h5ad not found at {TEST_PATH}; skipping orthogonal projection.")
    results_df = pd.DataFrame({
        "Model":          ["PCA+kNN", "MLP", "AE+kNN", "scANVI"],
        "Masked-set Acc": [acc_knn, acc_mlp, acc_ae, acc_scanvi],
        "Masked-set F1":  [f1_knn,  f1_mlp,  f1_ae,  f1_scanvi],
    })
    print(results_df.to_string(index=False))

# %% [markdown]
# | Model   | Accuracy | F1 Score |
# |---|---:|---:|
# | PCA+kNN | 0.5368 | 0.5767 |
# | MLP | 0.3145 | 0.3466 |
# | AE+kNN | 0.2999 | 0.3408 |
# | scANVI | 0.5275 | 0.5939 |
# 
# The test-set metrics show that the PCA+kNN and scANVI models have similar accuracy and F1 scores, suggesting that both models are able to generalize reasonably well to the test set. The MLP and AE+kNN models have lower accuracy and F1 scores, indicating that they may not be generalizing as well to the test set.

# %% [markdown]
# ### 2.5 Interpretability — Captum Integrated Gradients

# %%
try:
    from captum.attr import IntegratedGradients

    gene_names  = adata_mlp.var_names.tolist()
    unique_ct   = list(np.unique(y_test_str))
    ct_for_interp = unique_ct[:3]

    ig = IntegratedGradients(mlp)
    mlp.eval()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, ct in zip(axes, ct_for_interp):
        target_idx = int(np.where(le.classes_ == ct)[0][0])
        ct_mask = y_test_str == ct
        X_ct = torch.tensor(X_mlp_unlabeled[ct_mask], dtype=torch.float32).to(device)
        if len(X_ct) == 0:
            ax.set_visible(False)
            continue
        baseline = torch.zeros_like(X_ct[:1])
        attrs, _ = ig.attribute(
            X_ct[:min(50, len(X_ct))],
            baselines=baseline,
            target=target_idx,
            return_convergence_delta=True,
        )
        mean_attr = attrs.detach().cpu().numpy().mean(axis=0)
        top10_idx   = np.argsort(np.abs(mean_attr))[-10:][::-1]
        top10_genes = [gene_names[i] if i < len(gene_names) else f"gene_{i}" for i in top10_idx]
        top10_vals  = mean_attr[top10_idx]
        ax.barh(top10_genes[::-1], top10_vals[::-1], color="steelblue")
        ax.set_xlabel("Attribution score")
        ax.set_title(f"Top-10 genes — {ct}")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_5_ig_top10_genes.png", dpi=100)
    plt.show()
    print("Saved Integrated Gradients plot.")

except ImportError:
    print("captum not installed. Skipping interpretability section.")
    print("Install with:  pip install captum")

# %% [markdown]
# ![Interpretability](figures/2_5_ig_top10_genes.png)
# The above plot shows the top-10 genes with the highest absolute attribution scores for three selected cell types from the unlabeled test set, as computed by Integrated Gradients on the MLP model.  The attribution scores indicate how much each gene contributed to the MLP's prediction for that cell type, with positive scores indicating a contribution towards predicting that cell type and negative scores indicating a contribution against it. It makes sense that certain genes have high attribution scores for specific cell types, as these genes may be key markers or drivers of the identity of those cell types. However, the specific genes and their scores would need to be interpreted in the context of known biology and marker genes for those cell types to draw meaningful conclusions.

# %%
import glob as _glob
print("\n" + "=" * 70)
print("Pipeline complete. Generated output files:")
print("=" * 70)
for f in sorted(_glob.glob(str(FIGURES_DIR / "*.png"))):
    print(f"  {f}")

# %% [markdown]
# ## References:
# 
# * https://github.com/sivasatvik/nyu-ai-in-genomics-a2/tasks/251358b9-3d19-481f-964c-a07a03217891
# * Various improvements in local vscode copilot chat on top of the above reference.
# 


