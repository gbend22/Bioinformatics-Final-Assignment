import os
import gc

import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

TULIP_32_TUMOR_TYPES = [
    "uveal melanoma",
    "adrenocortical cancer",
    "bladder urothelial carcinoma",
    "brain lower grade glioma",
    "breast invasive carcinoma",
    "cervical & endocervical cancer",
    "cholangiocarcinoma",
    "colon adenocarcinoma",
    "diffuse large b-cell lymphoma",
    "esophageal carcinoma",
    "glioblastoma multiforme",
    "head & neck squamous cell carcinoma",
    "kidney chromophobe",
    "kidney clear cell carcinoma",
    "kidney papillary cell carcinoma",
    "liver hepatocellular carcinoma",
    "lung adenocarcinoma",
    "lung squamous cell carcinoma",
    "mesothelioma",
    "ovarian serous cystadenocarcinoma",
    "pancreatic adenocarcinoma",
    "pheochromocytoma & paraganglioma",
    "prostate adenocarcinoma",
    "rectum adenocarcinoma",
    "sarcoma",
    "skin cutaneous melanoma",
    "stomach adenocarcinoma",
    "testicular germ cell tumor",
    "thymoma",
    "thyroid carcinoma",
    "uterine carcinosarcoma",
    "uterine corpus endometrioid carcinoma",
]


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def load_protein_coding_genes(pc_url, pc_path):
    if not os.path.exists(pc_path):
        response = requests.get(pc_url)
        with open(pc_path, "wb") as f:
            f.write(response.content)
        print("Downloaded protein coding genes list")
    else:
        print("Protein coding genes list already exists")

    pc_genes_df = pd.read_csv(
        pc_path, sep="\t", header=None, names=["gene_id", "gene_name"]
    )
    pc_genes_df["gene_id_base"] = pc_genes_df["gene_id"].str.split(".").str[0]
    pc_gene_ids = set(pc_genes_df["gene_id_base"].tolist())
    print(f"Protein coding genes: {len(pc_gene_ids):,}")
    return pc_gene_ids


def load_phenotype(phenotype_path):
    labels_full = pd.read_csv(phenotype_path, sep="\t", index_col=0)
    print(f"Total samples in phenotype file: {len(labels_full)}")

    primary_mask = labels_full["sample_type"] == "Primary Tumor"
    primary_tumor_idx = labels_full[primary_mask].index
    print(f"Primary tumor samples: {len(primary_tumor_idx)}")

    return labels_full, primary_tumor_idx


def load_expression_data(expr_path, pc_gene_ids, primary_tumor_idx, chunk_size=5000):
    expr_chunks = []

    for chunk in pd.read_csv(expr_path, sep="\t", index_col=0, chunksize=chunk_size):
        chunk.index = chunk.index.str.split(".").str[0]

        pc_in_chunk = chunk.index.intersection(pc_gene_ids)
        if len(pc_in_chunk) == 0:
            continue
        chunk = chunk.loc[pc_in_chunk]

        chunk_T = chunk.T
        chunk_filtered = chunk_T.loc[chunk_T.index.intersection(primary_tumor_idx)]

        if len(chunk_filtered) == 0:
            continue

        expr_chunks.append(chunk_filtered)

    expr_full = pd.concat(expr_chunks, axis=1).astype(np.float32)
    expr_full.index.name = "sample"

    del expr_chunks
    gc.collect()

    print(f"Expression matrix shape: {expr_full.shape}")
    print(f"Genes matched: {expr_full.shape[1]:,} / {len(pc_gene_ids):,}")
    print(f"Samples matched: {expr_full.shape[0]:,} / {len(primary_tumor_idx):,}")

    expr_full = (2.0 ** expr_full) - 1.0
    expr_full[expr_full < 0] = 0

    return expr_full


def align_samples(expr_full, labels_full, primary_tumor_idx):
    shared = expr_full.index.intersection(primary_tumor_idx)
    expr = expr_full.loc[shared]
    labels = labels_full.loc[shared]

    print(f"Aligned samples: {len(shared)}")
    print(f"Expression: {expr.shape}")
    print(f"Labels: {labels.shape}")

    return expr, labels


def filter_to_tulip_types(expr, labels):
    y_raw = labels["_primary_disease"].astype(str).str.strip().str.lower()

    mask = y_raw.isin(TULIP_32_TUMOR_TYPES)
    expr = expr.loc[mask]
    y_raw = y_raw.loc[mask]
    labels = labels.loc[mask]

    print(f"After filtering to 32 TULIP types: {len(expr):,} samples, {y_raw.nunique()} classes")

    return expr, y_raw, labels


def normalize_and_pad(expr, pad_target=19800):
    row_sums = expr.sum(axis=1)
    expr = expr.div(row_sums, axis=0) * 1_000_000

    expr[expr <= 0] = 0.000001
    expr = expr.astype(np.float64).apply(np.log10).astype(np.float32)
    expr[expr < 0] = 0

    print(f"Value range after log10(TPM): [{expr.values.min():.4f}, {expr.values.max():.4f}]")

    n_genes = expr.shape[1]
    n_pad = pad_target - n_genes
    if n_pad > 0:
        pad_cols = pd.DataFrame(
            np.zeros((len(expr), n_pad), dtype=np.float32),
            index=expr.index,
            columns=[f"PAD_{i}" for i in range(n_pad)],
        )
        expr = pd.concat([expr, pad_cols], axis=1)
    elif n_pad < 0:
        expr = expr.iloc[:, :pad_target]

    X = expr.to_numpy(dtype=np.float32, copy=False)
    gc.collect()
    print(f"X shape: {X.shape}")

    return X


def encode_labels(y_raw):
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"NUM_CLASSES: {num_classes}")

    return y, le, num_classes


def split_data(X, y, test_size=0.20, val_ratio=0.50, random_state=42):
    indices = np.arange(len(y))

    idx_train, idx_temp = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=val_ratio, random_state=random_state, stratify=y[idx_temp]
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size=128, num_workers=2):
    train_loader = DataLoader(
        NumpyDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        NumpyDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        NumpyDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
