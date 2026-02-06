# Bioinformatics Final Assignment

## აბსტრაქტი

სიმსივნის ტიპის ზუსტი იდენტიფიკაცია RNA-seq გენის ექსპრესიის მონაცემებზე დაყრდნობით კრიტიკულ როლს ასრულებს precision oncology-ში. წინამდებარე ნაშრომში ჩვენ ვიკვლევთ Deep Learning-ზე დაფუძნებულ მიდგომებს პირველადი სიმსივნის ტიპის კლასიფიკაციისთვის, TULIP-ის (Tumor Type Prediction Tool Using Convolutional Neural Networks) კვლევის საფუძველზე, რომელმაც CNN არქიტექტურა გამოიყენა ამ ამოცანის გადასაწყვეტად. ჩვენი მიზანი იყო TULIP-ის შედეგების რეპლიკაცია ალტერნატიული არქიტექტურებით - კერძოდ, **LSTM** (Long Short-Term Memory) და **Transformer** მოდელებით. ორივე მოდელი გავწვრთენით TCGA (The Cancer Genome Atlas) RNA-seq მონაცემთა ბაზიდან მიღებულ 9,185 Primary Tumor ნიმუშზე, 32 სხვადასხვა სიმსივნის ტიპის კლასიფიკაციისთვის. Preprocessing მოიცავდა Protein-Coding გენების ფილტრაციას, FPKM-დან TPM-ში კონვერტაციას, Log10 ტრანსფორმაციას და Library Size ნორმალიზაციას. Transformer მოდელმა მიაღწია **95.76%** სიზუსტეს Test Set-ზე, ხოლო LSTM-მა - **93.25%**. შედეგები ადასტურებს, რომ Sequence-based Deep Learning არქიტექტურები ეფექტიანად ახერხებენ გენის ექსპრესიის პროფილებიდან სიმსივნის ტიპის ამოცნობას და წარმოადგენენ CNN-ის ღირსეულ ალტერნატივას ამ ამოცანისთვის.

---

## Project Overview

This project replicates and extends the work of the [TULIP paper](https://doi.org/10.1016/j.cmpb.2023.107720), which used a CNN to classify primary tumor types from RNA-seq gene expression data. Instead of CNNs, we implement and compare two alternative deep learning architectures:

- **LSTM** (Bidirectional, 2-layer)
- **Transformer** (3-layer encoder with multi-head self-attention)

Both models are trained on TCGA RNA-seq data to classify **32 cancer types**.

## Dataset

- **Source:** [Xena](https://xenabrowser.net/datapages/)
- **Expression Data:** `tcga_RSEM_gene_fpkm.gz` (FPKM values)
- **Phenotype Data:** `tcga_phenotype.tsv.gz`
- **Samples:** 9,185 primary tumor samples
- **Features:** 19,740 protein-coding genes (padded to 19,800)
- **Classes:** 32 cancer types
- **Split:** 80% train / 10% validation / 10% test (stratified)

## Preprocessing

The preprocessing pipeline (`preprocessing/preprocess.py`) applies the following steps:

1. **Gene filtering** - keep only protein-coding genes (~19,740 genes)
2. **Sample filtering** - keep only Primary Tumor samples
3. **Cancer type filtering** - retain only the 32 TULIP-defined cancer types
4. **FPKM → TPM conversion** - `(2^FPKM) - 1`, negatives clipped to 0
5. **Library size normalization** - scale each sample to 1,000,000 (TPM)
6. **Log10 transformation** - `log10(TPM)` with floor at 1e-6
7. **Padding** - pad to 19,800 genes for uniform input size
8. **Label encoding** - `LabelEncoder` for the 32 cancer type strings

## Models

### LSTM

| Component | Details |
|-----------|---------|
| Input reshape | (batch, 19800) → (batch, 198, 100) |
| LSTM | 2 layers, hidden=128, bidirectional, dropout=0.3 |
| Post-LSTM | LayerNorm → Dropout(0.3) → Linear(256→32) |
| Optimizer | Adam (lr=5e-4, weight_decay=1e-5) |
| Parameters | **639,520** |

### Transformer

| Component | Details |
|-----------|---------|
| Input projection | Linear(19800→2048) → ReLU → Linear(2048→1024) |
| Reshape | (batch, 1024) → (batch, 4, 256) |
| Positional encoding | Sinusoidal (Vaswani et al.) |
| Encoder | 3 layers, d_model=256, 8 heads, FFN=1024, GELU |
| Classifier | Linear(1024→512→256→32) with ReLU + Dropout |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Parameters | **45,684,256** |

Both models use **CrossEntropyLoss**, **ReduceLROnPlateau** scheduler, **gradient clipping** (max_norm=1.0), and **early stopping**.

## Results

| Metric | LSTM | Transformer |
|--------|------|-------------|
| Test Accuracy | 93.25% | **95.76%** |
| Precision | 0.94 | **0.96** |
| Recall | 0.93 | **0.96** |
| F1 (weighted) | 0.93 | **0.96** |
| Parameters | 639K | 45.7M |
| Epochs trained | 51 | 70 |

The Transformer outperforms the LSTM by ~2.5% accuracy, though at the cost of 71x more parameters. Both models struggle with cancer types that have small sample sizes (e.g., cholangiocarcinoma) or similar expression profiles (e.g., rectum vs. colon adenocarcinoma).

## Project Structure

```
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py          # Data loading & preprocessing pipeline
├── models/
│   ├── __init__.py
│   ├── lstm.py                # LSTM classifier
│   └── transformer.py         # Transformer classifier
├── lstm.ipynb                 # LSTM training & evaluation (Colab)
├── transformer.ipynb          # Transformer training & evaluation (Colab)
└── README.md
```

## How to Run

Both notebooks are designed to run on **Google Colab** with GPU. The data files should be placed in Google Drive and the paths updated accordingly in the notebooks.

```
pip install torch scikit-learn pandas numpy matplotlib seaborn tqdm
```

## References

- Lyu, B., & Haque, A. (2023). *TULIP - An RNA-seq-based Primary Tumor Type Prediction Tool Using Convolutional Neural Networks.* Computer Methods and Programs in Biomedicine. The paper link: [link](https://journals.sagepub.com/doi/epub/10.1177/11769351221139491)
- Expression Data - [link](https://xenabrowser.net/datapages/?dataset=tcga_RSEM_gene_fpkm&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
- Phenotype Data - [link](https://xenabrowser.net/datapages/?dataset=TCGA_phenotype_denseDataOnlyDownload.tsv&host=https%3A%2F%2Fpancanatlas.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
- Protein coding gene data - [link](https://raw.githubusercontent.com/CBIIT/TULIP/main/gene_lists/protein_coding_genes.txt)
