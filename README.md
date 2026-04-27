# 🧬 Classification of Cancer Driver vs Passenger Mutations

> Using Sequence & Genomic Context Features

A single-notebook machine-learning pipeline that distinguishes **cancer driver mutations** from **passenger mutations** using features derived from the [COSMIC](https://cancer.sanger.ac.uk/cosmic) database (v103, GRCh37).

---

## 🔬 The Problem

Every tumour carries thousands of somatic mutations, but only a tiny minority — **driver mutations** — actually cause the cancer to grow and spread. The vast majority are **passenger mutations** that accumulate by chance with no functional impact. Reliably telling them apart is one of the hardest open problems in precision oncology.

This project frames the problem as a **binary classification task**: given a somatic mutation's sequence-level and genomic-context features, predict whether it lies in a known cancer driver gene (`is_CGC = 1`) or not (`is_CGC = 0`).

---

## 🏆 Key Results

| Model | ROC-AUC | PR-AUC | MCC | F1 | Precision | Recall |
|-------|---------|--------|-----|-------|-----------|--------|
| Logistic Regression | 0.586 | 0.109 | 0.067 | 0.169 | 0.101 | 0.503 |
| XGBoost (default thresh) | 0.988 | 0.929 | 0.825 | 0.838 | 0.797 | 0.883 |
| **XGBoost (thresh 0.59)** | **0.988** | **0.929** | **0.850** | **0.859** | **0.912** | **0.813** |
| LightGBM | 0.955 | 0.810 | 0.587 | 0.602 | 0.471 | 0.836 |
| Ensemble (XGB + LGBM) | 0.978 | 0.894 | 0.743 | 0.760 | 0.682 | 0.859 |

**Best model:** XGBoost with optimised threshold (0.59) — **PR-AUC 0.929**, **Precision 91.2%**, **F1 0.859**.
<p align="center">
  <img src="eda/roc_pr_curves.png" alt="ROC and Precision-Recall Curves" width="800">
  <br>
  <em><strong>Left:</strong> ROC curves. <strong>Right:</strong> Precision-Recall curves. XGBoost and the Ensemble heavily outperform the linear baseline on this severely imbalanced dataset.</em>
</p>
---

## 📁 Project Structure

```
.
├── README.md                                  # This file
├── cancer_mutation_classification.ipynb       # The complete end-to-end pipeline
├── project_report.docx                        # The comprehensive project thesis/report (ignored in git)
├── thesis_figure_generation.ipynb             # Notebook dedicated to generating report figures
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore rules
│
├── eda/                                       # Additional saved EDA figures
├── thesis_figures/                            # High-quality figures organized by chapter for the report
└── papers/                                    # Reference research papers and literature (ignored in git)
```

> **`cancer_mutation_classification.ipynb`** is the main notebook containing the entire pipeline from raw data ingestion through to final model evaluation, all in one place.
> **`thesis_figure_generation.ipynb`** is a supplementary notebook used specifically to generate high-quality visuals for the `project_report.docx`, saving them into the `thesis_figures/` directory.

---

## 🔄 Pipeline Overview

The master notebook is organised into four stages:

### Stage 1 — Data Preprocessing
- Chunk-reads the 12 GB raw COSMIC TSV in 100K-row blocks (memory-safe)
- Filters for confirmed somatic, protein-altering mutations (missense, stop-gained, frameshift, splice, in-frame)
- Merges with the Cancer Gene Census to create the binary `is_CGC` label
- Deduplicates and samples 500K rows for modelling

### Stage 2 — Exploratory Data Analysis
- Visualises class distribution (~92% passenger vs ~8% driver)
- Analyses top mutation types and most frequently mutated genes
- Identifies patterns that inform feature engineering

<p align="center">
  <img src="eda/class distribution.png" alt="Class Distribution" width="400">
  <br>
  <em>The massive class imbalance (roughly 12:1 ratio) constrained our metrics, necessitating careful threshold tuning and the use of PR-AUC.</em>
</p>

<p align="center">
  <img src="eda/top mutation.png" alt="Top Mutation Types" width="45%">
  &nbsp;
  <img src="eda/top genes.png" alt="Top Mutated Genes" width="45%">
  <br>
  <em><strong>Left:</strong> Missense mutations significantly dominate the data. <strong>Right:</strong> Massive genes like TTN accumulate passengers by sheer length, whereas tumor suppressors like TP53 are dense hot-spots for genuine drivers.</em>
</p>

### Stage 3 — Feature Engineering & Model Training
- Engineers **65 features** across five categories:
  - **Mutation type** — one-hot encoded consequence
  - **Amino-acid properties** — physicochemical class, radical-change flag, position
  - **Nucleotide substitution** — 12 SNV transitions + indel/complex
  - **Chromosomal context** — one-hot chromosome + genomic coordinate
  - **Gene-level frequency** — per-gene mutation count (leak-free)
- Trains Logistic Regression, XGBoost (with RandomizedSearchCV tuning), LightGBM, and an ensemble
- Uses 5-fold stratified cross-validation with PR-AUC as the primary metric

### Stage 4 — Evaluation & Interpretability
- Evaluates all models on a held-out 20% test set
- Optimises the decision threshold (0.50 → 0.59) to maximise F1
- Generates ROC & PR curves, confusion matrices, and SHAP feature importance plots
- Performs McNemar's test for statistical significance between models

<p align="center">
  <img src="eda/confusion_matrix_xgb_tuned.png" alt="XGBoost Confusion Matrix" width="45%">
  &nbsp;
  <img src="eda/shap_summary.png" alt="SHAP Feature Importance" width="45%">
  <br>
  <em><strong>Left:</strong> The optimally tuned XGBoost model reduces False Positives to just 618 while maintaining high recall. <strong>Right:</strong> SHAP breakdown of feature attributes showing that Gene Mutation Frequency is overwhelmingly the most predictive signal, followed by absolute genomic coordinate positions.</em>
</p>

---

## 📊 Data

| Dataset | Source | Description |
|---------|--------|-------------|
| **GenomeScreensMutant v103** | COSMIC | ~13.6 M somatic mutation records |
| **CancerGeneCensus v103** | COSMIC CGC | Known cancer driver genes (Tier 1 & 2) |

### Data Setup

The raw COSMIC files are too large (~12 GB) for GitHub and are excluded via `.gitignore`.

**Option A — Download from Kaggle (recommended):**

1. Download the preprocessed dataset from Kaggle:
   👉 **[Cancer Driver vs Passenger Somatic Mutations](https://www.kaggle.com/datasets/arjunb1204/cancer-driver-vs-passenger-somatic-mutations)**
2. Place the downloaded `gsm_clean_with_cgc.tsv` file in the `preprocessed data/` directory
3. Run `cancer_mutation_classification.ipynb` — the notebook will automatically detect the preprocessed file and skip raw data ingestion

**Option B — Run from raw data:**

1. Register for a free academic account at [COSMIC Downloads](https://cancer.sanger.ac.uk/cosmic/download)
2. Download:
   - `Cosmic_GenomeScreensMutant_v103_GRCh37.tsv`
   - `Cosmic_CancerGeneCensus_v103_GRCh37.tsv`
3. Place both files in the `raw data/` directory
4. Run all cells in `cancer_mutation_classification.ipynb` — Stage 1 will process the raw files from scratch

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Classification-of-Cancer-Driver-vs-Passenger-Mutations.git
cd Classification-of-Cancer-Driver-vs-Passenger-Mutations

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
jupyter notebook cancer_mutation_classification.ipynb
```

Run all cells top-to-bottom. If raw data is missing, the notebook will print:

```
Raw file not found: raw data/Cosmic_GenomeScreensMutant_v103_GRCh37.tsv
Skipping raw ingestion — using preprocessed data instead.
```

…and proceed automatically.



## 📚 References

The algorithms, methodologies, and definitions employed in this project are drawn from the following comprehensive literature (you can find the full PDFs organized within the `papers/` directory):

**Cancer Genomics & Biological Baselines:**
1. Sondka, Z. et al. (2018). *The COSMIC Cancer Gene Census.* Nature Reviews Cancer, 18(11).
2. Tate, J.G. et al. (2019). *COSMIC: the Catalogue Of Somatic Mutations In Cancer.* NAR, 47(D1).
3. Vogelstein, B. et al. (2013). *Cancer genome landscapes.* Science, 339(6127).
4. Bailey, M.H. et al. (2018). *Comprehensive Characterization of Cancer Driver Genes and Mutations.* Cell, 173(2).
5. Hanahan, D. & Weinberg, R. A. (2011). *Hallmarks of cancer: the next generation.* Cell, 144(5).
6. Stratton, M. R. et al. (2009). *The cancer genome.* Nature, 458(7239).
7. Alexandrov, L. B. et al. (2013). *Signatures of mutational processes in human cancer.* Nature, 500(7463).
8. Martincorena, I. et al. (2017). *Universal patterns of selection in cancer and somatic tissues.* Cell, 171(5).

**Variant Effect Classifiers & Target Identification Tools:**
9. Adzhubei, I. A. et al. (2010). *A method and server for predicting damaging missense mutations (PolyPhen-2).* Nature Methods, 7(4).
10. Ng, P. C. & Henikoff, S. (2001). *Predicting deleterious amino acid substitutions (SIFT).* Genome Research, 11(5).
11. Kircher, M. et al. (2014). *A general framework for estimating the relative pathogenicity of human genetic variants (CADD).* Nature Genetics, 46(3).
12. Carter, H. et al. (2009). *Cancer-specific high-throughput annotation of somatic mutations (CHASM).* Cancer Research, 69(16).
13. Tokheim, C. & Karchin, R. (2019). *CHASMplus reveals the scope of somatic missense mutations driving human cancers.* Cell Systems, 9(1).
14. Lawrence, M. S. et al. (2013). *Mutational heterogeneity in cancer and the search for new cancer-associated genes (MutSigCV).* Nature, 499(7457).

**Machine Learning & Statistics:**
15. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
16. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
17. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1).
18. Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics, 29(5).
19. Chawla, N. V. et al. (2002). *SMOTE: synthetic minority over-sampling technique.* JAIR, 16.
20. Lundberg, S.M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.
21. Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot.* PLoS ONE, 10(3).
22. Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12.

---

## 📄 License

This project is for academic and research purposes. COSMIC data is subject to the [COSMIC Terms of Use](https://cancer.sanger.ac.uk/cosmic/terms).
