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

The algorithms, methodologies, and definitions employed in this project are drawn from the following comprehensive literature (full PDF collection located in `papers/`):

1. Adzhubei, I. A., Schmidt, S., Peshkin, L., Ramensky, V. E., Gerasimova, A., Bork, P., Kondrashov, A. S., & Sunyaev, S. R. (2010). A method and server for predicting damaging missense mutations. *Nature Methods*, 7(4), 248-249.
2. Alexandrov, L. B., Nik-Zainal, S., Wedge, D. C., Aparicio, S. A. J. R., Behjati, S., Biankin, A. V., ... & Stratton, M. R. (2013). Signatures of mutational processes in human cancer. *Nature*, 500(7463), 415-421.
3. Bailey, M. H., Tokheim, C., Porta-Pardo, E., Sengupta, S., Bertrand, D., Weerasinghe, A., ... & Ding, L. (2018). Comprehensive characterization of cancer driver genes and mutations. *Cell*, 173(2), 371-385.
4. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
5. Carter, H., Chen, S., Isik, L., Tyekucheva, S., Velculescu, V. E., Kinzler, K. W., Vogelstein, B., & Karchin, R. (2009). Cancer-specific high-throughput annotation of somatic mutations: Computational prediction of driver missense mutations. *Cancer Research*, 69(16), 6660-6667.
6. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
7. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *In Proceedings of the 22nd ACM SIGKDD International Conference* (pp. 785-794).
8. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
9. Gonzalez-Perez, A., & Lopez-Bigas, N. (2012). Functional impact bias reveals cancer drivers. *Nucleic Acids Research*, 40(21), e169.
10. Hanahan, D., & Weinberg, R. A. (2011). Hallmarks of cancer: The next generation. *Cell*, 144(5), 646-674.
11. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
12. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
13. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
14. Kircher, M., Witten, D. M., Jain, P., O'Roak, B. J., Cooper, G. M., & Shendure, J. (2014). A general framework for estimating the relative pathogenicity of human genetic variants. *Nature Genetics*, 46(3), 310-315.
15. Lawrence, M. S., Stojanov, P., Polak, P., Kryukov, G. V., Cibulskis, K., Sivachenko, A., ... & Getz, G. (2013). Mutational heterogeneity in cancer and the search for new cancer-associated genes. *Nature*, 499(7457), 214-218.
16. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
17. Martincorena, I., Raine, K. M., Gerstung, M., Dawson, K. J., Haase, K., Van Loo, P., Davies, H., ... & Campbell, P. J. (2017). Universal patterns of selection in cancer and somatic tissues. *Cell*, 171(5), 1029-1041.
18. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
19. Ng, P. C., & Henikoff, S. (2003). SIFT: Predicting amino acid changes that affect protein function. *Nucleic Acids Research*, 31(13), 3812-3814.
20. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
21. Reva, B., Antipin, Y., & Sander, C. (2011). Predicting the functional impact of protein mutations: Application to cancer genomics. *Nucleic Acids Research*, 39(17), e118.
22. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
23. Shihab, H. A., Gough, J., Cooper, D. N., Stenson, P. D., Barker, G. L. A., Edwards, K. J., ... & Gaunt, T. R. (2013). Predicting the functional, molecular, and phenotypic consequences of amino acid substitutions using hidden Markov models. *Human Mutation*, 34(1), 57-65.
24. Sondka, Z., Bamford, S., Cole, C. G., Ward, S. A., Dunham, I., & Forbes, S. A. (2018). The COSMIC Cancer Gene Census: Describing genetic dysfunction across all human cancers. *Nature Reviews Cancer*, 18(11), 696-705.
25. Stratton, M. R., Campbell, P. J., & Futreal, P. A. (2009). The cancer genome. *Nature*, 458(7239), 719-724.
26. Tamborero, D., Gonzalez-Perez, A., & Lopez-Bigas, N. (2013). OncodriveCLUST: Exploiting the positional clustering of somatic mutations to identify cancer genes. *Bioinformatics*, 29(18), 2238-2244.
27. Tate, J. G., Bamford, S., Jubb, H. C., Sondka, Z., Beare, D. M., Bindal, N., ... & Forbes, S. A. (2019). COSMIC: The Catalogue Of Somatic Mutations In Cancer. *Nucleic Acids Research*, 47(D1), D941-D947.
28. Tokheim, C. J., Papadopoulos, N., Kinzler, K. W., Vogelstein, B., & Karchin, R. (2016). Evaluating the evaluation of cancer driver genes. *Proceedings of the National Academy of Sciences*, 113(50), 14330-14335.
29. Vogelstein, B., Papadopoulos, N., Velculescu, V. E., Zhou, S., Diaz, L. A., & Kinzler, K. W. (2013). Cancer genome landscapes. *Science*, 339(6127), 1546-1558.

---

## 📄 License

This project is for academic and research purposes. COSMIC data is subject to the [COSMIC Terms of Use](https://cancer.sanger.ac.uk/cosmic/terms).
