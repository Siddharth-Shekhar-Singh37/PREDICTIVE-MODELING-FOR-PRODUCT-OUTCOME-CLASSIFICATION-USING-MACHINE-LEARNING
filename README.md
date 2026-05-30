# 🤖 Predictive Modeling for Product Outcome Classification Using Machine Learning

> **Classifying product issue consequences (Injury vs. Malfunction) across 50,646 records using Lasso Regression, Logistic Regression, and KNN — with cross-validation and AUC-based model evaluation**

[![Language](https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r)](https://www.r-project.org/)
[![Models](https://img.shields.io/badge/Models-Lasso%20%7C%20Logistic%20Regression%20%7C%20KNN-orange?style=for-the-badge)]()
[![Validation](https://img.shields.io/badge/Validation-10--Fold%20Cross%20Validation-green?style=for-the-badge)]()

---

## 🔍 Project Overview

This project applies supervised machine learning to classify **product issue consequences** — predicting whether a reported product incident results in **Injury** or **Malfunction** — using a large-scale product safety dataset of 50,646 records.

The analysis follows a rigorous pipeline: exploratory data analysis, feature engineering via Lasso regularisation, binary classification using Logistic Regression and KNN, and model evaluation using accuracy, AUC, confusion matrices, and ROC curves. The dataset is segmented into three subsets to test model stability across different data contexts.

---

## 📊 Key Results at a Glance

### Lasso Regression — Feature Selection & Accuracy

| Subset | Records | Accuracy |
|---|---|---|
| Subset 1 (rows 1–16,000) | 16,000 | **80.71%** |
| Subset 2 (rows 16,001–32,000) | 16,000 | **98.29%** |
| Subset 3 (rows 32,001–50,646) | 18,646 | **97.30%** |

### Logistic Regression — Accuracy & AUC

| Subset | Accuracy | AUC |
|---|---|---|
| Subset 1 | 79.98% | 0.763 |
| Subset 2 | **99.08%** | 0.7916 |
| Subset 3 | **98.57%** | **0.918** |

### KNN Classification (k=1, 10-Fold CV)

| Subset | AUC | vs. Logistic Regression |
|---|---|---|
| Subset 1 | **0.8236** | +0.06 improvement over LR |
| Subset 2 | Matched LR | Comparable performance |

> **Key finding:** Subset 3's Logistic Regression achieved the highest AUC of **0.918**, indicating exceptional ability to discriminate between Injury and Malfunction outcomes. KNN outperformed Logistic Regression on Subset 1 by AUC, making it the preferred model where raw discrimination power is the priority.

---

## 🎯 Problem Statement

Product safety databases contain thousands of incident reports with varied consequences — from malfunctions to serious injuries. Manual classification is slow and inconsistent. This project builds a **binary classifier** to automate consequence prediction, enabling:

- Faster regulatory triage of product safety reports
- Identification of high-risk product characteristics
- Data-driven prioritisation of safety investigations

**Target variable:** `Issue_Consequence_new`
- `Injury` → encoded as **1**
- `Malfunction` (includes Death, Other, No answer provided, Malfunction) → encoded as **0**

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Total records** | 50,646 |
| **Train/Test split** | 70% / 30% (stratified) |
| **Target variable** | `Issue_Consequence_new` (binary) |
| **Key predictors** | Product codes, manufacturer location, legal announcement frequency, root cause descriptions, brand names, reporter job codes, product field descriptions |
| **Preprocessing** | Removed ID column, handled NA/zero values, converted character columns, re-levelled target variable into binary classes |

### Data Segmentation Strategy
The full dataset was divided into three subsets for Lasso analysis to manage dimensionality and test model consistency across different data segments:

```
Subset 1: rows 1      – 16,000
Subset 2: rows 16,001 – 32,000
Subset 3: rows 32,001 – 50,646
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis
- `SmartEDA::ExpCatViz()` used to visualise categorical feature distributions against the target variable
- Key EDA findings: geographic disparities in product issue frequency; manufacturer state and country strongly associated with outcome type; legal announcement frequency a significant risk indicator

### Step 2 — Target Variable Engineering
The original multi-class target (`Death`, `Injury`, `Malfunction`, `Other`, `No answer provided`) was collapsed into a **binary classification problem**:
```r
Issue_Consequence_new = case_when(
  Proudct.issue.consequence == "Injury" ~ "Injury",
  TRUE ~ "Malfunction"   # Death, Other, No answer, Malfunction → Malfunction
)
```

### Step 3 — Lasso Regression (Feature Selection)
- 10-fold cross-validation used to identify optimal lambda (`lambda.min`)
- `glmnet` with `alpha=1` (full Lasso penalty) and `family="binomial"`
- Top predictors extracted per subset — coefficients plotted for interpretability
- Key predictors identified: `last_four_years_legal_announcementing_firm_num_uniq`, `product.field_description`, `last_two_years_root_cause_description_most_freq`, `product.manufacturer_country`, `reporter_job_code`

### Step 4 — Logistic Regression
- Features from Lasso used to build parsimonious logistic regression models per subset
- Model diagnostics: VIF (multicollinearity check), residual analysis, confusion matrix
- ROC curves plotted with AUC printed for each subset
- 0.5 decision threshold applied for binary classification

### Step 5 — KNN Classification
- `caret::train()` with `method="knn"`, k values tested from 1–10
- Features pre-processed: centred and scaled
- 10-fold cross-validation applied during training
- Confusion matrix and ROC curves generated for evaluation

### Step 6 — K-Fold Cross Validation
- 10-fold CV applied to Logistic Regression across all three subsets
- Metrics reported: Accuracy, Kappa, Accuracy SD, Kappa SD
- Validates model generalisability beyond the fixed train/test split

---

## 📈 Visualisations Produced

- Lasso coefficient plots (regularisation path across lambda values)
- Top 10 predictor bar charts per subset
- Confusion matrices for both Logistic Regression and KNN
- ROC curves with printed AUC values
- Target variable distribution before and after re-levelling
- Categorical feature distributions via SmartEDA

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **R 4.2** | Core programming language |
| `glmnet` | Lasso regularisation and feature selection |
| `caret` | KNN training, cross-validation, confusion matrix |
| `pROC` | ROC curves and AUC calculation |
| `SmartEDA` | Automated exploratory data analysis |
| `tidyverse` | Data manipulation and pipeline |
| `ggplot2` | Custom visualisations |
| `caTools` | Train/test splitting utilities |
| `MASS` | Statistical modelling support |
| `car` | VIF multicollinearity diagnostics |
| `dplyr` | Data transformation |
| `kableExtra` | Publication-quality coefficient tables |
| `viridis` | Colour palettes for visualisations |

---

## 📁 Repository Structure

```
PREDICTIVE-MODELING-FOR-PRODUCT-OUTCOME-CLASSIFICATION-USING-MACHINE-LEARNING/
│
├── Advance_Analytics_and_Machine_Learning_Assignment.Rmd   # Full analysis source code
└── README.md
```

---

## 🚀 How to Reproduce

```r
# Install required packages
install.packages(c("SmartEDA", "tidyverse", "glmnet", "caTools", "MASS",
                   "car", "dplyr", "pROC", "kableExtra", "caret",
                   "viridis", "ggplot2"))

# Open the .Rmd file in RStudio
# Load your dataset via file.choose() when prompted
# Knit to Word or run chunks sequentially
```

> **Note:** The dataset is loaded interactively via `file.choose()`. Ensure you have the product safety CSV file available locally before running.

---

## 💡 Key Insights & Conclusions

1. **Lasso is a powerful feature selector** — it reduced a high-dimensional predictor space to a small set of meaningful variables, improving model interpretability without sacrificing accuracy.

2. **Legal announcement frequency is the strongest predictor** — firms with a higher number of unique legal announcements in the past four years show significantly increased association with product issue outcomes, suggesting ongoing compliance or quality challenges.

3. **Geographic origin matters** — manufacturer state and country are consistently significant predictors, pointing to regional disparities in manufacturing standards and regulatory enforcement.

4. **Logistic Regression vs KNN trade-off:**
   - Use **Logistic Regression** when interpretability is required (coefficient-level insights into predictor influence)
   - Use **KNN** when maximum discrimination accuracy is the goal (AUC of 0.8236 on Subset 1 vs 0.763 for LR)

5. **Subset 3 is the most predictable segment** — AUC of 0.918 indicates that the features in this data segment (product report codes, reporter job codes) are highly informative for classifying injury outcomes.

6. **Class imbalance awareness** — the multi-class target was deliberately collapsed to binary to address sparsity in Death and Other categories, improving model stability.

---

## 📚 References

- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society*, Series B, 58(1), 267–288.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22.

---

## 👨‍💻 Author

**Siddharth Shekhar Singh**  
Data Analyst | NI Water, Belfast  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/siddharth-shekhar-singh)

---

*This project was completed as part of an Advanced Analytics and Machine Learning assignment. The full source code is available in the .Rmd file above.*
