# 🏥 Healthcare Analytics: Predicting Hospital Readmissions for Diabetes Patients

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

This project applies machine learning to predict **30-day hospital readmissions** for diabetic patients using the [UCI ML Repository — Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

Hospital readmissions are a key quality metric in healthcare: in 2011, over **3.3 million patients** were readmitted within 30 days at a cost of ~**$41 billion**. Early identification of high-risk patients enables targeted interventions and cost reduction.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Records** | 101,766 patient encounters |
| **Features** | 50 (demographics, diagnoses, medications, lab results) |
| **Time span** | 1999–2008, 130 US hospitals |
| **Target** | `readmitted` → binary: readmitted <30 days vs. not |
| **License** | CC BY 4.0 |

### Key Features
- Patient demographics: age, race, gender
- Clinical metrics: time in hospital, number of diagnoses, lab procedures
- Medication info: insulin, metformin, diabetes medications
- HbA1c test results and blood glucose levels
- Prior admission history: outpatient, emergency, inpatient visits

---

## 🤖 Models

Three classifiers are trained and compared:

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear classifier, interpretable coefficients |
| **Random Forest** | Ensemble of decision trees, robust to noise |
| **XGBoost** | Gradient boosted trees, typically best performance |

All models use **class-weight balancing** to handle the imbalanced target distribution.

---

## 📁 Project Structure

```
healthcare-readmission/
├── data/
│   └── diabetic_data.csv         # UCI dataset (download separately)
├── src/
│   ├── preprocess.py             # Data cleaning & feature engineering
│   ├── model.py                  # Model training & evaluation
│   └── visualize.py              # EDA & result plots
├── models/
│   └── best_model.pkl            # Saved best model
├── reports/
│   └── figures/                  # All generated plots
│       ├── 01_target_distribution.png
│       ├── 02_age_distribution.png
│       ├── 03_numeric_distributions.png
│       ├── 04_correlation_heatmap.png
│       ├── 05_roc_curves.png
│       ├── 06_confusion_matrices.png
│       ├── 07_feature_importance.png
│       └── 08_model_comparison.png
├── notebooks/
│   └── diabetes_readmission_analysis.ipynb
├── main.py                       # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/healthcare-readmission.git
cd healthcare-readmission
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset

**Option A** — Using `ucimlrepo` (recommended):
```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

ds = fetch_ucirepo(id=296)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
df.to_csv('data/diabetic_data.csv', index=False)
```

**Option B** — Direct download from [UCI Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place the CSV in `data/`.

### 4. Run the full pipeline
```bash
python main.py
```

### 5. Explore the notebook
```bash
jupyter notebook notebooks/diabetes_readmission_analysis.ipynb
```

---

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | ~0.72 | ~0.38 | ~0.65 | ~0.48 | ~0.68 |
| Random Forest | ~0.70 | ~0.36 | ~0.62 | ~0.46 | ~0.66 |
| Logistic Regression | ~0.65 | ~0.34 | ~0.60 | ~0.43 | ~0.64 |

> **Note:** Recall (sensitivity) is prioritised over precision in this clinical context — it is more important to identify high-risk patients correctly than to avoid false alarms.

### Key Predictors of Readmission
1. Number of inpatient visits in prior year
2. Time in hospital
3. Number of diagnoses
4. Number of medications
5. HbA1c test result

---

## 🔑 Key Insights

- Patients with **more prior inpatient visits** are at significantly higher readmission risk
- **Longer hospital stays** correlate with higher 30-day readmission probability
- **HbA1c testing** and medication changes (insulin adjustments) are strong clinical signals
- **Older patients** (60–80 age group) have the highest readmission rates
- **XGBoost** consistently outperforms simpler models on AUC-ROC

---

## 🔬 Methodology

1. **Data Cleaning** — Handle missing values (`?`), remove duplicates per patient, drop high-cardinality/low-info features
2. **Feature Engineering** — Age bins → numeric midpoints, categorical label encoding, standard scaling
3. **Target Encoding** — Binary: readmitted within 30 days (1) vs. not (0)
4. **Class Imbalance** — Class weighting on all models
5. **Evaluation** — Stratified 80/20 train-test split, metrics: Accuracy, Precision, Recall, F1, ROC-AUC

---

## 📚 References

- Strack B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International. [doi:10.1155/2014/781670](https://doi.org/10.1155/2014/781670)
- [UCI ML Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## 📄 License

This project is licensed under the MIT License. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
