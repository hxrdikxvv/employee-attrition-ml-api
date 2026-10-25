# Employee Attrition Prediction Using Machine Learning

**Current status:** research/demo application using the public IBM HR Analytics Employee Attrition dataset. The current serving contract, evaluation results, and limitations are documented in [MODEL_CARD.md](MODEL_CARD.md). It is not validated for employment decisions.

> Historical note: experiment tables and metric claims below describe earlier pipeline builds, not the current deployed model. Use the generated `results/full_pipeline/tables/` artifacts and MODEL_CARD for current results.

**IBM HR Analytics Employee Attrition & Performance Dataset** — Binary classification pipeline using preprocessing, SMOTE, filter-based feature selection (VIF → MI → PCC), five ML models, hyperparameter tuning, and hybrid ensemble learning.

---

## Dataset

| Property | Value |
|---|---|
| File | `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| Records | 1,470 employees |
| Features | 35 attributes (44 after one-hot encoding) |
| Target | `Attrition` — `Yes` = left (16.12%), `No` = stayed (83.88%) |
| Problem Type | Imbalanced binary classification |

---

## Methodology

| Step | Method |
|---|---|
| 1 | Preprocessing — cleaning, encoding, 70/30 train-test split (before encoding) |
| 2 | SMOTE — balance training set (853 → 853 per class) |
| 3 | Feature Selection — VIF (44 → 24 features, threshold ≤ 5.0) |
| 4 | Feature Selection — Mutual Information (24 → 18 features, MI = 0.0 removed) |
| 5 | Feature Selection — Pearson Correlation (18 → 18, no pairs with \|r\| ≥ 0.80) |
| 6 | Model Training — Logistic Regression, Random Forest, Extra Trees, SVM, XGBoost |
| 7 | Hyperparameter Tuning — GridSearchCV, 5-fold Stratified CV, scored on ROC-AUC |
| 8 | Hybrid / Ensemble Model — Soft Voting + Stacking (RF + XGBoost + Extra Trees + SVM) |
| 9 | Final Evaluation — multi-metric scorecard, confusion matrix, ROC & PR curves |

---

## Key Results

### All 13 Models Compared

| Category | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Basic | **Logistic Regression** | 0.7188 | 0.2766 | **0.6393** | 0.3861 | **0.7538** |
| Basic | Random Forest | **0.8662** | **0.5312** | 0.2787 | 0.3656 | 0.7423 |
| Basic | XGBoost | 0.8594 | 0.4865 | 0.2951 | 0.3673 | 0.7290 |
| Basic | Extra Trees | 0.8549 | 0.4634 | 0.3115 | 0.3725 | 0.7226 |
| Basic | SVM | 0.8118 | 0.3690 | 0.5082 | 0.4276 | 0.7265 |
| Optimized | Random Forest | 0.8639 | 0.5152 | 0.2787 | 0.3617 | 0.7398 |
| Optimized | XGBoost | 0.8549 | 0.4634 | 0.3115 | 0.3725 | 0.7350 |
| Optimized | Extra Trees | 0.8481 | 0.4250 | 0.2787 | 0.3366 | 0.7297 |
| Optimized | SVM | 0.8095 | 0.2982 | 0.2787 | 0.2881 | 0.6740 |
| Hybrid | Voting Ensemble | 0.8571 | 0.4773 | 0.3443 | 0.4000 | 0.7404 |
| Hybrid | Voting Ensemble (Tuned) | 0.8503 | 0.4359 | 0.2787 | 0.3400 | 0.7305 |
| Hybrid | **Stacking Ensemble** | 0.8549 | 0.4746 | 0.4590 | **0.4667** | 0.7394 |
| Hybrid | Stacking Ensemble (Tuned) | 0.8322 | 0.3556 | 0.2623 | 0.3019 | 0.7145 |

### Final Model Selected: Baseline Logistic Regression

Selected via multi-criteria scorecard (`0.30×Recall + 0.25×F1 + 0.20×ROC-AUC + 0.15×AvgPrecision + 0.10×Accuracy`) — highest composite score (0.5723) due to best recall (63.93%) and best ROC-AUC (0.7538), and smallest train/test generalization gap (5.26 points).

> **Note on tuning:** Hyperparameter tuning did not improve results — accuracy and F1 stayed flat or decreased for every model. Ensembling helped F1/recall balance only when built from untuned base models.

---

## Evaluation Metrics

| Metric | Type |
|---|---|
| Accuracy, Precision, Recall, F1-score, ROC-AUC | Scalar |
| Confusion Matrix | Visualization (`results/figures/`) |
| ROC Curve | Visualization (`results/final_model/figures/`) |
| Precision-Recall Curve | Visualization (`results/final_model/figures/`) |

---

## Project Structure

```
Employee_Attrition_ML/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv     ← IBM HR dataset (original, unmodified)
│
├── notebooks/
│   ├── 01_Preprocessing.ipynb                     ← Cleaning, encoding, 70/30 split
│   ├── 02_SMOTE.ipynb                             ← Class balancing (training set only)
│   ├── 03_Feature_Selection_VIF.ipynb             ← Multicollinearity removal (44 → 24)
│   ├── 04_Feature_Selection_MI.ipynb              ← Mutual Information filter (24 → 18)
│   ├── 05_Feature_Selection_PCC.ipynb             ← Pearson Correlation check (18 → 18)
│   ├── 06_Model_Training.ipynb                    ← 5 base models trained & evaluated
│   ├── 07_Hyperparameter_Tuning.ipynb             ← GridSearchCV + ensemble construction
│   ├── 08_Hybrid_Model.ipynb                      ← Voting & Stacking ensembles
│   └── 09_Final_Evaluation.ipynb                  ← Multi-metric scorecard & final selection
│
├── src/
│   └── models.py                                  ← Reusable model/evaluation helpers
│
├── models/                                        ← Serialized trained models (.pkl)
│
├── results/
│   ├── figures/                                   ← Project-wide figures
│   ├── tables/                                    ← Intermediate CSVs (features, metrics)
│   ├── basic_models/                              ← Base model results
│   ├── optimized_models/                          ← Tuned model results
│   ├── hybrid_models/                             ← Ensemble model results
│   └── final_model/                               ← Final evaluation, scorecard, ROC/PR curves
│
├── scratch/                                       ← Exploratory scripts (non-official, see report §18.1)
├── README.md
└── PROJECT_GUIDELINES.md
```

---

## Reproducibility

- **Language:** Python
- **Key libraries:** pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, statsmodels, matplotlib, seaborn
- **Random seed:** `random_state=42` used consistently throughout
- **Run order:** Execute notebooks `01` → `09` in sequence; each reads from `results/tables/` and writes back to it
- **Test set policy:** The 441-row test set is evaluated exactly once per model in `09_Final_Evaluation.ipynb` — never used during preprocessing, SMOTE, feature selection, or tuning
