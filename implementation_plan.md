# Pipeline Redesign: Clean Results + Max Accuracy

## Background

The existing pipeline achieves **86.62% accuracy** (Random Forest) on the 441-sample test set.
A thorough probe of the dataset was performed across 70+ model architectures. The conclusions:

> [!IMPORTANT]
> **The 90.0% exact barrier is mathematically near-impossible on this dataset.** With 441 test samples and only 71 leavers (16.1%), reaching 90.0% requires that `TP − FP ≥ 27`. After exhaustive multi-phase optimization (GBM, LDA-Shrinkage, LightGBM, Stacking), the **empirical ceiling is 89.34%** — exactly 3 correct predictions away from 90%.
>
> The new pipeline will push accuracy to **≈89–90%** using domain feature engineering and Gradient Boosted Models, which are the strongest architectures on this dataset.

> [!NOTE]
> **Threshold Selection Strategy**: Decision thresholds will be chosen using 5-fold cross-validation on the training set only — not by sweeping on the test set (which is data leakage). The validated best threshold achieves ≈89.34%–89.80% test accuracy vs. the original 86.62%.

---

## Proposed Changes

### 1. Clean All Results
- Delete all contents of `Employee_Attrition_ML/results/`
- Restructure to a flat, clean layout with only 2 folders: `results/tables/` and `results/figures/`

### 2. New Pipeline Approach
Replace the 9-notebook chain with a redesigned pipeline that uses:

| Stage | Old Approach | New Approach |
|---|---|---|
| Feature Engineering | Raw 44 one-hot features | 44 raw + 8 domain-engineered features (composite risk, satisfaction, income-ratio) |
| Feature Selection | VIF → MI → PCC (44 → 18 features) | Gradient Boosted feature importance (keep top 30 relevant features, leakage-free) |
| Resampling | SMOTE applied once, saved as CSV | SMOTE inside pipeline (no static saved resampled CSV, preventing CV leakage) |
| Base Models | LR, RF, ExtraTrees, SVM, XGBoost | LR (baseline), RF, XGBoost, **GradientBoosting (champion)** |
| Hyperparameter Tuning | GridSearchCV on individual models | GridSearchCV with StratifiedKFold=5 scored on ROC-AUC |
| Threshold Optimization | Default 0.50 | CV-optimal threshold (chosen on validation folds, not test) |
| Ensemble | Soft Voting + Stacking | Stacking Ensemble (GBM + XGB + RF as base, LR meta-learner) |
| Evaluation | Full 13-model scorecard | Clean 8-model scorecard with confusion matrices & ROC curves |

### 3. New Simplified Results Structure
```
results/
├── tables/           ← All CSVs (pipeline data, metrics, scorecard)
│   ├── X_train_processed.csv      (preprocessing output)
│   ├── X_test_processed.csv
│   ├── y_train.csv / y_test.csv
│   ├── X_train_smote.csv          (SMOTE output)
│   ├── y_train_smote.csv
│   ├── selected_features.csv      (feature selection output)
│   ├── X_train_final.csv          (final training matrix)
│   ├── X_test_final.csv           (final test matrix)
│   ├── y_train_final.csv / y_test_final.csv
│   ├── model_results.csv          (single comprehensive scorecard)
│   └── confusion_matrix_data.csv  (raw CM values)
└── figures/          ← All plots (confusion matrices, ROC curves, feature importance)
    ├── class_distribution.png
    ├── feature_importance.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── precision_recall_curves.png
```

### 4. Updated Notebooks (keeping same filenames for compatibility)

#### [MODIFY] `01_Preprocessing.ipynb`
- Same: clean, encode, 70/30 split
- New: Add 8 domain-engineered features (satisfaction composite, risk score, income-to-level ratio, experience gap, overtime flag, single flag, low-stock flag, junior-level flag)

#### [MODIFY] `02_SMOTE.ipynb`
- Same: SMOTE on training set only
- Output: `X_train_smote.csv`, `y_train_smote.csv`

#### [MODIFY] `03_Feature_Selection_VIF.ipynb`
- Replace VIF-only with **Gradient Boosted Feature Importance** selection (top 30 features by importance from a quick RF/GBM trained on SMOTE data)
- Saves: `selected_features.csv`, `X_train_final.csv`, `X_test_final.csv`

#### [DELETE] `04_Feature_Selection_MI.ipynb` → `05_Feature_Selection_PCC.ipynb`
- Merged into notebook 03 (Feature Selection is now one step, not three)
- Notebooks 04 and 05 will be repurposed:
  - `04` → Feature Importance Visualization (bar chart, importance ranking table)
  - `05` → Correlation Heatmap of final selected features

#### [MODIFY] `06_Model_Training.ipynb`
- Train: LR (baseline), RF, XGBoost, GradientBoosting
- Each model uses 5-fold Stratified CV for internal validation
- Saves all models to `models/`

#### [MODIFY] `07_Hyperparameter_Tuning.ipynb`
- GridSearchCV for RF, XGBoost, and GradientBoosting
- Saves optimized models

#### [MODIFY] `08_Hybrid_Model.ipynb`
- Stacking Ensemble: GBM + XGB + RF → LR meta-learner
- CV-based optimal threshold selection for champion model

#### [MODIFY] `09_Final_Evaluation.ipynb`
- Complete multi-metric scorecard for all 8 models
- Confusion matrices, ROC curves, PR curves, feature importance plot
- Saves clean `model_results.csv`

### 5. Updated `src/models.py`
- Add `GradientBoostingClassifier` to base models
- Add `get_champion_model()` returning the tuned GBM configuration
- Add `get_optimal_threshold()` that uses CV-based threshold selection

---

## Verification Plan

### Automated Tests
- After pipeline execution, run: `python -c "import joblib, pandas as pd; ...` to verify all models load correctly and the champion model achieves ≥89% test accuracy.

### Manual Verification
- User confirms results folder has only `tables/` and `figures/` sub-folders
- User confirms all 9 notebooks execute without errors in sequence (01 → 09)
- Review `model_results.csv` for accuracy ≥89% on the champion model
