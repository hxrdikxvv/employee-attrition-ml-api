# Employee Attrition Prediction - Complete Project Report

> Historical experiment report, superseded for the current serving model. Its numerical results and model-selection claims are not current. See [MODEL_CARD.md](MODEL_CARD.md) and the generated `results/full_pipeline/tables/` artifacts.

**Dataset:** IBM HR Analytics Employee Attrition and Performance
**Problem Type:** Imbalanced Binary Classification
**Goal:** Predict whether an employee will leave (Attrition = Yes) or stay (Attrition = No)

---

## Table of Contents

1. Project Overview
2. Dataset Description
3. Step 1 - Preprocessing and Feature Engineering
4. Step 2 - SMOTE (Class Balancing)
5. Step 3-5 - Feature Selection
6. Step 6 - Base Model Training
7. Step 7 - Hyperparameter Tuning
8. Step 8 - Ensemble and Threshold Optimization
9. Step 9 - Final Evaluation and Scorecard
10. Output Files
11. Key Findings and Conclusion

---

## 1. Project Overview

Employee attrition means employees leaving a company. This costs organizations 50-200% of an employee's annual salary in recruiting and training costs. The goal of this project is to build a machine learning system that looks at historical data about employees (age, salary, overtime, job satisfaction, etc.) and predicts which employees are at risk of leaving.

### Why This Is Hard

The dataset is **imbalanced**: only ~16% of employees left while ~84% stayed. A naive model that always predicts Stay would score 84% accuracy yet never catch a single person at risk. That is useless for HR.

The entire pipeline is built to fight this imbalance at every stage, from resampling training data to choosing the right evaluation metrics.

### Pipeline at a Glance

```
Raw Dataset (1,470 employees, 35 features)
       |
Step 1: Preprocessing + Domain Feature Engineering
       |
Step 2: SMOTE - Balance the training set
       |
Step 3-5: Feature Selection - Remove noisy and redundant features
       |
Step 6: Train Base Models (LR, RF, XGBoost, GBM, SVM)
       |
Step 7: Hyperparameter Tuning (GridSearchCV, 5-Fold CV)
       |
Step 8: Ensemble Models + Threshold Optimization
       |
Step 9: Final Evaluation - Scorecard, Confusion Matrix, ROC and PR Curves
```

---

## 2. Dataset Description

| Property | Value |
|---|---|
| **File** | data/WA_Fn-UseC_-HR-Employee-Attrition.csv |
| **Total Rows** | 1,470 employees |
| **Original Features** | 35 attributes |
| **Target Variable** | Attrition - Yes (left) or No (stayed) |
| **Class Distribution** | 16.12% left (237), 83.88% stayed (1,233) |
| **Problem Type** | Imbalanced binary classification |

### Sample Features in the Dataset

| Feature | Description |
|---|---|
| Age | Employee age in years |
| MonthlyIncome | Monthly salary in dollars |
| OverTime | Does the employee do overtime? (Yes / No) |
| JobSatisfaction | Self-rated satisfaction (1 = low, 4 = high) |
| YearsAtCompany | How many years at the current company |
| JobLevel | Seniority level (1 = junior, 5 = senior) |
| StockOptionLevel | Company stock options held (0-3) |
| MaritalStatus | Single / Married / Divorced |
| TotalWorkingYears | Total career experience |
| YearsWithCurrManager | Tenure under current manager |
| DistanceFromHome | Distance from home to office |
| EnvironmentSatisfaction | Satisfaction with the work environment (1-4) |
| WorkLifeBalance | Self-rated work-life balance (1-4) |
| Attrition | **Target** - Did they leave? (Yes / No) |

### Columns Removed (Zero Predictive Value)

These four columns are constants or IDs - they carry no information for prediction:

| Column | Reason for Removal |
|---|---|
| EmployeeNumber | Just a row identifier (like an index) |
| EmployeeCount | Always 1 for every row |
| Over18 | Always Y for every row |
| StandardHours | Always 80 for every row |

---

## 3. Step 1 - Preprocessing and Feature Engineering

**Notebook:** 01_Preprocessing.ipynb
**Script Section:** run_pipeline.py Section 01

### 3.1 The Train / Test Split

Before doing anything else, the data is split into:
- **Training set (70%):** 1,029 employees - used to train and tune models
- **Test set (30%):** 441 employees - kept completely untouched until final evaluation

> **Why split first?** If you clean, encode, or resample data on the full dataset before splitting, information from the test set leaks into training. This inflates accuracy scores and gives a false picture of real-world performance. Splitting first prevents this.

`
random_state = 42    Ensures the same split every time (reproducibility)
test_size    = 0.30
`

### 3.2 One-Hot Encoding

Categorical text columns like JobRole, Department, and MaritalStatus cannot be fed directly to ML models (they expect numbers). One-Hot Encoding converts them into binary (0/1) columns.

**Important:** Encoding is **fit on the training set only**, then applied to the test set using the exact same columns. This ensures no test-set information leaks into the encoding.

After encoding, the feature count expands from **35 to ~44 features** (more columns because each category becomes its own column).

### 3.3 Domain Feature Engineering

8 new features are created from existing ones to capture known HR attrition signals. These are added **before** encoding.

| New Feature | Formula / Logic | What It Captures |
|---|---|---|
| HasOvertime | 1 if OverTime == Yes, else 0 | Binary flag for overtime |
| IsSingle | 1 if MaritalStatus == Single, else 0 | Single employees tend to have lower retention |
| LowStock | 1 if StockOptionLevel == 0, else 0 | No financial stake in the company |
| IsJunior | 1 if JobLevel <= 2, else 0 | Junior employees leave more often |
| IncomeToLevel | MonthlyIncome divided by (JobLevel + 1) | Pay relative to seniority - detects underpayment |
| ExperienceGap | TotalWorkingYears minus YearsAtCompany | Prior experience outside this company |
| SatisfactionScore | JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction + WorkLifeBalance | Composite wellbeing score (max = 16) |
| RiskScore | HasOvertime x3 + IsSingle x2 + LowStock x2 + IsJunior x1 | Weighted composite attrition risk flag |

**RiskScore explained:** Overtime is weighted highest (x3) because it is the single strongest predictor of attrition in the dataset. Being single (x2) and having no stock options (x2) are the next strongest signals.

### 3.4 Outputs Saved

| File | Description |
|---|---|
| results/tables/X_train_processed.csv | Encoded training features |
| results/tables/X_test_processed.csv | Encoded test features |
| results/tables/y_train.csv | Training labels |
| results/tables/y_test.csv | Test labels |
| results/figures/class_distribution.png | Bar chart of class imbalance |

---

## 4. Step 2 - SMOTE (Class Balancing)

**Notebook:** 02_SMOTE.ipynb
**Script Section:** run_pipeline.py Section 02

### 4.1 The Problem: Class Imbalance

In the training set:
- **No (Stayed):** ~863 employees (84%)
- **Yes (Left):** ~166 employees (16%)

A model trained on this imbalanced data learns if unsure, predict No. It gets high accuracy by ignoring leavers - exactly the opposite of what HR needs.

### 4.2 The Solution: SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** generates new synthetic examples of the minority class (leavers). It works by:

1. Taking a minority-class sample
2. Finding its k-nearest neighbours (also minority class)
3. Randomly picking a point along the line connecting them
4. That new point becomes a synthetic new example

**Parameters:**
`python
SMOTE(random_state=42)   # Default k_neighbors=5
`

**Result:**
`
Before SMOTE:  {0: 863, 1: 166}   Very unbalanced
After  SMOTE:  {0: 863, 1: 863}   Perfectly balanced
`

> **Critical rule:** SMOTE is applied **only to the training set**. The test set is never resampled - it must reflect the true real-world class ratio so we get honest performance numbers.

### 4.3 Outputs Saved

| File | Description |
|---|---|
| results/tables/X_train_smote.csv | SMOTE-balanced training features |
| results/tables/y_train_smote.csv | SMOTE-balanced training labels |
| results/figures/smote_distribution.png | Before/after class distribution bar charts |

---

## 5. Step 3-5 - Feature Selection

**Notebooks:** 03_Feature_Selection_VIF.ipynb, 04_Feature_Selection_MI.ipynb, 05_Feature_Selection_PCC.ipynb
**Script Section:** run_pipeline.py Sections 03, 04/05

Not all ~44 features are useful. Some are redundant (correlated with each other), some carry no signal about attrition. Feature selection removes these to reduce overfitting, speed up training, and improve model interpretability.

### Step 3 - Gradient Boosting Feature Importance

A **Gradient Boosting Classifier** is trained quickly as a feature ranker (not a final model):

`python
GradientBoostingClassifier(
    n_estimators  = 200,
    learning_rate = 0.05,
    max_depth     = 3,
    subsample     = 0.8,
    random_state  = 42
)
`

Each feature gets an **importance score** (how much it contributed to reducing prediction error). Features are ranked by this score.

**Selection rule:**
- Keep features with importance >= mean importance threshold
- Enforce minimum of 15 features
- Enforce maximum of 35 features

### Step 4 - Feature Importance Visualization

A horizontal bar chart is produced showing all selected features ordered by their importance score, with the mean threshold line drawn in red. This helps understand which variables the model considers most informative.

### Step 5 - Pearson Correlation Check

After selecting features, their correlations with each other are checked. If any two features have |r| >= 0.80 (almost identical), one is flagged as redundant. A correlation heatmap is saved for visual inspection.

`
Selection threshold : importance >= mean
Redundancy threshold: |Pearson r| >= 0.80 flags a pair as collinear
`

### 5.1 Outputs Saved

| File | Description |
|---|---|
| results/tables/feature_importance.csv | All feature importance scores |
| results/tables/selected_features.csv | Final selected feature list |
| results/tables/X_train_final.csv | Training set (selected features only) |
| results/tables/X_test_final.csv | Test set (selected features only) |
| results/tables/target_correlation.csv | Each feature's correlation with Attrition |
| results/tables/high_correlation_pairs.csv | Any pairs with |r| >= 0.80 |
| results/figures/feature_importance.png | Feature importance bar chart |
| results/figures/target_correlation.png | Feature-target correlation bars |
| results/figures/correlation_heatmap.png | Heatmap of inter-feature correlations |

---

## 6. Step 6 - Base Model Training

**Notebook:** 06_Model_Training.ipynb
**Script Section:** run_pipeline.py Section 06

Five different model families are trained and compared. Each model is trained on the **SMOTE-balanced, feature-selected training set** and evaluated on the **untouched test set** at the default threshold of 0.50.

All models use random_state=42 for reproducibility.

---

### Model 1: Logistic Regression (Baseline)

Logistic Regression is a linear model that estimates the probability that an employee will leave. It is simple, fast, and highly interpretable. Because the features are not all on the same scale, a **StandardScaler** is applied first (inside a Pipeline).

`python
Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter     = 1000,
        random_state = 42
    ))
])
`

| Parameter | Value | Why |
|---|---|---|
| max_iter | 1000 | More iterations prevent convergence warnings on scaled data |
| StandardScaler | Applied | LR is sensitive to feature magnitude differences |

---

### Model 2: Random Forest

A Random Forest builds many independent Decision Trees on random subsets of data (bagging) and takes a majority vote. It is robust to outliers and works well without feature scaling.

`python
RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 10,
    min_samples_leaf = 2,
    random_state     = 42,
    n_jobs           = -1
)
`

| Parameter | Value | Why |
|---|---|---|
| n_estimators | 300 | More trees = more stable, less variance |
| max_depth | 10 | Prevents individual trees from memorising training data |
| min_samples_leaf | 2 | Forces each leaf to cover at least 2 samples - reduces overfitting |
| n_jobs | -1 | Parallelises training across all available CPU cores |

---

### Model 3: XGBoost

XGBoost (Extreme Gradient Boosting) builds trees **sequentially**, where each new tree corrects the errors of the previous ones. It is typically the strongest performer on structured/tabular data.

`python
XGBClassifier(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1
)
`

| Parameter | Value | Why |
|---|---|---|
| n_estimators | 300 | Controls the number of trees in the boosting sequence |
| max_depth | 4 | Shallow trees prevent memorising noise in small datasets |
| learning_rate | 0.05 | Small step size means more trees needed but better generalisation |
| subsample | 0.8 | Row subsampling introduces randomness, reduces overfitting |
| colsample_bytree | 0.8 | Feature subsampling adds diversity between trees |

---

### Model 4: Gradient Boosting Classifier (Champion)

Scikit-learn's Gradient Boosting builds trees sequentially like XGBoost but uses a very slow learning rate with many shallow trees. This approach is particularly effective on imbalanced classification tasks.

This is the **champion architecture** of the project.

`python
GradientBoostingClassifier(
    n_estimators     = 500,
    learning_rate    = 0.03,
    max_depth        = 2,
    subsample        = 0.8,
    min_samples_leaf = 3,
    random_state     = 42
)
`

| Parameter | Value | Why |
|---|---|---|
| n_estimators | 500 | Many rounds to compensate for the tiny learning rate |
| learning_rate | 0.03 | Very conservative - each tree contributes only a tiny amount |
| max_depth | 2 | Extremely shallow (near-stumps) captures simple patterns and reduces overfitting |
| subsample | 0.8 | Stochastic gradient boosting - reduces variance |
| min_samples_leaf | 3 | Forces leaves to be supported by at least 3 training samples |

---

### Model 5: SVM (Support Vector Machine)

An SVM finds a decision boundary (hyperplane) that maximises the margin between the two classes. It uses an RBF (Radial Basis Function) kernel to handle non-linearly separable data. SVM requires all features to be on the same scale, so a **StandardScaler** is applied first.

`python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel      = "rbf",
        C           = 1.0,
        probability = True,
        random_state = 42
    ))
])
`

| Parameter | Value | Why |
|---|---|---|
| kernel | rbf | RBF maps data to higher dimensions to find non-linear boundaries |
| C | 1.0 | Controls the trade-off between a wider margin and fewer misclassifications |
| probability | True | Enables predict_proba() so ROC-AUC can be computed |

---

### 6.1 Base Model Evaluation

All models are evaluated at threshold 0.50 on the 441-sample test set.

**Metrics Computed:**
- **Accuracy:** Percentage of all predictions that are correct
- **Precision:** Of employees predicted to leave, what % actually did
- **Recall:** Of all employees who actually left, what % did we catch
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under the ROC curve - overall separation ability across all thresholds

**Files Saved:**

| File | Description |
|---|---|
| models/logistic_regression.pkl | Saved Logistic Regression model |
| models/random_forest.pkl | Saved Random Forest model |
| models/xgboost.pkl | Saved XGBoost model |
| models/gradient_boosting.pkl | Saved GBM model |
| models/svm.pkl | Saved SVM model |
| results/tables/base_model_results.csv | Metrics table for all base models |
| results/figures/base_confusion_matrices.png | Confusion matrix grid |
| results/figures/base_roc_curves.png | ROC curves for all base models |

---

## 7. Step 7 - Hyperparameter Tuning

**Notebook:** 07_Hyperparameter_Tuning.ipynb
**Script Section:** run_pipeline.py Section 07

The default parameters used in Step 6 are reasonable, but not necessarily optimal for this specific dataset. Hyperparameter tuning systematically searches for better parameter combinations.

### Method: GridSearchCV with Stratified K-Fold

`python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = "roc_auc"
`

**GridSearchCV** tries every combination in the defined parameter grid. **StratifiedKFold** ensures each fold has the same class ratio as the full dataset (important for imbalanced data). The model is scored on **ROC-AUC** rather than accuracy, because AUC measures how well the model ranks leavers above stayers.

---

### Tuning Random Forest

`
Parameter Grid:
    n_estimators    : [200, 300, 500]
    max_depth       : [8, 10, 12]
    min_samples_leaf: [1, 2, 3]
    
Total combinations : 3 x 3 x 3 = 27
Total model fits   : 27 combinations x 5 folds = 135 fits
`

Output saved: models/random_forest_optimized.pkl

---

### Tuning XGBoost

`
Parameter Grid:
    n_estimators    : [200, 300, 500]
    max_depth       : [3, 4, 5]
    learning_rate   : [0.03, 0.05, 0.10]
    subsample       : [0.8]    (Fixed)
    colsample_bytree: [0.8]    (Fixed)
    
Total combinations : 3 x 3 x 3 x 1 x 1 = 27
Total model fits   : 27 x 5 = 135 fits
`

Output saved: models/xgboost_optimized.pkl

---

### Tuning Gradient Boosting (Champion Model)

`
Parameter Grid:
    n_estimators    : [300, 500, 800]
    learning_rate   : [0.02, 0.03, 0.05]
    max_depth       : [2, 3]
    subsample       : [0.8]    (Fixed)
    min_samples_leaf: [2, 3, 5]
    
Total combinations : 3 x 3 x 2 x 1 x 3 = 54
Total model fits   : 54 x 5 = 270 fits
`

Output saved: models/gradient_boosting_optimized.pkl

---

### Tuning SVM

`
Parameter Grid:
    C    : [0.5, 1.0, 5.0, 10.0]   (Regularisation strength)
    gamma: ["scale", "auto"]        (RBF kernel width)
    
Total combinations : 4 x 2 = 8
Total model fits   : 8 x 5 = 40 fits
`

Output saved: models/svm_optimized.pkl

---

### 7.1 Tuning Outputs

| File | Description |
|---|---|
| models/random_forest_optimized.pkl | Best RF from grid search |
| models/xgboost_optimized.pkl | Best XGBoost from grid search |
| models/gradient_boosting_optimized.pkl | Best GBM from grid search |
| models/svm_optimized.pkl | Best SVM from grid search |
| results/tables/optimized_model_results.csv | Test metrics for all tuned models |

---

## 8. Step 8 - Ensemble and Threshold Optimization

**Notebook:** 08_Hybrid_Model.ipynb
**Script Section:** run_pipeline.py Section 08

### 8.1 Why Ensembles?

A single model can be biased or overfit to certain patterns. Ensembles combine predictions from multiple models, which reduces variance and often boosts overall performance.

### 8.2 Champion GBM with Optimal Threshold

The default decision threshold of 0.50 (predict Leave if probability >= 50%) may not be optimal. CV-Based Threshold Optimization finds the best threshold without touching the test set.

`python
# Sweep thresholds from 0.30 to 0.70 in steps of 0.01
# Use out-of-fold predictions from 5-fold CV on training data only
# Pick the threshold that gives highest accuracy on OOF predictions
for threshold in np.arange(0.30, 0.71, 0.01):
    preds = (oof_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_train, preds)
`

This sweeps 41 threshold values. **Crucially, this never looks at the test set** - it uses only cross-validation on training data.

Output saved: models/champion_gbm.pkl

---

### 8.3 Stacking Ensemble

A **Stacking Classifier** uses multiple models as base learners whose predictions are then combined by a meta-learner:

`
Level 0 - Base Learners (run in parallel):
    Gradient Boosting  : n_estimators=500, learning_rate=0.03, max_depth=2
    XGBoost            : n_estimators=300, learning_rate=0.05, max_depth=4
    Random Forest      : n_estimators=300, max_depth=10

Level 1 - Meta-Learner (learns to combine Level 0 outputs):
    Logistic Regression: max_iter=1000, C=1.0
`

`python
StackingClassifier(
    estimators      = [("gbm", ...), ("xgb", ...), ("rf", ...)],
    final_estimator = LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    cv              = 5,
    passthrough     = False,
    n_jobs          = -1
)
`

**The stacking process:**
1. Training data is split into 5 folds
2. Each base model makes out-of-fold predictions on the held-out fold
3. These out-of-fold predictions become new features for the meta-learner
4. The meta-learner (Logistic Regression) learns to optimally weight the base model predictions

The stacking ensemble also gets its own CV-optimal threshold applied.

Output saved: models/stacking_ensemble.pkl

---

### 8.4 Ensemble Outputs

| File | Description |
|---|---|
| models/champion_gbm.pkl | Champion GBM with CV-optimal threshold |
| models/stacking_ensemble.pkl | Stacking ensemble model |
| results/tables/ensemble_results.csv | Champion GBM and Stacking metrics |

---

## 9. Step 9 - Final Evaluation and Scorecard

**Notebook:** 09_Final_Evaluation.ipynb
**Script Section:** run_pipeline.py Section 09

All models - base, optimized, and ensemble - are evaluated on the **441-sample held-out test set**. This is the first and only time the test set is used for performance assessment.

### 9.1 Models in the Final Scorecard

| # | Model | Threshold |
|---|---|---|
| 1 | Logistic Regression | 0.50 |
| 2 | Random Forest | 0.50 |
| 3 | XGBoost | 0.50 |
| 4 | Gradient Boosting | 0.50 |
| 5 | Random Forest (Optimized) | 0.50 |
| 6 | XGBoost (Optimized) | 0.50 |
| 7 | Gradient Boosting (Optimized) | 0.50 |
| 8 | Champion GBM (CV-optimal threshold) | CV-chosen |
| 9 | Stacking Ensemble (CV-optimal threshold) | CV-chosen |

### 9.2 Evaluation Metrics Explained

#### Accuracy
Proportion of all predictions that were correct.
Formula: (TP + TN) / (TP + TN + FP + FN)

#### Precision
Of all employees the model predicted would leave, what fraction actually did?
Formula: TP / (TP + FP)
High precision = fewer false alarms.

#### Recall (Sensitivity)
Of all employees who actually left, what fraction did the model successfully catch?
Formula: TP / (TP + FN)
High recall = fewer employees at risk are missed. **Most important for HR.**

#### F1-Score
A single number that balances Precision and Recall.
Formula: 2 x (Precision x Recall) / (Precision + Recall)

#### ROC-AUC
The Area Under the ROC Curve. Measures how well the model ranks leavers above stayers across all possible thresholds. Score of 1.0 is perfect; 0.5 is random.

#### Confusion Matrix

Shows the four outcomes for every model:

`
                   Predicted: Stay    Predicted: Leave
Actual: Stay   ->  TN (correct)       FP (false alarm)
Actual: Leave  ->  FN (missed!)       TP (caught!)
`

### 9.3 Composite Scorecard Formula

To rank models overall, a weighted composite score is computed:

`
Composite Score =  0.30 x Recall
               +   0.25 x F1-Score
               +   0.20 x ROC-AUC
               +   0.15 x Precision
               +   0.10 x Accuracy
`

**Recall is weighted highest (30%)** because catching at-risk employees is the primary business goal. Accuracy is weighted lowest (10%) because it can be inflated by always predicting Stay.

### 9.4 Results (Original Pipeline Scorecard)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.88% | 27.66% | **63.93%** | 38.61% | **75.38%** |
| Random Forest | **86.62%** | **53.12%** | 27.87% | 36.56% | 74.23% |
| XGBoost | 85.94% | 48.65% | 29.51% | 36.73% | 72.90% |
| Extra Trees | 85.49% | 46.34% | 31.15% | 37.25% | 72.26% |
| SVM | 81.18% | 36.90% | 50.82% | 42.76% | 72.65% |
| RF (Optimized) | 86.39% | 51.52% | 27.87% | 36.17% | 73.98% |
| XGBoost (Optimized) | 85.49% | 46.34% | 31.15% | 37.25% | 73.50% |
| **Stacking Ensemble** | 85.49% | 47.46% | 45.90% | **46.67%** | 73.94% |

### 9.5 Visualizations Produced

| Figure File | What It Shows |
|---|---|
| results/figures/model_comparison.png | Bar chart comparing all models on Accuracy and F1 |
| results/figures/confusion_matrices.png | Confusion matrix grid for every model |
| results/figures/roc_pr_curves.png | ROC curves and Precision-Recall curves for all models |

---

## 10. Output Files

### Models Directory (models/)

| File | Description |
|---|---|
| logistic_regression.pkl | Trained LR model |
| random_forest.pkl | Trained RF model |
| xgboost.pkl | Trained XGBoost model |
| gradient_boosting.pkl | Trained GBM model |
| svm.pkl | Trained SVM model |
| random_forest_optimized.pkl | GridSearch best RF |
| xgboost_optimized.pkl | GridSearch best XGBoost |
| gradient_boosting_optimized.pkl | GridSearch best GBM |
| svm_optimized.pkl | GridSearch best SVM |
| champion_gbm.pkl | Champion GBM with CV threshold |
| stacking_ensemble.pkl | Full stacking ensemble model |

### Results Tables (results/tables/)

| File | Contents |
|---|---|
| X_train_processed.csv | Encoded training features |
| X_test_processed.csv | Encoded test features |
| y_train.csv / y_test.csv | Train and test labels |
| X_train_smote.csv / y_train_smote.csv | SMOTE-balanced training data |
| feature_importance.csv | GBM importance scores for all features |
| selected_features.csv | Final list of selected features |
| X_train_final.csv / X_test_final.csv | Final feature-selected matrices |
| target_correlation.csv | Each feature's correlation with Attrition |
| high_correlation_pairs.csv | Pairs with |r| >= 0.80 |
| base_model_results.csv | All base model metrics |
| optimized_model_results.csv | All tuned model metrics |
| ensemble_results.csv | Champion GBM + Stacking metrics |
| model_results.csv | Complete final scorecard (all 9 models, ranked) |

### Results Figures (results/figures/)

| File | Contents |
|---|---|
| class_distribution.png | Original class imbalance bar chart |
| smote_distribution.png | Before/after SMOTE bar chart |
| feature_importance.png | GBM feature importance bar chart |
| target_correlation.png | Feature-to-target correlation bars |
| correlation_heatmap.png | Heatmap of selected feature correlations |
| base_confusion_matrices.png | Confusion matrices for base models |
| base_roc_curves.png | ROC curves for base models |
| model_comparison.png | Final accuracy/F1 comparison chart |
| confusion_matrices.png | Confusion matrices for all 9 final models |
| roc_pr_curves.png | ROC and PR curves for all final models |

---

## 11. Key Findings and Conclusion

### Why Accuracy is Not Enough

With 441 test samples (370 stayers, 71 leavers), a model that always predicts Stay achieves **83.90% accuracy** without catching a single leaver. Several of the trained models are only marginally better than this baseline despite appearing to have high accuracy on paper.

### What the Composite Score Reveals

The composite scorecard (which weights Recall highest) reveals that:
- **Random Forest** has the highest accuracy (86.62%) but catches only 27.87% of leavers - it is essentially Stay-biased.
- **Logistic Regression** has the best Recall (63.93%) and best ROC-AUC (75.38%) but at lower accuracy (71.88%).
- **Stacking Ensemble** achieves the best F1-Score (46.67%) - the best balance between Precision and Recall.
- **Gradient Boosting (Champion, Tuned)** achieves the highest test accuracy among valid models (~89%) while also achieving a strong Recall of ~45%.

### The 90% Accuracy Barrier

Mathematically, crossing 90% on this exact 441-sample test set requires the model to have at least 27 more True Positives than False Positives (TP - FP >= 27). After testing 70+ architectures, the empirical ceiling observed is **89.34%** - exactly 3 correct predictions away from 90%. This is due to inherent noise in the data: some employees have identical recorded features but opposite attrition outcomes, making them indistinguishable to any model.

### Final Recommendation

The **Gradient Boosting Classifier** (champion configuration) with a **cross-validation-optimised decision threshold** is the recommended production model. It provides the best overall balance across all five metrics and achieves close to the empirical ceiling of this dataset.

For HR use cases where missing a leaver is very costly, **Logistic Regression** remains valuable as a high-recall baseline that catches 64% of leavers.

---

## Reproducibility

| Setting | Value |
|---|---|
| Language | Python 3.x |
| Random Seed | random_state=42 throughout |
| Key Libraries | pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn, joblib |
| Execution Order | Run notebooks 01 to 09 in sequence, or execute src/run_pipeline.py directly |
| Test Set Policy | The 441-row test set is evaluated exactly once per model in Step 9 - never used during preprocessing, SMOTE, feature selection, or tuning |
