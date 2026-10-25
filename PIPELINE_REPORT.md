# Employee Attrition Prediction Pipeline Report

> Historical experiment report, superseded for the current serving model. Its numerical results and model-selection claims are not current. See [MODEL_CARD.md](MODEL_CARD.md) and the generated `results/full_pipeline/tables/` artifacts.

## 1. Project Objective

This project builds a binary classification pipeline to predict whether an employee will leave the company (`Attrition = Yes`) or stay (`Attrition = No`). The dataset is imbalanced because only about 16% of employees left, so we use a combination of preprocessing, class balancing, feature selection, model evaluation, tuning, and hybrid ensembling.

The project uses the IBM HR Analytics Employee Attrition dataset and focuses on creating a robust, measurable prediction workflow from raw data to final model selection.

---

## 2. End-to-End Pipeline Overview

### Step 1: Data Loading and Cleaning
- Load the raw HR dataset from `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Remove non-useful columns such as `EmployeeCount`, `EmployeeNumber`, `Over18`, and `StandardHours`
- Convert target labels to binary values:
  - `Yes` -> `1`
  - `No` -> `0`
- Engineer domain-based features such as:
  - `HasOvertime`
  - `IsSingle`
  - `LowStock`
  - `IsJunior`
  - `IncomeToLevel`
  - `ExperienceGap`
  - `SatisfactionScore`
  - `RiskScore`

These engineered features encode business intuition that may improve the predictive power of the models.

### Step 2: Train/Test Split
- Use a 70/30 train-test split with `random_state=42`
- This is done before encoding and feature selection to avoid leakage
- The test set remains untouched until final evaluation

### Step 3: One-Hot Encoding
- Convert categorical variables into numerical form
- Reindex the test data against the training columns so the model sees the same feature space

### Step 4: Class Balancing with SMOTE
- Use Synthetic Minority Over-sampling Technique (SMOTE) only on the training data
- This balances the minority class and helps models learn from both classes more fairly
- Data shape after SMOTE: the training set is balanced so positive and negative classes are roughly equal

### Step 5: Feature Selection
The active implementation uses a Gradient Boosting feature-importance selector rather than a full VIF/MI/PCC chain in the final pipeline:
- Train a Gradient Boosting Classifier on SMOTE-balanced data
- Rank features by importance
- Keep the most important features above the mean-importance threshold
- Result: smaller, non-redundant feature set suitable for modeling

### Step 6: Base Model Training
A set of conventional and tree-based models are trained on the processed training data and evaluated on the holdout test set.

### Step 7: Hyperparameter Tuning
GridSearchCV with 5-fold Stratified Cross-Validation is used to tune the best-performing models using ROC-AUC as the optimization metric.

### Step 8: Hybrid / Ensemble Modeling
Three types of hybrid modeling are used:
- Soft Voting Ensemble
- Stacking Ensemble
- Additional “More Models” combinations using tree + boosting models

### Step 9: Model Evaluation
Each model is evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Probability threshold analysis

---

## 3. Techniques Used in This Pipeline

### 3.1 Data preprocessing
- Cleaning and feature engineering
- One-hot encoding for categorical variables
- Fixed random seed (`random_state=42`) for reproducibility

### 3.2 Class imbalance handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- This is used only on the training split to reduce bias toward the majority class

### 3.3 Feature selection
- Gradient Boosting feature importance based selection
- Filters the low-information features and keeps the strongest predictors

### 3.4 Model classes used
- Linear model: Logistic Regression
- Tree ensembles: Random Forest, Extra Trees
- Boosted trees: XGBoost, Gradient Boosting, LightGBM, CatBoost, AdaBoost
- Margin-based model: SVM
- Ensembling: Voting and Stacking

### 3.5 Hyperparameter optimization
- GridSearchCV
- Stratified K-Fold (5 folds)
- Scoring metric: ROC-AUC

---

## 4. Models Used in the Main Pipeline

### 4.1 Logistic Regression
Hyperparameters:
- `StandardScaler()`
- `LogisticRegression(max_iter=1000, random_state=42)`

Why used:
- Strong baseline for tabular classification
- Easy to interpret
- Good reference point for more complex models

### 4.2 Random Forest
Hyperparameters:
- `n_estimators=300`
- `max_depth=10`
- `min_samples_leaf=2`
- `random_state=42`
- `n_jobs=-1`

Why used:
- Robust ensemble with lower variance than single trees
- Handles non-linearity well

### 4.3 XGBoost
Hyperparameters:
- `n_estimators=300`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `eval_metric='logloss'`
- `random_state=42`
- `n_jobs=-1`

Why used:
- Strong gradient boosting method for structured/tabular data
- Often performs very well for classification tasks like attrition risk prediction

### 4.4 Gradient Boosting (Champion Model)
Hyperparameters:
- `n_estimators=500`
- `learning_rate=0.03`
- `max_depth=2`
- `subsample=0.8`
- `min_samples_leaf=3`
- `random_state=42`

Why used:
- Sequential boosting approach
- Helps capture subtle nonlinear patterns in the data
- This is the model selected as the champion in the project report

### 4.5 SVM (Support Vector Machine)
Hyperparameters:
- `Pipeline(StandardScaler(), SVC(probability=True, kernel='rbf', C=1.0, random_state=42))`

Why used:
- Strong for margin-based classification
- Works well when feature scaling is handled correctly

### 4.6 Extra Trees
Hyperparameters in the expanded model set:
- `n_estimators=500`
- `max_features='sqrt'`
- `min_samples_leaf=1`
- `random_state=42`
- `n_jobs=-1`

Why used:
- Ensemble of randomized decision trees
- Good for nonlinear, high-dimensional tabular data

---

## 5. Hyperparameter Tuning Search Spaces

### 5.1 Random Forest tuning
- `n_estimators`: `[200, 300, 500]`
- `max_depth`: `[8, 10, 12]`
- `min_samples_leaf`: `[1, 2, 3]`
- Scoring metric: `roc_auc`

### 5.2 XGBoost tuning
- `n_estimators`: `[200, 300, 500]`
- `max_depth`: `[3, 4, 5]`
- `learning_rate`: `[0.03, 0.05, 0.10]`
- `subsample`: `[0.8]`
- `colsample_bytree`: `[0.8]`
- Scoring metric: `roc_auc`

### 5.3 Gradient Boosting tuning
- `n_estimators`: `[300, 500, 800]`
- `learning_rate`: `[0.02, 0.03, 0.05]`
- `max_depth`: `[2, 3]`
- `subsample`: `[0.8]`
- `min_samples_leaf`: `[2, 3, 5]`
- Scoring metric: `roc_auc`

### 5.4 SVM tuning
- `C`: `[0.5, 1.0, 5.0, 10.0]`
- `gamma`: `['scale', 'auto']`
- Scoring metric: `roc_auc`

---

## 6. Hybrid / Ensemble Models Used

### 6.1 Soft Voting Ensemble
Used in the main pipeline:
- Random Forest
- XGBoost
- Extra Trees
- SVM

Voting method:
- `voting='soft'`
- Averaged predicted probabilities are used to decide the final class

### 6.2 Stacking Ensemble
Used in the main pipeline:
- Base learners:
  - Gradient Boosting
  - XGBoost
  - Random Forest
- Meta-learner:
  - Logistic Regression

Stacking configuration:
- `cv=5`
- `passthrough=False`
- `final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42)`

### 6.3 Additional “More Models” combinations
These were added as part of the extended model set:
- `Random Forest + LightGBM`
- `Extra Trees + LightGBM`
- `Extra Trees + CatBoost`
- `Random Forest + AdaBoost`

Each used soft voting to combine model probabilities.

---

## 7. Additional Models Added in the “More models” Folder

### 7.1 LightGBM
Hyperparameters:
- `objective='binary'`
- `n_estimators=500`
- `learning_rate=0.05`
- `max_depth=-1`
- `num_leaves=31`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0.1`
- `reg_lambda=1.0`
- `random_state=42`
- `n_jobs=-1`
- `verbosity=-1`

### 7.2 AdaBoost
Hyperparameters:
- `AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=300, learning_rate=0.05, random_state=42)`

### 7.3 CatBoost
Hyperparameters:
- `iterations=500`
- `learning_rate=0.05`
- `depth=6`
- `loss_function='Logloss'`
- `random_seed=42`
- `verbose=False`

### 7.4 Extra Trees
Hyperparameters:
- `n_estimators=500`
- `max_features='sqrt'`
- `min_samples_leaf=1`
- `random_state=42`
- `n_jobs=-1`

---

## 8. Metric Definitions and Interpretation

### Accuracy
Accuracy tells how many total predictions were correct.

Formula:
- Accuracy = Correct predictions / Total predictions

Use case:
- Good for a quick overall view
- Less useful when the dataset is imbalanced, because a model can look good by predicting the majority class often

### Precision
Precision tells us: among all employees predicted as likely to leave, how many actually left?

Formula:
- Precision = True Positives / (True Positives + False Positives)

Interpretation:
- High precision means fewer false alarms
- Very important in HR attrition context because false positives can cause unnecessary retention actions

### Recall
Recall tells us: among all actual leavers, how many did the model successfully catch?

Formula:
- Recall = True Positives / (True Positives + False Negatives)

Interpretation:
- High recall means the model did a better job identifying actual attrition cases
- This is often more important than raw accuracy in attrition prediction because missing a real risk is costly

### F1-score
F1-score is the harmonic mean of precision and recall.

Formula:
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

Interpretation:
- Gives a single score that balances both false alarms and missed cases
- Especially useful for imbalanced data like employee attrition prediction

### ROC-AUC
ROC-AUC measures the model’s ability to rank positive cases above negative cases across all classification thresholds.

Interpretation:
- Higher is better
- A value near 1.0 is excellent
- A value near 0.5 is weak or random

### Confusion Matrix
A confusion matrix shows:
- True Positive (correctly predicted attrition)
- False Positive (predicted attrition but stayed)
- False Negative (actual attrition but predicted stay)
- True Negative (correctly predicted stay)

This is useful for understanding whether the model is mostly missing risk cases or making too many false alarms.

### Threshold
The threshold is the probability cutoff used to convert model scores into class predictions.
- Default threshold: `0.50`
- In some tuned models, an optimized threshold is selected for better balance between recall and precision

---

## 9. Main Pipeline Results Summary

### 9.1 Base and tuned models

| Category | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Basic | Logistic Regression | 0.7188 | 0.2766 | 0.6393 | 0.3861 | 0.7538 |
| Basic | Random Forest | 0.8662 | 0.5312 | 0.2787 | 0.3656 | 0.7423 |
| Basic | XGBoost | 0.8594 | 0.4865 | 0.2951 | 0.3673 | 0.7290 |
| Basic | Extra Trees | 0.8549 | 0.4634 | 0.3115 | 0.3725 | 0.7226 |
| Basic | SVM | 0.8118 | 0.3690 | 0.5082 | 0.4276 | 0.7265 |
| Optimized | Random Forest | 0.8639 | 0.5152 | 0.2787 | 0.3617 | 0.7398 |
| Optimized | XGBoost | 0.8549 | 0.4634 | 0.3115 | 0.3725 | 0.7350 |
| Optimized | Extra Trees | 0.8481 | 0.4250 | 0.2787 | 0.3366 | 0.7297 |
| Optimized | SVM | 0.8095 | 0.2982 | 0.2787 | 0.2881 | 0.6740 |
| Hybrid | Voting Ensemble | 0.8571 | 0.4773 | 0.3443 | 0.4000 | 0.7404 |
| Hybrid | Voting Ensemble (Tuned) | 0.8503 | 0.4359 | 0.2787 | 0.3400 | 0.7305 |
| Hybrid | Stacking Ensemble | 0.8549 | 0.4746 | 0.4590 | 0.4667 | 0.7394 |
| Hybrid | Stacking Ensemble (Tuned) | 0.8322 | 0.3556 | 0.2623 | 0.3019 | 0.7145 |

### Important interpretation
- Logistic Regression had the best recall and strongest ROC-AUC among the single models
- Stacking Ensemble had the best F1 among the original project models
- Tuning did not consistently improve performance; in many cases, it made recall and F1 worse
- The best-performing original model from the project’s main pipeline was the baseline Stacking Ensemble because it balanced recall and precision best

---

## 10. Results from the Added “More models” Set

### 10.1 Results table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest + LightGBM | 0.8753 | 0.5682 | 0.4098 | 0.4762 | 0.8045 |
| LightGBM | 0.8662 | 0.5208 | 0.4098 | 0.4587 | 0.7994 |
| AdaBoost | 0.8821 | 0.6286 | 0.3607 | 0.4583 | 0.7893 |
| Extra Trees + LightGBM | 0.8730 | 0.5610 | 0.3770 | 0.4510 | 0.7829 |
| CatBoost | 0.8594 | 0.4894 | 0.3770 | 0.4259 | 0.7888 |
| Random Forest + AdaBoost | 0.8776 | 0.6061 | 0.3279 | 0.4255 | 0.7972 |
| Extra Trees + CatBoost | 0.8617 | 0.5000 | 0.3279 | 0.3960 | 0.7770 |
| Extra Trees | 0.8617 | 0.5000 | 0.2787 | 0.3579 | 0.7500 |

### 10.2 Interpretation of these extra results
- `Random Forest + LightGBM` performed best in the expanded comparison with F1 = `0.4762` and ROC-AUC = `0.8045`
- `AdaBoost` achieved the highest accuracy (`0.8821`) but lower recall, showing a trade-off between recall and precision
- `LightGBM` and `Extra Trees + LightGBM` also performed strongly, with good recall and manageable precision
- The ensemble models generally improved the F1-score relative to many standalone models

---

## 11. Which Model Should Be Preferred?

The answer depends on the business goal:

### If the goal is to catch as many real attrition risks as possible
- Prefer models with high Recall
- Logistic Regression and Stacking Ensemble performed strongly in this regard

### If the goal is balanced detection with fewer false alarms
- Prefer models with high F1-score and strong ROC-AUC
- The best expanded model was `Random Forest + LightGBM`

### If the goal is to maximize interpretability and simple deployment
- Logistic Regression remains the easiest model to explain and monitor

---

## 12. Final Conclusion

This pipeline is a complete machine learning lifecycle for employee attrition prediction:
- Data cleaning and feature engineering
- Stratified validation and class balancing with SMOTE
- Feature importance-based filtering
- Training of multiple models and hybrids
- Hyperparameter tuning using cross-validation
- Final comparison using precision, recall, F1, and ROC-AUC

The project demonstrates both classical and modern modeling approaches, including:
- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- Gradient Boosting
- SVM
- LightGBM
- AdaBoost
- CatBoost
- Voting and Stacking ensembles

The expanded results show that tree-boosting combinations, particularly `Random Forest + LightGBM`, are very competitive and often outperform the original project models on the F1-score metric.

---

## 13. Files in the Project

Relevant project files include:
- `src/models.py` — model definitions and evaluation logic
- `src/run_pipeline.py` — end-to-end execution script
- `results/` — model output tables and evaluation files
- `simple_results/` — simplified project summary files
- `More models/` — expanded models and extra results

This report summarizes the actual implementation as it exists in the project pipeline.
