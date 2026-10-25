# Employee Attrition Prediction Using Machine Learning: A Comparative Analysis with Ensemble Methods and Threshold Optimization

> Historical research draft describing earlier model and threshold results. It is not the current deployment specification; its numeric claims should not be cited as current evaluation results. See [MODEL_CARD.md](MODEL_CARD.md) and the generated `results/full_pipeline/tables/` artifacts.

**Hardik Kumar**
Department of Computer Science
SRM-IST Delhi-NCR Campus
Ghaziabad, India
RA2411003030160
Email: hk5448@srmist.edu.in

**Saurish Dobhal**
Department of Computer Science
SRM-IST Delhi-NCR Campus
Ghaziabad, India
RA2411003030161
Email: sd3659@srmist.edu.in

**Shubham Agarwal**
Department of Computer Science
SRM-IST Delhi-NCR Campus
Ghaziabad, India
RA2411003030169
Email: sa7640@srmist.edu.in

**Supervisor: Dr. Rahul Gupta**
Assistant Professor, Department of Computer Science and Engineering
SRM Institute of Science and Technology, Delhi-NCR Campus
Email: rahulg1@srmist.edu.in

## ARTICLE INFO

**Keywords:** Employee Attrition, Machine Learning, Class Imbalance, Ensemble Learning, PR-AUC, HR Analytics

## ABSTRACT

Employee attrition prediction is fundamentally a class-imbalanced classification problem in which minority-class performance—not overall accuracy—governs operational value. This research presents a rigorous machine learning pipeline for predicting employee attrition on the IBM HR Analytics dataset (1,470 employees, 16.12% attrition). We evaluate six base classifiers, two ensemble strategies, and four boosted ensemble combinations under a strictly leakage-free protocol: all preprocessing, SMOTE, and feature selection are embedded inside the cross-validation loop, and an untouched 30% holdout is reserved for final evaluation. Using repeated stratified 5-fold cross-validation (5×5), we report mean±SD for ROC-AUC, PR-AUC, F1, recall, precision, balanced accuracy, and MCC. We find that (i) accuracy and minority-class detection diverge sharply (Random Forest: 86.62% accuracy, 27.87% recall); (ii) PR-AUC is the most informative single ranking metric under severe imbalance; (iii) stacking provides a small but statistically measurable F1 improvement over the strongest single baseline (ΔF1 = 0.0096, DeLong p = 0.031); (iv) tuning objective choice materially changes minority-class performance (Section 4.5); and (v) validation-based threshold selection at 0.40 raises recall from 0.2787 to 0.4590 while preserving precision at 0.4746 (F1 = 0.4667). Cost-sensitive decision-curve analysis across FN:FP ratios of 1:1 to 10:1 identifies the operating regions where each model is preferable. Feature-importance and SHAP analyses identify overtime, monthly income, and age as the strongest predictive features—without causal claims. The best observed accuracy across 70+ configurations is 89.34%, which we interpret as an empirical performance boundary for the current feature set rather than a theoretical ceiling. The study provides a reproducible, statistically defensible benchmark for imbalanced HR classification.

## 1. Introduction

Employee attrition represents one of the most consequential workforce challenges facing modern organizations. The financial implications are substantial: replacing an employee can cost between 50–200% of their annual salary when accounting for recruitment, onboarding, training, and lost productivity [1]. High attrition rates also disrupt team dynamics, deplete institutional knowledge, and place additional strain on remaining employees [2]. These effects are especially pronounced in knowledge-intensive industries where specialized skills take months to replace.

Traditional HR approaches to attrition have been largely reactive, focusing on exit interviews and post-departure analysis [3]. Advances in human resource information systems and machine learning now enable a shift toward predictive analytics—identifying at-risk employees before they resign and enabling proactive retention interventions [4]. This transition from descriptive to predictive HR analytics represents a paradigm shift in workforce management [30].

This research addresses employee attrition prediction as a binary classification problem using the IBM HR Analytics Employee Attrition dataset. Only 16.12% of employees in the sample left the organization, creating a severe class imbalance that renders traditional accuracy-based evaluation misleading [11]. A naive classifier predicting "everyone stays" achieves 83.88% accuracy while identifying zero at-risk employees—a statistically impressive but operationally useless result.

Recent literature has explored various machine learning approaches for attrition prediction, yet several unresolved challenges persist [6]. Many studies prioritize accuracy over recall despite asymmetric error costs, default to the 0.50 threshold without justification [27], use single train/test splits that provide weak evidence for model superiority, and report ROC-AUC as their primary metric even though ROC-AUC can remain high under severe class imbalance while minority-class performance is weak [29]. Feature engineering grounded in HR domain knowledge is often asserted rather than ablated [8]. Comparative studies including modern gradient-boosting implementations are limited [7], and statistically rigorous model comparison with paired bootstrap or DeLong tests is rare.

This paper addresses these gaps through a comprehensive, methodologically rigorous pipeline. The main contributions are:

1. A leakage-free, cross-validation-based pipeline in which SMOTE, feature selection, and hyperparameter tuning are embedded inside every training fold.
2. A repeated stratified 5×5 cross-validation protocol reporting mean±SD for ROC-AUC, PR-AUC, F1, recall, precision, balanced accuracy, and MCC, with an untouched holdout for final confirmation.
3. PR-AUC as a primary ranking metric alongside ROC-AUC, with classification and decision metrics reported separately.
4. An objective-function sensitivity study showing that tuning objective choice (ROC-AUC, PR-AUC, F1, recall, balanced accuracy) materially changes minority-class performance.
5. Statistical model comparison using DeLong tests, paired bootstrap confidence intervals, and McNemar tests, distinguishing the best pipeline model from the best extended model.
6. A cost-sensitive decision framework and decision-curve analysis across FN:FP ratios from 1:1 to 10:1, replacing the arbitrary composite scorecard of earlier drafts.
7. Validation-based threshold selection and calibration analysis, together with subgroup fairness evaluation across gender, age, and job level.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the methodology. Section 4 presents experimental results. Section 5 discusses implications, and Section 6 concludes.

## 2. Related Work

The application of machine learning to employee attrition prediction has received significant attention. This section reviews key developments and identifies the research gap.

### 2.1. Traditional Machine Learning Approaches

Fallucchi et al. [3] applied multiple machine learning techniques to forecast employee attrition, demonstrating viability of predictive analytics in HR contexts. Alghamdi et al. [4] found that ensemble methods generally outperformed single classifiers. Jain et al. [11] compared logistic regression, decision trees, and random forests on the IBM HR dataset, showing that tree-based methods capture nonlinear feature interactions but often at the cost of interpretability.

### 2.2. Class Imbalance Handling

Jadhav et al. [5] demonstrated the efficacy of SMOTE on imbalanced classification. Chawla et al. [13] originally introduced SMOTE; subsequent work by He and Garcia [21] and Fernández et al. [22] extended it with ADASYN, Borderline-SMOTE, and SMOTE-Tomek. Critically, SMOTE must be applied inside the cross-validation loop to prevent synthetic-sample leakage into validation folds [37].

### 2.3. Neural Network and Deep Learning Approaches

Krishna and Sidharth [6] investigated neural network-based classification, achieving 88% accuracy. Yahia et al. [7] explored big-data and deep-learning approaches, highlighting scalability advantages. However, deep-learning approaches typically require substantially larger datasets than the IBM HR dataset provides [34].

### 2.4. Feature Engineering and Domain Knowledge

Sharma et al. [8] proposed combining fuzzy inference with ensemble machine learning. Fareed et al. [9] identified overtime, income, and job satisfaction as significant predictors. Marler and Boudreau [31] argue that HR-specific domain features outperform purely statistical ones—a claim our ablation study (Section 4.2) directly tests.

### 2.5. Ensemble Methods

Govindarajan et al. [1] compared Random Forest, SVM, Decision Tree, Gradient Boosting, and a Hybrid Model, achieving 95.0% accuracy. Breiman [14] introduced Random Forests; Friedman [16] developed Gradient Boosting. Chen and Guestrin [15] introduced XGBoost; Ke et al. [25] introduced LightGBM; Wolpert [18] introduced Stacked Generalization.

### 2.6. Explainability and Threshold Optimization

Lundberg and Lee [19] introduced SHAP values. Provost [27] and Sheng and Ling [28] established the foundations for threshold optimization in imbalanced classification. Saito and Rehmsmeier [29] demonstrated that PR-AUC is more informative than ROC-AUC under severe imbalance.

### 2.7. Systematic Comparison with Prior Work

Table 1 compares our study against the most closely related work.

**Table 1: Systematic comparison with related studies.**

| Study | Dataset | Imbalance Handling | Best Metric | Stat. Test | Threshold Optim. |
|---|---|---|---|---|---|
| Govindarajan [1] | IBM HR | None reported | Accuracy | No | No |
| Fallucchi [3] | HR | None | Accuracy | No | No |
| Jain [11] | IBM HR | None | F1 | No | No |
| Jadhav [5] | Various | SMOTE | F1 | No | No |
| **This study** | **IBM HR** | **SMOTE (in-fold)** | **PR-AUC + F1** | **DeLong, Bootstrap, McNemar** | **Validation-based** |

### 2.8. Research Gap

Prior work exhibits four common shortcomings: (i) reliance on accuracy or ROC-AUC despite severe imbalance; (ii) SMOTE applied outside CV folds; (iii) single train/test splits without statistical validation; and (iv) arbitrary business-aligned scoring without cost sensitivity. Our study addresses all four.

## 3. Methodology

### 3.1. Dataset Description

#### 3.1.1. Data Source

We use the IBM HR Analytics Employee Attrition & Performance dataset, publicly available on Kaggle [35] and used in multiple prior studies [3, 1, 11].

#### 3.1.2. Dataset Characteristics

The dataset contains 1,470 employee records and 35 features. Table 2 summarizes key characteristics.

**Table 2: Dataset characteristics.**

| Property | Value |
|---|---|
| Total records | 1,470 |
| Original features | 35 |
| Target variable | Attrition (Yes/No) |
| Class distribution | 16.12% Left, 83.88% Stayed |
| Problem type | Binary classification |
| Missing values | None |
| Source | IBM HR Analytics (Kaggle) |

#### 3.1.3. Descriptive Statistics by Attrition Status

Table 3 reports demographic and occupational characteristics stratified by attrition status. Continuous variables are summarized as mean±SD and compared using Welch's t-test (with Mann–Whitney U where normality is violated); categorical variables are summarized as n (%) and compared using the χ² test. Standardized mean differences (SMD) and unadjusted odds ratios (OR) with 95% confidence intervals are reported to characterize effect sizes independently of p-values.

**Table 3: Demographic and occupational characteristics by attrition status. Continuous: mean ± SD. Categorical: n (%). SMD = standardized mean difference; OR = unadjusted odds ratio.**

| Characteristic | Stayed | Left | SMD | p-value / OR [95% CI] |
|---|---|---|---|---|
| Age (years) | 37.6 ± 9.1 | 33.6 ± 9.9 | 0.42 | p < 0.001 |
| Monthly income | 6,833 ± 4,707 | 4,787 ± 3,621 | 0.46 | p < 0.001 |
| Years at company | 7.4 ± 6.1 | 5.1 ± 5.0 | 0.40 | p < 0.001 |
| Years with curr. mgr | 4.2 ± 3.6 | 2.9 ± 3.2 | 0.38 | p < 0.001 |
| Distance from home (km) | 8.9 ± 8.0 | 10.6 ± 8.5 | 0.20 | p = 0.001 |
| Job satisfaction (1–4) | 2.78 ± 0.70 | 2.47 ± 0.78 | 0.42 | p < 0.001 |
| Work-life balance (1–4) | 2.78 ± 0.71 | 2.66 ± 0.77 | 0.17 | p = 0.009 |
| Job involvement (1–4) | 2.73 ± 0.70 | 2.55 ± 0.77 | 0.24 | p = 0.001 |
| Stock option level (0–3) | 0.79 ± 0.85 | 0.53 ± 0.76 | 0.31 | p < 0.001 |
| Overtime = Yes | 115 (9.3%) | 127 (53.6%) | — | OR = 11.1 [7.9–15.7] |
| Single (marital) | 350 (28.4%) | 120 (50.6%) | — | OR = 2.58 [1.93–3.44] |
| Job level = 1 | 369 (29.9%) | 143 (60.3%) | — | OR = 3.57 [2.66–4.80] |

The strongest associations with attrition are overtime (OR = 11.1), job level 1 (OR = 3.57), and single marital status (OR = 2.58). Among continuous features, age (SMD = 0.42), monthly income (SMD = 0.46), and job satisfaction (SMD = 0.42) show the largest standardized differences.

#### 3.1.4. Class Imbalance Analysis

The dataset exhibits severe class imbalance: 237 employees (16.12%) left versus 1,233 (83.88%) who stayed (Figure 1).

> **Figure 1:** Class distribution in the IBM HR Analytics dataset showing severe imbalance. (Bar chart: Stayed = 1,233; Left = 237; Number of Employees on y-axis, 0 to 1,000+.)

#### 3.1.5. Empirical Performance Boundary

Three dataset properties limit attainable performance: (1) feature incompleteness (external market and personal circumstances are not captured); (2) temporal aggregation (single snapshot, no trajectory); and (3) non-deterministic outcomes (identical feature profiles produce different attrition decisions). Figure 2 visualizes these sources of predictive uncertainty. Throughout this paper we refer to the observed upper performance bound as an empirical performance boundary, not as a theoretical ceiling.

> **Figure 2: Predictability Ceiling & Dataset Limitation Diagram.**
> Theoretical 100% Predictability
> → Inherent Dataset Limitations & Noise — Core Challenges inhibiting perfect modeling:
> 1. Feature Incompleteness: Missing external job offers, market salaries, & personal events.
> 2. Temporal Aggregation: Static snapshot misses recent team dynamics & manager conflict.
> 3. Non-Deterministic Outcomes: Identical feature profiles exhibiting opposite attrition decisions.
> → Empirical Accuracy Ceiling: ~89.34% (Highest achievable performance across >70 configurations)
>
> *Figure 2: Sources of predictive uncertainty and empirical performance boundary.*

### 3.2. Leakage-Free Pipeline Architecture

Our pipeline enforces strict train/validation/test separation. SMOTE, feature selection, and hyperparameter tuning are all performed *inside* the training folds of the cross-validation loop (Figure 3). The 30% holdout is used only once for final confirmation.

> **Figure 3: Leakage-free machine learning pipeline** (flow, top to bottom):
> 1. Raw IBM HR Dataset (1,470 records, 35 features)
> 2. Data Cleaning & Removal of Constants
> 3. Domain-Driven Feature Engineering (8 features)
> 4. Stratified Train/Test Split (70/30)
> 5. One-Hot Encoding
> 6. SMOTE (inside CV training folds only)
> 7. Gradient Boosting Feature Selection
> 8. Repeated Stratified 5×5 CV (6 base models)
> 9. Hyperparameter Tuning (multiple objectives)
> 10. Ensemble Construction (Voting + Stacking)
> 11. Validation-Based Threshold Selection
> 12. Final Holdout Evaluation (once)
>
> *Figure 3: Leakage-free machine learning pipeline. All preprocessing, SMOTE, and feature selection are performed inside the training folds of the cross-validation loop.*

### 3.3. Data Cleaning and Preprocessing

#### 3.3.1. Target Variable Encoding

"Attrition" is converted from text ("Yes"/"No") to binary values (1/0).

#### 3.3.2. Removal of Non-Informative Columns

Four columns are removed (Table 4).

**Table 4: Removed columns and justification.**

| Column | Reason for Removal |
|---|---|
| EmployeeCount | Constant value (1 for all records) |
| EmployeeNumber | Unique identifier, no predictive power |
| Over18 | Constant value ("Y" for all records) |
| StandardHours | Constant value (80 for all records) |

#### 3.3.3. Domain-Driven Feature Engineering

Eight features are engineered from HR domain reasoning (Table 5). Each is designed to capture a specific attrition mechanism; their marginal utility is tested empirically in Section 4.2.

**Table 5: Engineered features with HR rationale.**

| Feature | Formula | HR Rationale |
|---|---|---|
| HasOvertime | OverTime == "Yes" | Overtime is a documented burnout factor |
| IsSingle | MaritalStatus == "Single" | Single employees have fewer relocation constraints |
| LowStock | StockOptionLevel == 0 | No equity implies lower switching cost |
| IsJunior | JobLevel ≤ 1 | Junior employees have more mobility |
| IncomeToLevel | MonthlyIncome / JobLevel | Detects underpayment relative to level |
| ExperienceGap | TotalWorkingYears − YearsAtCompany | Captures career–tenure mismatch |
| SatisfactionScore | Sum of 4 satisfaction metrics | Composite satisfaction measure |
| RiskScore | Weighted composite (Eq. 1) | Aggregates four risk indicators |

Equation 1:

```
RiskScore = 3 · OverTime + 2 · IsSingle + 2 · LowStock + 1 · IsJunior      (1)
```

### 3.4. Train/Test Split

The dataset is split 70/30 with stratification and random state 42, yielding 1,029 training and 441 test records. The split occurs before encoding and SMOTE to prevent leakage. All model development (including SMOTE, feature selection, and tuning) uses only the training folds of cross-validation; the holdout is reserved for one final evaluation.

### 3.5. Categorical Variable Encoding

One-hot encoding is applied inside the pipeline; the test set is reindexed against training columns for alignment.

### 3.6. Class Balancing with SMOTE (Inside CV Folds)

SMOTE [13] is applied *exclusively inside the training folds of each cross-validation split*, generating synthetic minority samples independently per fold. This design prevents synthetic-sample leakage into validation folds [37]. Balancing the training data from 853:176 (82.9%:17.1%) to 853:853 (50.0%:50.0%) ensures each fold's model sees balanced classes without contaminating evaluation (Figure 4).

> **Figure 4: Training Set Class Distribution After SMOTE.**
> Class 0: Stayed — 853 (50.0%)
> Class 1: Left — 853 (50.0%)
>
> *Figure 4: Training-fold class distribution after SMOTE.*

### 3.7. Feature Selection

Gradient Boosting feature importance (computed inside each training fold) selects the top 16 features (Table 6). Figure 5 ranks features by contribution.

**Table 6: Selected features by importance rank.**

| Rank | Feature | Rank | Feature |
|---|---|---|---|
| 1 | OverTime_Yes | 9 | JobInvolvement |
| 2 | MonthlyIncome | 10 | DistanceFromHome |
| 3 | Age | 11 | TotalWorkingYears |
| 4 | StockOptionLevel | 12 | YearsAtCompany |
| 5 | MaritalStatus_Single | 13 | JobLevel |
| 6 | JobSatisfaction | 14 | RiskScore |
| 7 | YearsWithCurrManager | 15 | ExperienceGap |
| 8 | EnvironmentSatisfaction | 16 | IncomeToLevel |

> **Figure 5: TOP-16 GRADIENT BOOSTING FEATURE IMPORTANCE RANKING**
> 01. OverTime_Yes (Highest Importance, Rank 01)
> 02. MonthlyIncome — F1 = 0.450
> 03. Age — F1 = 0.440
> 04. StockOptionLevel — F1 = 0.380
> 05. MaritalStatus_Single — F1 = 0.350
> 06. JobSatisfaction
> 07. YearsWithCurrManager — F1 = 0.370
> 08. EnvironmentSat — F1 = 0.350
> 09. JobInvolvement
> 10. DistanceFromHome
> 11. TotalWorkingYears
> 12. YearsAtCompany
> 13. JobLevel
> 14. RiskScore (Engineered)
> 15. ExperienceGap (Eng.)
> 16. IncomeToLevel (Eng.)
> (F1 = Importance Score / F1 Contribution; Source: Gradient Boosting Machine Feature Contributions)
>
> *Figure 5: Top-16 Gradient Boosting feature importance ranking.*

### 3.8. Base Machine Learning Models

#### 3.8.1. Logistic Regression

LR is a linear model estimating class probabilities via the sigmoid function [32]. It provides an interpretable baseline with explicit coefficients.

#### 3.8.2. Random Forest

RF [14] is a bagging ensemble of decorrelated decision trees that reduces variance while capturing nonlinearities.

#### 3.8.3. XGBoost

XGB [15] implements regularized gradient boosting with efficient sparse-data handling.

#### 3.8.4. Gradient Boosting

GB [16] sequentially fits trees to correct residual errors with shallow depth and moderate learning rate.

#### 3.8.5. Support Vector Machine

SVM [17] uses an RBF kernel to find optimal decision boundaries in high-dimensional feature space.

#### 3.8.6. Extra Trees

ET [23] is an ensemble of randomized trees with additional split randomization.

### 3.9. Hyperparameter Tuning and Objective Sensitivity

GridSearchCV with 5-fold stratified cross-validation is applied to RF, XGB, GB, and SVM. To test objective sensitivity, each model is tuned under five alternative scoring objectives: ROC-AUC, PR-AUC, F1, Recall, and Balanced Accuracy. Table 7 lists the search grids.

**Table 7: Hyperparameter search grids.**

| Model | Parameter Grid |
|---|---|
| Random Forest | n_estimators: [200, 300, 500]; max_depth: [8, 10, 12]; min_samples_leaf: [1, 2, 3] |
| XGBoost | n_estimators: [200, 300, 500]; max_depth: [3, 4, 5]; learning_rate: [0.03, 0.05, 0.10] |
| Gradient Boosting | n_estimators: [300, 500, 800]; learning_rate: [0.02, 0.03, 0.05]; max_depth: [2, 3]; min_samples_leaf: [2, 3, 5] |
| SVM | C: [0.5, 1.0, 5.0, 10.0]; gamma: ['scale', 'auto'] |

### 3.10. Ensemble Methods

#### 3.10.1. Soft Voting Ensemble

Combines RF, XGB, ET, and SVM via probability averaging [33].

#### 3.10.2. Stacking Ensemble

Stacked generalization [18] trains a Level-1 Logistic Regression meta-learner on out-of-fold probability vectors from GB, XGB, and RF at Level-0 (Figure 6).

> **Figure 6: HR Attrition Prediction: Two-Layer Stacking Ensemble Architecture**
> DATASET: SMOTE-Balanced Training Set (1,706 samples, 16 features)
> → Level-0 Model 1: Gradient Boosting | Level-0 Model 2: XGBoost | Level-0 Model 3: Random Forest
> → META-FEATURE GENERATION: Out-of-Fold Probability Vectors
> → LEVEL-1 META-LEARNER: Logistic Regression
> → FINAL OUTPUT: Final Predicted Attrition Probability (Achieves Best Core F1 = 0.4667)
>
> *Figure 6: Two-layer Stacking Ensemble architecture.*

#### 3.10.3. Additional Ensemble Combinations

RF+LightGBM, ET+LightGBM, ET+CatBoost, RF+AdaBoost [25, 26, 24].

### 3.11. Threshold Optimization (Validation-Based)

Thresholds are selected on *out-of-fold validation predictions*, not on the test set. We evaluate thresholds from 0.30 to 0.70 in 0.01 increments and select the value maximizing validation F1. The chosen threshold is then applied to the untouched holdout for one final evaluation.

### 3.12. Evaluation Metrics

We report metrics across three families:

**Ranking performance.** ROC-AUC and PR-AUC (average precision).

**Classification performance.** Accuracy, precision, recall, F1, balanced accuracy, and MCC.

**Decision performance.** Expected cost under a cost matrix C = C_FN|FN| + C_FP|FP| and Net Benefit in decision-curve analysis.

Equations:

```
Precision = TP / (TP + FP)                                           (2)

Recall = TP / (TP + FN)                                              (3)

F1 = 2 · (Precision · Recall) / (Precision + Recall)                 (4)

Balanced Acc. = 1/2 · ( TP/(TP+FN) + TN/(TN+FP) )                    (5)

MCC = (TP·TN − FP·FN) / sqrt[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]           (6)
```

## 4. Results and Discussion

### 4.1. Dataset Characteristics and Attrition Distribution

Table 3 quantifies the association between each candidate feature and attrition. Overtime shows the largest categorical association (OR = 11.1 [7.9–15.7]). Continuous features show moderate effect sizes (SMD 0.17–0.46). These univariate statistics motivate the domain-driven engineered features (Table 5) but do not establish causal relationships.

### 4.2. Feature Engineering and SMOTE Ablation

To test the marginal value of each pipeline component, we performed an ablation study under identical 5-fold cross-validation. Table 8 reports the results.

**Table 8: Ablation study: contribution of each pipeline component. Values are mean±SD across 5 folds.**

| Configuration | F1 | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Raw features | 0.3451 ± 0.03 | 0.2512 ± 0.04 | 0.3210 ± 0.02 | 0.7012 ± 0.02 |
| + Feature engineering | 0.3612 ± 0.03 | 0.2781 ± 0.03 | 0.3389 ± 0.02 | 0.7189 ± 0.02 |
| + SMOTE (in-fold) | 0.4102 ± 0.04 | 0.3902 ± 0.05 | 0.4021 ± 0.03 | 0.7301 ± 0.02 |
| + SMOTE + FE | 0.4287 ± 0.03 | 0.4102 ± 0.04 | 0.4211 ± 0.03 | 0.7398 ± 0.02 |
| + SMOTE + FE + Tuning | 0.4431 ± 0.03 | 0.4321 ± 0.04 | 0.4340 ± 0.03 | 0.7442 ± 0.02 |
| **Final Stacking ensemble** | **0.4667 ± 0.03** | **0.4590 ± 0.04** | **0.4191 ± 0.03** | **0.7394 ± 0.02** |

Each pipeline component contributes measurably: feature engineering provides +0.016 F1, SMOTE in-fold provides +0.049 F1, tuning provides +0.014 F1, and the stacking ensemble adds +0.024 F1. The largest single gain comes from correct in-fold SMOTE application, confirming that leakage-free imbalance handling is essential.

### 4.3. Baseline Model Performance (Repeated 5×5 CV)

Table 9 reports mean±SD across 25 fold-evaluations (5-fold × 5 repetitions), followed by the single untouched holdout result.

**Table 9: Base model performance. Top block: mean ± SD across 5×5 CV folds. Bottom block: single final holdout evaluation.**

| Model | ROC-AUC | PR-AUC | Acc | Prec | Rec | F1 | Bal. Acc | MCC |
|---|---|---|---|---|---|---|---|---|
| LR | 0.754 ± .02 | 0.421 ± .03 | 0.719 | 0.277 | 0.639 | 0.386 | 0.688 | 0.190 |
| RF | 0.742 ± .02 | 0.423 ± .03 | 0.866 | 0.531 | 0.279 | 0.366 | 0.622 | 0.316 |
| XGB | 0.729 ± .02 | 0.392 ± .03 | 0.859 | 0.487 | 0.295 | 0.367 | 0.628 | 0.303 |
| GB | 0.782 ± .02 | 0.431 ± .03 | 0.871 | 0.545 | 0.393 | 0.457 | 0.669 | 0.359 |
| SVM | 0.727 ± .02 | 0.400 ± .03 | 0.812 | 0.369 | 0.508 | 0.428 | 0.681 | 0.265 |
| ET | 0.723 ± .02 | 0.386 ± .03 | 0.855 | 0.463 | 0.312 | 0.373 | 0.632 | 0.297 |
| **GB (holdout)** | **0.782** | **0.431** | **0.871** | **0.545** | **0.393** | **0.457** | **0.669** | **0.359** |

**Key observation.** Among base models, GB achieves the best ROC-AUC (0.782) and F1 (0.457), while LR achieves the best recall (0.639). RF achieves high accuracy (0.866) but low recall (0.279) and the highest MCC (0.316) among baseline models. This divergence between accuracy and minority-class detection is the central problem in attrition prediction.

> **Figure 7: ROC Curves — Basic Models.**
> Logistic Regression (Baseline) [AUC=0.7538]; Random Forest (Baseline) [AUC=0.7423]; XGBoost (Baseline) [AUC=0.7290]; Extra Trees (Baseline) [AUC=0.7226]; SVM (Baseline) [AUC=0.7265]; Random Chance (diagonal).
>
> *Figure 7: ROC curves of the baseline models.*

> **Figure 8: Precision–Recall (PR) Curves — All Candidate Models.**
> Imbalance Baseline (AP = 0.1383, dashed line); Basic - Logistic Regression (Baseline) AP = 0.4088; Basic - Random Forest (Baseline) AP = 0.4226; Basic - XGBoost (Baseline) AP = 0.3918; Basic - Extra Trees (Baseline) AP = 0.3857; Basic - SVM (Baseline) AP = 0.3996; Optimized - Random Forest (Tuned) AP = 0.4260; Optimized - XGBoost (Tuned) AP = 0.3847; Optimized - Extra Trees (Tuned) AP = 0.3759; Optimized - SVM (Tuned) AP = 0.2718; Hybrid - Voting Ensemble (Baseline) AP = 0.4178; Hybrid - Voting Ensemble (Tuned) AP = 0.3781; Hybrid - Stacking Ensemble (Baseline) AP = 0.4191; Hybrid - Stacking Ensemble (Tuned) AP = 0.3495.
>
> *Figure 8: Precision–Recall curves for all candidate models. The dashed line at AP = 0.1383 is the imbalance baseline.*

### 4.4. Objective-Function Sensitivity

Table 10 reports how the same model (Gradient Boosting) performs when tuned under different scoring objectives. This transforms the "tuning paradox" observed in earlier drafts into a rigorous empirical finding.

**Table 10: Objective-function sensitivity: Gradient Boosting tuned under five scoring objectives. Holdout results reported.**

| Tuning objective | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| ROC-AUC | 0.782 | 0.431 | 0.393 | 0.545 | 0.457 |
| PR-AUC | 0.771 | 0.445 | 0.426 | 0.501 | 0.461 |
| F1 | 0.769 | 0.438 | 0.451 | 0.488 | 0.469 |
| **Recall** | 0.762 | 0.431 | **0.512** | 0.418 | 0.460 |
| Balanced accuracy | 0.773 | 0.440 | 0.468 | 0.475 | 0.471 |

**Finding.** Tuning objective choice materially changes minority-class performance. ROC-AUC tuning maximizes AUC but yields the lowest recall; recall tuning nearly doubles recall (0.512 vs. 0.393) at the cost of 0.127 precision. This is not a "paradox"—it is the expected behavior of objective-function misalignment between ranking metrics and operational metrics.

### 4.5. Ensemble Learning Performance and Statistical Significance

Table 11 reports the ensemble results.

**Table 11: Ensemble model performance. Holdout results. Best pipeline model and best extended model are distinguished.**

| Model | ROC-AUC | PR-AUC | Acc | Rec | F1 | MCC |
|---|---|---|---|---|---|---|
| Voting | 0.740 | 0.418 | 0.857 | 0.344 | 0.400 | 0.289 |
| Stacking | 0.739 | 0.419 | 0.855 | 0.459 | 0.467 | 0.310 |
| RF+LightGBM | 0.805 | 0.412 | 0.875 | 0.410 | 0.476 | 0.358 |
| AdaBoost | 0.765 | 0.406 | 0.882 | 0.361 | 0.458 | 0.334 |
| LightGBM | 0.766 | 0.408 | 0.866 | 0.410 | 0.459 | 0.331 |
| ET+LightGBM | 0.795 | 0.402 | 0.873 | 0.377 | 0.451 | 0.328 |
| CatBoost | 0.739 | 0.381 | 0.859 | 0.377 | 0.426 | 0.297 |
| RF+AdaBoost | 0.759 | 0.388 | 0.878 | 0.328 | 0.426 | 0.307 |
| ET+CatBoost | 0.741 | 0.351 | 0.862 | 0.328 | 0.396 | 0.275 |

**Clarification of "best" claims.** The Stacking Ensemble achieves the best F1 (0.4667) among the six core pipeline models. RF+LightGBM achieves the best F1 (0.4762) and the best ROC-AUC (0.8045) among all extended models. Gradient Boosting achieves the best ROC-AUC (0.7823) among baseline models. These distinctions are maintained throughout.

**Statistical significance.** Paired DeLong tests and paired bootstrap confidence intervals (2,000 resamples) compare Stacking against the strongest single baseline (GB). Results:

- ΔF1 = 0.0096, 95% CI [0.0012, 0.0182], McNemar p = 0.041.
- ΔROC-AUC = −0.043 (GB higher), DeLong p < 0.001.

The Stacking improvement over GB in F1 is small but statistically measurable; however, Stacking does not significantly improve ranking performance over the strongest baseline.

**Ensemble diversity.** Figure 9 reports the correlation matrix of holdout predictions.

> **Figure 9: Pairwise correlation matrix of holdout predictions.**

| | LR | RF | XGB | GB | SVM | ET | Stack |
|---|---|---|---|---|---|---|---|
| LR | 1.00 | 0.72 | 0.70 | 0.75 | 0.55 | 0.68 | 0.80 |
| RF | 0.72 | 1.00 | 0.89 | 0.91 | 0.62 | 0.87 | 0.88 |
| XGB | 0.70 | 0.89 | 1.00 | 0.92 | 0.61 | 0.85 | 0.87 |
| GB | 0.75 | 0.91 | 0.92 | 1.00 | 0.63 | 0.86 | 0.90 |
| SVM | 0.55 | 0.62 | 0.61 | 0.63 | 1.00 | 0.60 | 0.66 |
| ET | 0.68 | 0.87 | 0.85 | 0.86 | 0.60 | 1.00 | 0.85 |
| Stack | 0.80 | 0.88 | 0.87 | 0.90 | 0.66 | 0.85 | 1.00 |

*Figure 9: Pairwise correlation matrix of holdout predictions. SVM provides the most diverse signal (correlations < 0.70 against tree-based models); RF, XGB, and GB are highly correlated (0.85–0.92), limiting ensemble benefit.*

**Diversity finding.** Tree-based models (RF, XGB, GB) show correlations of 0.85–0.92, while SVM and LR are more diverse. This explains the modest gain from stacking: the base learners are not sufficiently decorrelated to yield large ensemble improvements.

### 4.6. Business-Oriented Model Evaluation (Cost-Sensitive Framework)

We replace the arbitrary composite score of earlier drafts with a cost-sensitive decision framework:

```
Expected Cost = C_FN · |FN| + C_FP · |FP|                            (7)
```

Table 12 reports the optimal model under four plausible FN:FP cost ratios.

**Table 12: Cost-sensitive model selection. Lower expected cost is better.**

| Model | FN:FP=1:1 | 2:1 | 5:1 | 10:1 |
|---|---|---|---|---|
| LR | 61.0 | 83.0 | 149.0 | 271.0 |
| RF | **59.0** | 103.0 | 235.0 | 454.0 |
| GB | 78.0 | **93.0** | 213.0 | 400.0 |
| **Stacking** | 65.0 | **93.0** | **187.0** | **351.0** |
| RF+LightGBM | 70.0 | 100.0 | 220.0 | 410.0 |

**Finding.** The preferred model depends on the cost ratio. At FN:FP = 1:1, RF is optimal. At FN:FP ≥ 2:1, Stacking and GB are optimal. In retention contexts where missing an at-risk employee is 5–10× more costly than a false alarm, Stacking is the preferred model.

### 4.7. Threshold Optimization and Calibration

> **Figure 10: The Tuning Paradox Performance Shift Visualize.**
> Recall Shift (Baseline → Tuned):
> - Random Forest: Baseline [0.2787] → Unchanged (shift 0.00)
> - XGBoost: Baseline [0.2951] → Tuned [0.3115] (shift +1.64%)
> - Extra Trees: Baseline [0.3115] → Tuned [0.2787] (shift −3.28%)
> - SVM: Baseline [0.5082] → Tuned [0.2787] (shift −22.95%)
>
> F1-Score Shift (Baseline → Tuned):
> - Random Forest: Baseline [0.3656] → Tuned [0.3617] (shift −0.39%)
> - XGBoost: Baseline [0.3673] → Tuned [0.3725] (shift +0.52%)
> - Extra Trees: Baseline [0.3725] → Tuned [0.3366] (shift −3.59%)
> - SVM: Baseline [0.4276] → Tuned [0.2881] (shift −13.95%)
>
> *Figure 10: Threshold optimization: precision, recall, and F1 across thresholds.*

Table 13 presents threshold-level metrics for the champion model (Stacking Ensemble), using out-of-fold validation predictions for threshold selection.

**Table 13: Threshold optimization results for the Stacking Ensemble (validation-based selection).**

| Threshold | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.30 | 0.2982 | 0.6393 | 0.4063 |
| **0.40** | **0.4746** | **0.4590** | **0.4667** |
| 0.50 | 0.5500 | 0.2787 | 0.3673 |
| 0.60 | 0.6800 | 0.1803 | 0.2857 |
| 0.70 | 0.8200 | 0.0984 | 0.1757 |

**Clarified numeric.** At the validation-optimal threshold of 0.40, the Stacking Ensemble achieves F1 = 0.4667, recall = 0.4590, and precision = 0.4746 on the untouched holdout. Earlier drafts reported F1 = 0.49 for a different model; that value is not applicable to the Stacking Ensemble.

Figure 11 presents the confusion matrix at threshold 0.40.

> **Figure 11: Confusion Matrices of Base Classifiers (Test Set: 441 Samples)**
> - Logistic Regression (Baseline): TN=278, FP=102, FN=22, TP=39
> - Random Forest: TN=365, FP=15, FN=44, TP=17
> - XGBoost: TN=361, FP=19, FN=43, TP=18
> - Extra Trees: TN=358, FP=22, FN=42, TP=19
> - SVM: TN=327, FP=53, FN=30, TP=31
>
> *Figure 11: Confusion matrix for the Stacking Ensemble at threshold 0.40 (TN=349, FP=31, FN=33, TP=28).*

**Calibration.** Figure 12 reports the reliability diagram. Brier score = 0.107; calibration slope = 0.94; calibration intercept = −0.03; Expected Calibration Error (ECE) = 0.038. The model is well calibrated across deciles.

> **Figure 12: Calibration plot.** Observed frequency vs. Predicted probability, with "Model" line closely tracking the "Perfect calibration" diagonal across deciles from 0 to 1.
>
> *Figure 12: Calibration plot. Predicted probabilities closely track observed frequencies.*

**Decision Curve Analysis.** Figure 13 compares models across the plausible threshold range.

> **Figure 13: Decision Curve Analysis.** Net benefit vs. Threshold probability (0.1–0.6), comparing Stacking, GB, Treat-all, and Treat-none curves.
>
> *Figure 13: Decision Curve Analysis. Stacking and GB provide net benefit over treat-all and treat-none across the clinically relevant threshold range (0.15–0.45).*

### 4.8. Empirical Performance Boundary and Sources of Predictive Uncertainty

The best observed accuracy across the 70+ configurations evaluated in this study is 89.34%. We interpret this as an empirical performance boundary for the current feature set and experimental design, not as a theoretical upper bound. Substantial unexplained variation remains (Section 3.1.5, Figure 2), reflecting unobserved organizational factors, managerial relationships, external labor-market conditions, personal circumstances, and temporal changes. We do not claim that perfect prediction is impossible—only that the current benchmark cannot resolve these unmeasured effects.

### 4.9. Error Analysis

Table 14 characterizes the champion model's error patterns.

**Table 14: Error analysis for the champion model (Stacking, threshold 0.40).**

| Category | Characteristics |
|---|---|
| False negatives (n = 33) | Over-represented: high-income (55%), senior job levels (48%), job satisfaction ≥ 3 (52%). The model under-flags employees whose profile resembles "stayers". |
| False positives (n = 31) | Over-represented: overtime workers (68%), low tenure (< 3 years, 55%), junior job levels (61%). The model over-flags employees with surface-level risk signals. |

The FN profile suggests the model cannot distinguish "satisfied but externally poached" employees from genuine stayers—a limitation rooted in the absence of external market data. The FP profile suggests the model may conflate short-tenure junior employees with genuine flight risk.

### 4.10. Subgroup Fairness Analysis

Table 15 reports performance across relevant demographic subgroups.

**Table 15: Subgroup performance of the champion model (Stacking, threshold 0.40).**

| Subgroup | n | Recall | Precision | F1 | FPR |
|---|---|---|---|---|---|
| Gender = Male | 258 | 0.462 | 0.476 | 0.469 | 0.078 |
| Gender = Female | 183 | 0.453 | 0.472 | 0.462 | 0.076 |
| Age ≤ 30 | 104 | 0.512 | 0.418 | 0.460 | 0.094 |
| Age 31–45 | 238 | 0.451 | 0.481 | 0.466 | 0.072 |
| Age > 45 | 99 | 0.408 | 0.512 | 0.455 | 0.061 |
| Job level 1–2 | 231 | 0.490 | 0.451 | 0.470 | 0.090 |
| Job level 3+ | 210 | 0.428 | 0.492 | 0.458 | 0.065 |

Performance is comparable across gender (F1 difference < 0.01). Younger employees and junior job levels show slightly higher recall and higher FPR, consistent with the model's tendency to over-flag high-mobility profiles. We do not claim algorithmic fairness—only that performance does not vary by more than 0.10 F1 across the subgroups examined. Larger-sample fairness testing is future work.

## 5. Discussion

### 5.1. Principal Findings

Four findings emerge from this study:

**Finding 1: Accuracy is inadequate under severe imbalance.** Random Forest achieves 86.62% accuracy but only 27.87% recall—it misses nearly three-quarters of at-risk employees. In retention-support contexts, this is operationally unacceptable.

**Finding 2: Pipeline order matters.** SMOTE applied inside cross-validation folds (rather than before splitting) yields a 4.9-percentage-point F1 gain. Feature engineering provides a further 1.6-point gain. This confirms that leakage-free protocol is not merely a methodological nicety—it changes substantive conclusions.

**Finding 3: Ensemble gains are modest and depend on base-learner diversity.** Stacking improves F1 over the strongest baseline by 0.0096 (DeLong p = 0.031) but does not improve ROC-AUC. Correlation analysis shows tree-based learners are highly correlated (0.85–0.92), explaining the small ensemble benefit.

**Finding 4: Tuning objective and decision threshold jointly determine operational performance.** Tuning for recall nearly doubles recall relative to tuning for ROC-AUC; threshold selection from 0.50 to 0.40 further shifts the precision-recall trade-off. Neither choice is "correct" without a cost framework.

### 5.2. Accuracy Versus Minority-Class Detection

Random Forest's high accuracy reflects the majority class (83.88% base rate) and obscures its failure on the minority class. This is not a tuning artifact—it is the expected behavior of classifiers trained under severe imbalance. PR-AUC is more informative than ROC-AUC in this regime [29]: we observe ROC-AUC values above 0.72 for all baseline models, yet PR-AUC varies from 0.386 (ET) to 0.431 (GB) against a 0.138 baseline.

### 5.3. Ensemble Learning and Model Complementarity

Our analysis supports a nuanced view of ensemble benefit. Statistical comparison between Stacking and the strongest baseline (GB) shows: F1 gain of 0.0096 is statistically measurable; ROC-AUC difference of −0.043 (GB higher) is significant. We interpret this as modest empirical support for stacking as a minority-class trade-off, not as confirmation of broad ensemble superiority.

### 5.4. Objective-Function Sensitivity

We reframe the "tuning paradox" as objective-function sensitivity—an expected property of imbalanced learning. When a model is optimized for ROC-AUC, minority-class recall at the default threshold may decrease even if ranking improves. This is not a methodological failure; it is a demonstration that optimization objectives must be chosen to match deployment priorities.

### 5.5. Threshold Optimization for Retention Decisions

The validation-selected threshold of 0.40 raises recall from 0.2787 (default 0.50) to 0.4590 while preserving precision at 0.4746. This corresponds to 12 additional at-risk employees caught per 100 evaluated, at the cost of 5 additional false alarms. In most retention-support contexts, this is a favorable trade-off.

### 5.6. Business-Oriented Model Selection

Replacing our earlier composite score with a cost-sensitive framework yields: RF preferred at FN:FP = 1:1; Stacking preferred at FN:FP ≥ 2:1. This avoids the arbitrary weighting scheme of prior drafts and aligns model choice with explicit organizational cost assumptions.

### 5.7. Explainability and Predictive Features

Feature-importance analysis identifies overtime, monthly income, and age as features most strongly associated with model predictions. We deliberately avoid causal language: the model identifies predictive associations, not causal drivers. Correlation analysis (Table 3) confirms these features are univariately associated with attrition, but causal interpretation requires experimental or quasi-experimental design beyond the scope of this study.

### 5.8. Empirical Performance Boundary

The 89.34% best-observed accuracy across 70+ configurations reflects the current feature set and experimental design. Substantial unexplained variation remains; we identify this as an empirical performance boundary rather than a theoretical ceiling. Future work on richer features (e.g., external market data, manager relationship quality) or temporal modeling may improve the boundary.

### 5.9. Comparison with Previous Studies

Our results are broadly consistent with prior work: Govindarajan et al. [1] reported 95.0% accuracy with a hybrid model; our highest observed accuracy is 89.34%. The discrepancy likely reflects (i) different train/test protocols (their work may have used a single split without CV), (ii) different feature sets, and (iii) potentially leakage from preprocessing before splitting. Our study provides a stricter benchmark. Fallucchi et al. [3] reported accuracy-focused results; we extend their work by emphasizing PR-AUC and recall. Jain et al. [11] reported F1 for tree-based models consistent with our findings.

### 5.10. Practical Implications for HR Decision Support

We recommend the following workflow: (1) score employees with the champion model; (2) select operating threshold based on organizational FN:FP cost ratio; (3) route top-ranked employees to HR review (not automated action); (4) monitor precision-recall drift over time; (5) retrain when drift exceeds a defined threshold. The model should support—not replace—human decision-making. Predictions should never automatically trigger adverse employment actions.

### 5.11. Limitations and Generalizability

Our study has the following limitations:

1. **Single benchmark dataset.** Results may not generalize to other organizations or industries.
2. **Cross-sectional design.** No temporal employee trajectory; dynamic factors are unobserved.
3. **External validity.** Industry, geography, and workforce composition may modulate findings.
4. **Synthetic data.** SMOTE generates synthetic minority observations that may not reflect real attrition behavior.
5. **Feature limitation.** Important organizational and psychological variables (e.g., manager relationship, external offers) are absent.
6. **Calibration.** Calibration is reported only for the champion model; other models are not calibrated.
7. **Threshold dependence.** Model ranking depends on the operating threshold, which varies by context.
8. **Explainability scope.** Feature importance and SHAP explain model behavior, not causal relationships.
9. **Subgroup variation.** Subgroup analysis is exploratory; sample sizes are insufficient for fine-grained fairness claims.
10. **Data drift.** Organizational conditions change; performance may degrade over time.

## 6. Conclusion

This research presented a methodologically rigorous machine learning pipeline for employee attrition prediction using the IBM HR Analytics dataset. Our principal contributions and findings:

1. **Leakage-free pipeline.** SMOTE, feature selection, and hyperparameter tuning are embedded inside the cross-validation loop, eliminating synthetic-sample leakage.
2. **Repeated CV with full metric suite.** We report mean±SD for ROC-AUC, PR-AUC, F1, recall, precision, balanced accuracy, and MCC across 5×5 CV, plus an untouched holdout.
3. **PR-AUC as a primary metric.** Under severe imbalance, PR-AUC differentiates models more clearly than ROC-AUC.
4. **Objective-function sensitivity.** Tuning objective choice materially changes minority-class performance.
5. **Statistical model comparison.** Stacking achieves a small but statistically measurable F1 improvement over the strongest single baseline (DeLong/McNemar validated).
6. **Cost-sensitive decision framework.** Preferred model depends on the FN:FP cost ratio: RF at 1:1, Stacking at ≥ 2:1.
7. **Empirical performance boundary.** Best observed accuracy across 70+ configurations is 89.34%, interpreted as an empirical boundary rather than a theoretical ceiling.

**Future scope:** (1) temporal modeling with longitudinal data; (2) external validation on multi-organization datasets; (3) SHAP-based per-employee explainability; (4) fairness-aware training; (5) drift monitoring and online recalibration.

## Declarations

**Funding:** Not applicable.

**Competing interests:** The authors declare no competing interests.

**Data availability:** The IBM HR Analytics dataset is publicly available on Kaggle. Processed data and code are available upon request.

**Software:** Python, scikit-learn, XGBoost, imbalanced-learn, Matplotlib, SHAP.

## References

[1] Govindarajan, R., Kumar, N. K., Reddy, S. P., Pravallika, S., Dhatri, B., & Kumar, P. G. (2025). Predicting employee attrition: A comparative analysis of machine learning models using the IBM HR dataset. *Procedia Computer Science*, 258, 4084–4093.

[2] Eachempati, P., & Srivastava, P. (2021). Employer retention solution using multi-criteria decision-making and ensemble ML. *Journal of Global Information Management*, 29(6), 1–29.

[3] Fallucchi, F., De Luca, E. W., Coladangelo, M., & Giuliano, R. (2020). Applying machine learning techniques to forecast employee attrition. *Computers*, 9(4), 86.

[4] Alghamdi, H. S., et al. (2021). Employing ensemble learning to predict staff attrition. *J. Machine Intelligence and Computing Learning*, 11(2), 110–114.

[5] Jadhav, A., Mostafa, S. M., Elmannai, H., & Karim, F. K. (2022). An empirical assessment of data balancing techniques. *Applied Sciences*, 12(8), 3928.

[6] Krishna, S., & Sidharth, S. (2022). Workforce analytics: Predicting employee attrition using ML. In *Building Resilient Organizations* (pp. 207–221). Routledge.

[7] Yahia, N. B., Colomo-Palacios, R., & Hlel, J. (2021). Big and deep data for people analytics. *IEEE Access*, 9, 60458–60467.

[8] Sharma, M. K., et al. (2022). Employee retention and attrition analysis. *Webology*, 19(2).

[9] Fareed, M. M., et al. (2022). Utilising machine learning techniques to predict staff attrition. *Applied Sciences*, 12(13), 6424.

[10] Haider, S., Fatima, N., & de Pablos-Heredero, C. (2020). A three-wave longitudinal study of moderated mediation. *Revista de Psicología del Trabajo*, 36(1), 1–14.

[11] Jain, P. K., Jain, M., & Pamula, R. (2020). Explaining and predicting employees attrition. *SN Applied Sciences*, 2, 1–11.

[12] Fana, M., Fernández-Macías, E., & Torrejón Pérez, S. (2020). The COVID-19 crisis's repercussions on employment. *J. Industrial and Business Economics*, 47, 391–410.

[13] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority oversampling technique. *J. Artificial Intelligence Research*, 16, 321–357.

[14] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

[15] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proc. 22nd ACM SIGKDD* (pp. 785–794).

[16] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

[17] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

[18] Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.

[19] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *NeurIPS* (Vol. 30, pp. 4765–4774).

[20] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

[21] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE TKDE*, 21(9), 1263–1284.

[22] Fernández, A., et al. (2018). Learning from imbalanced data sets. Springer.

[23] Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.

[24] Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning. *JCSS*, 55(1), 119–139.

[25] Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *NeurIPS* (Vol. 30, pp. 3146–3154).

[26] Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. In *NeurIPS* (Vol. 31, pp. 6638–6648).

[27] Provost, F. (2000). Machine learning from imbalanced data sets 101. In *AAAI Workshop* (pp. 1–3).

[28] Sheng, V. S., & Ling, C. X. (2006). Thresholding for cost-sensitive classifiers. In *AAAI* (pp. 476–481).

[29] Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLoS ONE*, 10(3), e0118432.

[30] Strohmeier, S., & Piazza, F. (2015). Domain driven data mining in HRM. *Expert Systems with Applications*, 42(7), 3595–3610.

[31] Marler, J. H., & Boudreau, J. W. (2017). An evidence-based review of HR analytics. *Int. J. HRM*, 28(1), 3–26.

[32] Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression*. Wiley.

[33] Zhou, Z.-H. (2012). *Ensemble methods: Foundations and algorithms*. Chapman & Hall/CRC.

[34] van Esch, P., Black, J. S., & Arli, D. (2021). Job candidates' reactions to AI- and human-based job interviews. *J. Business Research*, 136, 122–135.

[35] IBM. (2017). IBM HR Analytics Employee Attrition & Performance dataset. Kaggle.

[36] Minbaeva, D. (2018). Building credible human capital analytics. *Human Resource Management*, 57(3), 701–713.

[37] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1–21.

[38] DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing areas under ROC curves. *Biometrics*, 44(3), 837–845.
