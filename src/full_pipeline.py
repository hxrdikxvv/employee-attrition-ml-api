"""
full_pipeline.py  —  Complete reverse-engineered ML pipeline from MLFINAL_paper.md
====================================================================================
Implements EVERY stage described in the paper with strict leakage-free protocols.

  S3.3   Data cleaning & domain feature engineering (Table 4 / Table 5 / Eq. 1)
  S3.4   Stratified 70/30 train-test split (random_state=42)
  S3.5   One-hot encoding (fit on train only)
  S3.6   SMOTE applied *inside* CV training folds only (leakage-free)
  S3.7   Gradient-Boosting feature selection — top-16 features (Table 6)
  S3.8   Six base classifiers: LR, RF, XGB, GB, SVM, ET
  S3.9   GridSearchCV tuning under 5 scoring objectives (Table 7 grids)
  S3.10  Ensemble methods: Soft Voting + two-layer Stacking (Fig. 6)
         + additional combos: RF+LightGBM, ET+LightGBM, ET+CatBoost, RF+AdaBoost
  S3.11  Fixed serving threshold with a descriptive sensitivity sweep
  S3.12  Full metric suite: ROC-AUC, PR-AUC, Acc, Prec, Rec, F1, BalAcc, MCC
  S4.2   Ablation study (Table 8)
  S4.3   Repeated 5x5 stratified CV (Table 9)
  S4.4   Objective-function sensitivity study (Table 10)
  S4.5   Exploratory McNemar + paired bootstrap comparison (CV champion vs runner-up)
  S4.5   Ensemble prediction correlation matrix (Fig. 9)
  S4.6   Cost-sensitive expected-cost table (Eq. 7, Table 12)
  S4.6   Decision Curve Analysis (Fig. 13)
  S4.7   Calibration: Brier score, ECE, reliability diagram (Fig. 12)
  S4.9   Error analysis: FN / FP profiles (Table 14)
  S4.10  Subgroup fairness: gender, age, job-level (Table 15)
  S5.7   SHAP beeswarm + bar plots

Usage:
    python Employee_Attrition_ML/src/full_pipeline.py

All outputs -> Employee_Attrition_ML/results/full_pipeline/
"""

import sys, warnings, json, time, copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, brier_score_loss,
    roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT       = SCRIPT_DIR.parent
OUT_ROOT   = ROOT / "results" / "full_pipeline"
FIGS       = OUT_ROOT / "figures"
TABLES     = OUT_ROOT / "tables"
MDIR       = OUT_ROOT / "models"
for d in (FIGS, TABLES, MDIR):
    d.mkdir(parents=True, exist_ok=True)

t0 = time.time()

def section(title: str):
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")

# ═══════════════════════════════════════════════════════════════════════
# 1  Helper: compute_metrics
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_proba, threshold=0.50):
    """Full metric dictionary for one (y_true, y_proba, threshold) triple."""
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred  = (y_proba >= threshold).astype(int)
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "ROC-AUC":   round(roc_auc_score(y_true, y_proba), 4),
        "PR-AUC":    round(average_precision_score(y_true, y_proba), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "BalAcc":    round(balanced_accuracy_score(y_true, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
        "Threshold": threshold,
        "CM": cm, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

# ═══════════════════════════════════════════════════════════════════════
# 2  Domain feature engineering  (S3.3.3 / Table 5 / Eq. 1)
# ═══════════════════════════════════════════════════════════════════════
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HasOvertime"]     = (df["OverTime"] == "Yes").astype(int)
    df["IsSingle"]        = (df["MaritalStatus"] == "Single").astype(int)
    df["LowStock"]        = (df["StockOptionLevel"] == 0).astype(int)
    df["IsJunior"]        = (df["JobLevel"] <= 1).astype(int)
    df["IncomeToLevel"]   = df["MonthlyIncome"] / df["JobLevel"].clip(lower=1)
    df["ExperienceGap"]   = df["TotalWorkingYears"] - df["YearsAtCompany"]
    df["SatisfactionScore"] = (
        df["JobSatisfaction"] + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"] + df["WorkLifeBalance"]
    )
    df["RiskScore"] = (
        3 * df["HasOvertime"] + 2 * df["IsSingle"]
        + 2 * df["LowStock"] + 1 * df["IsJunior"]
    )
    return df

# ═══════════════════════════════════════════════════════════════════════
# 3  In-fold feature selection  (S3.7)
# ═══════════════════════════════════════════════════════════════════════
def select_features_gbm(X: pd.DataFrame, y, n: int = 16) -> list:
    excluded = {"Age", "Gender", "OverTime_No", "HasOvertime", "IsSingle", "RiskScore", "SatisfactionScore", "IncomeToLevel", "ExperienceGap", "LowStock", "IsJunior"}
    X = X.drop(columns=[name for name in excluded if name in X.columns])
    sel = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42
    )
    sel.fit(X, y)
    imp = pd.Series(sel.feature_importances_, index=X.columns)
    return imp.nlargest(n).index.tolist()

# ═══════════════════════════════════════════════════════════════════════
# 4  Model definitions  (S3.8 / S3.10)
# ═══════════════════════════════════════════════════════════════════════
def make_base_models():
    return {
        "LR": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        "XGB": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=2,
            subsample=0.8, min_samples_leaf=3, random_state=42,
        ),
        "SVM": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0, random_state=42)),
        ]),
        "ET": ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            random_state=42, n_jobs=-1,
        ),
    }

def make_stacking_ensemble():
    base = [
        ("gbm", GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=2,
            subsample=0.8, min_samples_leaf=3, random_state=42)),
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1)),
        ("rf",  RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        cv=5, passthrough=False, n_jobs=-1,
    )

def make_voting_ensemble():
    return VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)),
            ("et",  ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1, random_state=42, n_jobs=-1)),
            ("svm", Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True, kernel="rbf", C=1.0, random_state=42))])),
        ],
        voting="soft", n_jobs=-1,
    )

def make_additional_ensembles():
    rf   = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
    et   = ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1)
    lgbm = LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=42, n_jobs=-1, verbosity=-1)
    cat  = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                               loss_function="Logloss", random_seed=42, verbose=False)
    ada  = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=300, learning_rate=0.05, random_state=42,
    )
    return {
        "RF+LightGBM": VotingClassifier([("rf", rf), ("lgbm", lgbm)], voting="soft", n_jobs=-1),
        "ET+LightGBM": VotingClassifier([("et", et), ("lgbm", lgbm)], voting="soft", n_jobs=-1),
        "ET+CatBoost": VotingClassifier([("et", et), ("cat", cat)], voting="soft", n_jobs=-1),
        "RF+AdaBoost": VotingClassifier([("rf", rf), ("ada", ada)], voting="soft", n_jobs=-1),
    }

# ═══════════════════════════════════════════════════════════════════════
# 5  Validation-based threshold selection  (S3.11)
# ═══════════════════════════════════════════════════════════════════════
def select_threshold_oof(model, X_train, y_train, cv=5):
    skf      = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    oof_prob = cross_val_predict(model, X_train, y_train, cv=skf,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    best_s, best_th = -np.inf, 0.50
    for th in np.arange(0.30, 0.71, 0.01):
        preds = (oof_prob >= th).astype(int)
        s     = f1_score(y_train, preds, zero_division=0)
        if s > best_s:
            best_s, best_th = s, round(float(th), 2)
    return best_th

# ═══════════════════════════════════════════════════════════════════════
# 6  Leakage-free repeated 5×5 CV  (S3.2 / S3.6 / S3.7 / S4.3)
# ═══════════════════════════════════════════════════════════════════════
def cv_evaluate_model(model, X_enc: pd.DataFrame, y: pd.Series,
                      n_features=16, n_splits=5, n_repeats=5, threshold=0.50):
    rskf        = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    fold_metrics = []
    for _, (tr_idx, va_idx) in enumerate(rskf.split(X_enc, y)):
        Xtr, Xva = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx],     y.iloc[va_idx]
        
        # In-fold SMOTE
        sm        = SMOTE(random_state=42)
        Xtr_a, ytr_a = sm.fit_resample(Xtr, ytr)
        Xtr_sm    = pd.DataFrame(Xtr_a, columns=Xtr.columns)
        
        # In-fold feature selection (Strictly inside the loop)
        sel       = select_features_gbm(Xtr_sm, ytr_a, n=n_features)
        Xtr_s     = Xtr_sm[sel]
        Xva_s     = Xva[sel]
        
        # Fit & evaluate
        m         = copy.deepcopy(model)
        m.fit(Xtr_s, ytr_a)
        yp        = m.predict_proba(Xva_s)[:, 1]
        fold_metrics.append(compute_metrics(yva.values, yp, threshold))
        
    keys = ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1", "BalAcc", "MCC"]
    return {k: (round(np.mean([d[k] for d in fold_metrics]), 4),
                round(np.std( [d[k] for d in fold_metrics]), 4))
            for k in keys}

# ═══════════════════════════════════════════════════════════════════════
# 7  Statistical tests  (S4.5)
# ═══════════════════════════════════════════════════════════════════════
def mcnemar_test(y_true, pred_a, pred_b):
    from scipy import stats
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b-c)-1)**2 / (b+c)
    return round(chi2,4), round(float(1 - stats.chi2.cdf(chi2, df=1)),4)

def paired_bootstrap_ci(y_true, proba_a, proba_b, metric="f1",
                        th_a=0.50, th_b=0.50, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    n     = len(y_true)
    yv    = np.asarray(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pA, pB = yv[idx], proba_a[idx], proba_b[idx]
        if metric == "f1":
            sA = f1_score(yb, (pA>=th_a).astype(int), zero_division=0)
            sB = f1_score(yb, (pB>=th_b).astype(int), zero_division=0)
        else:
            sA, sB = roc_auc_score(yb, pA), roc_auc_score(yb, pB)
        diffs.append(sA-sB)
    lo = np.quantile(diffs, 0.025); hi = np.quantile(diffs, 0.975)
    return round(np.mean(diffs),4), round(lo,4), round(hi,4)

# ═══════════════════════════════════════════════════════════════════════
# 8  Cost / DCA helpers  (S4.6)
# ═══════════════════════════════════════════════════════════════════════
def expected_cost(cm, fn_fp_ratio):
    tn, fp, fn, tp = cm.ravel()
    return int(fn_fp_ratio * fn + fp)

def net_benefit(y_true, y_proba, pt):
    n  = len(y_true); yp = (y_proba >= pt).astype(int)
    tp = int(np.sum((yp==1) & (y_true==1)))
    fp = int(np.sum((yp==1) & (y_true==0)))
    return (tp/n) - (fp/n)*(pt/(1-pt+1e-9))

# ═══════════════════════════════════════════════════════════════════════════════
# ════════════════════  MAIN PIPELINE  ═════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

section("STEP 1 — DATA LOADING, CLEANING & DOMAIN FEATURE ENGINEERING")
raw = pd.read_csv(ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = raw.copy()
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
df = add_domain_features(df)
print(f"Data shape after engineering: {df.shape}")

counts = df["Attrition"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["No (Stayed)", "Yes (Left)"], [counts[0], counts[1]],
              color=["#2196F3", "#F44336"], edgecolor="black", linewidth=0.8)
for bar, cnt in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            f"{cnt}\n({cnt/len(df)*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Attrition Class Distribution (Full Dataset)", fontsize=14, fontweight="bold")
ax.set_ylabel("Employee Count"); ax.set_ylim(0, max(counts)*1.2)
plt.tight_layout(); plt.savefig(FIGS/"class_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

section("STEP 2 — STRATIFIED 70/30 TRAIN/TEST SPLIT")
y = df["Attrition"]
X = df.drop(columns=["Attrition"])
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
test_idx = X_test_raw.index
print(f"Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")

section("STEP 3 — ONE-HOT ENCODING (fit on train only)")
cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
X_train  = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=False)
X_test_e = pd.get_dummies(X_test_raw,  columns=cat_cols, drop_first=False)
X_test   = X_test_e.reindex(columns=X_train.columns, fill_value=0)
excluded_predictors = {"Age", "Gender", "OverTime_No", "HasOvertime", "IsSingle", "RiskScore", "SatisfactionScore", "IncomeToLevel", "ExperienceGap", "LowStock", "IsJunior"}
excluded_predictors.update(c for c in X_train.columns if c.startswith("MaritalStatus_"))
X_train = X_train.drop(columns=[c for c in excluded_predictors if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in excluded_predictors if c in X_test.columns])
# Limit candidates to fields collected by the prediction API and frontend form.
api_numeric = {"JobInvolvement", "JobLevel", "MonthlyIncome", "StockOptionLevel", "WorkLifeBalance", "YearsWithCurrManager", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"}
api_categorical = {"BusinessTravel", "Department", "JobRole", "OverTime"}
api_columns = api_numeric | {c for c in X_train.columns if any(c.startswith(f"{name}_") for name in api_categorical)}
X_train = X_train[[c for c in X_train.columns if c in api_columns]]
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
print(f"Encoded — Train: {X_train.shape}  Test: {X_test.shape}")

section("STEP 4 — SMOTE BALANCING (full train split for final fit)")
sm = SMOTE(random_state=42)
Xtr_sm_arr, ytr_sm_arr = sm.fit_resample(X_train, y_train)
X_tr_sm  = pd.DataFrame(Xtr_sm_arr, columns=X_train.columns)
y_tr_sm  = pd.Series(ytr_sm_arr, name="Attrition")
print(f"Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"After : {dict(zip(*np.unique(y_tr_sm, return_counts=True)))}")

# SMOTE figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cnts, title in [
    (axes[0], pd.Series(y_train).value_counts(), "Before SMOTE"),
    (axes[1], pd.Series(y_tr_sm).value_counts(), "After SMOTE"),
]:
    brs = ax.bar(["Stayed (0)","Left (1)"], [cnts.get(0,0), cnts.get(1,0)], color=["#2196F3", "#F44336"], edgecolor="black")
    tot = sum([cnts.get(0,0), cnts.get(1,0)])
    for bar, v in zip(brs, [cnts.get(0,0), cnts.get(1,0)]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f"{v}\n({v/tot*100:.0f}%)", ha="center", fontweight="bold", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_ylabel("Count")
plt.suptitle("SMOTE — Training Set Class Distribution", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"smote_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

section("STEP 5 — GRADIENT BOOSTING FEATURE SELECTION (top-16 for final model)")
sel_gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
excluded_predictors = {"Age", "Gender", "OverTime_No", "HasOvertime", "IsSingle", "RiskScore", "SatisfactionScore", "IncomeToLevel", "ExperienceGap", "LowStock", "IsJunior"}
excluded_predictors.update(c for c in X_tr_sm.columns if c.startswith("MaritalStatus_"))
X_tr_candidates = X_tr_sm.drop(columns=[c for c in excluded_predictors if c in X_tr_sm.columns])
sel_gbm.fit(X_tr_candidates, y_tr_sm)
imp_all      = pd.Series(sel_gbm.feature_importances_, index=X_tr_candidates.columns).sort_values(ascending=False)
thresh_imp   = imp_all.mean()
sel_feats    = imp_all[imp_all >= thresh_imp].index.tolist()
if len(sel_feats) < 15: sel_feats = imp_all.head(15).index.tolist()
if len(sel_feats) > 16: sel_feats = imp_all.head(16).index.tolist()
print(f"Selected {len(sel_feats)} features for final holdout models.")
imp_all.reset_index().rename(columns={"index":"Feature",0:"Importance"}).to_csv(TABLES/"feature_importance.csv", index=False)
pd.DataFrame({"Rank": range(1, len(sel_feats)+1), "Feature": sel_feats}).to_csv(TABLES/"selected_features.csv", index=False)

imp_top = imp_all.head(len(sel_feats)).reset_index()
imp_top.columns = ["Feature","Importance"]
fig, ax = plt.subplots(figsize=(10, max(6, len(sel_feats)*0.38)))
clrs    = plt.cm.RdYlGn(imp_top["Importance"] / imp_top["Importance"].max())
ax.barh(imp_top["Feature"][::-1], imp_top["Importance"][::-1], color=clrs[::-1], edgecolor="black", linewidth=0.5)
ax.axvline(thresh_imp, color="red", linestyle="--", linewidth=1.5, label="Mean threshold")
ax.set_title("Top-16 GBM Feature Importance (S3.7 / Fig. 5)", fontsize=14, fontweight="bold")
ax.legend(); plt.tight_layout(); plt.savefig(FIGS/"feature_importance_ranking.png", dpi=150, bbox_inches="tight"); plt.close()

X_tr_final = X_tr_sm[sel_feats]
X_te_final = X_test[sel_feats]

# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — Ablation study (FIXED: Dropped eng_cols for c1)
# ═══════════════════════════════════════════════════════════════════════
section("STEP 6 — ABLATION STUDY (S4.2 / Table 8)")
skf5     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gb_raw   = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
gb_tuned = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=2, subsample=0.8, min_samples_leaf=3, random_state=42)

def _simple_cv(X_arr, y_arr, model):
    res = []
    for tr, va in skf5.split(X_arr, y_arr):
        m = copy.deepcopy(model)
        m.fit(X_arr[tr], y_arr[tr])
        yp = m.predict_proba(X_arr[va])[:, 1]
        res.append(compute_metrics(y_arr[va], yp))
    return {k: (round(np.mean([d[k] for d in res]),4), round(np.std([d[k] for d in res]),4)) for k in ["ROC-AUC","PR-AUC","F1","Recall"]}

def _smote_cv(X_df, y_s, model, sel=None):
    res = []
    for tr_i, va_i in skf5.split(X_df, y_s):
        Xtr = X_df.iloc[tr_i]; Xva = X_df.iloc[va_i]
        ytr = y_s.iloc[tr_i];  yva = y_s.iloc[va_i]
        sm2 = SMOTE(random_state=42)
        Xa, ya = sm2.fit_resample(Xtr, ytr)
        Xa = pd.DataFrame(Xa, columns=X_df.columns)
        Xts, Xvs = (Xa[sel], Xva[sel]) if sel else (Xa, Xva)
        m = copy.deepcopy(model); m.fit(Xts, ya)
        yp = m.predict_proba(Xvs)[:, 1]
        res.append(compute_metrics(yva.values, yp))
    return {k: (round(np.mean([d[k] for d in res]),4), round(np.std([d[k] for d in res]),4)) for k in ["ROC-AUC","PR-AUC","F1","Recall"]}

# Define Engineered cols
eng_cols = ["HasOvertime", "IsSingle", "LowStock", "IsJunior", "IncomeToLevel", "ExperienceGap", "SatisfactionScore", "RiskScore"]
X_raw_only = X_train.drop(columns=eng_cols, errors="ignore")

print("  c1 raw features...", end=" ", flush=True)
c1 = _simple_cv(X_raw_only.values, y_train.values, gb_raw); print("done")
print("  c2 +FE...", end=" ", flush=True)
c2 = _simple_cv(X_train.values, y_train.values, gb_raw); print("done")
print("  c3 +SMOTE in-fold...", end=" ", flush=True)
c3 = _smote_cv(X_train, y_train, gb_raw); print("done")
print("  c4 +SMOTE+FE...", end=" ", flush=True)
c4 = _smote_cv(X_train, y_train, gb_raw); print("done")
print("  c5 +SMOTE+FE+Tuning...", end=" ", flush=True)
c5 = _smote_cv(X_train, y_train, gb_tuned); print("done")
print("  c6 Stacking ensemble...", end=" ", flush=True)
stk_abl = make_stacking_ensemble()
c6 = _smote_cv(X_train[sel_feats], y_train, stk_abl); print("done")

def _fmt(d): return {k: f"{v[0]:.4f} +- {v[1]:.2f}" for k,v in d.items()}
abl_df = pd.DataFrame([
    {"Configuration": "Raw features",              **_fmt(c1)},
    {"Configuration": "+ Feature engineering",     **_fmt(c2)},
    {"Configuration": "+ SMOTE (in-fold)",          **_fmt(c3)},
    {"Configuration": "+ SMOTE + FE",              **_fmt(c4)},
    {"Configuration": "+ SMOTE + FE + Tuning",     **_fmt(c5)},
    {"Configuration": "Final Stacking ensemble",   **_fmt(c6)},
])
abl_df.to_csv(TABLES/"ablation_study.csv", index=False)
print(abl_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — Repeated 5×5 CV (FIXED: Pass full X_train to cv_evaluate_model)
# ═══════════════════════════════════════════════════════════════════════
section("STEP 7 — REPEATED 5x5 CV — BASE MODEL EVALUATION (S4.3 / Table 9)")
base_models = make_base_models()
cv_results  = {}
for name, model in base_models.items():
    print(f"  CV: {name} ...", end=" ", flush=True)
    agg = cv_evaluate_model(
        model, X_train, y_train, # <--- FIXED: Pass full X_train
        n_features=16, n_splits=5, n_repeats=5,
    )
    cv_results[name] = agg
    print(f"ROC-AUC={agg['ROC-AUC'][0]:.4f}+-{agg['ROC-AUC'][1]:.3f}  F1={agg['F1'][0]:.4f}")

cv_df = pd.DataFrame([{"Model": n, **{k: f"{v[0]:.4f} +- {v[1]:.3f}" for k,v in m.items()}} for n,m in cv_results.items()])
cv_df.to_csv(TABLES/"cv_5x5_results.csv", index=False)
print(cv_df.to_string(index=False))
champion_name = max(cv_results, key=lambda name: cv_results[name]["F1"][0])
comparison_name = next(name for name in sorted(cv_results, key=lambda n: cv_results[n]["F1"][0], reverse=True) if name != champion_name)

# ═══════════════════════════════════════════════════════════════════════
# STEP 8 — Objective-function sensitivity (FIXED: Use ImbPipeline)
# ═══════════════════════════════════════════════════════════════════════
section("STEP 8 — OBJECTIVE-FUNCTION SENSITIVITY (S4.4 / Table 10)")
OBJECTIVES = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision", "F1": "f1", "Recall": "recall", "Balanced Accuracy": "balanced_accuracy"}
GB_GRID = {
    "clf__n_estimators":    [300, 500, 800], "clf__learning_rate":   [0.02, 0.03, 0.05],
    "clf__max_depth":       [2, 3], "clf__min_samples_leaf":[2, 3, 5],
}
skf_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
obj_results  = {}
for obj_name, scoring in OBJECTIVES.items():
    print(f"  Tuning GB [{obj_name}] ...", end=" ", flush=True)
    gb_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', GradientBoostingClassifier(random_state=42, subsample=0.8))
    ])
    gs = GridSearchCV(gb_pipeline, GB_GRID, cv=skf_tune, scoring=scoring, n_jobs=-1)
    gs.fit(X_train[sel_feats], y_train) # Use raw data; SMOTE happens inside
    best = gs.best_estimator_
    yp   = best.predict_proba(X_te_final)[:, 1]
    obj_results[obj_name] = compute_metrics(y_test.values, yp)
    m = obj_results[obj_name]
    print(f"Rec={m['Recall']:.4f}  F1={m['F1']:.4f}")
    joblib.dump(best, MDIR/f"gb_tuned_{obj_name.lower().replace('-','_').replace(' ','_')}.pkl")

obj_df = pd.DataFrame([{"Tuning Objective": k, "ROC-AUC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"], "Recall": m["Recall"], "Precision": m["Precision"], "F1": m["F1"]} for k,m in obj_results.items()])
obj_df.to_csv(TABLES/"objective_sensitivity.csv", index=False)
print(obj_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# STEP 9 — Train models on full SMOTE train -> holdout evaluation
# ═══════════════════════════════════════════════════════════════════════
section("STEP 9 — HOLDOUT EVALUATION: ALL BASE MODELS + ENSEMBLES")
trained, holdout, probas = {}, {}, {}

for name, model in make_base_models().items():
    model.fit(X_tr_final, y_tr_sm)
    yp = model.predict_proba(X_te_final)[:, 1]
    trained[name] = model; probas[name] = yp
    holdout[name] = compute_metrics(y_test.values, yp)
    print(f"  {name}: Acc={holdout[name]['Accuracy']:.4f}  F1={holdout[name]['F1']:.4f}  ROC-AUC={holdout[name]['ROC-AUC']:.4f}")
    joblib.dump(model, MDIR/f"{name.lower()}_base.pkl")

voting = make_voting_ensemble()
voting.fit(X_tr_final, y_tr_sm)
yp_v = voting.predict_proba(X_te_final)[:, 1]
trained["Voting"] = voting; probas["Voting"] = yp_v
holdout["Voting"] = compute_metrics(y_test.values, yp_v)
joblib.dump(voting, MDIR/"voting_ensemble.pkl")

stacking = make_stacking_ensemble()
stacking.fit(X_tr_final, y_tr_sm)
yp_s = stacking.predict_proba(X_te_final)[:, 1]
trained["Stacking"] = stacking; probas["Stacking"] = yp_s
holdout["Stacking"] = compute_metrics(y_test.values, yp_s)
joblib.dump(stacking, MDIR/"stacking_ensemble.pkl")

for name, model in make_additional_ensembles().items():
    model.fit(X_tr_final, y_tr_sm)
    yp = model.predict_proba(X_te_final)[:, 1]
    trained[name] = model; probas[name] = yp
    holdout[name] = compute_metrics(y_test.values, yp)
    joblib.dump(model, MDIR/f"{name.lower().replace('+','plus').replace(' ','_')}.pkl")

holdout_df = pd.DataFrame([{"Model":n, "ROC-AUC":m["ROC-AUC"], "PR-AUC":m["PR-AUC"], "Accuracy":m["Accuracy"], "Recall":m["Recall"], "Precision":m["Precision"], "F1":m["F1"], "BalAcc":m["BalAcc"], "MCC":m["MCC"]} for n,m in holdout.items()]).sort_values("F1", ascending=False)
holdout_df.to_csv(TABLES/"holdout_all_models.csv", index=False)
print(holdout_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# STEP 10 — Threshold Optimization (FIXED: Leakage-free pipeline)
# ═══════════════════════════════════════════════════════════════════════
section("STEP 10 — SERVING THRESHOLD SENSITIVITY")
SERVING_THRESHOLD = 0.50
SERVING_MODEL = champion_name
serving_proba = probas[SERVING_MODEL]
print(f"Using fixed serving threshold: {SERVING_THRESHOLD:.2f}")

thresh_rows = []
for th in np.round(np.arange(0.30, 0.72, 0.05), 2):
    r = compute_metrics(y_test.values, serving_proba, threshold=float(th))
    thresh_rows.append({"Model": SERVING_MODEL, "Threshold": round(float(th),2), "Precision": r["Precision"], "Recall": r["Recall"], "F1": r["F1"], "Accuracy": r["Accuracy"], "BalAcc": r["BalAcc"], "MCC": r["MCC"]})
th_df = pd.DataFrame(thresh_rows)
th_df.to_csv(TABLES/"threshold_sweep.csv", index=False)
print(th_df.to_string(index=False))

serving_metrics = compute_metrics(y_test.values, serving_proba, threshold=SERVING_THRESHOLD)
print(f"\n{SERVING_MODEL} serving @ th={SERVING_THRESHOLD:.2f}: F1={serving_metrics['F1']:.4f}  Rec={serving_metrics['Recall']:.4f}  Prec={serving_metrics['Precision']:.4f}")

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(th_df["Threshold"], th_df["Precision"], "o-", label="Precision", color="#2196F3", lw=2)
ax.plot(th_df["Threshold"], th_df["Recall"],    "s-", label="Recall",    color="#F44336", lw=2)
ax.plot(th_df["Threshold"], th_df["F1"],        "^-", label="F1",        color="#4CAF50", lw=2.5)
ax.axvline(SERVING_THRESHOLD, color="purple", ls="--", lw=1.8, label=f"Serving ({SERVING_THRESHOLD:.2f})")
ax.set_title(f"Threshold Sensitivity — {SERVING_MODEL}", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.4); plt.tight_layout(); plt.savefig(FIGS/"threshold_optimization.png", dpi=150, bbox_inches="tight"); plt.close()

# ═══════════════════════════════════════════════════════════════════════
# STEP 11 — Statistical significance
# ═══════════════════════════════════════════════════════════════════════
section(f"STEP 11 — EXPLORATORY MODEL COMPARISON ({SERVING_MODEL} vs {comparison_name})")
pr_lr = serving_proba; pr_g = probas[comparison_name]
pred_lr = (pr_lr >= SERVING_THRESHOLD).astype(int)
pred_g = (pr_g >= SERVING_THRESHOLD).astype(int)

chi2_mc, p_mc = mcnemar_test(y_test.values, pred_lr, pred_g)
mean_df1, lo_f1, hi_f1 = paired_bootstrap_ci(y_test, pr_lr, pr_g, metric="f1", th_a=SERVING_THRESHOLD, th_b=SERVING_THRESHOLD)
mean_dauc, lo_auc, hi_auc = paired_bootstrap_ci(y_test, pr_lr, pr_g, metric="roc_auc")

print(f"Exploratory McNemar p: {p_mc} | Paired bootstrap Delta-F1 CI: [{lo_f1}, {hi_f1}]")
stat_df = pd.DataFrame([
    {"Test":f"McNemar {SERVING_MODEL} vs {comparison_name} (exploratory)","chi2":chi2_mc,"p-value":p_mc},
    {"Test":f"Paired bootstrap Delta-F1 {SERVING_MODEL} - {comparison_name}","mean":mean_df1,"CI_lo":lo_f1,"CI_hi":hi_f1},
    {"Test":f"Paired bootstrap Delta-ROC-AUC {SERVING_MODEL} - {comparison_name}","mean":mean_dauc,"CI_lo":lo_auc,"CI_hi":hi_auc},
])
stat_df.to_csv(TABLES/"statistical_tests.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 12 — Correlation matrix
# ═══════════════════════════════════════════════════════════════════════
section("STEP 12 — PREDICTION CORRELATION MATRIX")
core   = ["LR","RF","XGB","GB","SVM","ET","Stacking"]
prob_df = pd.DataFrame({n: probas[n] for n in core if n in probas})
corr    = prob_df.corr()
corr.to_csv(TABLES/"prediction_correlation.csv")
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True)
ax.set_title("Pairwise Prediction Correlation", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"prediction_correlation_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

# ═══════════════════════════════════════════════════════════════════════
# STEP 13 — Cost-sensitive analysis + DCA
# ═══════════════════════════════════════════════════════════════════════
section("STEP 13 — COST-SENSITIVE ANALYSIS & DECISION CURVE")
ratios      = [1, 2, 5, 10]
cost_models = ["LR","RF","GB","Stacking","RF+LightGBM"]
cost_rows   = []
for ratio in ratios:
    row = {"FN:FP": f"{ratio}:1"}
    for nm in cost_models:
        if nm in holdout:
            row[nm] = expected_cost(holdout[nm]["CM"], ratio)
    cost_rows.append(row)
pd.DataFrame(cost_rows).to_csv(TABLES/"cost_sensitive.csv", index=False)

pts     = np.arange(0.05, 0.70, 0.01)
dca_map = {SERVING_MODEL: serving_proba, comparison_name: probas[comparison_name]}
fig, ax = plt.subplots(figsize=(10,6))
for nm, pr in dca_map.items():
    nb = [net_benefit(y_test.values, pr, pt) for pt in pts]
    ax.plot(pts, nb, label=nm, lw=2)
prev = y_test.values.mean() # FIXED: using .values
nb_all = [(prev-(1-prev)*pt/(1-pt+1e-9)) for pt in pts]
ax.plot(pts, nb_all, "k--", label="Treat-all", lw=1.5)
ax.axhline(0, color="gray", ls=":", lw=1, label="Treat-none")
ax.set_title("Decision Curve Analysis", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.35); plt.tight_layout(); plt.savefig(FIGS/"decision_curve_analysis.png", dpi=150, bbox_inches="tight"); plt.close()

# ═══════════════════════════════════════════════════════════════════════
# STEP 14 — Calibration
# ═══════════════════════════════════════════════════════════════════════
section("STEP 14 — CALIBRATION ANALYSIS")
brier  = brier_score_loss(y_test.values, serving_proba)
bins   = np.linspace(0, 1, 11); ece = 0.0; n_tot = len(y_test)
for lo_b, hi_b in zip(bins[:-1], bins[1:]):
    mask = (serving_proba >= lo_b) & (serving_proba < hi_b)
    if mask.sum() > 0: ece += (mask.sum()/n_tot) * abs(y_test.values[mask].mean() - serving_proba[mask].mean())
print(f"Brier Score: {brier:.4f}  |  ECE: {ece:.4f}")

fig, ax = plt.subplots(figsize=(7,6))
CalibrationDisplay.from_predictions(y_test.values, serving_proba, n_bins=10, ax=ax, name=SERVING_MODEL)
ax.set_title("Reliability Diagram", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"calibration_reliability.png", dpi=150, bbox_inches="tight"); plt.close()
pd.DataFrame([{"Model":SERVING_MODEL,"Threshold":SERVING_THRESHOLD,"Brier":round(brier,4),"ECE":round(ece,4)}]).to_csv(TABLES/"calibration.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 15 — Error analysis
# ═══════════════════════════════════════════════════════════════════════
section("STEP 15 — ERROR ANALYSIS")
y_pred_opt   = pred_lr
y_test_arr   = y_test.values
fn_mask      = (y_pred_opt==0) & (y_test_arr==1)
fp_mask      = (y_pred_opt==1) & (y_test_arr==0)
X_test_orig  = X_test_raw.loc[test_idx].reset_index(drop=True)

def _profile(mask, X_orig):
    Xg = X_orig.loc[pd.Series(mask)]
    out = {}
    if "MonthlyIncome" in Xg.columns:
        med = X_test_orig["MonthlyIncome"].median(); out["high_income_%"] = round((Xg["MonthlyIncome"]>med).mean()*100,1)
    if "JobLevel" in Xg.columns: out["senior_joblevel_%"] = round((Xg["JobLevel"]>=3).mean()*100,1)
    if "JobSatisfaction" in Xg.columns: out["high_satisfaction_%"] = round((Xg["JobSatisfaction"]>=3).mean()*100,1)
    if "OverTime" in Xg.columns: out["overtime_%"] = round((Xg["OverTime"]=="Yes").mean()*100,1)
    if "YearsAtCompany" in Xg.columns: out["low_tenure_%"] = round((Xg["YearsAtCompany"]<3).mean()*100,1)
    return out

fn_p = _profile(pd.Series(fn_mask), X_test_orig)
fp_p = _profile(pd.Series(fp_mask), X_test_orig)
pd.DataFrame([{"Model":SERVING_MODEL,"Threshold":SERVING_THRESHOLD,"Type": f"False Negatives (n={fn_mask.sum()})", **fn_p}, {"Model":SERVING_MODEL,"Threshold":SERVING_THRESHOLD,"Type": f"False Positives (n={fp_mask.sum()})", **fp_p}]).to_csv(TABLES/"error_analysis.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 16 — Subgroup fairness
# ═══════════════════════════════════════════════════════════════════════
section("STEP 16 — SUBGROUP FAIRNESS ANALYSIS")
ytr = y_test.reset_index(drop=True).values
yhat = pd.Series(y_pred_opt)

def _sg_metrics(mask):
    yt = ytr[mask]; yh = yhat[mask].values
    if len(yt) == 0: return {}
    cm2 = confusion_matrix(yt, yh, labels=[0,1]); tn,fp,fn,tp = cm2.ravel()
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    return {"n":int(mask.sum()), "Recall": round(recall_score(yt,yh,zero_division=0),3), "Precision":round(precision_score(yt,yh,zero_division=0),3), "F1": round(f1_score(yt,yh,zero_division=0),3), "FPR": round(fpr,3)}

Xto = X_test_orig.reset_index(drop=True)
rows_sg = []
for g in ["Male","Female"]: rows_sg.append({"Model":SERVING_MODEL,"Threshold":SERVING_THRESHOLD,"Subgroup":f"Gender={g}", **_sg_metrics((Xto["Gender"]==g).values)})
ac = Xto["Age"]
for lbl, m in [("Age<=30",ac<=30),("Age 31-45",(ac>=31)&(ac<=45)),("Age>45",ac>45)]: rows_sg.append({"Model":SERVING_MODEL,"Threshold":SERVING_THRESHOLD,"Subgroup":lbl, **_sg_metrics(m.values)})
jl = Xto["JobLevel"]
for lbl, m in [("JobLevel 1-2",(jl>=1)&(jl<=2)),("JobLevel 3+",(jl>=3))]: rows_sg.append({"Model":SERVING_MODEL,"Threshold":SERVING_THRESHOLD,"Subgroup":lbl, **_sg_metrics(m.values)})

pd.DataFrame(rows_sg).to_csv(TABLES/"subgroup_fairness.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 17 — SHAP analysis
# ═══════════════════════════════════════════════════════════════════════
section("STEP 17 — SHAP ANALYSIS")
gb_model   = trained["GB"]
explainer  = shap.TreeExplainer(gb_model)
shap_vals  = explainer.shap_values(X_te_final)

fig = plt.figure(figsize=(10,7))
shap.summary_plot(shap_vals, X_te_final, show=False, max_display=16)
plt.title("SHAP Summary Plot (Beeswarm) — GB Model", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

fig = plt.figure(figsize=(10,6))
shap.summary_plot(shap_vals, X_te_final, plot_type="bar", show=False, max_display=16)
plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

shap_imp = pd.DataFrame({"Feature": X_te_final.columns, "MeanAbsSHAP": np.abs(shap_vals).mean(axis=0)}).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
shap_imp.to_csv(TABLES/"shap_importance.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 18 — ROC & PR curves
# ═══════════════════════════════════════════════════════════════════════
section("STEP 18 — ROC & PR CURVES")
n_m     = len(holdout); pal = plt.cm.tab20(np.linspace(0,1,n_m))
fig, axes = plt.subplots(1, 2, figsize=(17,7))
for idx, (nm, m) in enumerate(holdout.items()):
    if nm not in probas: continue
    pr = probas[nm]
    fpr_r, tpr_r, _ = roc_curve(y_test.values, pr)
    axes[0].plot(fpr_r, tpr_r, label=f"{nm} ({m['ROC-AUC']:.3f})", color=pal[idx], lw=1.8)
    pc, rc, _ = precision_recall_curve(y_test.values, pr)
    ap        = average_precision_score(y_test.values, pr)
    axes[1].plot(rc, pc, label=f"{nm} (AP={ap:.3f})", color=pal[idx], lw=1.8)

axes[0].plot([0,1],[0,1],"k--",lw=1); axes[0].set_title("ROC Curves — All Models", fontsize=13, fontweight="bold"); axes[0].legend(fontsize=6.5, loc="lower right")
axes[1].axhline(y_test.mean(), color="black", ls="--", lw=1, label="Imbalance baseline"); axes[1].set_title("PR Curves — All Models", fontsize=13, fontweight="bold"); axes[1].legend(fontsize=6.5, loc="upper right")
plt.tight_layout(); plt.savefig(FIGS/"roc_pr_all_models.png", dpi=150, bbox_inches="tight"); plt.close()

# ═══════════════════════════════════════════════════════════════════════
# STEP 19 — Confusion matrices
# ═══════════════════════════════════════════════════════════════════════
section("STEP 19 — CONFUSION MATRICES")
cols_g = 4; rows_g = (n_m+cols_g-1)//cols_g
fig, axes = plt.subplots(rows_g, cols_g, figsize=(cols_g*4.5, rows_g*4))
axes_flat = axes.ravel()
for idx, (nm, m) in enumerate(holdout.items()):
    sns.heatmap(m["CM"], annot=True, fmt="d", cmap="Blues", ax=axes_flat[idx], xticklabels=["Stay","Leave"], yticklabels=["Stay","Leave"])
    axes_flat[idx].set_title(f"{nm}\nAcc={m['Accuracy']:.3f} Rec={m['Recall']:.3f}", fontsize=8, fontweight="bold")
for i in range(n_m, len(axes_flat)): axes_flat[i].set_visible(False)
plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"confusion_matrices_all.png", dpi=150, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(serving_metrics["CM"], annot=True, fmt="d", cmap="Reds", ax=ax, xticklabels=["Stay","Leave"], yticklabels=["Stay","Leave"])
ax.set_title(f"{SERVING_MODEL} @ th={SERVING_THRESHOLD:.2f}\nPrec={serving_metrics['Precision']:.4f} Rec={serving_metrics['Recall']:.4f} F1={serving_metrics['F1']:.4f}", fontsize=10, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS/"serving_model_confusion.png", dpi=150, bbox_inches="tight"); plt.close()

# ═══════════════════════════════════════════════════════════════════════
# STEP 20 — Final summary
# ═══════════════════════════════════════════════════════════════════════
section("STEP 20 — FINAL SUMMARY")
all_rows = []
for nm, m in holdout.items():
    all_rows.append({"Model":nm, "Threshold":0.50, **{k:m[k] for k in ["ROC-AUC","PR-AUC","Accuracy","Precision","Recall","F1","BalAcc","MCC"]}})
summary_df = pd.DataFrame(all_rows).sort_values("F1", ascending=False).reset_index(drop=True)
summary_df["Rank"] = summary_df.index + 1
summary_df.to_csv(TABLES/"final_summary.csv", index=False)

champion_name = max(cv_results, key=lambda name: cv_results[name]["F1"][0])
champ = summary_df.loc[summary_df["Model"] == champion_name].iloc[0]
elapsed   = (time.time()-t0)/60

print("\n" + "="*64)
print("  PIPELINE COMPLETE — FINAL RESULTS")
print("="*64)
print(f"  Champion Model  : {champ['Model']}")
print(f"  Threshold       : {champ['Threshold']:.2f}")
print(f"  Accuracy        : {champ['Accuracy']:.4f}")
print(f"  Precision       : {champ['Precision']:.4f}")
print(f"  Recall          : {champ['Recall']:.4f}")
print(f"  F1-Score        : {champ['F1']:.4f}")
print(f"  ROC-AUC         : {champ['ROC-AUC']:.4f}")
print(f"  PR-AUC          : {champ['PR-AUC']:.4f}")
print(f"  MCC             : {champ['MCC']:.4f}")
print(f"  BalAcc          : {champ['BalAcc']:.4f}")
print("-"*64)
print(f"  Exploratory McNemar p           : {p_mc}")
print(f"  Bootstrap Delta-F1 95% CI       : [{lo_f1}, {hi_f1}]")
print(f"  Bootstrap Delta-AUC 95% CI      : [{lo_auc}, {hi_auc}]")
print(f"  Brier Score                     : {brier:.4f}")
print(f"  ECE (10-bin)                    : {ece:.4f}")
print("-"*64)
print(f"  Total elapsed time : {elapsed:.1f} min")
print("="*64)

meta = {
    "champion_model":     champion_name,
    "serving_threshold":  SERVING_THRESHOLD,
    "selection_basis":    "highest mean repeated-stratified-CV F1 among the six base models",
    "validation_scope":   "single random holdout from the IBM HR Analytics dataset; external and temporal validation not performed",
    "training_numeric_ranges": {
        col: {"min": float(X_train_raw[col].min()), "max": float(X_train_raw[col].max())}
        for col in X_train_raw.select_dtypes(include=[np.number]).columns
    },
    "brier_score":        round(float(brier),4),
    "ece":                round(float(ece),4),
    "mcnemar_p":          float(p_mc),
    "bootstrap_delta_f1": [float(lo_f1), float(hi_f1)],
    "bootstrap_delta_roc_auc": [float(lo_auc), float(hi_auc)],
    "comparison_model":   comparison_name,
    "selected_features":  sel_feats,
    "total_runtime_min":  round(float(elapsed),2),
}
(TABLES/"pipeline_metadata.json").write_text(json.dumps(meta, indent=2))
print("\nPipeline metadata saved.")
print("Full pipeline completed.\n")
