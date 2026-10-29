# Model card: Employee attrition research demo

## Intended use

This is a research/demo model trained on the public IBM HR Analytics Employee Attrition dataset. It is not validated for any employer, workforce, time period, or employment decision. Do not use its scores to rank, monitor, retain, promote, discipline, or otherwise make decisions about employees. A human-review warning is not a substitute for validation or authorization.

## Current serving contract

The selected model, fixed classification threshold, selected feature list, source-training numeric ranges, and evaluation scope are recorded in `results/full_pipeline/tables/pipeline_metadata.json`. The API loads that metadata to choose its default model and input columns. LR is selected to match the final paper; CV and holdout rankings are reported separately and do not silently override the selected champion. The API flags numeric inputs outside observed training ranges.

Direct age, gender, and marital-status features are excluded from model predictors. Exact duplicate aliases and hand-built composite features are also excluded. This does not establish fairness: remaining work and compensation variables can still proxy for protected or sensitive characteristics.

## Current evaluation snapshot

The generated scorecard in `results/full_pipeline/tables/` uses one stratified random holdout of 441 records from the same 1,470-row source dataset. It is not external or temporal validation. Repeated cross-validation is run on the training portion only. See the artifact consistency issue below: supporting generated analyses describe XGB while the deployment champion is LR.

Current serving model: Logistic Regression (`lr_base`), fixed threshold 0.50, selected to match `RESEARCH_PAPER_FINAL`.

**Artifact consistency issue:** `RESEARCH_PAPER_FINAL` reports LR holdout F1 0.5241, recall 0.5352, precision 0.5241, accuracy 0.8435, and MCC 0.4307. The bundled `final_summary.csv` instead reports LR F1 0.4375, recall 0.4930, precision 0.3933, accuracy 0.7959, and MCC 0.3178. Its repeated-CV table also ranks XGB first (F1 0.5218) and LR at 0.4735. The bundled calibration, threshold, subgroup, error-analysis, and statistical-test artifacts identify XGB as the evaluated serving model. These are different experiment snapshots; do not combine them as if they describe the deployed LR. Reproduce the paper's exact split and training procedure and regenerate all dependent artifacts before using performance or fairness claims for LR.

In the bundled XGB holdout run, accuracy is 0.8413, precision 0.5085, recall 0.4225, F1 0.4615, ROC-AUC 0.7763, and PR-AUC 0.4683. This is 30 true positives, 41 false negatives, 29 false positives, and 341 true negatives. These point estimates have sampling uncertainty and should not be treated as expected performance elsewhere.

In the bundled XGB run, repeated 5x5 CV mean F1 is 0.5218 (standard deviation 0.065); mean ROC-AUC is 0.8120 (standard deviation 0.033). Its holdout calibration is summarized by Brier score 0.1262 and 10-bin ECE 0.0862. Calibration on this small sample does not guarantee that an individual probability is accurate.

The current subgroup table shows recall of 0.083 for age >45 (n=80), versus 0.571 for age <=30 (n=111), and recall of 0.200 for job levels 3+ (n=108), versus 0.459 for job levels 1-2 (n=333). These are risk signals requiring investigation, not a complete fairness assessment or legal conclusion. Excluding direct demographic features does not prove fairness.

## Known limitations and required work before any real use

- Only 1,470 historical records from one public benchmark are available; there is no external-employer or temporal evaluation.
- Attrition is a historical outcome, not a causal explanation, an employee's intent, or proof that any intervention will help.
- The target is imbalanced (237 leavers, 16.12%); at the fixed threshold, the holdout misses 41 of 71 leavers.
- The 0.50 threshold is a transparent demo default, not a cost-optimized or operationally approved cutoff.
- Group sizes and positive outcome counts are limited; subgroup rates are unstable and incomplete. Proxy effects may remain.
- Scores may be unreliable outside the source population, time period, feature ranges, categories, and joint input distributions. Range checks detect only univariate numeric extrapolation, not all out-of-distribution inputs or drift.
- Holdout comparisons are exploratory, use one split, and are not evidence of broad model superiority or causality.
- This project does not supply an organization-specific privacy, security, labor-law, accessibility, or impact assessment.

Before any proposed operational use, require lawful-purpose review, representative data, independent temporal and external validation, a full impact/fairness assessment with uncertainty intervals, calibrated probabilities for the intended population, a stakeholder-approved threshold/cost policy, privacy/security review, monitoring and rollback controls, and a process preventing scores from triggering adverse action.

## Source of truth

This model card and the current generated artifacts supersede numeric model claims in earlier reports. `README.md`, `PIPELINE_REPORT.md`, `PROJECT_REPORT.md`, and `MLFINAL_paper.md` contain historical experiment results and are not the current serving-model specification.
