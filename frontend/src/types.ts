export type ModelInfo = { id: string; artifact: string };

export type EmployeeInput = {
  BusinessTravel: 'Non-Travel' | 'Travel_Frequently' | 'Travel_Rarely';
  Department: 'Human Resources' | 'Research & Development' | 'Sales';
  JobInvolvement: number;
  JobLevel: number;
  JobRole: string;
  MonthlyIncome: number;
  OverTime: 'Yes' | 'No';
  StockOptionLevel: number;
  WorkLifeBalance: number;
  YearsWithCurrManager: number;
  JobSatisfaction: number;
  EnvironmentSatisfaction: number;
  RelationshipSatisfaction: number;
};

export type PredictionResult = {
  model: string;
  prediction: 'Yes' | 'No';
  attrition_probability: number;
  threshold: number;
  out_of_training_range_fields: string[];
};

export type ModelMetrics = {
  Model: string;
  Threshold: number;
  'ROC-AUC': number;
  'PR-AUC': number;
  Accuracy: number;
  Precision: number;
  Recall: number;
  F1: number;
  BalAcc: number;
  MCC: number;
  Rank: number;
};

export type CrossValidationMetrics = {
  Model: string;
  'ROC-AUC': string;
  'PR-AUC': string;
  Accuracy: string;
  Precision: string;
  Recall: string;
  F1: string;
  BalAcc: string;
  MCC: string;
};

export type FeatureImportance = { Feature: string; Importance: number };
export type ShapImportance = { Feature: string; MeanAbsSHAP: number };
export type FairnessGroup = {
  Model: string;
  Threshold: number;
  Subgroup: string;
  n: number;
  Recall: number;
  Precision: number;
  F1: number;
  FPR: number;
};
export type ErrorProfile = {
  Model: string;
  Threshold: number;
  Type: string;
  'high_income_%': number;
  'senior_joblevel_%': number;
  'high_satisfaction_%': number;
  'overtime_%': number;
  'low_tenure_%': number;
};

export type PerformanceData = {
  criterion: string;
  champion: ModelMetrics | null;
  holdout: ModelMetrics[];
  all_models: Omit<ModelMetrics, 'Rank' | 'Threshold'>[];
  cross_validation: CrossValidationMetrics[];
  threshold_sweep: Array<{ Model: string; Threshold: number; Precision: number; Recall: number; F1: number; Accuracy: number; BalAcc: number; MCC: number }>;
  calibration: Array<{ Model: string; Threshold: number; Brier: number; ECE: number }>;
  statistical_tests: Array<Record<string, string | number | null>>;
  figures: { roc_pr: string; confusion_matrices: string; decision_curve: string };
};

export type InsightsData = {
  feature_importance: FeatureImportance[];
  shap_importance: ShapImportance[];
  subgroup_fairness: FairnessGroup[];
  error_analysis: ErrorProfile[];
  selected_features: Array<{ Rank: number; Feature: string }>;
};

export type HistoryEntry = {
  id: string;
  createdAt: string;
  model: string;
  prediction: PredictionResult['prediction'];
  probability: number;
};
