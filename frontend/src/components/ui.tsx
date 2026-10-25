import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { AlertCircle, LoaderCircle } from 'lucide-react';

export function PageHeader({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="page-heading-row">
      <div>
        {eyebrow && <div className="page-eyebrow">{eyebrow}</div>}
        <h1 className="page-title">{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {action && <div className="page-heading-action">{action}</div>}
    </div>
  );
}

export function Panel({ title, description, action, children, className = '' }: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || description || action) && <div className="panel-header">
        <div>{title && <h2 className="panel-title">{title}</h2>}{description && <p className="panel-description">{description}</p>}</div>
        {action && <div className="panel-action">{action}</div>}
      </div>}
      {children}
    </section>
  );
}

export function MetricCard({ label, value, note, icon: Icon, tone = 'neutral' }: {
  label: string;
  value: string;
  note: string;
  icon: LucideIcon;
  tone?: 'neutral' | 'green' | 'amber' | 'blue';
}) {
  return (
    <article className="metric-card">
      <div className="metric-top"><span>{label}</span><span className={`metric-icon tone-${tone}`}><Icon size={16} strokeWidth={1.8} /></span></div>
      <div className="metric-value tabular">{value}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}

export function LoadingState({ label = 'Loading evaluation data' }: { label?: string }) {
  return <div className="state-message"><LoaderCircle className="spin" size={18} /><span>{label}</span></div>;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="error-state" role="alert"><AlertCircle size={18} /><div><strong>Could not load this view</strong><p>{message}</p>{onRetry && <button className="text-button" onClick={onRetry}>Try again</button>}</div></div>;
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="empty-state"><strong>{title}</strong><p>{description}</p></div>;
}

export function modelLabel(id: string) {
  const labels: Record<string, string> = {
    et_base: 'Extra Trees', etpluscatboost: 'Extra Trees + CatBoost', etpluslightgbm: 'Extra Trees + LightGBM',
    gb_base: 'Gradient Boosting', gb_tuned_balanced_accuracy: 'Gradient Boosting (Balanced Accuracy)',
    gb_tuned_f1: 'Gradient Boosting (F1)', gb_tuned_pr_auc: 'Gradient Boosting (PR-AUC)',
    gb_tuned_recall: 'Gradient Boosting (Recall)', gb_tuned_roc_auc: 'Gradient Boosting (ROC-AUC)',
    lr_base: 'Logistic Regression', rf_base: 'Random Forest', rfplusadaboost: 'Random Forest + AdaBoost',
    rfpluslightgbm: 'Random Forest + LightGBM', stacking_ensemble: 'Stacking Ensemble',
    svm_base: 'Support Vector Machine', voting_ensemble: 'Soft Voting Ensemble', xgb_base: 'XGBoost',
  };
  return labels[id] ?? id.replaceAll('_', ' ');
}

export const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
export const compact = (value: number) => new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);