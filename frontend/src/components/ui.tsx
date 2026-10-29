import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { AlertCircle } from 'lucide-react';

/* ── PAGE HEADER ──────────────────────────────────────────────────────────── */
export function PageHeader({
  eyebrow, title, description, action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
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

/* ── PANEL ────────────────────────────────────────────────────────────────── */
export function Panel({
  title, description, action, children, className = '',
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || description || action) && (
        <div className="panel-header">
          <div>
            {title && <h2 className="panel-title">{title}</h2>}
            {description && <p className="panel-description">{description}</p>}
          </div>
          {action && <div className="panel-action">{action}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/* ── METRIC CARD ──────────────────────────────────────────────────────────── */
export type MetricTone = 'default' | 'accent' | 'red' | 'green' | 'slate';

export function MetricCard({
  label, value, note, icon: Icon, tone = 'default',
}: {
  label: string;
  value: string;
  note: string;
  icon: LucideIcon;
  tone?: MetricTone;
}) {
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span className="metric-label">{label}</span>
        <span className={`metric-icon ${tone}`}>
          <Icon size={15} strokeWidth={1.8} />
        </span>
      </div>
      <div className="metric-value tabular">{value}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}

/* ── LOADING STATE — shimmer skeleton ────────────────────────────────────── */
export function LoadingState({ rows = 1 }: { rows?: number }) {
  return (
    <div className="state-loading fade-in">
      <div className="shimmer-bar" style={{ marginBottom: 4 }} />
      <div className="loading-shimmer-grid">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="shimmer-card">
            <div className="shimmer-line shimmer-block h-sm w-60" />
            <div className="shimmer-line shimmer-block h-lg" style={{ marginTop: 8 }} />
            <div className="shimmer-line shimmer-block h-sm w-40" style={{ marginTop: 6 }} />
          </div>
        ))}
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="shimmer-card shimmer-block"
          style={{ height: 200, borderRadius: 8 }}
        />
      ))}
    </div>
  );
}

/* ── ERROR STATE ──────────────────────────────────────────────────────────── */
export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-error" role="alert">
      <AlertCircle size={17} style={{ flexShrink: 0 }} />
      <div>
        <strong>Could not load this view</strong>
        <p>{message}</p>
        {onRetry && (
          <button className="text-link" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    </div>
  );
}

/* ── EMPTY STATE ──────────────────────────────────────────────────────────── */
export function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  );
}

/* ── MODEL LABEL MAP ──────────────────────────────────────────────────────── */
export function modelLabel(id: string) {
  const labels: Record<string, string> = {
    et_base:                    'Extra Trees',
    etpluscatboost:             'Extra Trees + CatBoost',
    etpluslightgbm:             'Extra Trees + LightGBM',
    gb_base:                    'Gradient Boosting',
    gb_tuned_balanced_accuracy: 'GB (Balanced Acc.)',
    gb_tuned_f1:                'GB (F1)',
    gb_tuned_pr_auc:            'GB (PR-AUC)',
    gb_tuned_recall:            'GB (Recall)',
    gb_tuned_roc_auc:           'GB (ROC-AUC)',
    lr_base:                    'Logistic Regression',
    rf_base:                    'Random Forest',
    rfplusadaboost:             'RF + AdaBoost',
    rfpluslightgbm:             'RF + LightGBM',
    stacking_ensemble:          'Stacking Ensemble',
    svm_base:                   'Support Vector Machine',
    voting_ensemble:            'Soft Voting Ensemble',
    xgb_base:                   'XGBoost',
  };
  return labels[id] ?? id.replaceAll('_', ' ');
}

export const percent = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
export const compact = (v: number) =>
  new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(v);