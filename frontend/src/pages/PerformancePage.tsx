import { useEffect, useState } from 'react';
import { Activity, Award, Target, TrendingUp } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import type { PerformanceData } from '../types';
import { ErrorState, LoadingState, MetricCard, modelLabel, PageHeader, Panel, percent } from '../components/ui';

export function PerformancePage() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const load = () => {
    setLoading(true); setError('');
    api.performance().then(setData).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Metrics unavailable.')).finally(() => setLoading(false));
  };
  useEffect(load, []);

  if (loading) return <><PageHeader eyebrow="Evaluation" title="Model performance" description="Compare real holdout results and repeated cross-validation metrics." /><LoadingState /></>;
  if (error || !data) return <><PageHeader eyebrow="Evaluation" title="Model performance" description="Compare real holdout results and repeated cross-validation metrics." /><ErrorState message={error || 'Metrics unavailable.'} onRetry={load} /></>;

  const champion = data.champion;
  const chartRows = [...data.all_models].sort((a, b) => b.F1 - a.F1).map((row) => ({
    model: modelLabel(row.Model), Precision: row.Precision, Recall: row.Recall, F1: row.F1,
  }));
  const accuracyRows = [...data.all_models].sort((a, b) => b.Accuracy - a.Accuracy).map((row) => ({
    model: modelLabel(row.Model), Accuracy: row.Accuracy,
  }));
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Evaluation" title="Model performance" description="Real evaluation results for the trained models. The held-out test set is reported separately from repeated cross-validation." />
      <div className="champion-strip"><div className="champion-icon"><Award size={18} /></div><div><span>Selected by {data.criterion}</span><strong>{champion ? modelLabel(champion.Model) : 'No champion available'}</strong></div>{champion && <div className="champion-f1"><span>F1</span><strong>{percent(champion.F1, 2)}</strong></div>}</div>

      {champion && <div className="metric-grid four-metrics">
        <MetricCard label="Holdout ROC-AUC" value={percent(champion['ROC-AUC'], 2)} note={modelLabel(champion.Model)} icon={Activity} tone="blue" />
        <MetricCard label="Holdout precision" value={percent(champion.Precision, 2)} note="Positive predictions correct" icon={Target} />
        <MetricCard label="Holdout recall" value={percent(champion.Recall, 2)} note="Actual attrition detected" icon={TrendingUp} tone="amber" />
        <MetricCard label="Holdout MCC" value={champion.MCC.toFixed(3)} note="Balanced correlation metric" icon={Award} tone="green" />
      </div>}

      <Panel title="Holdout accuracy comparison" description="Accuracy by model on the held-out test set. Read alongside precision, recall and F1 below.">
        <div className="chart-wrap accuracy-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={accuracyRows} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--chart-grid)" />
          <XAxis type="number" domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tick={{ fill: 'var(--text-subtle)', fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="model" width={165} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip formatter={(value) => percent(Number(value ?? 0), 1)} contentStyle={{ borderColor: 'var(--border)', background: 'var(--surface)' }} />
          <Bar dataKey="Accuracy" fill="var(--chart-secondary)" radius={[0, 4, 4, 0]} barSize={11} />
        </BarChart></ResponsiveContainer></div>
      </Panel>

      <Panel title="Precision, recall & F1" description="Metrics shown side by side because attrition is an imbalanced classification problem.">
        <div className="chart-wrap performance-chart"><ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartRows} margin={{ top: 8, right: 14, bottom: 70, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--chart-grid)" />
            <XAxis dataKey="model" angle={-34} textAnchor="end" interval={0} height={82} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fill: 'var(--text-subtle)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(value) => percent(Number(value ?? 0), 1)} contentStyle={{ borderColor: 'var(--border)', background: 'var(--surface)' }} />
            <Legend />
            <Bar dataKey="Precision" fill="var(--chart-secondary)" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Recall" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            <Bar dataKey="F1" fill="var(--chart-tertiary)" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer></div>
      </Panel>

      <Panel title="Holdout model scorecard" description="Sorted by F1-score; all values are from the saved 30% holdout evaluation.">
        <div className="table-scroll"><table className="data-table"><thead><tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>MCC</th></tr></thead>
          <tbody>{data.holdout.map((row) => <tr key={`${row.Rank}-${row.Model}`} className={row.Rank === 1 ? 'top-row' : ''}>
            <td><span className="rank-pill">{row.Rank}</span></td><td className="model-cell">{modelLabel(row.Model)}</td>
            <td>{percent(row.Accuracy, 1)}</td><td>{percent(row.Precision, 1)}</td><td>{percent(row.Recall, 1)}</td><td className="metric-strong">{percent(row.F1, 1)}</td><td>{percent(row['ROC-AUC'], 1)}</td><td>{percent(row['PR-AUC'], 1)}</td><td>{row.MCC.toFixed(3)}</td>
          </tr>)}</tbody></table></div>
      </Panel>

      <Panel title="Repeated 5×5 cross-validation" description="Mean ± standard deviation across repeated stratified folds. These results are distinct from the holdout scorecard.">
        <div className="table-scroll"><table className="data-table"><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Balanced accuracy</th><th>MCC</th></tr></thead>
          <tbody>{data.cross_validation.map((row) => <tr key={row.Model}><td className="model-cell">{modelLabel(row.Model)}</td><td>{row['ROC-AUC']}</td><td>{row['PR-AUC']}</td><td>{row.Accuracy}</td><td>{row.Precision}</td><td>{row.Recall}</td><td>{row.F1}</td><td>{row.BalAcc}</td><td>{row.MCC}</td></tr>)}</tbody>
        </table></div>
      </Panel>

      <div className="grid-two">
        <Panel title="ROC and precision–recall curves" description="Generated from model predictions on the holdout set."><img className="evaluation-figure" src={data.figures.roc_pr} alt="ROC and precision-recall curves for the evaluated models" loading="lazy" /></Panel>
        <Panel title="Confusion matrices" description="Holdout classification counts by model at the evaluation threshold."><img className="evaluation-figure" src={data.figures.confusion_matrices} alt="Confusion matrices for the evaluated models" loading="lazy" /></Panel>
      </div>
      <p className="source-note">Do not select a model by accuracy alone. The underlying dataset has a 16.12% attrition rate; precision, recall, F1 and MCC provide additional context.</p>
    </div>
  );
}