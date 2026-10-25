import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import type { InsightsData } from '../types';
import { ErrorState, LoadingState, PageHeader, Panel, percent } from '../components/ui';

export function InsightsPage() {
  const [data, setData] = useState<InsightsData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const load = () => {
    setLoading(true); setError('');
    api.insights().then(setData).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Insights unavailable.')).finally(() => setLoading(false));
  };
  useEffect(load, []);
  if (loading) return <><PageHeader eyebrow="Evaluation" title="Employee insights" description="Aggregate feature, subgroup and error-analysis results from the model pipeline." /><LoadingState /></>;
  if (error || !data) return <><PageHeader eyebrow="Evaluation" title="Employee insights" description="Aggregate feature, subgroup and error-analysis results from the model pipeline." /><ErrorState message={error || 'Insights unavailable.'} onRetry={load} /></>;

  const importance = data.feature_importance.slice(0, 12).map((item) => ({ name: item.Feature.replaceAll('_', ' '), value: item.Importance }));
  const shap = data.shap_importance.slice(0, 12).map((item) => ({ name: item.Feature.replaceAll('_', ' '), value: item.MeanAbsSHAP }));
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Evaluation" title="Employee insights" description="Explore aggregate patterns captured by the feature selector and validation analyses." />
      <div className="grid-two">
        <Panel title="Feature selection importance" description="Gradient-boosting selector importance across the training features.">
          <div className="chart-wrap insights-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={importance} layout="vertical" margin={{ top: 6, right: 18, bottom: 0, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--chart-grid)" />
            <XAxis type="number" tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--text-subtle)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={172} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(value) => `${(Number(value ?? 0) * 100).toFixed(2)}%`} contentStyle={{ borderColor: 'var(--border)', background: 'var(--surface)' }} />
            <Bar dataKey="value" fill="var(--accent)" radius={[0, 4, 4, 0]} barSize={11} />
          </BarChart></ResponsiveContainer></div>
        </Panel>
        <Panel title="Global SHAP importance" description="Mean absolute SHAP value across the evaluated sample; global, not employee-specific explanations.">
          <div className="chart-wrap insights-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={shap} layout="vertical" margin={{ top: 6, right: 18, bottom: 0, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--chart-grid)" />
            <XAxis type="number" tick={{ fill: 'var(--text-subtle)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={172} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(value) => Number(value ?? 0).toFixed(3)} contentStyle={{ borderColor: 'var(--border)', background: 'var(--surface)' }} />
            <Bar dataKey="value" fill="var(--chart-secondary)" radius={[0, 4, 4, 0]} barSize={11} />
          </BarChart></ResponsiveContainer></div>
        </Panel>
      </div>

      <Panel title="Subgroup evaluation" description="Holdout recall, precision and false-positive rate by the pipeline's reported groups.">
        <div className="table-scroll"><table className="data-table"><thead><tr><th>Subgroup</th><th>Employees</th><th>Recall</th><th>Precision</th><th>F1</th><th>False-positive rate</th></tr></thead><tbody>
          {data.subgroup_fairness.map((group) => <tr key={group.Subgroup}><td className="model-cell">{group.Subgroup}</td><td>{group.n}</td><td>{percent(group.Recall, 1)}</td><td>{percent(group.Precision, 1)}</td><td>{percent(group.F1, 1)}</td><td>{percent(group.FPR, 1)}</td></tr>)}
        </tbody></table></div>
        <p className="panel-footnote">These descriptive slices are not a complete fairness audit or evidence of causal relationships. Small subgroups may produce unstable estimates.</p>
      </Panel>

      <Panel title="Error profiles" description="Aggregate characteristics of false-negative and false-positive cases in the saved analysis.">
        <div className="table-scroll"><table className="data-table"><thead><tr><th>Error type</th><th>High income</th><th>Senior job level</th><th>High satisfaction</th><th>Overtime</th><th>Low tenure</th></tr></thead><tbody>
          {data.error_analysis.map((item) => <tr key={item.Type}><td className="model-cell">{item.Type}</td><td>{percent(item['high_income_%'] / 100, 1)}</td><td>{percent(item['senior_joblevel_%'] / 100, 1)}</td><td>{percent(item['high_satisfaction_%'] / 100, 1)}</td><td>{percent(item['overtime_%'] / 100, 1)}</td><td>{percent(item['low_tenure_%'] / 100, 1)}</td></tr>)}
        </tbody></table></div>
      </Panel>

      <Panel title="Selected model inputs" description="The 16 features selected for the final models by the pipeline.">
        <div className="feature-chip-list">{data.selected_features.map((item) => <span className="feature-chip" key={item.Rank}><b>{String(item.Rank).padStart(2, '0')}</b>{item.Feature.replaceAll('_', ' ')}</span>)}</div>
      </Panel>
      <p className="source-note">Importance values describe model behavior in this dataset. They should not be interpreted as causal effects or reasons to make an employment decision.</p>
    </div>
  );
}