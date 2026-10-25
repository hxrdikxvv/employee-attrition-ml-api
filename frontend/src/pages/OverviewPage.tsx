import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowRight, CircleHelp, Target, TrendingUp, Users } from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';
import { api } from '../api';
import type { InsightsData, PerformanceData } from '../types';
import { ErrorState, LoadingState, MetricCard, modelLabel, PageHeader, Panel, percent } from '../components/ui';

const datasetSplit = [
  { name: 'Stayed', value: 1233, color: '#1d715b' },
  { name: 'Left', value: 237, color: '#d59745' },
];

export function OverviewPage() {
  const [performance, setPerformance] = useState<PerformanceData | null>(null);
  const [insights, setInsights] = useState<InsightsData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    setError('');
    Promise.all([api.performance(), api.insights()])
      .then(([metrics, insightData]) => { setPerformance(metrics); setInsights(insightData); })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Could not load dashboard data.'))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  if (loading) return <><PageHeader eyebrow="HR analytics" title="Overview" description="A concise view of the employee attrition model and evaluation cohort." /><LoadingState /></>;
  if (error || !performance || !insights) return <><PageHeader eyebrow="HR analytics" title="Overview" description="A concise view of the employee attrition model and evaluation cohort." /><ErrorState message={error || 'Evaluation data is unavailable.'} onRetry={load} /></>;

  const rows = performance.all_models;
  const bestAccuracy = Math.max(...rows.map((row) => row.Accuracy));
  const bestAuc = Math.max(...rows.map((row) => row['ROC-AUC']));
  const bestAccuracyModel = rows.find((row) => row.Accuracy === bestAccuracy)?.Model ?? '—';
  const bestAucModel = rows.find((row) => row['ROC-AUC'] === bestAuc)?.Model ?? '—';
  const comparison = [...rows].sort((a, b) => b.F1 - a.F1).slice(0, 7).map((row) => ({
    name: modelLabel(row.Model), F1: row.F1, 'ROC-AUC': row['ROC-AUC'],
  }));
  const topFeatures = insights.feature_importance.slice(0, 6).map((row) => ({
    name: row.Feature.replaceAll('_', ' '), importance: row.Importance,
  }));

  return (
    <div className="page-stack">
      <PageHeader eyebrow="HR analytics" title="Overview" description="A concise view of the employee attrition model and evaluation cohort." action={<Link className="button button-primary" to="/predict">New prediction <ArrowRight size={15} /></Link>} />

      <div className="metric-grid">
        <MetricCard label="Employees in dataset" value="1,470" note="Source dataset records" icon={Users} />
        <MetricCard label="Observed attrition" value="16.12%" note="237 employees left in source data" icon={Activity} tone="amber" />
        <MetricCard label="Best holdout accuracy" value={percent(bestAccuracy, 2)} note={`${modelLabel(bestAccuracyModel)} · holdout`} icon={Target} tone="blue" />
        <MetricCard label="Best holdout ROC-AUC" value={percent(bestAuc, 2)} note={`${modelLabel(bestAucModel)} · holdout`} icon={TrendingUp} tone="green" />
      </div>

      <div className="grid-two overview-grid">
        <Panel title="Dataset attrition distribution" description="Observed labels in the original 1,470-row dataset." action={<span className="data-tag">Source data</span>}>
          <div className="distribution-layout">
            <div className="donut-wrap">
              <ResponsiveContainer width="100%" height={210}>
                <PieChart>
                  <Pie data={datasetSplit} dataKey="value" nameKey="name" innerRadius={62} outerRadius={86} paddingAngle={2} stroke="none">
                    {datasetSplit.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                  </Pie>
                  <Tooltip formatter={(value) => [Number(value ?? 0), 'Employees']} />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center"><strong>1,470</strong><span>employees</span></div>
            </div>
            <div className="legend-list">
              {datasetSplit.map((entry) => (
                <div className="legend-row" key={entry.name}><span className="legend-name"><i style={{ backgroundColor: entry.color }} />{entry.name}</span><strong>{entry.value.toLocaleString()} <small>{percent(entry.value / 1470, 1)}</small></strong></div>
              ))}
              <div className="insight-note"><CircleHelp size={14} /><span>Attrition is the minority class; accuracy alone can be misleading.</span></div>
            </div>
          </div>
        </Panel>

        <Panel title="Model comparison" description="Top models by held-out F1; ROC-AUC shown for ranking context." action={<Link className="panel-link" to="/performance">Full comparison <ArrowRight size={14} /></Link>}>
          <div className="chart-wrap chart-tall">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparison} layout="vertical" margin={{ top: 4, right: 14, bottom: 0, left: 8 }} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--chart-grid)" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tick={{ fill: 'var(--text-subtle)', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={150} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(value) => percent(Number(value ?? 0), 1)} contentStyle={{ borderColor: 'var(--border)', background: 'var(--surface)' }} />
                <Bar dataKey="F1" fill="var(--accent)" radius={[0, 4, 4, 0]} barSize={10} />
                <Bar dataKey="ROC-AUC" fill="var(--chart-secondary)" radius={[0, 4, 4, 0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-legend"><span><i className="legend-accent" />F1</span><span><i className="legend-secondary" />ROC-AUC</span><span>Holdout evaluation</span></div>
        </Panel>
      </div>

      <div className="grid-two overview-grid lower-grid">
        <Panel title="Most influential features" description="Gradient-boosting importance from the trained feature selector." action={<Link className="panel-link" to="/insights">Explore insights <ArrowRight size={14} /></Link>}>
          <div className="chart-wrap feature-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topFeatures} layout="vertical" margin={{ top: 2, right: 18, bottom: 0, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--chart-grid)" />
                <XAxis type="number" tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} tick={{ fill: 'var(--text-subtle)', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={168} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(value) => `${(Number(value ?? 0) * 100).toFixed(2)}%`} contentStyle={{ borderColor: 'var(--border)', background: 'var(--surface)' }} />
                <Bar dataKey="importance" fill="var(--accent)" radius={[0, 4, 4, 0]} barSize={12} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel title="Model selection" description="The project selects its champion using held-out F1, not accuracy alone.">
          <div className="champion-card">
            <div className="champion-mark"><Target size={18} /></div>
            <div><span className="data-tag">F1 rank #1</span><h3>{performance.champion ? modelLabel(performance.champion.Model) : 'Unavailable'}</h3><p>Best F1 score on the held-out evaluation set.</p></div>
            {performance.champion && <strong className="champion-score tabular">{percent(performance.champion.F1, 2)}<small>F1</small></strong>}
          </div>
          <div className="notice-line"><CircleHelp size={15} /><span>The dataset is imbalanced. Compare precision and recall alongside accuracy when reviewing results.</span></div>
        </Panel>
      </div>
      <p className="source-note">Source: IBM HR Analytics Employee Attrition dataset · Model metrics are from the saved holdout evaluation.</p>
    </div>
  );
}