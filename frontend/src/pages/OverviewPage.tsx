import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowRight, CircleHelp, Target, TrendingUp, Users } from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { api } from '../api';
import type { InsightsData, PerformanceData } from '../types';
import { ErrorState, LoadingState, MetricCard, modelLabel, PageHeader, Panel, percent } from '../components/ui';

const DS_COLORS = { stayed: '#52a882', left: '#e07b39' };
const datasetSplit = [
  { name: 'Stayed', value: 1233, color: DS_COLORS.stayed },
  { name: 'Left',   value: 237,  color: DS_COLORS.left   },
];

const TT_STYLE = {
  background: '#121215',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: 6,
  color: '#f4f4f5',
  fontSize: 11,
  boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
};

export function OverviewPage() {
  const [perf, setPerf]   = useState<PerformanceData | null>(null);
  const [ins,  setIns]    = useState<InsightsData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true); setError('');
    Promise.all([api.performance(), api.insights()])
      .then(([p, i]) => { setPerf(p); setIns(i); })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Could not load data.'))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  if (loading)
    return <><PageHeader eyebrow="HR Analytics" title="Overview" /><LoadingState rows={2} /></>;
  if (error || !perf || !ins)
    return <><PageHeader eyebrow="HR Analytics" title="Overview" /><ErrorState message={error || 'Data unavailable.'} onRetry={load} /></>;

  const rows       = perf.all_models;
  const bestAcc    = Math.max(...rows.map(r => r.Accuracy));
  const bestAuc    = Math.max(...rows.map(r => r['ROC-AUC']));
  const bestAccMdl = rows.find(r => r.Accuracy    === bestAcc)?.Model ?? '—';
  const bestAucMdl = rows.find(r => r['ROC-AUC']  === bestAuc)?.Model ?? '—';

  const comparison  = [...rows].sort((a, b) => b.F1 - a.F1).slice(0, 7).map(r => ({
    name: modelLabel(r.Model), F1: r.F1, 'ROC-AUC': r['ROC-AUC'],
  }));
  const topFeatures = ins.feature_importance.slice(0, 6).map(r => ({
    name: r.Feature.replaceAll('_', ' '), importance: r.Importance,
  }));

  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="HR Analytics"
        title="Overview"
        description="Model summary and evaluation cohort at a glance."
        action={
          <Link className="btn btn-primary" to="/predict">
            New prediction <ArrowRight size={14} />
          </Link>
        }
      />

      {/* KPI row */}
      <div className="grid-4">
        <MetricCard label="Dataset size"        value="1,470"              note="IBM HR Analytics employees"     icon={Users}      tone="default" />
        <MetricCard label="Observed attrition"  value="16.12%"             note="237 of 1,470 left"              icon={Activity}   tone="accent"  />
        <MetricCard label="Best accuracy"       value={percent(bestAcc,2)} note={`${modelLabel(bestAccMdl)} · holdout`} icon={Target} tone="slate" />
        <MetricCard label="Best ROC-AUC"        value={percent(bestAuc,2)} note={`${modelLabel(bestAucMdl)} · holdout`} icon={TrendingUp} tone="green" />
      </div>

      {/* Charts row */}
      <div className="grid-2">
        <Panel
          title="Attrition distribution"
          description="Observed class labels in the source 1,470-row dataset."
          action={<span className="tag">Source data</span>}
        >
          <div className="dist-layout">
            <div className="donut-wrap">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={datasetSplit} dataKey="value" nameKey="name"
                    innerRadius={58} outerRadius={82} paddingAngle={3} stroke="none"
                  >
                    {datasetSplit.map(e => <Cell key={e.name} fill={e.color} />)}
                  </Pie>
                  <Tooltip
                    formatter={v => [Number(v).toLocaleString(), 'Employees']}
                    contentStyle={TT_STYLE}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center">
                <strong>1,470</strong>
                <span>employees</span>
              </div>
            </div>
            <div className="legend-rows">
              {datasetSplit.map(e => (
                <div className="legend-row" key={e.name}>
                  <span className="legend-row-label">
                    <i style={{ background: e.color }} />{e.name}
                  </span>
                  <strong>
                    {e.value.toLocaleString()}
                    <small> {percent(e.value / 1470, 1)}</small>
                  </strong>
                </div>
              ))}
              <div className="callout" style={{ marginTop: 6 }}>
                <CircleHelp size={13} />
                <span className="callout-body">
                  Attrition is the minority class — accuracy alone is misleading.
                </span>
              </div>
            </div>
          </div>
        </Panel>

        <Panel
          title="Model comparison"
          description="Top 7 models by held-out F1. ROC-AUC shown for context."
          action={<Link className="panel-link" to="/performance">Full table <ArrowRight size={13} /></Link>}
        >
          <div className="chart-wrap chart-h-sm">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparison} layout="vertical" margin={{ top: 4, right: 12, bottom: 0, left: 4 }} barGap={3}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number" domain={[0,1]}
                  tickFormatter={(v: number) => `${Math.round(v*100)}%`}
                  tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <YAxis
                  type="category" dataKey="name" width={146}
                  tick={{ fill: '#a1a1aa', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <Tooltip formatter={v => percent(Number(v), 1)} contentStyle={TT_STYLE} />
                <Bar dataKey="F1"      fill="#e07b39" radius={[0,3,3,0]} barSize={9} />
                <Bar dataKey="ROC-AUC" fill="#6b7fa3" radius={[0,3,3,0]} barSize={9} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-footer">
            <span className="legend-item"><span className="legend-swatch" style={{ background: '#e07b39' }} />F1</span>
            <span className="legend-item"><span className="legend-swatch" style={{ background: '#6b7fa3' }} />ROC-AUC</span>
            <span className="legend-item" style={{ marginLeft: 'auto', color: '#3f3f46' }}>Holdout evaluation</span>
          </div>
        </Panel>
      </div>

      {/* Lower row */}
      <div className="grid-2">
        <Panel
          title="Feature importance"
          description="Top 6 features from gradient-boosting selector."
          action={<Link className="panel-link" to="/insights">Explore <ArrowRight size={13} /></Link>}
        >
          <div className="chart-wrap chart-h-xs">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topFeatures} layout="vertical" margin={{ top: 2, right: 14, bottom: 0, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number" tickFormatter={(v: number) => `${(v*100).toFixed(0)}%`}
                  tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <YAxis
                  type="category" dataKey="name" width={164}
                  tick={{ fill: '#a1a1aa', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <Tooltip formatter={v => `${(Number(v)*100).toFixed(2)}%`} contentStyle={TT_STYLE} />
                <Bar dataKey="importance" fill="#e07b39" radius={[0,3,3,0]} barSize={11} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Selected model" description="Champion selected to match RESEARCH_PAPER_FINAL.">
          <div className="champion-card">
            <div className="champion-badge"><Target size={17} /></div>
            <div>
              <span className="tag" style={{ marginBottom: 4 }}>Paper champion</span>
              <h3>{perf.champion ? modelLabel(perf.champion.Model) : 'Unavailable'}</h3>
              <p>Holdout score reported separately from CV selection.</p>
            </div>
            {perf.champion && (
              <div className="champion-score">
                <strong className="tabular">{percent(perf.champion.F1, 2)}</strong>
                <small>F1 holdout</small>
              </div>
            )}
          </div>
          <div className="notice-line">
            <CircleHelp size={14} />
            <span>Imbalanced dataset — compare precision, recall, and F1 alongside accuracy.</span>
          </div>
        </Panel>
      </div>

      <p className="source-note">
        Source: IBM HR Analytics Employee Attrition dataset. Metrics from saved holdout evaluation.
      </p>
    </div>
  );
}
