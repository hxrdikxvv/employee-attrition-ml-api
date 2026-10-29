import { useEffect, useState } from 'react';
import { Activity, Award, Target, TrendingUp } from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { api } from '../api';
import type { PerformanceData } from '../types';
import { ErrorState, LoadingState, MetricCard, modelLabel, PageHeader, Panel, percent } from '../components/ui';

const TT_STYLE = {
  background: '#121215',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: 6,
  color: '#f4f4f5',
  fontSize: 11,
  boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
};

export function PerformancePage() {
  const [data, setData]   = useState<PerformanceData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true); setError('');
    api.performance()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Metrics unavailable.'))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  if (loading)
    return <><PageHeader eyebrow="Evaluation" title="Model Performance" /><LoadingState rows={3} /></>;
  if (error || !data)
    return <><PageHeader eyebrow="Evaluation" title="Model Performance" /><ErrorState message={error || 'Metrics unavailable.'} onRetry={load} /></>;

  const champ       = data.champion;
  const chartRows   = [...data.all_models].sort((a,b) => b.F1 - a.F1).map(r => ({
    model: modelLabel(r.Model), Precision: r.Precision, Recall: r.Recall, F1: r.F1,
  }));
  const accRows     = [...data.all_models].sort((a,b) => b.Accuracy - a.Accuracy).map(r => ({
    model: modelLabel(r.Model), Accuracy: r.Accuracy,
  }));

  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="Evaluation"
        title="Model Performance"
        description="The selected champion follows the final paper; the bundled scorecard is from a separate XGBoost experiment run."
      />

      {/* Champion */}
      <div className="champion-strip">
        <div className="champion-icon"><Award size={17} /></div>
        <div className="champion-strip-info">
          <span>Selected by {data.criterion}</span>
          <strong>{champ ? modelLabel(champ.Model) : 'No champion'}</strong>
        </div>
        {champ && (
          <div className="champion-f1">
            <span>Holdout F1</span>
            <strong className="tabular">{percent(champ.F1, 2)}</strong>
          </div>
        )}
      </div>

      {/* Champion KPIs */}
      {champ && (
        <div className="grid-4">
          <MetricCard label="ROC-AUC"    value={percent(champ['ROC-AUC'],2)} note={modelLabel(champ.Model)} icon={Activity}   tone="slate"  />
          <MetricCard label="Precision"  value={percent(champ.Precision,2)}  note="Positive predictions correct" icon={Target} tone="default" />
          <MetricCard label="Recall"     value={percent(champ.Recall,2)}     note="Actual attrition caught"  icon={TrendingUp} tone="accent" />
          <MetricCard label="MCC"        value={champ.MCC.toFixed(3)}        note="Balanced correlation"      icon={Award}     tone="green"  />
        </div>
      )}

      {/* Accuracy chart */}
      <Panel
        title="Holdout accuracy"
        description="Accuracy per model on the 30% holdout test set — read alongside precision, recall and F1."
      >
        <div className="chart-wrap chart-h-xl">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={accRows} layout="vertical" margin={{ top: 4, right: 14, bottom: 0, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
              <XAxis
                type="number" domain={[0,1]}
                tickFormatter={(v: number) => `${Math.round(v*100)}%`}
                tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <YAxis
                type="category" dataKey="model" width={162}
                tick={{ fill: '#a1a1aa', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <Tooltip formatter={v => percent(Number(v),1)} contentStyle={TT_STYLE} />
              <Bar dataKey="Accuracy" fill="#6b7fa3" radius={[0,3,3,0]} barSize={10} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      {/* Precision / Recall / F1 chart */}
      <Panel
        title="Precision, Recall & F1"
        description="Side-by-side comparison — critical for imbalanced classification."
      >
        <div className="chart-wrap chart-h-xl">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartRows} margin={{ top: 6, right: 12, bottom: 72, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="model" angle={-34} textAnchor="end" interval={0} height={80}
                tick={{ fill: '#a1a1aa', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <YAxis
                domain={[0,1]} tickFormatter={(v: number) => `${Math.round(v*100)}%`}
                tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <Tooltip formatter={v => percent(Number(v),1)} contentStyle={TT_STYLE} />
              <Legend
                wrapperStyle={{ paddingTop: 12, fontSize: 11, color: '#a1a1aa' }}
              />
              <Bar dataKey="Precision" fill="#6b7fa3" radius={[3,3,0,0]} />
              <Bar dataKey="Recall"    fill="#e07b39" radius={[3,3,0,0]} />
              <Bar dataKey="F1"        fill="#8a6bbf" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      {/* Holdout scorecard table */}
      <Panel
        title="Holdout scorecard"
        description="All models sorted by F1 — from the saved 30% holdout evaluation."
      >
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th><th>Model</th><th>Accuracy</th><th>Precision</th>
                <th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>MCC</th>
              </tr>
            </thead>
            <tbody>
              {data.holdout.map(row => (
                <tr key={`${row.Rank}-${row.Model}`} className={row.Rank === 1 ? 'top-row' : ''}>
                  <td><span className="rank-pip">{row.Rank}</span></td>
                  <td className="model-cell">{modelLabel(row.Model)}</td>
                  <td>{percent(row.Accuracy, 1)}</td>
                  <td>{percent(row.Precision, 1)}</td>
                  <td>{percent(row.Recall, 1)}</td>
                  <td className="val-accent">{percent(row.F1, 1)}</td>
                  <td>{percent(row['ROC-AUC'], 1)}</td>
                  <td>{percent(row['PR-AUC'], 1)}</td>
                  <td>{row.MCC.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* CV table */}
      <Panel
        title="Repeated 5×5 cross-validation"
        description="Mean ± std across repeated stratified folds — distinct from the holdout scorecard."
      >
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Accuracy</th>
                <th>Precision</th><th>Recall</th><th>F1</th><th>Bal. Acc.</th><th>MCC</th>
              </tr>
            </thead>
            <tbody>
              {data.cross_validation.map(row => (
                <tr key={row.Model}>
                  <td className="model-cell">{modelLabel(row.Model)}</td>
                  <td>{row['ROC-AUC']}</td><td>{row['PR-AUC']}</td>
                  <td>{row.Accuracy}</td><td>{row.Precision}</td>
                  <td>{row.Recall}</td><td>{row.F1}</td>
                  <td>{row.BalAcc}</td><td>{row.MCC}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Figures */}
      <div className="grid-2">
        <Panel title="ROC & PR curves" description="Generated from holdout predictions.">
          <img className="eval-figure" src={data.figures.roc_pr} alt="ROC and PR curves" loading="lazy" />
        </Panel>
        <Panel title="Confusion matrices" description="Holdout classification counts at evaluation threshold.">
          <img className="eval-figure" src={data.figures.confusion_matrices} alt="Confusion matrices" loading="lazy" />
        </Panel>
      </div>

      <p className="source-note">
        Accuracy alone is insufficient on this 16.12% attrition-rate dataset. Review F1, recall and MCC together.
      </p>
    </div>
  );
}
