import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';
import type { InsightsData } from '../types';
import { ErrorState, LoadingState, PageHeader, Panel, percent } from '../components/ui';

const TT_STYLE = {
  background: '#121215',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: 6,
  color: '#f4f4f5',
  fontSize: 11,
  boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
};

export function InsightsPage() {
  const [data, setData]   = useState<InsightsData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true); setError('');
    api.insights()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Insights unavailable.'))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  if (loading)
    return <><PageHeader eyebrow="Evaluation" title="Employee Insights" /><LoadingState rows={3} /></>;
  if (error || !data)
    return <><PageHeader eyebrow="Evaluation" title="Employee Insights" /><ErrorState message={error || 'Unavailable.'} onRetry={load} /></>;

  const importance = data.feature_importance.slice(0,12).map(d => ({
    name: d.Feature.replaceAll('_',' '), value: d.Importance,
  }));
  const shap = data.shap_importance.slice(0,12).map(d => ({
    name: d.Feature.replaceAll('_',' '), value: d.MeanAbsSHAP,
  }));

  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="Evaluation"
        title="Employee Insights"
        description="Aggregate patterns from feature selection and validation analyses."
      />

      {/* Charts */}
      <div className="grid-2">
        <Panel
          title="Feature selection importance"
          description="Training-set importance — age, gender, and marital-status excluded."
        >
          <div className="chart-wrap chart-h-lg">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importance} layout="vertical" margin={{ top: 4, right: 14, bottom: 0, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number" tickFormatter={(v: number) => `${(v*100).toFixed(0)}%`}
                  tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <YAxis
                  type="category" dataKey="name" width={168}
                  tick={{ fill: '#a1a1aa', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <Tooltip formatter={v => `${(Number(v)*100).toFixed(2)}%`} contentStyle={TT_STYLE} />
                <Bar dataKey="value" fill="#e07b39" radius={[0,3,3,0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel
          title="Global SHAP importance"
          description="Gradient Boosting reference model — global, not employee-specific explanations."
        >
          <div className="chart-wrap chart-h-lg">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shap} layout="vertical" margin={{ top: 4, right: 14, bottom: 0, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number"
                  tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <YAxis
                  type="category" dataKey="name" width={168}
                  tick={{ fill: '#a1a1aa', fontSize: 10 }} axisLine={false} tickLine={false}
                />
                <Tooltip formatter={v => Number(v).toFixed(3)} contentStyle={TT_STYLE} />
                <Bar dataKey="value" fill="#6b7fa3" radius={[0,3,3,0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      {/* Subgroup table */}
      <Panel
        title="Subgroup evaluation"
        description={`Holdout results for ${data.subgroup_fairness[0]?.Model ?? 'the model'} at ${percent(data.subgroup_fairness[0]?.Threshold ?? 0.5, 0)} threshold — descriptive, not a fairness audit.`}
      >
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Subgroup</th><th>n</th><th>Recall</th>
                <th>Precision</th><th>F1</th><th>FPR</th>
              </tr>
            </thead>
            <tbody>
              {data.subgroup_fairness.map(g => (
                <tr key={g.Subgroup}>
                  <td className="model-cell">{g.Subgroup}</td>
                  <td>{g.n}</td>
                  <td>{percent(g.Recall, 1)}</td>
                  <td>{percent(g.Precision, 1)}</td>
                  <td>{percent(g.F1, 1)}</td>
                  <td>{percent(g.FPR, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="panel-footnote">
          Proxy effects may remain even after excluding protected attributes. Small groups and few positive outcomes make these estimates unstable — not a complete fairness audit.
        </p>
      </Panel>

      {/* Error profiles */}
      <Panel
        title="Error profiles"
        description={`Aggregate errors for ${data.error_analysis[0]?.Model ?? 'the model'} at ${percent(data.error_analysis[0]?.Threshold ?? 0.5, 0)} threshold.`}
      >
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Error type</th><th>High income</th><th>Senior level</th>
                <th>High satisfaction</th><th>Overtime</th><th>Low tenure</th>
              </tr>
            </thead>
            <tbody>
              {data.error_analysis.map(item => (
                <tr key={item.Type}>
                  <td className="model-cell">{item.Type}</td>
                  <td>{percent(item['high_income_%'] / 100, 1)}</td>
                  <td>{percent(item['senior_joblevel_%'] / 100, 1)}</td>
                  <td>{percent(item['high_satisfaction_%'] / 100, 1)}</td>
                  <td>{percent(item['overtime_%'] / 100, 1)}</td>
                  <td>{percent(item['low_tenure_%'] / 100, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Selected features */}
      <Panel
        title="Selected model inputs"
        description={`${data.selected_features.length} features selected by the pipeline.`}
      >
        <div className="chip-list">
          {data.selected_features.map(item => (
            <span className="chip" key={item.Rank}>
              <b>{String(item.Rank).padStart(2,'0')}</b>
              {item.Feature.replaceAll('_',' ')}
            </span>
          ))}
        </div>
      </Panel>

      <p className="source-note">
        Importance values describe model behavior — not causal effects or grounds for employment decisions.
      </p>
    </div>
  );
}
