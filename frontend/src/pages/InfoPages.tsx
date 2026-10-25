import { ExternalLink, Moon, ShieldCheck, Sun } from 'lucide-react';
import { API_ORIGIN } from '../api';
import { PageHeader, Panel } from '../components/ui';

type Theme = 'light' | 'dark';

export function AboutPage() {
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Reference" title="About the model" description="How the employee attrition pipeline was evaluated and how to interpret its output." />
      <div className="grid-two">
        <Panel title="Dataset" description="IBM HR Analytics Employee Attrition & Performance">
          <dl className="definition-list">
            <div><dt>Employees</dt><dd>1,470</dd></div>
            <div><dt>Observed attrition</dt><dd>237 employees · 16.12%</dd></div>
            <div><dt>Evaluation split</dt><dd>70% train · 30% holdout</dd></div>
            <div><dt>Class balance</dt><dd>Imbalanced binary classification</dd></div>
          </dl>
        </Panel>
        <Panel title="Model and selection" description="The prediction service includes 17 saved model artifacts.">
          <div className="method-list">
            <div><span className="method-index">01</span><div><strong>Preprocessing</strong><p>Domain features, training-only encoding, and in-fold SMOTE.</p></div></div>
            <div><span className="method-index">02</span><div><strong>Feature selection</strong><p>Gradient-boosting importance selects from inputs the prediction form collects; age, gender, marital status, and redundant aliases are excluded as predictors.</p></div></div>
            <div><span className="method-index">03</span><div><strong>Evaluation</strong><p>Repeated stratified 5×5 cross-validation and one holdout evaluation.</p></div></div>
            <div><span className="method-index">04</span><div><strong>Champion criterion</strong><p>The service model is selected by mean repeated cross-validation F1 among six base models; the holdout is reported separately.</p></div></div>
          </div>
        </Panel>
      </div>
      <Panel title="Responsible interpretation" description="Predictions are estimates based on historical data, not explanations or decisions.">
        <div className="notice-box"><ShieldCheck size={18} /><div><strong>Research/demo only</strong><p>Do not use these scores to rank, monitor, retain, promote, or otherwise make employment decisions. The public dataset is small and not representative evidence for another employer. No external validation or complete fairness audit has been performed; human review does not by itself make an unvalidated score safe to use.</p></div></div>
      </Panel>
      <a className="external-link" href="https://github.com/hxrdikxvv/employee-attrition-ml-api" target="_blank" rel="noreferrer">Project source repository <ExternalLink size={14} /></a>
    </div>
  );
}

export function SettingsPage({ theme, setTheme, service }: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  service: { status: string; available_models: number } | null;
}) {
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Workspace" title="Settings" description="Manage local display preferences and inspect service connection details." />
      <Panel title="Appearance" description="This preference is saved in this browser only.">
        <div className="setting-row"><div><strong>Color theme</strong><p>Choose a light or dark workspace.</p></div><div className="segmented-control" role="group" aria-label="Color theme">
          <button className={theme === 'light' ? 'selected' : ''} onClick={() => setTheme('light')} aria-pressed={theme === 'light'}><Sun size={15} /> Light</button>
          <button className={theme === 'dark' ? 'selected' : ''} onClick={() => setTheme('dark')} aria-pressed={theme === 'dark'}><Moon size={15} /> Dark</button>
        </div></div>
      </Panel>
      <Panel title="Model service" description="Connection information for the deployed FastAPI service.">
        <dl className="definition-list settings-list">
          <div><dt>Service status</dt><dd><span className={`table-outcome ${service?.status === 'ok' ? 'stay' : 'risk'}`}>{service?.status === 'ok' ? 'Online' : 'Unavailable'}</span></dd></div>
          <div><dt>Available model artifacts</dt><dd>{service?.available_models ?? '—'}</dd></div>
          <div><dt>API origin</dt><dd><code>{API_ORIGIN}</code></dd></div>
          <div><dt>Documentation</dt><dd><a href={`${API_ORIGIN}/docs`} target="_blank" rel="noreferrer">Open API docs <ExternalLink size={13} /></a></dd></div>
          <div><dt>API key</dt><dd>Provided by the service administrator and entered only when requesting a prediction.</dd></div>
        </dl>
      </Panel>
      <div className="settings-security"><ShieldCheck size={16} /><span>The frontend does not persist API keys or employee input. Session history contains only prediction metadata.</span></div>
    </div>
  );
}
