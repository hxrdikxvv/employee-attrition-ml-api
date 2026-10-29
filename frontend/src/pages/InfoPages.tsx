import { ExternalLink, Moon, ShieldCheck, Sun } from 'lucide-react';
import { API_ORIGIN } from '../api';
import { PageHeader, Panel } from '../components/ui';

type Theme = 'light' | 'dark';

export function AboutPage() {
  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="Reference"
        title="About the Model"
        description="How the employee attrition pipeline was evaluated and how to interpret its output."
      />

      <div className="grid-2">
        <Panel title="Dataset" description="IBM HR Analytics Employee Attrition & Performance">
          <dl className="def-list">
            <div><dt>Employees</dt><dd>1,470</dd></div>
            <div><dt>Observed attrition</dt><dd>237 · 16.12%</dd></div>
            <div><dt>Evaluation split</dt><dd>70% train · 30% holdout</dd></div>
            <div><dt>Class balance</dt><dd>Imbalanced binary</dd></div>
          </dl>
        </Panel>

        <Panel title="Methodology" description="17 saved model artifacts in the prediction service.">
          <div className="method-list">
            <div>
              <span className="method-num">01</span>
              <div>
                <strong>Preprocessing</strong>
                <p>Domain features, training-only encoding, and in-fold SMOTE.</p>
              </div>
            </div>
            <div>
              <span className="method-num">02</span>
              <div>
                <strong>Feature selection</strong>
                <p>Gradient-boosting importance; age, gender, marital status, and redundant aliases excluded.</p>
              </div>
            </div>
            <div>
              <span className="method-num">03</span>
              <div>
                <strong>Evaluation</strong>
                <p>Repeated stratified 5×5 cross-validation and one holdout evaluation.</p>
              </div>
            </div>
            <div>
              <span className="method-num">04</span>
              <div>
                <strong>Champion criterion</strong>
                <p>Logistic Regression is selected to match the final research paper. The bundled generated evaluation tables come from a separate XGBoost run.</p>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="Responsible interpretation" description="Predictions are estimates from historical data, not decisions.">
        <div className="callout amber">
          <ShieldCheck size={15} />
          <div className="callout-body">
            <strong>Research / demo only</strong>
            Do not use these scores to rank, monitor, retain, promote, or make employment decisions. The public dataset is small and unvalidated for another employer. No external validation or complete fairness audit has been performed.
          </div>
        </div>
      </Panel>

      <a className="external-link" href="https://github.com/hxrdikxvv/employee-attrition-ml-api" target="_blank" rel="noreferrer">
        Project source repository <ExternalLink size={13} />
      </a>
    </div>
  );
}

export function SettingsPage({
  theme, setTheme, service,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
  service: { status: string; available_models: number } | null;
}) {
  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="Workspace"
        title="Settings"
        description="Manage local display preferences and inspect service connection details."
      />

      <Panel title="Appearance" description="Saved in this browser only.">
        <div className="setting-row">
          <div>
            <strong>Color theme</strong>
            <p>Choose light or dark workspace.</p>
          </div>
          <div className="seg-control" role="group" aria-label="Color theme">
            <button
              className={theme === 'light' ? 'active' : ''}
              onClick={() => setTheme('light')}
              aria-pressed={theme === 'light'}
            >
              <Sun size={14} /> Light
            </button>
            <button
              className={theme === 'dark' ? 'active' : ''}
              onClick={() => setTheme('dark')}
              aria-pressed={theme === 'dark'}
            >
              <Moon size={14} /> Dark
            </button>
          </div>
        </div>
      </Panel>

      <Panel title="Model service" description="Connection information for the deployed FastAPI service.">
        <dl className="def-list">
          <div>
            <dt>Service status</dt>
            <dd>
              <span className={`tag ${service?.status === 'ok' ? 'green' : 'red'}`}>
                {service?.status === 'ok' ? 'Online' : 'Unavailable'}
              </span>
            </dd>
          </div>
          <div><dt>Available models</dt><dd>{service?.available_models ?? '—'}</dd></div>
          <div><dt>API origin</dt><dd><code>{API_ORIGIN}</code></dd></div>
          <div>
            <dt>Documentation</dt>
            <dd>
              <a href={`${API_ORIGIN}/docs`} target="_blank" rel="noreferrer">
                Open API docs <ExternalLink size={12} />
              </a>
            </dd>
          </div>
          <div>
            <dt>API key</dt>
            <dd style={{ textAlign: 'right', maxWidth: 260, fontSize: 11, color: '#71717a', fontWeight: 400 }}>
              Provided by admin — entered per prediction, never stored.
            </dd>
          </div>
        </dl>
      </Panel>

      <div className="security-note">
        <ShieldCheck size={15} />
        <span>The frontend does not persist API keys or employee input fields. Session history stores prediction metadata only.</span>
      </div>
    </div>
  );
}
