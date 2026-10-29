import { useEffect, useState } from 'react';
import { ArrowRight, Eye, EyeOff, RotateCcw, ShieldCheck, Sparkles } from 'lucide-react';
import { api } from '../api';
import type { EmployeeInput, HistoryEntry, ModelInfo, PredictionResult } from '../types';
import { modelLabel, PageHeader, percent } from '../components/ui';

type Draft = Record<keyof EmployeeInput, string>;

const blankDraft: Draft = {
  BusinessTravel: '', Department: '', JobInvolvement: '', JobLevel: '', JobRole: '',
  MonthlyIncome: '', OverTime: '', StockOptionLevel: '', WorkLifeBalance: '',
  YearsWithCurrManager: '', JobSatisfaction: '', EnvironmentSatisfaction: '', RelationshipSatisfaction: '',
};

function toEmployee(d: Draft): EmployeeInput {
  return {
    BusinessTravel:           d.BusinessTravel as EmployeeInput['BusinessTravel'],
    Department:               d.Department as EmployeeInput['Department'],
    JobInvolvement:           Number(d.JobInvolvement),
    JobLevel:                 Number(d.JobLevel),
    JobRole:                  d.JobRole,
    MonthlyIncome:            Number(d.MonthlyIncome),
    OverTime:                 d.OverTime as EmployeeInput['OverTime'],
    StockOptionLevel:         Number(d.StockOptionLevel),
    WorkLifeBalance:          Number(d.WorkLifeBalance),
    YearsWithCurrManager:     Number(d.YearsWithCurrManager),
    JobSatisfaction:          Number(d.JobSatisfaction),
    EnvironmentSatisfaction:  Number(d.EnvironmentSatisfaction),
    RelationshipSatisfaction: Number(d.RelationshipSatisfaction),
  };
}

function SField({
  label, name, value, options, onChange, required = true,
}: {
  label: string; name: keyof EmployeeInput; value: string;
  options: { value: string; label: string }[];
  onChange: (n: keyof EmployeeInput, v: string) => void; required?: boolean;
}) {
  return (
    <div className="form-field">
      <label htmlFor={name}>
        {label}<b className="required-mark" aria-hidden="true">*</b>
      </label>
      <select
        id={name} name={name} value={value} required={required}
        onChange={e => onChange(name, e.target.value)}
      >
        <option value="" disabled>Select…</option>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function NField({
  label, name, value, min, max, step = 1, onChange, hint,
}: {
  label: string; name: keyof EmployeeInput; value: string;
  min: number; max: number; step?: number;
  onChange: (n: keyof EmployeeInput, v: string) => void; hint?: string;
}) {
  return (
    <div className="form-field">
      <label htmlFor={name}>
        {label}<b className="required-mark" aria-hidden="true">*</b>
      </label>
      <input
        id={name} name={name} type="number" value={value}
        min={min} max={max} step={step} required
        placeholder={hint ?? `${min}–${max}`}
        onChange={e => onChange(name, e.target.value)}
      />
    </div>
  );
}

export function PredictionPage() {
  const [draft, setDraft]             = useState<Draft>(blankDraft);
  const [models, setModels]           = useState<ModelInfo[]>([]);
  const [selectedModel, setModel]     = useState('lr_base');
  const [apiKey, setApiKey]           = useState('');
  const [showKey, setShowKey]         = useState(false);
  const [result, setResult]           = useState<PredictionResult | null>(null);
  const [error, setError]             = useState('');
  const [submitting, setSubmitting]   = useState(false);

  useEffect(() => {
    let active = true;
    api.models()
      .then(d => { if (active) { setModels(d.models); setModel(d.default_model); } })
      .catch((e: unknown) => { if (active) setError(e instanceof Error ? e.message : 'Could not load models.'); });
    return () => { active = false; };
  }, []);

  const update = (n: keyof EmployeeInput, v: string) =>
    setDraft(prev => ({ ...prev, [n]: v }));

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!apiKey.trim()) { setError('Enter your service API key to run a prediction.'); return; }
    setError(''); setResult(null); setSubmitting(true);
    try {
      const pred = await api.predict(selectedModel, toEmployee(draft), apiKey.trim());
      setResult(pred);
      const entry: HistoryEntry = {
        id: crypto.randomUUID(), createdAt: new Date().toISOString(),
        model: pred.model, prediction: pred.prediction, probability: pred.attrition_probability,
      };
      try {
        const hist = JSON.parse(sessionStorage.getItem('attritioniq-history') ?? '[]') as HistoryEntry[];
        sessionStorage.setItem('attritioniq-history', JSON.stringify([entry, ...hist].slice(0, 100)));
      } catch { /* optional */ }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed. Please try again.');
    } finally { setSubmitting(false); }
  }

  function reset() {
    setDraft(blankDraft); setApiKey(''); setResult(null); setError(''); setShowKey(false);
  }

  const opts  = (vals: string[]) => vals.map(v => ({ value: v, label: v.replaceAll('_',' ') }));
  const lvls  = [1,2,3,4].map(n => ({ value: String(n), label: `${n}${n===1?' · Low':n===4?' · High':''}` }));
  const leaving = result?.prediction === 'Yes';

  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="Prediction Workspace"
        title="Predict Attrition"
        description="Research/demo estimate from work-related inputs. Age, gender, and marital status are not used."
      />

      <div className="predict-grid">
        {/* ── FORM ── */}
        <form className="panel" onSubmit={submit}>
          <div className="predict-form-body">

            {/* Section 1 */}
            <div className="form-section">
              <div className="form-section-head">
                <span className="step-num">01</span>
                <div>
                  <h2>Employee information</h2>
                  <p>Role and compensation details used by the research model.</p>
                </div>
              </div>
              <div className="form-grid">
                <SField label="Department"     name="Department"     value={draft.Department}     options={opts(['Sales','Research & Development','Human Resources'])} onChange={update} />
                <SField label="Job role"       name="JobRole"        value={draft.JobRole}        options={opts(['Healthcare Representative','Human Resources','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist','Sales Executive','Sales Representative'])} onChange={update} />
                <NField label="Monthly income" name="MonthlyIncome"  value={draft.MonthlyIncome}  min={0} max={500000} step={1} onChange={update} />
                <NField label="Stock option level" name="StockOptionLevel" value={draft.StockOptionLevel} min={0} max={3} onChange={update} />
              </div>
            </div>

            {/* Section 2 */}
            <div className="form-section">
              <div className="form-section-head">
                <span className="step-num">02</span>
                <div>
                  <h2>Work & satisfaction</h2>
                  <p>Work conditions and satisfaction ratings on the training scale of 1–4.</p>
                </div>
              </div>
              <div className="form-grid">
                <SField label="Business travel"     name="BusinessTravel"         value={draft.BusinessTravel}         options={opts(['Non-Travel','Travel_Frequently','Travel_Rarely'])} onChange={update} />
                <SField label="Overtime"            name="OverTime"               value={draft.OverTime}               options={opts(['Yes','No'])} onChange={update} />
                <NField label="Yrs w/ current mgr" name="YearsWithCurrManager"   value={draft.YearsWithCurrManager}   min={0} max={50} onChange={update} />
                <SField label="Job level"           name="JobLevel"               value={draft.JobLevel}               options={[1,2,3,4,5].map(n=>({value:String(n),label:`Level ${n}`}))} onChange={update} />
                <SField label="Job involvement"     name="JobInvolvement"         value={draft.JobInvolvement}         options={lvls} onChange={update} />
                <SField label="Work-life balance"   name="WorkLifeBalance"        value={draft.WorkLifeBalance}        options={lvls} onChange={update} />
                <SField label="Job satisfaction"    name="JobSatisfaction"        value={draft.JobSatisfaction}        options={lvls} onChange={update} />
                <SField label="Environment satisfaction" name="EnvironmentSatisfaction" value={draft.EnvironmentSatisfaction} options={lvls} onChange={update} />
                <SField label="Relationship satisfaction" name="RelationshipSatisfaction" value={draft.RelationshipSatisfaction} options={lvls} onChange={update} />
              </div>
            </div>

            {/* Section 3 */}
            <div className="form-section">
              <div className="form-section-head">
                <span className="step-num">03</span>
                <div>
                  <h2>Model & access</h2>
                  <p>API key is used for this session only and never stored.</p>
                </div>
              </div>
              <div className="model-access-grid">
                <div className="form-field">
                  <label htmlFor="model-select">Model</label>
                  <select
                    id="model-select"
                    value={selectedModel}
                    onChange={e => setModel(e.target.value)}
                    disabled={!models.length}
                  >
                    {models.map(m => <option key={m.id} value={m.id}>{modelLabel(m.id)}</option>)}
                  </select>
                </div>
                <div className="form-field">
                  <label htmlFor="api-key">
                    Service API key<b className="required-mark" aria-hidden="true">*</b>
                  </label>
                  <div className="secret-wrap">
                    <input
                      id="api-key"
                      type={showKey ? 'text' : 'password'}
                      value={apiKey}
                      onChange={e => setApiKey(e.target.value)}
                      autoComplete="off"
                      required
                      aria-label="Service API key"
                    />
                    <button
                      className="secret-toggle"
                      type="button"
                      onClick={() => setShowKey(s => !s)}
                      aria-label={showKey ? 'Hide key' : 'Show key'}
                    >
                      {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                    </button>
                  </div>
                </div>
              </div>
              <p className="settings-hint">
                <ShieldCheck size={13} /> Ask the service administrator for an authorized API key.
              </p>
            </div>

          </div>

          {error && (
            <div className="form-alert" role="alert" style={{ margin: '0 20px 4px' }}>
              <span className="form-alert-icon">!</span>
              <span>{error}</span>
            </div>
          )}

          <div className="form-actions" style={{ padding: '14px 20px 16px' }}>
            <span className="privacy-note">
              <ShieldCheck size={13} /> Employee fields are not stored in history.
            </span>
            <button
              className="btn btn-primary"
              type="submit"
              disabled={submitting || !models.length}
            >
              {submitting
                ? <><span style={{ opacity: 0.6 }}>Running model…</span></>
                : <>Predict attrition <ArrowRight size={14} /></>}
            </button>
          </div>
        </form>

        {/* ── RESULT ── */}
        <aside className="panel result-panel" aria-live="polite">
          <div className="result-header">
            <div className="result-overline">Model output</div>
            <h2>Assessment</h2>
            <p>{result ? modelLabel(result.model) : 'Awaiting profile'}</p>
          </div>

          {result ? (
            <div className="result-body">
              <div className={`outcome-badge ${leaving ? 'risk' : 'stay'}`}>
                <span className="outcome-dot" />
                {leaving ? 'At risk of leaving' : 'Likely to stay'}
              </div>

              <div className={`risk-num tabular ${leaving ? 'is-risk' : 'is-stay'}`}>
                {percent(result.attrition_probability, 1)}
              </div>
              <div className="risk-sublabel">Predicted attrition probability</div>

              <div className="prob-track"
                role="meter"
                aria-label="Attrition probability"
                aria-valuemin={0} aria-valuemax={100}
                aria-valuenow={Math.round(result.attrition_probability * 100)}
              >
                <div
                  className={`prob-fill ${leaving ? 'risk' : ''}`}
                  style={{ width: `${Math.round(result.attrition_probability * 100)}%` }}
                />
              </div>

              <div className="result-stat">
                <span>Classification threshold</span>
                <strong>{percent(result.threshold, 0)}</strong>
              </div>

              <p className="result-interp">
                {leaving
                  ? 'This estimate meets or exceeds the classification threshold.'
                  : 'This estimate is below the classification threshold.'}
              </p>

              {result.out_of_training_range_fields?.length > 0 && (
                <p className="panel-footnote">
                  Out of training range: {result.out_of_training_range_fields.join(', ')}. Treat this estimate as especially uncertain.
                </p>
              )}

              <button className="btn btn-secondary full" type="button" onClick={reset}>
                <RotateCcw size={14} /> Make another prediction
              </button>
            </div>
          ) : (
            <div className="result-empty">
              <Sparkles size={18} />
              <strong>No prediction yet</strong>
              <p>Complete the employee profile to receive a model estimate.</p>
            </div>
          )}

          <div className="result-disclaimer">
            Estimates are decision support only. Do not use as the sole basis for employment decisions.
          </div>
        </aside>
      </div>
    </div>
  );
}
