import { useEffect, useState } from 'react';
import { ArrowRight, Eye, EyeOff, LoaderCircle, RotateCcw, ShieldCheck, Sparkles } from 'lucide-react';
import { api } from '../api';
import type { EmployeeInput, HistoryEntry, ModelInfo, PredictionResult } from '../types';
import { modelLabel, PageHeader, percent } from '../components/ui';

type Draft = Record<keyof EmployeeInput, string>;

const blankDraft: Draft = {
  Age: '', BusinessTravel: '', Department: '', JobInvolvement: '', JobLevel: '', JobRole: '',
  MaritalStatus: '', MonthlyIncome: '', OverTime: '', StockOptionLevel: '', WorkLifeBalance: '',
  YearsWithCurrManager: '', JobSatisfaction: '', EnvironmentSatisfaction: '', RelationshipSatisfaction: '',
};

function toEmployee(draft: Draft): EmployeeInput {
  return {
    Age: Number(draft.Age),
    BusinessTravel: draft.BusinessTravel as EmployeeInput['BusinessTravel'],
    Department: draft.Department as EmployeeInput['Department'],
    JobInvolvement: Number(draft.JobInvolvement),
    JobLevel: Number(draft.JobLevel),
    JobRole: draft.JobRole,
    MaritalStatus: draft.MaritalStatus as EmployeeInput['MaritalStatus'],
    MonthlyIncome: Number(draft.MonthlyIncome),
    OverTime: draft.OverTime as EmployeeInput['OverTime'],
    StockOptionLevel: Number(draft.StockOptionLevel),
    WorkLifeBalance: Number(draft.WorkLifeBalance),
    YearsWithCurrManager: Number(draft.YearsWithCurrManager),
    JobSatisfaction: Number(draft.JobSatisfaction),
    EnvironmentSatisfaction: Number(draft.EnvironmentSatisfaction),
    RelationshipSatisfaction: Number(draft.RelationshipSatisfaction),
  };
}

function SelectField({ label, name, value, options, onChange, required = true }: {
  label: string; name: keyof EmployeeInput; value: string; options: Array<{ value: string; label: string }>;
  onChange: (name: keyof EmployeeInput, value: string) => void; required?: boolean;
}) {
  return <label className="form-field"><span>{label}<b className="required-mark" aria-hidden="true">*</b></span>
    <select name={name} value={value} onChange={(event) => onChange(name, event.target.value)} required={required}>
      <option value="" disabled>Select {label.toLowerCase()}</option>
      {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
    </select>
  </label>;
}

function NumberField({ label, name, value, min, max, step = 1, onChange, hint }: {
  label: string; name: keyof EmployeeInput; value: string; min: number; max: number; step?: number;
  onChange: (name: keyof EmployeeInput, value: string) => void; hint?: string;
}) {
  return <label className="form-field"><span>{label}<b className="required-mark" aria-hidden="true">*</b></span>
    <input name={name} type="number" value={value} min={min} max={max} step={step} required placeholder={hint ?? `${min}–${max}`} onChange={(event) => onChange(name, event.target.value)} />
  </label>;
}

export function PredictionPage() {
  const [draft, setDraft] = useState<Draft>(blankDraft);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState('lr_base');
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    let active = true;
    api.models().then((data) => {
      if (!active) return;
      setModels(data.models);
      setSelectedModel(data.default_model);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : 'Could not load model list.');
    });
    return () => { active = false; };
  }, []);

  const update = (name: keyof EmployeeInput, value: string) => setDraft((current) => ({ ...current, [name]: value }));

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKey.trim()) { setError('Enter your service API key to run a prediction.'); return; }
    setError('');
    setResult(null);
    setIsSubmitting(true);
    try {
      const prediction = await api.predict(selectedModel, toEmployee(draft), apiKey.trim());
      setResult(prediction);
      const entry: HistoryEntry = {
        id: crypto.randomUUID(), createdAt: new Date().toISOString(), model: prediction.model,
        prediction: prediction.prediction, probability: prediction.attrition_probability,
      };
      try {
        const existing = JSON.parse(sessionStorage.getItem('attritioniq-history') ?? '[]') as HistoryEntry[];
        sessionStorage.setItem('attritioniq-history', JSON.stringify([entry, ...existing].slice(0, 100)));
      } catch { /* Session history is optional; prediction success must not depend on it. */ }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Prediction failed. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  }

  function reset() {
    setDraft(blankDraft);
    setApiKey('');
    setResult(null);
    setError('');
    setShowKey(false);
  }

  const options = (values: string[]) => values.map((value) => ({ value, label: value.replaceAll('_', ' ') }));
  const levels = [1, 2, 3, 4].map((value) => ({ value: String(value), label: `${value}${value === 1 ? ' · Low' : value === 4 ? ' · High' : ''}` }));
  const leaving = result?.prediction === 'Yes';

  return (
    <div className="page-stack">
      <PageHeader eyebrow="Prediction workspace" title="Predict attrition" description="Evaluate one employee profile with any trained model. Inputs are sent to the prediction service and are not saved in history." />
      <div className="prediction-layout">
        <form className="panel prediction-form" onSubmit={submit}>
          <div className="form-section">
            <div className="form-section-title"><span className="step-index">01</span><div><h2>Employee information</h2><p>Demographic and role details accepted by the model.</p></div></div>
            <div className="form-grid">
              <NumberField label="Age" name="Age" value={draft.Age} min={18} max={100} onChange={update} />
              <SelectField label="Department" name="Department" value={draft.Department} options={options(['Sales', 'Research & Development', 'Human Resources'])} onChange={update} />
              <SelectField label="Job role" name="JobRole" value={draft.JobRole} options={options(['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'])} onChange={update} />
              <NumberField label="Monthly income" name="MonthlyIncome" value={draft.MonthlyIncome} min={0} max={500000} step={1} onChange={update} />
              <SelectField label="Marital status" name="MaritalStatus" value={draft.MaritalStatus} options={options(['Single', 'Married', 'Divorced'])} onChange={update} />
              <NumberField label="Stock option level" name="StockOptionLevel" value={draft.StockOptionLevel} min={0} max={3} onChange={update} />
            </div>
          </div>

          <div className="form-section">
            <div className="form-section-title"><span className="step-index">02</span><div><h2>Work &amp; satisfaction</h2><p>Work conditions and satisfaction ratings use the training scale of 1 to 4.</p></div></div>
            <div className="form-grid">
              <SelectField label="Business travel" name="BusinessTravel" value={draft.BusinessTravel} options={options(['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'])} onChange={update} />
              <SelectField label="Overtime" name="OverTime" value={draft.OverTime} options={options(['Yes', 'No'])} onChange={update} />
              <NumberField label="Years with current manager" name="YearsWithCurrManager" value={draft.YearsWithCurrManager} min={0} max={50} onChange={update} />
              <SelectField label="Job level" name="JobLevel" value={draft.JobLevel} options={[1, 2, 3, 4, 5].map((n) => ({ value: String(n), label: `Level ${n}` }))} onChange={update} />
              <SelectField label="Job involvement" name="JobInvolvement" value={draft.JobInvolvement} options={levels} onChange={update} />
              <SelectField label="Work-life balance" name="WorkLifeBalance" value={draft.WorkLifeBalance} options={levels} onChange={update} />
              <SelectField label="Job satisfaction" name="JobSatisfaction" value={draft.JobSatisfaction} options={levels} onChange={update} />
              <SelectField label="Environment satisfaction" name="EnvironmentSatisfaction" value={draft.EnvironmentSatisfaction} options={levels} onChange={update} />
              <SelectField label="Relationship satisfaction" name="RelationshipSatisfaction" value={draft.RelationshipSatisfaction} options={levels} onChange={update} />
            </div>
          </div>

          <div className="form-section model-section">
            <div className="form-section-title"><span className="step-index">03</span><div><h2>Model &amp; access</h2><p>The API key remains in this page session only.</p></div></div>
            <div className="model-access-grid">
              <label className="form-field"><span>Model</span><select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={!models.length}>
                {models.map((model) => <option key={model.id} value={model.id}>{modelLabel(model.id)}</option>)}
              </select></label>
              <label className="form-field"><span>Service API key <b className="required-mark" aria-hidden="true">*</b></span>
                <div className="secret-input"><input type={showKey ? 'text' : 'password'} value={apiKey} onChange={(event) => setApiKey(event.target.value)} autoComplete="off" required aria-label="Service API key" />
                  <button className="secret-toggle" type="button" onClick={() => setShowKey(!showKey)} aria-label={showKey ? 'Hide API key' : 'Show API key'}>{showKey ? <EyeOff size={16} /> : <Eye size={16} />}</button>
                </div>
              </label>
            </div>
            <p className="settings-link"><ShieldCheck size={14} /> Ask the service administrator for an authorized API key.</p>
          </div>

          {error && <div className="form-error" role="alert"><span className="error-indicator">!</span><span>{error}</span></div>}
          <div className="form-actions">
            <span className="privacy-note"><ShieldCheck size={14} /> Employee fields are not stored in prediction history.</span>
            <button className="button button-primary" type="submit" disabled={isSubmitting || !models.length}>
              {isSubmitting ? <><LoaderCircle className="spin" size={16} /> Running model</> : <>Predict attrition <ArrowRight size={15} /> </>}
            </button>
          </div>
        </form>

        <aside className="panel prediction-result" aria-live="polite">
          <div className="result-heading"><div className="section-overline">MODEL OUTPUT</div><h2>Assessment</h2><p>{result ? modelLabel(result.model) : 'Awaiting profile'}</p></div>
          {result ? <div className="result-detail">
            <div className={`outcome-banner ${leaving ? 'outcome-risk' : 'outcome-stay'}`}><span className="outcome-dot" /><strong>{leaving ? 'At risk of leaving' : 'Likely to stay'}</strong></div>
            <div className="risk-number tabular">{percent(result.attrition_probability, 1)}</div>
            <div className="risk-label">Predicted attrition probability</div>
            <div className="probability-track" role="meter" aria-label="Predicted attrition probability" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(result.attrition_probability * 100)}><span style={{ width: `${Math.round(result.attrition_probability * 100)}%` }} /></div>
            <div className="result-stat"><span>Classification threshold</span><strong>{percent(result.threshold, 0)}</strong></div>
            <p className="result-interpretation">{leaving ? 'This model estimate is at or above its classification threshold.' : 'This model estimate is below its classification threshold.'}</p>
            <button className="button button-secondary full-width" type="button" onClick={reset}><RotateCcw size={15} /> Make another prediction</button>
          </div> : <div className="result-empty"><Sparkles size={19} /><strong>No prediction yet</strong><p>Complete the employee profile to receive a model estimate.</p></div>}
          <div className="result-disclaimer">Estimates are decision support only. They must not be used as the sole basis for employment decisions.</div>
        </aside>
      </div>
    </div>
  );
}
