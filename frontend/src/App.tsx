import { useEffect, useState } from 'react'
import {
  Activity, BrainCircuit, Database, LayoutDashboard, FlaskConical,
  Boxes, Target, Settings, Bell, Search, RefreshCw, Users,
  TrendingUp, Clock3, Zap,FileText
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar
} from 'recharts'
import './App.css'
import ReportsPanel from './ReportsPanel'
import DatasetPanel from './DatasetPanel'
import SettingsPanel from './SettingsPanel'
type Stats = {
  total_predictions: number
  total_buyers: number
  total_non_buyers: number
  buyer_percentage: number
  average_purchase_probability: number
  average_processing_time_seconds: number
}

type Period = {
  total_predictions: number
  buyers: number
  non_buyers: number
  buyer_percentage: number
}

type Analytics = {
  today: Period
  last_7_days: Period
  last_30_days: Period
}

const navigation = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Datasets', icon: Database },
  { label: 'Model Training', icon: BrainCircuit },
  { label: 'Experiments', icon: FlaskConical },
  { label: 'Model Registry', icon: Boxes },
  { label: 'Predictions', icon: Target },
  { label: 'Reports', icon: FileText },
]


function ModelTrainingPanel() {
  const [datasets, setDatasets] = useState<any[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<any>(null)
  const [mode, setMode] = useState('')

  useEffect(() => {
    fetch('/api/datasets?page=1&limit=50')
      .then(r => r.json())
      .then(j => {
        if (j.success) {
          setDatasets(j.datasets || [])
          if (j.datasets?.length) setDatasetId(j.datasets[0].dataset_id)
        } else setError(j.detail || 'Could not load datasets')
      })
      .catch(() => setError('Backend connection failed'))
  }, [])

  async function execute(action: 'train' | 'tune') {
    if (!datasetId || busy) return
    if (action === 'train' && !window.confirm(
      'Training will activate a new production model and replace the currently active model. Continue?'
    )) return

    setBusy(true)
    setError('')
    setMessage('')
    setResult(null)
    setMode(action)

    try {
      const response = await fetch(`/api/datasets/${datasetId}/${action}`, {
        method: 'POST',
        headers: action === 'tune' ? { 'Content-Type': 'application/json' } : undefined,
        body: action === 'tune' ? JSON.stringify({
          c_values: [0.01, 0.1, 1, 10, 100],
          random_state: 42,
          test_size: 0.15,
          validation_size: 0.15,
          threshold_target_precision: 0.6
        }) : undefined
      })
      const json = await response.json()
      if (!response.ok || !json.success) {
        throw new Error(typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail || 'Training failed'))
      }
      setResult(json)
      setMessage(action === 'train' ? 'Model training completed successfully.' : 'Hyperparameter tuning completed.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Operation failed')
    } finally {
      setBusy(false)
    }
  }

  const metrics = result?.result?.final_test_metrics || result?.best_experiment?.validation_metrics
  const modelName = result?.result?.best_model || result?.best_experiment?.name
  const split = result?.result?.split

  return (
    <div className="dataset-module">
      <div className="dataset-toolbar">
        <div><h2>Model Training Studio</h2><p>Train machine learning models and optimize hyperparameters.</p></div>
      </div>

      <div className="panel training-panel">
        <h3>Select training dataset</h3>
        <select value={datasetId} onChange={e => { setDatasetId(e.target.value); setResult(null) }}>
          <option value="">Choose a dataset</option>
          {datasets.map(d => <option key={d.dataset_id} value={d.dataset_id}>{d.filename}</option>)}
        </select>
        {datasets.length === 0 && <p>No datasets available. Upload a dataset first.</p>}
        <div className="training-actions">
          <button className="dataset-primary" disabled={!datasetId || busy} onClick={() => execute('train')}>
            {busy && mode === 'train' ? 'Training...' : 'Train Production Model'}
          </button>
          <button className="dataset-secondary" disabled={!datasetId || busy} onClick={() => execute('tune')}>
            {busy && mode === 'tune' ? 'Tuning...' : 'Auto Tune (5 experiments)'}
          </button>
        </div>
        <p className="training-note">Training activates the new model in production. Auto Tune runs experiments separately and does not change the active production model.</p>
      </div>

      {busy && <div className="panel"><p>Model is processing. Please wait...</p></div>}
      {error && <div className="error-banner">{error}</div>}
      {message && <div className="dataset-success">{message}</div>}

      {result && (
        <>
          <div className="panel dataset-detail-panel">
            <div className="panel-heading"><div><h3>Training Results</h3><p>{modelName || 'Model'} · {result.result?.version || result.best_experiment?.model_version || 'Experiment'}</p></div></div>
            <div className="dataset-metrics">
              {Object.entries(metrics || {}).filter(([k]) => ['accuracy','precision','recall','f1_score','roc_auc'].includes(k)).map(([k,v]) => (
                <div key={k}><small>{k.replace('_',' ').toUpperCase()}</small><strong>{(Number(v) * 100).toFixed(2)}%</strong></div>
              ))}
            </div>
            {split && <div className="training-note">Data split — Train: {split.train} · Validation: {split.validation} · Test: {split.test}</div>}
            {result.result && <div className="training-note">Selected threshold: {result.result.selected_threshold} · Validation precision: {result.result.validation_precision} · Validation recall: {result.result.validation_recall}</div>}
            {result.best_experiment && <div className="training-note">Best experiment: {result.best_experiment.name} · Production model changed: No</div>}
            {result.total_experiments !== undefined && <div className="training-note">Experiments: {result.completed_experiments} completed · {result.failed_experiments} failed</div>}
            {result.result?.validation_model_results && (
              <div className="dataset-subsection">
                <h4>Validation model comparison (ROC-AUC)</h4>
                {Object.entries(result.result.validation_model_results).map(([name, value]: [string, any]) => (
                  <div className="quality-row" key={name}><span>{name}</span><strong>{value.roc_auc}</strong></div>
                ))}
              </div>
            )}
            {result.results && (
              <div className="dataset-subsection">
                <h4>Experiment results</h4>
                {result.results.map((r: any, i: number) => (
                  <div className="quality-row" key={r.experiment_id || i}>
                    <span>Experiment #{r.experiment_id} · {r.model_name || r.error || 'Completed'}</span>
                    <strong>{r.validation_metrics?.roc_auc ?? '—'}</strong>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}



function ExperimentsPanel() {
  const [experiments, setExperiments] = useState<any[]>([])
  const [datasets, setDatasets] = useState<any[]>([])
  const [name, setName] = useState('My ML Experiment')
  const [datasetId, setDatasetId] = useState('')
  const [model, setModel] = useState('logistic_regression')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [selectedA, setSelectedA] = useState('')
  const [selectedB, setSelectedB] = useState('')
  const [comparison, setComparison] = useState<any>(null)

  async function load() {
    try {
      const [er, dr] = await Promise.all([
        fetch('/api/experiments?page=1&limit=100'),
        fetch('/api/datasets?page=1&limit=50')
      ])
      const ej = await er.json()
      const dj = await dr.json()
      if (!er.ok || !ej.success) throw new Error(ej.detail || 'Could not load experiments')
      setExperiments(ej.experiments || [])
      if (dj.success) {
        setDatasets(dj.datasets || [])
        if (!datasetId && dj.datasets?.length) setDatasetId(dj.datasets[0].dataset_id)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Loading failed')
    }
  }

  useEffect(() => { load() }, [])

  async function createExperiment() {
    if (!name.trim() || !datasetId) return
    setBusy(true); setError(''); setMessage('')
    try {
      const r = await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment_name: name,
          dataset_id: datasetId,
          model_name: model,
          hyperparameters: { C: 1, max_iter: 1000 },
          training_config: { random_state: 42, test_size: 0.15, validation_size: 0.15 }
        })
      })
      const j = await r.json()
      if (!r.ok || !j.success) throw new Error(j.detail || 'Create experiment failed')
      setMessage('Experiment created successfully.')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Create failed')
    } finally { setBusy(false) }
  }

  async function runExperiment(id: string) {
    setBusy(true); setError(''); setMessage('')
    try {
      const r = await fetch(`/api/experiments/${id}/run`, { method: 'POST' })
      const j = await r.json()
      if (!r.ok || !j.success) throw new Error(j.detail || 'Experiment run failed')
      setMessage('Experiment completed successfully.')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Run failed')
    } finally { setBusy(false) }
  }

  async function compareExperiments() {
    if (!selectedA || !selectedB || selectedA === selectedB) return
    setBusy(true); setError(''); setComparison(null)
    try {
      const params = new URLSearchParams({
        experiment_a: selectedA,
        experiment_b: selectedB,
        metric_name: 'roc_auc'
      })
      const r = await fetch(`/api/experiments/compare?${params}`)
      const j = await r.json()
      if (!r.ok || !j.success) throw new Error(j.detail || 'Comparison failed')
      setComparison(j)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Comparison failed')
    } finally { setBusy(false) }
  }

  return (
    <div className="dataset-module">
      <div className="dataset-toolbar">
        <div><h2>Experiment Tracking</h2><p>Create, run and compare machine learning experiments.</p></div>
        <button className="refresh-button" onClick={load} disabled={busy}><RefreshCw size={15}/> Refresh</button>
      </div>

      <div className="panel training-panel">
        <h3>Create New Experiment</h3>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="Experiment name" />
        <select value={datasetId} onChange={e => setDatasetId(e.target.value)}>
          <option value="">Select dataset</option>
          {datasets.map(d => <option key={d.dataset_id} value={d.dataset_id}>{d.filename}</option>)}
        </select>
        <select value={model} onChange={e => setModel(e.target.value)}>
          <option value="logistic_regression">Logistic Regression</option>
          <option value="random_forest">Random Forest</option>
          <option value="decision_tree">Decision Tree</option>
        </select>
        <button className="dataset-primary" disabled={busy || !datasetId || !name.trim()} onClick={createExperiment}>
          {busy ? 'Please wait...' : 'Create Experiment'}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="dataset-success">{message}</div>}

      <div className="panel dataset-list-panel">
        <div className="panel-heading"><div><h3>Experiment History</h3><p>{experiments.length} experiment(s)</p></div></div>
        {experiments.length === 0 ? <div className="empty-state">No experiments yet. Create your first experiment.</div> : (
          <div className="dataset-table-wrap">
            <table className="dataset-table">
              <thead><tr><th>Name</th><th>Model</th><th>Status</th><th>ROC-AUC</th><th>Created</th><th>Action</th></tr></thead>
              <tbody>
                {experiments.map(e => (
                  <tr key={e.id}>
                    <td><strong>{e.experiment_name}</strong><small>{String(e.id)}</small></td>
                    <td>{e.model_name}</td>
                    <td>{e.status}</td>
                    <td>{e.validation_metrics?.roc_auc != null ? Number(e.validation_metrics.roc_auc).toFixed(4) : '—'}</td>
                    <td>{e.created_at ? new Date(e.created_at).toLocaleDateString() : '—'}</td>
                    <td>
                      <button className="dataset-view" disabled={busy || e.status === 'running' || e.status === 'completed'} onClick={() => runExperiment(String(e.id))}>
                        Run
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="panel training-panel">
        <h3>Compare Experiments</h3>
        <div className="training-actions">
          <select value={selectedA} onChange={e => setSelectedA(e.target.value)}>
            <option value="">Experiment A</option>
            {experiments.map(e => <option key={e.id} value={e.id}>{e.experiment_name}</option>)}
          </select>
          <select value={selectedB} onChange={e => setSelectedB(e.target.value)}>
            <option value="">Experiment B</option>
            {experiments.map(e => <option key={e.id} value={e.id}>{e.experiment_name}</option>)}
          </select>
          <button className="dataset-secondary" disabled={busy || !selectedA || !selectedB || selectedA === selectedB} onClick={compareExperiments}>Compare ROC-AUC</button>
        </div>
        {comparison && <pre className="training-note">{JSON.stringify(comparison, null, 2)}</pre>}
      </div>
    </div>
  )
}



function ModelRegistryPanel() {
  const [models, setModels] = useState<any[]>([])
  const [active, setActive] = useState<any>(null)
  const [versionA, setVersionA] = useState('')
  const [versionB, setVersionB] = useState('')
  const [comparison, setComparison] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  async function load() {
    setError('')
    try {
      const [mr, ar] = await Promise.all([
        fetch('/api/models'),
        fetch('/api/models/active')
      ])
      const mj = await mr.json()
      const aj = await ar.json()

      if (!mr.ok || !mj.success) throw new Error(mj.detail || 'Could not load model registry')
      setModels(mj.models || [])
      setActive(ar.ok && aj.success ? aj.model : null)

      if (!versionA && mj.models?.length) setVersionA(mj.models[0].version)
      if (!versionB && mj.models?.length > 1) setVersionB(mj.models[1].version)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Registry loading failed')
    }
  }

  useEffect(() => { load() }, [])

  async function activate(version: string) {
    if (!window.confirm(`Activate ${version} as the production model?`)) return
    setBusy(true)
    setError('')
    setMessage('')
    try {
      const r = await fetch(`/api/models/${encodeURIComponent(version)}/activate`, {
        method: 'POST'
      })
      const j = await r.json()
      if (!r.ok || !j.success) throw new Error(j.detail || 'Model activation failed')
      setMessage(`${version} is now the active production model.`)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Activation failed')
    } finally {
      setBusy(false)
    }
  }

  async function compare() {
    if (!versionA || !versionB || versionA === versionB) return
    setBusy(true)
    setError('')
    setComparison(null)
    try {
      const params = new URLSearchParams({ version_a: versionA, version_b: versionB })
      const r = await fetch(`/api/models/compare?${params}`)
      const j = await r.json()
      if (!r.ok || !j.success) throw new Error(j.detail || 'Model comparison failed')
      setComparison(j.comparison)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Comparison failed')
    } finally {
      setBusy(false)
    }
  }

  const metricNames = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

  return (
    <div className="dataset-module">
      <div className="dataset-toolbar">
        <div>
          <h2>Model Registry</h2>
          <p>Manage production models, versions and performance.</p>
        </div>
        <button className="refresh-button" onClick={load} disabled={busy}>
          <RefreshCw size={15}/> Refresh
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="dataset-success">{message}</div>}

      <div className="panel dataset-detail-panel">
        <div className="panel-heading">
          <div><h3>Active Production Model</h3><p>Currently serving predictions</p></div>
          <span className="status-pill"><span/> {active ? 'Active' : 'No active model'}</span>
        </div>
        {active ? (
          <>
            <div className="dataset-metrics">
              <div><small>Version</small><strong>{active.version}</strong></div>
              <div><small>Model</small><strong>{active.model_name}</strong></div>
              <div><small>Threshold</small><strong>{active.threshold ?? '—'}</strong></div>
              <div><small>Training rows</small><strong>{active.training_rows?.toLocaleString?.() ?? '—'}</strong></div>
            </div>
            <div className="dataset-subsection">
              <h4>Active model performance</h4>
              <div className="dataset-metrics">
                {metricNames.map(k => (
                  <div key={k}>
                    <small>{k.replace('_', ' ').toUpperCase()}</small>
                    <strong>{active.metrics?.[k] != null ? `${(Number(active.metrics[k]) * 100).toFixed(2)}%` : '—'}</strong>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : <div className="empty-state">No active model found. Train a production model first.</div>}
      </div>

      <div className="panel dataset-list-panel">
        <div className="panel-heading">
          <div><h3>Registered Models</h3><p>{models.length} model version(s)</p></div>
        </div>
        {models.length === 0 ? (
          <div className="empty-state">No registered models found. Train your first production model.</div>
        ) : (
          <div className="dataset-table-wrap">
            <table className="dataset-table">
              <thead>
                <tr><th>Version</th><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>ROC-AUC</th><th>Status</th><th>Created</th><th>Action</th></tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.id}>
                    <td><strong>{m.version}</strong></td>
                    <td>{m.model_name}</td>
                    <td>{m.metrics?.accuracy != null ? `${(Number(m.metrics.accuracy) * 100).toFixed(2)}%` : '—'}</td>
                    <td>{m.metrics?.f1_score != null ? `${(Number(m.metrics.f1_score) * 100).toFixed(2)}%` : '—'}</td>
                    <td>{m.metrics?.roc_auc != null ? Number(m.metrics.roc_auc).toFixed(4) : '—'}</td>
                    <td>{m.is_active ? 'Active' : 'Inactive'}</td>
                    <td>{m.created_at ? new Date(m.created_at).toLocaleDateString() : '—'}</td>
                    <td>
                      {m.is_active ? <span>Current</span> : (
                        <button className="dataset-view" disabled={busy} onClick={() => activate(m.version)}>Activate</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="panel training-panel">
        <h3>Compare Model Versions</h3>
        <div className="training-actions">
          <select value={versionA} onChange={e => setVersionA(e.target.value)}>
            <option value="">Select version A</option>
            {models.map(m => <option key={m.version} value={m.version}>{m.version} · {m.model_name}</option>)}
          </select>
          <select value={versionB} onChange={e => setVersionB(e.target.value)}>
            <option value="">Select version B</option>
            {models.map(m => <option key={m.version} value={m.version}>{m.version} · {m.model_name}</option>)}
          </select>
          <button className="dataset-secondary" disabled={busy || !versionA || !versionB || versionA === versionB} onClick={compare}>
            Compare Models
          </button>
        </div>
        {comparison?.metrics && (
          <div className="dataset-subsection">
            <h4>{versionA} vs {versionB}</h4>
            <div className="dataset-table-wrap">
              <table className="dataset-table">
                <thead><tr><th>Metric</th><th>{versionA}</th><th>{versionB}</th><th>Difference (B − A)</th></tr></thead>
                <tbody>
                  {metricNames.map(k => {
                    const item = comparison.metrics[k]
                    return (
                      <tr key={k}>
                        <td>{k.replace('_', ' ').toUpperCase()}</td>
                        <td>{item?.[versionA] != null ? Number(item[versionA]).toFixed(4) : '—'}</td>
                        <td>{item?.[versionB] != null ? Number(item[versionB]).toFixed(4) : '—'}</td>
                        <td>{item?.difference_b_minus_a != null ? Number(item.difference_b_minus_a).toFixed(4) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {comparison.note && <p className="training-note">{comparison.note}</p>}
          </div>
        )}
      </div>
    </div>
  )
}


function PredictionsPanel() {
  const [form, setForm] = useState({
    age: '30',
    annual_income: '50000',
    website_visits: '10',
    previous_purchases: '2',
    avg_session_minutes: '5',
    marketing_emails_opened: '3'
  })
  const [result, setResult] = useState<any>(null)
  const [history, setHistory] = useState<any[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  async function loadHistory() {
    try {
      const response = await fetch('/api/predictions?page=1&limit=10')
      const json = await response.json()
      if (!response.ok || !json.success) {
        throw new Error(json.detail || 'Could not load prediction history')
      }
      setHistory(json.data || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'History loading failed')
    }
  }

  useEffect(() => { loadHistory() }, [])

  function updateField(name: string, value: string) {
    setForm(prev => ({ ...prev, [name]: value }))
  }

  async function predict() {
    setBusy(true)
    setError('')
    setMessage('')
    setResult(null)

    try {
      const payload = {
        age: Number(form.age),
        annual_income: Number(form.annual_income),
        website_visits: Number(form.website_visits),
        previous_purchases: Number(form.previous_purchases),
        avg_session_minutes: Number(form.avg_session_minutes),
        marketing_emails_opened: Number(form.marketing_emails_opened)
      }

      if (Object.values(payload).some(v => !Number.isFinite(v))) {
        throw new Error('Please enter valid numeric values.')
      }
      if (payload.age <= 0 || payload.age > 120 || payload.annual_income < 0 ||
          payload.website_visits < 0 || payload.previous_purchases < 0 ||
          payload.avg_session_minutes < 0 || payload.marketing_emails_opened < 0) {
        throw new Error('Please check the entered values.')
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const json = await response.json()
      if (!response.ok || !json.success) {
        throw new Error(json.detail || 'Prediction failed')
      }

      setResult(json.data)
      setMessage('Prediction completed successfully.')
      await loadHistory()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setBusy(false)
    }
  }

  const fields = [
    { key: 'age', label: 'Customer Age', min: 1, max: 120, step: '1' },
    { key: 'annual_income', label: 'Annual Income (₹)', min: 0, step: 'any' },
    { key: 'website_visits', label: 'Website Visits', min: 0, step: '1' },
    { key: 'previous_purchases', label: 'Previous Purchases', min: 0, step: '1' },
    { key: 'avg_session_minutes', label: 'Average Session (Minutes)', min: 0, step: 'any' },
    { key: 'marketing_emails_opened', label: 'Marketing Emails Opened', min: 0, step: '1' }
  ]

  return (
    <div className="dataset-module">
      <div className="dataset-toolbar">
        <div>
          <h2>Customer Predictions</h2>
          <p>Predict customer purchase probability using your active ML model.</p>
        </div>
        <button className="refresh-button" onClick={loadHistory} disabled={busy}>
          <RefreshCw size={15} /> Refresh History
        </button>
      </div>

      <div className="panel training-panel">
        <h3>Customer Information</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', margin: '18px 0' }}>
          {fields.map(field => (
            <label key={field.key} style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
              <span>{field.label}</span>
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step}
                value={form[field.key as keyof typeof form]}
                onChange={e => updateField(field.key, e.target.value)}
              />
            </label>
          ))}
        </div>
        <button className="dataset-primary" onClick={predict} disabled={busy}>
          {busy ? 'Analyzing Customer...' : 'Predict Purchase'}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="dataset-success">{message}</div>}

      {result && (
        <div className="panel dataset-detail-panel">
          <div className="panel-heading">
            <div>
              <h3>Prediction Result</h3>
              <p>{result.model_name} · {result.model_version}</p>
            </div>
            <span className="status-pill"><span /> Prediction Complete</span>
          </div>
          <div className="dataset-metrics">
            <div><small>Purchase Probability</small><strong>{(Number(result.purchase_probability) * 100).toFixed(2)}%</strong></div>
            <div><small>Decision</small><strong>{result.decision}</strong></div>
            <div><small>Threshold</small><strong>{(Number(result.threshold) * 100).toFixed(1)}%</strong></div>
            <div><small>Prediction</small><strong>{result.prediction}</strong></div>
          </div>
          <p className="training-note">Processing time: {result.processing_time_seconds ?? '—'} seconds</p>
        </div>
      )}

      <div className="panel dataset-list-panel">
        <div className="panel-heading">
          <div><h3>Prediction History</h3><p>{history.length} recent prediction(s)</p></div>
        </div>
        {history.length === 0 ? (
          <div className="empty-state">No predictions recorded yet.</div>
        ) : (
          <div className="dataset-table-wrap">
            <table className="dataset-table">
              <thead><tr><th>ID</th><th>Age</th><th>Income</th><th>Probability</th><th>Decision</th><th>Processing</th><th>Date</th></tr></thead>
              <tbody>
                {history.map(item => (
                  <tr key={item.id}>
                    <td>{item.id}</td>
                    <td>{item.age}</td>
                    <td>₹{Number(item.annual_income).toLocaleString()}</td>
                    <td>{(Number(item.purchase_probability) * 100).toFixed(2)}%</td>
                    <td>{item.decision}</td>
                    <td>{item.processing_time}s</td>
                    <td>{item.created_at ? new Date(item.created_at).toLocaleString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState('')
  const [activePage, setActivePage] = useState('Overview')

  async function loadDashboard() {
    setLoading(true)
    setError('')

    try {
      const [statsResponse, analyticsResponse] = await Promise.all([
        fetch('/api/dashboard/stats'),
        fetch('/api/dashboard/analytics'),
      ])

      if (!statsResponse.ok || !analyticsResponse.ok) {
        throw new Error('API request failed. Check that FastAPI is running on port 8001.')
      }

      const statsJson = await statsResponse.json()
      const analyticsJson = await analyticsResponse.json()

      if (!statsJson.success || !analyticsJson.success) {
        throw new Error('The API returned an unsuccessful response.')
      }

      setStats(statsJson.data)
      setAnalytics(analyticsJson.data)
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load dashboard data.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDashboard()
  }, [])

  const cards = [
    {
      title: 'Total Predictions',
      value: stats?.total_predictions ?? 0,
      icon: Activity,
      color: 'purple',
      detail: 'All recorded predictions',
    },
    {
      title: 'Predicted Buyers',
      value: stats?.total_buyers ?? 0,
      icon: Users,
      color: 'green',
      detail: 'Customers likely to purchase',
    },
    {
      title: 'Buyer Conversion',
      value: `${(stats?.buyer_percentage ?? 0).toFixed(2)}%`,
      icon: TrendingUp,
      color: 'blue',
      detail: 'Predicted buyer percentage',
    },
    {
      title: 'Avg. Probability',
      value: `${((stats?.average_purchase_probability ?? 0) * 100).toFixed(2)}%`,
      icon: Zap,
      color: 'orange',
      detail: 'Average purchase probability',
    },
  ]

  const chartData = analytics ? [
    { period: 'Today', predictions: analytics.today.total_predictions, buyers: analytics.today.buyers },
    { period: 'Last 7 days', predictions: analytics.last_7_days.total_predictions, buyers: analytics.last_7_days.buyers },
    { period: 'Last 30 days', predictions: analytics.last_30_days.total_predictions, buyers: analytics.last_30_days.buyers },
  ] : []

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><BrainCircuit size={23} /></div>
          <div>
            <h2>Intelli<span>Flow</span></h2>
            <small>AI DATA INTELLIGENCE</small>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <div className="workspace">
          <div className="workspace-avatar">IF</div>
          <div><strong>IntelliFlow</strong><small>Production workspace</small></div>
          <span className="workspace-dot" />
        </div>

        <div className="nav-label">PLATFORM</div>
        <nav>
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${activePage === label ? 'active' : ''}`}
              onClick={() => setActivePage(label)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === 'Overview' && <span className="nav-indicator" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <button
            className={`nav-item ${activePage === 'Settings' ? 'active' : ''}`}
            onClick={() => setActivePage('Settings')}
          >
            <Settings size={18} />
            <span>Settings</span>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>{activePage}</strong></div>
          <div className="top-actions">
            <div className="search-box"><Search size={16} /><span>Search...</span><kbd>⌘ K</kbd></div>
            <button className="icon-button" aria-label="Notifications"><Bell size={19} /></button>
          </div>
        </header>

        <section className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="live-dot" /> AI DATA INTELLIGENCE</div>
              <h1>{activePage}</h1>
              <p>
                {activePage === 'Datasets'
                  ? 'Upload CSV datasets and review quality metrics before training.'
                  : 'Monitor predictive performance, datasets, experiments, and model health.'}
              </p>
            </div>
            {activePage === 'Overview' && (
              <button className="refresh-button" onClick={loadDashboard} disabled={loading}>
                <RefreshCw size={16} className={loading ? 'spin' : ''} />
                Refresh data
              </button>
            )}
          </div>

          {activePage === 'Datasets' ? (
            <DatasetPanel />
          ) : activePage === 'Settings' ? (
            <SettingsPanel />
          ) : activePage === 'Model Training' ? (
            <ModelTrainingPanel />
          ) : activePage === 'Experiments' ? (
            <ExperimentsPanel />
          ) : activePage === 'Model Registry' ? (
            <ModelRegistryPanel />
          ) : activePage === 'Predictions' ? (
            <PredictionsPanel />
          ) : activePage === 'Reports' ? (
            <ReportsPanel />
          ) : activePage !== 'Overview' ? (
            <div className="notice">
              <BrainCircuit size={20} />
              <div><strong>{activePage} module</strong><p>This module will be connected to the live backend in the next development step.</p></div>
            </div>
          ) : (
            <>
          {error && <div className="error-banner">{error}<button onClick={loadDashboard}>Retry</button></div>}

          <div className="section-heading">
            <div><h3>Platform Overview</h3><p>Your prediction performance at a glance</p></div>
            <span className="updated">Last updated: {lastUpdated || '—'}</span>
          </div>

          <div className="stats-grid">
            {cards.map(({ title, value, icon: Icon, color, detail }) => (
              <div className="stat-card" key={title}>
                <div className="stat-top"><span>{title}</span><div className={`stat-icon ${color}`}><Icon size={19} /></div></div>
                <div className="stat-value">{loading ? '—' : value}</div>
                <div className="stat-detail">{detail}</div>
              </div>
            ))}
          </div>

          <div className="charts-grid">
            <div className="panel chart-panel">
              <div className="panel-heading">
                <div><h3>Prediction Activity</h3><p>Predictions across time periods</p></div>
                <span className="chart-tag">LIVE API DATA</span>
              </div>
              <div className="chart-area">
                {loading ? <div className="empty-state">Loading analytics...</div> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="predictionFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#8b7cff" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#8b7cff" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#252a3a" strokeDasharray="4 4" vertical={false} />
                      <XAxis dataKey="period" stroke="#777f95" tickLine={false} axisLine={false} />
                      <YAxis stroke="#777f95" tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: '#171c2b', border: '1px solid #30364a', borderRadius: 10 }} />
                      <Area type="monotone" dataKey="predictions" stroke="#9585ff" strokeWidth={3} fill="url(#predictionFill)" />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="panel chart-panel">
              <div className="panel-heading">
                <div><h3>Buyer Analysis</h3><p>Buyer vs non-buyer predictions</p></div>
                <span className="chart-tag">LIVE API DATA</span>
              </div>
              <div className="chart-area">
                {loading ? <div className="empty-state">Loading analytics...</div> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barGap={8}>
                      <CartesianGrid stroke="#252a3a" strokeDasharray="4 4" vertical={false} />
                      <XAxis dataKey="period" stroke="#777f95" tickLine={false} axisLine={false} />
                      <YAxis stroke="#777f95" tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: '#171c2b', border: '1px solid #30364a', borderRadius: 10 }} />
                      <Bar dataKey="buyers" name="Buyers" fill="#36c99a" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>

          <div className="bottom-grid">
            <div className="panel">
              <div className="panel-heading"><div><h3>Prediction Summary</h3><p>Overall customer classification</p></div></div>
              <div className="summary-row">
                <div className="summary-icon buyer"><TrendingUp size={19} /></div>
                <div className="summary-info"><strong>Predicted Buyers</strong><small>Likely to purchase</small></div>
                <strong>{stats?.total_buyers ?? 0}</strong>
              </div>
              <div className="summary-row">
                <div className="summary-icon nonbuyer"><Users size={19} /></div>
                <div className="summary-info"><strong>Predicted Non-buyers</strong><small>Unlikely to purchase</small></div>
                <strong>{stats?.total_non_buyers ?? 0}</strong>
              </div>
              <div className="summary-row">
                <div className="summary-icon clock"><Clock3 size={19} /></div>
                <div className="summary-info"><strong>Avg. Processing Time</strong><small>Per prediction</small></div>
                <strong>{(stats?.average_processing_time_seconds ?? 0).toFixed(4)}s</strong>
              </div>
            </div>

            <div className="panel health-panel">
              <div className="panel-heading"><div><h3>Platform Status</h3><p>Backend connectivity</p></div><span className={`status-pill ${
  error ? 'status-error' : loading ? 'status-loading' : 'status-connected'
}`}>
  <span />
  {error ? 'Disconnected' : loading ? 'Checking...' : 'Connected'}
</span></div>
              <div className="health-content">
                <div className="health-symbol"><Activity size={28} /></div>
                <div><strong>{error ? 'Connection issue' : loading ? 'Checking API...' : 'API response received'}</strong><p>{error || 'Dashboard statistics and analytics endpoints responded.'}</p></div>
              </div>
              <div className="health-footer"><span>FastAPI Backend</span><strong>Port 8001</strong></div>
              <div className="health-footer"><span>Frontend</span><strong>React + Vite</strong></div>
            </div>
          </div>

            </>
          )}

          <footer>IntelliFlow AI Platform <span>•</span> Project 01 <span>•</span> {activePage}</footer>
        </section>
      </main>
    </div>
  )
}

export default App
