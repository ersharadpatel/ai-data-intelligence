import { useEffect, useState } from 'react'
import { RefreshCw, ShieldCheck, Upload } from 'lucide-react'
import {
  analyzeCsvText,
  loadDemoDatasets,
  saveDemoDatasets,
  type DemoDataset,
} from './lib/clientDataset'
import { formatBytes, recordSuccessfulUpload, UPLOAD_LIMITS, validateUploadFile } from './lib/uploadSecurity'

type Dataset = DemoDataset

export default function DatasetPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState<Dataset | null>(null)
  const [preview, setPreview] = useState<Record<string, string>[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  function refreshLocal() {
    const local = loadDemoDatasets()
    setDatasets(local)
  }

  useEffect(() => {
    refreshLocal()
  }, [])

  async function processCsv(filename: string, text: string) {
    const dataset = analyzeCsvText(filename, text)
    const next = [dataset, ...loadDemoDatasets().filter((d) => d.filename !== dataset.filename)].slice(0, 12)
    saveDemoDatasets(next)
    setDatasets(next)
    setSelected(dataset)
    setPreview(dataset.preview)
      setMessage('Dataset analyzed successfully. Quality report is ready.')
    recordSuccessfulUpload()
  }

  async function uploadDataset() {
    const check = validateUploadFile(file)
    if (!check.ok) {
      setError(check.error)
      return
    }

    setBusy(true)
    setError('')
    setMessage('')
    setPreview([])

    try {
      const text = await file!.text()
      // Light content sniff: reject binary-looking payloads named .csv
      if (text.includes('\u0000')) {
        throw new Error('File content looks invalid for CSV.')
      }
      await processCsv(check.safeName, text)
      setFile(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to analyze dataset')
    } finally {
      setBusy(false)
    }
  }

  async function loadSample() {
    setBusy(true)
    setError('')
    setMessage('')
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}samples/synthetic_customers.csv`)
      if (!response.ok) throw new Error('Sample dataset could not be loaded.')
      const text = await response.text()
      await processCsv('synthetic_customers.csv', text)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Sample load failed')
    } finally {
      setBusy(false)
    }
  }

  function inspect(dataset: Dataset) {
    setSelected(dataset)
    setPreview(dataset.preview || [])
    setError('')
    setMessage('')
  }

  const report = selected?.quality_report

  return (
    <div className="dataset-module">
      <div className="demo-security-banner">
        <ShieldCheck size={18} />
        <div>
          <strong>Secure upload</strong>
          <p>
            CSV only · max {UPLOAD_LIMITS.maxBytes / (1024 * 1024)} MB · max{' '}
            {UPLOAD_LIMITS.maxRows.toLocaleString()} rows · processed locally in your browser.
          </p>
        </div>
      </div>

      <div className="dataset-toolbar">
        <div>
          <h2>Dataset Library</h2>
          <p>Upload, inspect, and prepare datasets for machine learning.</p>
        </div>
        <button className="refresh-button" onClick={refreshLocal} disabled={busy}>
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      <div className="panel dataset-upload">
        <div className="dataset-upload-icon">
          <Upload size={23} />
        </div>
        <div className="upload-copy">
          <h3>Upload a dataset</h3>
          <p>
            CSV files · Maximum {UPLOAD_LIMITS.maxBytes / (1024 * 1024)} MB · up to{' '}
            {UPLOAD_LIMITS.maxRows.toLocaleString()} rows
          </p>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => {
              setError('')
              setFile(e.target.files?.[0] || null)
            }}
          />
          {file && (
            <small>
              Selected: {file.name} · {formatBytes(file.size)}
            </small>
          )}
        </div>
        <div className="upload-actions">
          <button className="dataset-secondary" onClick={loadSample} disabled={busy}>
            Try sample CSV
          </button>
          <button className="dataset-primary" onClick={uploadDataset} disabled={!file || busy}>
            {busy ? 'Analyzing...' : 'Check Dataset'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="dataset-success">{message}</div>}

      <div className="panel dataset-list-panel">
        <div className="panel-heading">
          <div>
            <h3>Checked datasets</h3>
            <p>{datasets.length} dataset(s) in this browser session</p>
          </div>
        </div>
        {datasets.length === 0 ? (
          <div className="empty-state">No datasets yet. Upload a CSV or try the sample file.</div>
        ) : (
          <div className="dataset-table-wrap">
            <table className="dataset-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Rows</th>
                  <th>Columns</th>
                  <th>Missing</th>
                  <th>Created</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr key={d.dataset_id}>
                    <td>
                      <strong>{d.filename}</strong>
                      <small>{d.dataset_id.slice(0, 8)}… · local</small>
                    </td>
                    <td>{d.total_rows.toLocaleString()}</td>
                    <td>{d.total_columns}</td>
                    <td>{d.quality_report?.missing_values ?? '—'}</td>
                    <td>{new Date(d.created_at).toLocaleDateString()}</td>
                    <td>
                      <button className="dataset-view" onClick={() => inspect(d)}>
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selected && (
        <div className="panel dataset-detail-panel">
          <div className="panel-heading">
            <div>
              <h3>Dataset Quality Report</h3>
              <p>{selected.filename}</p>
            </div>
            <button className="dataset-view" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>

          <div className="dataset-metrics">
            <div>
              <small>Total rows</small>
              <strong>{selected.total_rows.toLocaleString()}</strong>
            </div>
            <div>
              <small>Columns</small>
              <strong>{selected.total_columns}</strong>
            </div>
            <div>
              <small>Missing values</small>
              <strong>{report?.missing_values ?? 0}</strong>
            </div>
            <div>
              <small>Duplicates</small>
              <strong>{report?.duplicate_rows ?? 0}</strong>
            </div>
            <div>
              <small>Completeness</small>
              <strong>{report?.completeness_score ?? '—'}%</strong>
            </div>
            <div>
              <small>Uniqueness</small>
              <strong>{report?.uniqueness_score ?? '—'}%</strong>
            </div>
          </div>

          {report?.missing_by_column && (
            <div className="dataset-subsection">
              <h4>Missing values by column</h4>
              {Object.entries(report.missing_by_column).map(([name, count]) => (
                <div className="quality-row" key={name}>
                  <span>{name}</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          )}

          {report?.data_types && (
            <div className="dataset-subsection">
              <h4>Column data types</h4>
              <div className="dataset-chips">
                {Object.entries(report.data_types).map(([name, type]) => (
                  <span key={name}>
                    {name} <b>{type}</b>
                  </span>
                ))}
              </div>
            </div>
          )}

          {report?.target_validation && (
            <div className="dataset-subsection">
              <h4>Target validation</h4>
              {Object.entries(report.target_validation).map(([name, value]) => (
                <div className="quality-row" key={name}>
                  <span>
                    {name}: invalid values {value.invalid_values?.join(', ') || 'None'}
                  </span>
                  <strong>{value.invalid_count ?? 0}</strong>
                </div>
              ))}
            </div>
          )}

          {report?.outliers && Object.keys(report.outliers).length > 0 && (
            <div className="dataset-subsection">
              <h4>Outlier summary</h4>
              {Object.entries(report.outliers).map(([name, value]) => (
                <div className="quality-row" key={name}>
                  <span>{name}</span>
                  <strong>{value.count ?? 0}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {preview.length > 0 && (
        <div className="panel dataset-detail-panel">
          <div className="panel-heading">
            <div>
              <h3>Preview</h3>
              <p>First five records</p>
            </div>
          </div>
          <div className="dataset-table-wrap">
            <table className="dataset-table">
              <thead>
                <tr>
                  {Object.keys(preview[0]).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j}>{String(v ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
