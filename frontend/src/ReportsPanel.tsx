
import { useEffect, useState } from 'react'
import {
  Download,
  FileText,
  RefreshCw,
  Printer
} from 'lucide-react'
import { apiFetch } from './lib/api'
import { IS_STATIC_HOST } from './lib/config'

type ReportData = {
  datasets: any[]
  models: any[]
  experiments: any[]
  predictions: any[]
}

export default function ReportsPanel() {
  const [data, setData] = useState<ReportData>({
    datasets: [],
    models: [],
    experiments: [],
    predictions: []
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Extract arrays from different API response formats
  function extract(json: any, key: string): any[] {
    if (Array.isArray(json?.[key])) return json[key]
    if (Array.isArray(json?.data)) return json.data
    if (Array.isArray(json?.data?.[key])) return json.data[key]
    if (Array.isArray(json?.items)) return json.items
    if (Array.isArray(json?.data?.items)) return json.data.items
    return []
  }

  // Load all report data
  async function loadReports() {
    setLoading(true)
    setError('')

    try {
      if (IS_STATIC_HOST) {
        setError('Reports need the full platform server. On GitHub Pages, use Datasets for CSV quality checks.')
        return
      }

      const endpoints = [
        '/datasets?page=1&limit=100',
        '/models',
        '/experiments?page=1&limit=100',
        '/predictions?page=1&limit=100'
      ]

      const responses = await Promise.all(
        endpoints.map(url => apiFetch(url))
      )

      const results = await Promise.all(
        responses.map(async response => {
          if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`)
          }
          return response.json()
        })
      )

      setData({
        datasets: extract(results[0], 'datasets'),
        models: extract(results[1], 'models'),
        experiments: extract(results[2], 'experiments'),
        predictions: extract(results[3], 'predictions')
      })
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load reports.'
      )
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadReports()
  }, [])

  // Escape CSV values
  function escapeCSV(value: any): string {
    const text =
      typeof value === 'object' && value !== null
        ? JSON.stringify(value)
        : String(value ?? '')

    return `"${text.replace(/"/g, '""')}"`
  }

  // Export data as CSV
  function exportCSV(filename: string, rows: any[]) {
    if (!rows.length) {
      setError('No data available to export.')
      return
    }

    const columns = Array.from(
      new Set(rows.flatMap(row => Object.keys(row)))
    )

    const csv = [
      columns.map(escapeCSV).join(','),
      ...rows.map(row =>
        columns.map(column => escapeCSV(row[column])).join(',')
      )
    ].join('\n')

    const blob = new Blob(
      ['\uFEFF' + csv],
      { type: 'text/csv;charset=utf-8;' }
    )

    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()

    URL.revokeObjectURL(url)
  }

  // Escape HTML values for printable reports
  function escapeHTML(value: any): string {
    const text =
      typeof value === 'object' && value !== null
        ? JSON.stringify(value)
        : String(value ?? '')

    return text.replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[char]!)
  }

  // Print report or save it as PDF
  function printReport(title: string, rows: any[]) {
    if (!rows.length) {
      setError('No data available for this report.')
      return
    }

    const columns = Array.from(
      new Set(rows.flatMap(row => Object.keys(row)))
    )

    const table = `
      <table>
        <thead>
          <tr>
            ${columns.map(column =>
              `<th>${escapeHTML(column)}</th>`
            ).join('')}
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              ${columns.map(column =>
                `<td>${escapeHTML(row[column])}</td>`
              ).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>
    `

    const win = window.open('', '_blank')

    if (!win) {
      setError('Please allow pop-ups to print this report.')
      return
    }

    win.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>${escapeHTML(title)}</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            padding: 28px;
            color: #222;
          }

          h1 { font-size: 22px; }

          p {
            color: #666;
            font-size: 12px;
          }

          table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
          }

          th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            font-size: 10px;
            overflow-wrap: anywhere;
          }

          th { background: #f2f2f2; }

          tr { page-break-inside: avoid; }

          @media print {
            body { padding: 0; }
          }
        </style>
      </head>
      <body>
        <h1>${escapeHTML(title)}</h1>
        <p>IntelliFlow AI</p>
        <p>Generated: ${escapeHTML(new Date().toLocaleString())}</p>
        <p>Total Records: ${rows.length}</p>
        ${table}
        <script>
          window.onload = () => window.print()
        </script>
      </body>
      </html>
    `)

    win.document.close()
  }

  const reports = [
    {
      title: 'Dataset Quality Report',
      description: 'Dataset information and quality assessment.',
      rows: data.datasets,
      filename: 'dataset_quality_report.csv'
    },
    {
      title: 'Model Performance Report',
      description: 'Registered models and performance metrics.',
      rows: data.models,
      filename: 'model_performance_report.csv'
    },
    {
      title: 'Experiment Report',
      description: 'Experiment configurations and results.',
      rows: data.experiments,
      filename: 'experiment_report.csv'
    },
    {
      title: 'Prediction History Report',
      description: 'Customer predictions and processing history.',
      rows: data.predictions,
      filename: 'prediction_history_report.csv'
    }
  ]

  return (
    <div className="dataset-module">

      <div className="dataset-toolbar">
        <div>
          <h2>Reports & Export</h2>
          <p>Generate reports and export platform data.</p>
        </div>

        <button
          className="refresh-button"
          onClick={loadReports}
          disabled={loading}
        >
          <RefreshCw size={15} />
          {loading ? 'Loading...' : 'Refresh Reports'}
        </button>
      </div>

      {error && (
        <div className="error-banner">
          {error}
        </div>
      )}

      <div className="dataset-metrics">
        <div>
          <small>Datasets</small>
          <strong>{data.datasets.length}</strong>
        </div>

        <div>
          <small>Registered Models</small>
          <strong>{data.models.length}</strong>
        </div>

        <div>
          <small>Experiments</small>
          <strong>{data.experiments.length}</strong>
        </div>

        <div>
          <small>Predictions</small>
          <strong>{data.predictions.length}</strong>
        </div>
      </div>

      <div className="panel dataset-list-panel">
        <div className="panel-heading">
          <div>
            <h3>Available Reports</h3>
            <p>Download CSV or print reports as PDF.</p>
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: 16
        }}>
          {reports.map(report => (
            <div
              key={report.title}
              className="panel"
              style={{ padding: 20 }}
            >
              <FileText size={24} />

              <h3>{report.title}</h3>
              <p>{report.description}</p>

              <p>
                <strong>{report.rows.length}</strong> records available
              </p>

              <div style={{
                display: 'flex',
                gap: 10,
                flexWrap: 'wrap'
              }}>
                <button
                  className="dataset-primary"
                  onClick={() =>
                    exportCSV(report.filename, report.rows)
                  }
                  disabled={!report.rows.length || loading}
                >
                  <Download size={15} />
                  CSV
                </button>

                <button
                  className="refresh-button"
                  onClick={() =>
                    printReport(report.title, report.rows)
                  }
                  disabled={!report.rows.length || loading}
                >
                  <Printer size={15} />
                  PDF / Print
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
