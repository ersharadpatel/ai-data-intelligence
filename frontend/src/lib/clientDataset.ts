import { UPLOAD_LIMITS } from './uploadSecurity'

export type QualityReport = {
  missing_values: number
  duplicate_rows: number
  completeness_score: number
  uniqueness_score: number
  missing_by_column: Record<string, number>
  data_types: Record<string, string>
  outliers: Record<string, { count: number }>
  target_validation?: Record<string, { invalid_count: number; invalid_values: string[] }>
}

export type DemoDataset = {
  dataset_id: string
  filename: string
  total_rows: number
  total_columns: number
  created_at: string
  quality_report: QualityReport
  preview: Record<string, string>[]
  source: 'local-demo'
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
      continue
    }
    if (ch === ',' && !inQuotes) {
      cells.push(current.trim())
      current = ''
      continue
    }
    current += ch
  }
  cells.push(current.trim())
  return cells
}

function inferType(values: string[]): string {
  let numeric = 0
  let booleanish = 0
  let filled = 0
  for (const raw of values) {
    if (raw === '') continue
    filled++
    const v = raw.toLowerCase()
    if (v === '0' || v === '1' || v === 'true' || v === 'false' || v === 'yes' || v === 'no') {
      booleanish++
    }
    if (!Number.isNaN(Number(raw)) && raw.trim() !== '') numeric++
  }
  if (!filled) return 'empty'
  if (booleanish / filled >= 0.9) return 'binary/categorical'
  if (numeric / filled >= 0.9) return 'numeric'
  return 'text'
}

function outlierCount(values: number[]): number {
  if (values.length < 8) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const q1 = sorted[Math.floor(sorted.length * 0.25)]
  const q3 = sorted[Math.floor(sorted.length * 0.75)]
  const iqr = q3 - q1
  if (iqr === 0) return 0
  const low = q1 - 1.5 * iqr
  const high = q3 + 1.5 * iqr
  return values.filter((v) => v < low || v > high).length
}

export function analyzeCsvText(filename: string, text: string): DemoDataset {
  const normalized = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  const lines = normalized.split('\n').filter((line) => line.trim().length > 0)

  if (lines.length < 2) {
    throw new Error('CSV must include a header row and at least one data row.')
  }

  const headers = parseCsvLine(lines[0]).map((h, i) => h || `column_${i + 1}`)
  if (headers.length > UPLOAD_LIMITS.maxColumns) {
    throw new Error(`Too many columns. Maximum is ${UPLOAD_LIMITS.maxColumns}.`)
  }

  const dataLines = lines.slice(1)
  if (dataLines.length > UPLOAD_LIMITS.maxRows) {
    throw new Error(`Too many rows. Maximum is ${UPLOAD_LIMITS.maxRows.toLocaleString()}.`)
  }

  const rows = dataLines.map((line) => {
    const cells = parseCsvLine(line)
    const row: Record<string, string> = {}
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? ''
    })
    return row
  })

  const missing_by_column: Record<string, number> = {}
  const data_types: Record<string, string> = {}
  const outliers: Record<string, { count: number }> = {}
  let missing_values = 0

  for (const header of headers) {
    const col = rows.map((r) => r[header] ?? '')
    const missing = col.filter((v) => v === '').length
    missing_by_column[header] = missing
    missing_values += missing
    data_types[header] = inferType(col)

    const nums = col.filter((v) => v !== '' && !Number.isNaN(Number(v))).map(Number)
    if (nums.length >= 8 && data_types[header] === 'numeric') {
      outliers[header] = { count: outlierCount(nums) }
    }
  }

  const rowKeys = rows.map((r) => JSON.stringify(r))
  const duplicate_rows = rowKeys.length - new Set(rowKeys).size
  const total_cells = rows.length * headers.length
  const completeness_score = total_cells
    ? Number((((total_cells - missing_values) / total_cells) * 100).toFixed(2))
    : 0
  const uniqueness_score = rows.length
    ? Number((((rows.length - duplicate_rows) / rows.length) * 100).toFixed(2))
    : 0

  const target_validation: QualityReport['target_validation'] = {}
  for (const header of headers) {
    if (!/purchas|target|label|churn|bought/i.test(header)) continue
    const invalid_values = Array.from(
      new Set(
        rows
          .map((r) => r[header])
          .filter((v) => v !== '' && !['0', '1', 'true', 'false', 'yes', 'no'].includes(v.toLowerCase()))
      )
    ).slice(0, 8)
    target_validation[header] = {
      invalid_count: rows.filter((r) => {
        const v = (r[header] || '').toLowerCase()
        return v !== '' && !['0', '1', 'true', 'false', 'yes', 'no'].includes(v)
      }).length,
      invalid_values,
    }
  }

  return {
    dataset_id: crypto.randomUUID(),
    filename,
    total_rows: rows.length,
    total_columns: headers.length,
    created_at: new Date().toISOString(),
    source: 'local-demo',
    preview: rows.slice(0, 5),
    quality_report: {
      missing_values,
      duplicate_rows,
      completeness_score,
      uniqueness_score,
      missing_by_column,
      data_types,
      outliers,
      ...(Object.keys(target_validation).length ? { target_validation } : {}),
    },
  }
}

const STORAGE_KEY = 'intelliflow_demo_datasets_v1'

export function loadDemoDatasets(): DemoDataset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveDemoDatasets(datasets: DemoDataset[]): void {
  const slim = datasets.slice(0, 12).map((d) => ({
    ...d,
    preview: d.preview.slice(0, 5),
  }))
  localStorage.setItem(STORAGE_KEY, JSON.stringify(slim))
}
