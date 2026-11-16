import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const sample = resolve(root, 'public/samples/synthetic_customers.csv')
const text = readFileSync(sample, 'utf8')

const MAX_BYTES = 2 * 1024 * 1024
const MAX_ROWS = 5000
const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/)
const headers = lines[0].split(',')
const rows = lines.slice(1)

const checks = {
  sampleExists: text.length > 0,
  underSizeLimit: Buffer.byteLength(text) <= MAX_BYTES,
  underRowLimit: rows.length <= MAX_ROWS,
  hasHeaderAndRows: lines.length >= 2,
  columnCount: headers.length,
  rowCount: rows.length,
}

const failed = Object.entries(checks).filter(([k, v]) => k.startsWith('under') || k.startsWith('sample') || k.startsWith('has') ? !v : false)
if (failed.length) {
  console.error('SMOKE FAIL', checks)
  process.exit(1)
}

console.log('SMOKE OK', checks)
