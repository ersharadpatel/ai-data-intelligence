/** Client-side upload guards for the public demo. */

export const UPLOAD_LIMITS = {
  maxBytes: 2 * 1024 * 1024, // 2 MB
  maxRows: 5_000,
  maxColumns: 40,
  maxFilenameLength: 120,
  allowedExtensions: ['.csv'] as const,
  allowedMimeTypes: new Set([
    'text/csv',
    'application/csv',
    'text/plain',
    'application/vnd.ms-excel',
    '',
  ]),
  /** Soft rate limit: uploads per browser tab session window */
  maxUploadsPerWindow: 8,
  windowMs: 10 * 60 * 1000,
} as const

const uploadTimestamps: number[] = []

export type UploadValidationOk = { ok: true; extension: '.csv'; safeName: string }
export type UploadValidationErr = { ok: false; error: string }
export type UploadValidation = UploadValidationOk | UploadValidationErr

function sanitizeFilename(name: string): string {
  const base = name.split(/[/\\]/).pop() || 'dataset.csv'
  return base
    .replace(/[^\w.\- ()[\]]+/g, '_')
    .replace(/_{2,}/g, '_')
    .slice(0, UPLOAD_LIMITS.maxFilenameLength)
}

export function validateUploadFile(file: File | null): UploadValidation {
  if (!file) return { ok: false, error: 'Choose a CSV file first.' }

  const safeName = sanitizeFilename(file.name)
  const lower = safeName.toLowerCase()
  const extension = (UPLOAD_LIMITS.allowedExtensions.find((ext) => lower.endsWith(ext)) ||
    null) as '.csv' | null

  if (!extension) {
    return { ok: false, error: 'Only CSV files are allowed.' }
  }

  if (file.size <= 0) {
    return { ok: false, error: 'Uploaded file is empty.' }
  }

  if (file.size > UPLOAD_LIMITS.maxBytes) {
    return {
      ok: false,
      error: `File is too large. Maximum size is ${UPLOAD_LIMITS.maxBytes / (1024 * 1024)} MB.`,
    }
  }

  if (file.type && !UPLOAD_LIMITS.allowedMimeTypes.has(file.type)) {
    return { ok: false, error: `Unsupported file type: ${file.type}` }
  }

  const now = Date.now()
  while (uploadTimestamps.length && now - uploadTimestamps[0] > UPLOAD_LIMITS.windowMs) {
    uploadTimestamps.shift()
  }
  if (uploadTimestamps.length >= UPLOAD_LIMITS.maxUploadsPerWindow) {
    return {
      ok: false,
      error: 'Too many uploads. Please wait a few minutes and try again.',
    }
  }

  return { ok: true, extension, safeName }
}

export function recordSuccessfulUpload(): void {
  uploadTimestamps.push(Date.now())
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}
