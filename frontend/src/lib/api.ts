import { API_PREFIX, IS_STATIC_HOST } from './config'

export class ApiUnavailableError extends Error {
  constructor(message = 'API is not available in this deployment.') {
    super(message)
    this.name = 'ApiUnavailableError'
  }
}

export function apiUrl(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`
  return `${API_PREFIX}${normalized}`
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  if (IS_STATIC_HOST) {
    throw new ApiUnavailableError(
      'This view needs the full platform server. On GitHub Pages, use Datasets for CSV quality checks.'
    )
  }
  return fetch(apiUrl(path), init)
}
