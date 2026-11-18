/**
 * GitHub Pages cannot run Python. On github.io we stay fully browser-local.
 * Full stack (same-origin UI + API): run FastAPI which also serves frontend/dist.
 */
function isGitHubPagesHost(): boolean {
  return typeof window !== 'undefined' && /\.github\.io$/i.test(window.location.hostname)
}

export const IS_STATIC_HOST =
  import.meta.env.VITE_DEMO_MODE === 'true' || isGitHubPagesHost()

/** Alias used by dataset browser-local flows */
export const IS_DEMO_MODE = IS_STATIC_HOST

/** Same-origin API prefix (Vite proxy locally, FastAPI strip in production) */
export const API_PREFIX = '/api'

export const APP_NAME = 'IntelliFlow'
export const APP_TAGLINE = 'AI Data Intelligence'
