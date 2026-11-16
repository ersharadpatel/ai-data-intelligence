/**
 * Browser-local dataset checks work without a backend (GitHub Pages friendly).
 * Set VITE_DEMO_MODE=false only when using API-backed uploads locally.
 */
export const IS_DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== 'false'

export const APP_NAME = 'IntelliFlow'
export const APP_TAGLINE = 'AI Data Intelligence'
