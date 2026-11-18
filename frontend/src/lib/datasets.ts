import { apiFetch } from './api'
import { IS_STATIC_HOST } from './config'
import { loadDemoDatasets, type DemoDataset } from './clientDataset'

export type WorkspaceDataset = {
  dataset_id: string
  filename: string
  total_rows: number
  total_columns: number
  created_at?: string
  source: 'local' | 'api'
  quality_report?: DemoDataset['quality_report']
  preview?: Record<string, string>[]
}

function fromLocal(): WorkspaceDataset[] {
  return loadDemoDatasets().map((d) => ({
    dataset_id: d.dataset_id,
    filename: d.filename,
    total_rows: d.total_rows,
    total_columns: d.total_columns,
    created_at: d.created_at,
    source: 'local' as const,
    quality_report: d.quality_report,
    preview: d.preview,
  }))
}

/** Prefer API datasets when online; always fall back to browser uploads. */
export async function loadWorkspaceDatasets(): Promise<WorkspaceDataset[]> {
  const local = fromLocal()

  if (IS_STATIC_HOST) return local

  try {
    const response = await apiFetch('/datasets?page=1&limit=50')
    const json = await response.json()
    if (!response.ok || !json.success) return local

    const apiList: WorkspaceDataset[] = (json.datasets || []).map((d: Record<string, unknown>) => ({
      dataset_id: String(d.dataset_id),
      filename: String(d.filename || 'dataset.csv'),
      total_rows: Number(d.total_rows || 0),
      total_columns: Number(d.total_columns || 0),
      created_at: d.created_at ? String(d.created_at) : undefined,
      source: 'api' as const,
      quality_report: d.quality_report as DemoDataset['quality_report'] | undefined,
    }))

    const apiIds = new Set(apiList.map((d) => d.dataset_id))
    const merged = [...apiList, ...local.filter((d) => !apiIds.has(d.dataset_id))]
    return merged
  } catch {
    return local
  }
}
