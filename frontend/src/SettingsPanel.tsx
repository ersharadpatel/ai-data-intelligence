import { ShieldCheck, Globe2, Lock, HardDrive } from 'lucide-react'
import { UPLOAD_LIMITS } from './lib/uploadSecurity'

export default function SettingsPanel() {
  return (
    <div className="dataset-module">
      <div className="dataset-toolbar">
        <div>
          <h2>Platform settings</h2>
          <p>Upload policy and data handling for this workspace.</p>
        </div>
      </div>

      <div className="settings-grid">
        <div className="panel settings-card">
          <div className="settings-icon">
            <Globe2 size={20} />
          </div>
          <h3>Access</h3>
          <p>Dataset quality checks are available without an account. Upload a CSV and inspect results immediately.</p>
        </div>

        <div className="panel settings-card">
          <div className="settings-icon">
            <Lock size={20} />
          </div>
          <h3>Upload policy</h3>
          <ul>
            <li>CSV only</li>
            <li>Max size {UPLOAD_LIMITS.maxBytes / (1024 * 1024)} MB</li>
            <li>Max {UPLOAD_LIMITS.maxRows.toLocaleString()} rows / {UPLOAD_LIMITS.maxColumns} columns</li>
            <li>Rate limit: {UPLOAD_LIMITS.maxUploadsPerWindow} checks / 10 min</li>
          </ul>
        </div>

        <div className="panel settings-card">
          <div className="settings-icon">
            <HardDrive size={20} />
          </div>
          <h3>Privacy</h3>
          <p>
            Files are parsed locally in your browser and kept in this device session. They are not shared with
            other visitors.
          </p>
        </div>

        <div className="panel settings-card">
          <div className="settings-icon">
            <ShieldCheck size={20} />
          </div>
          <h3>Full platform</h3>
          <p>
            Model training, experiments, registry, and live predictions use the FastAPI backend when it is
            connected.
          </p>
        </div>
      </div>
    </div>
  )
}
