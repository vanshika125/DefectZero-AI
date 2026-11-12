import { useCallback, useEffect, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

type SafeSettings = {
  inference_mode: string;
  k_shot: number;
  defect_threshold: number;
  max_upload_bytes: number;
  max_image_pixels: number;
  supported_categories: string[];
  persistence: string;
  model_cache_policy: string;
};
type Health = { status: string; model_ready: boolean; loaded_category: string | null; device: string | null; last_model_error: string | null };

export default function Settings() {
  const [settings, setSettings] = useState<SafeSettings | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [settingsError, setSettingsError] = useState("");
  const [serviceError, setServiceError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setSettingsError("");
    setServiceError("");
    const [settingsResult, healthResult, categoriesResult] = await Promise.allSettled([
      fetch(`${API_BASE}/settings`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json() as SafeSettings;
      }),
      fetch(`${API_BASE}/health`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json() as Health;
      }),
      fetch(`${API_BASE}/categories`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json() as string[];
      }),
    ]);

    if (settingsResult.status === "fulfilled") setSettings(settingsResult.value);
    else setSettingsError(`Runtime settings could not be loaded (${errorMessage(settingsResult.reason)}). Restart FastAPI from the project’s current source to enable /settings.`);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setServiceError(`Health check failed (${errorMessage(healthResult.reason)}).`);
    if (categoriesResult.status === "fulfilled") setCategories(categoriesResult.value);
    else if (healthResult.status === "rejected") setServiceError(`Could not reach the FastAPI service at ${API_BASE}.`);
    setLoading(false);
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const activeCategories = settings?.supported_categories ?? categories;
  const apiOnline = health?.status === "ok";

  return <section className="settings-page">
    <div className="settings-heading"><div><span className="history-eyebrow">SERVICE CONFIGURATION</span><h2>Settings</h2><p>Review service health, zero-shot inference, input limits, storage, and dashboard capabilities.</p></div><button className="history-secondary-button" type="button" onClick={() => void refresh()} disabled={loading}>{loading ? "Refreshing…" : "Refresh status"}</button></div>

    {serviceError && <div className="history-error" role="alert">{serviceError}<span>Check that FastAPI is running and that the dashboard API URL is correct.</span></div>}
    {settingsError && <div className="settings-api-note" role="status"><strong>Some live settings are unavailable</strong><span>{settingsError} Other settings below are shown only when verified by the health and categories endpoints.</span></div>}
    {loading && !health && !activeCategories.length && <div className="history-state">Loading service status…</div>}

    <div className="settings-status-banner">
      <span className={`settings-status-dot ${apiOnline ? "online" : health ? "offline" : ""}`} />
      <div><strong>{apiOnline ? "FastAPI service connected" : health ? "FastAPI reported a service problem" : "Waiting for backend status"}</strong><span>Dashboard API: {API_BASE}</span></div>
      <span className={`settings-model-state ${health?.model_ready ? "ready" : "idle"}`}>{health?.model_ready ? `Model ready · ${health.loaded_category ?? "category loaded"}` : "Pretrained model loads on first inspection"}</span>
    </div>

    {health?.last_model_error && <div className="history-error" role="status">Last model load error: {health.last_model_error}</div>}

    <section className="settings-panel"><div className="settings-panel-heading"><h3>Inference configuration</h3><span>Pretrained WinCLIP · zero-shot</span></div><div className="settings-grid">
      <SettingValue label="Inference mode" value={settings?.inference_mode ?? "Zero-shot (verified in service code)"} />
      <SettingValue label="Reference examples (k-shot)" value={settings ? String(settings.k_shot) : "0 (zero-shot)"} />
      <SettingValue label="Defect threshold" value={settings ? settings.defect_threshold.toFixed(4) : "Unavailable until /settings is active"} />
      <SettingValue label="Active device" value={health?.device ?? "Not loaded / unavailable"} />
      <SettingValue label="Loaded category" value={health?.loaded_category ?? "None"} />
      <SettingValue label="Category models" value={health?.model_ready ? "One category loaded" : "Loaded when first requested"} />
    </div></section>

    <section className="settings-panel"><div className="settings-panel-heading"><h3>Input and dataset</h3><span>Values are read from the running service</span></div><div className="settings-grid">
      <SettingValue label="Maximum upload size" value={settings ? `${(settings.max_upload_bytes / (1024 * 1024)).toFixed(0)} MiB` : "Unavailable until /settings is active"} />
      <SettingValue label="Maximum decoded pixels" value={settings ? settings.max_image_pixels.toLocaleString() : "Unavailable until /settings is active"} />
      <SettingValue label="Accepted image formats" value="PNG · JPG · JPEG · WEBP · BMP" />
      <SettingValue label="Available MVTec categories" value={String(activeCategories.length)} />
      <SettingValue label="Dataset access" value="Local MVTec AD categories" />
      <SettingValue label="Ground truth" value="Shown only for matched dataset images" />
    </div>{activeCategories.length > 0 && <div className="settings-category-list" aria-label="Supported categories">{activeCategories.map((category) => <span key={category}>{category.replaceAll("_", " ")}</span>)}</div>}</section>

    <section className="settings-panel"><div className="settings-panel-heading"><h3>History and reports</h3><span>Persistent inspection records</span></div><div className="settings-grid">
      <SettingValue label="Inspection metadata" value={settings?.persistence ?? "SQLite database"} />
      <SettingValue label="Image and visualization assets" value="Inspection-scoped files" />
      <SettingValue label="Report format" value="Individual PDF · batch summary PDF" />
    </div></section>

    <section className="settings-panel"><div className="settings-panel-heading"><h3>Dashboard features</h3><span>Available project workflows</span></div><div className="settings-feature-grid">
      <FeatureCard title="Single inspection" detail="Run zero-shot inference for one image and inspect its anomaly map." />
      <FeatureCard title="Batch inspection" detail="Process selected images sequentially with per-image outcomes." />
      <FeatureCard title="Inspection history" detail="Review saved metadata, input images, heatmaps, and available masks." />
      <FeatureCard title="Evaluation dashboard" detail="Review saved pilot metrics and per-category sample counts." />
    </div></section>

    <div className="settings-readonly-note"><strong>Configuration is read-only here</strong><span>Threshold, upload limits, dataset location, and CORS are controlled by FastAPI environment settings. The page does not change inference behavior. Model weights are pretrained; the project uses zero-shot inference and does not train or fine-tune the model.</span></div>
  </section>;
}

function errorMessage(reason: unknown) { return reason instanceof Error ? reason.message : "request failed"; }
function SettingValue({ label, value }: { label: string; value: string }) { return <div className="settings-value"><span>{label}</span><strong>{value}</strong></div>; }
function FeatureCard({ title, detail }: { title: string; detail: string }) { return <article className="settings-feature-card"><span className="settings-feature-indicator" aria-hidden="true" /><div><strong>{title}</strong><span>{detail}</span></div></article>; }
