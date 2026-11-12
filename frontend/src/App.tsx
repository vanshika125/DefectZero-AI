import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, FormEvent, ReactNode } from "react";
import InspectionHistory from "./components/InspectionHistory";
import ModelEvaluation from "./components/ModelEvaluation";
import BatchInspection from "./components/BatchInspection";
import DatasetExplorer from "./components/DatasetExplorer";
import Reports from "./components/Reports";
import Settings from "./components/Settings";
import GlobalSearch from "./components/GlobalSearch";
import Overview from "./components/Overview";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

type Health = {
  status: string;
  model_ready: boolean;
  loaded_category: string | null;
  device: string | null;
  last_model_error: string | null;
};

type Prediction = {
  category: string;
  anomaly_score: number;
  predicted_label: "normal" | "defective";
  threshold: number;
  anomaly_map_width: number;
  anomaly_map_height: number;
  anomaly_map_visualization_base64: string;
  anomaly_map_visualization_mime_type: string;
  ground_truth_available: boolean;
  ground_truth_defect_type: string | null;
  ground_truth_mask_base64: string | null;
  ground_truth_mask_mime_type: string | null;
  ground_truth_mask_width: number | null;
  ground_truth_mask_height: number | null;
  inference_time_ms: number;
  device: string;
  inference_mode: "zero-shot";
  k_shot: 0;
};

type ViewMode = "comparison" | "overlay" | "original";

function Icon({ name, size = 19 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  const paths: Record<string, ReactNode> = {
    logo: <><path d="M12 2.5v2.2M12 19.3v2.2M4.4 4.4 6 6M18 18l1.6 1.6M2.5 12h2.2m14.6 0h2.2M4.4 19.6 6 18m12-12 1.6-1.6"/><circle cx="12" cy="12" r="8.2"/><circle cx="12" cy="12" r="3.1"/></>,
    overview: <><rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="3" y="15" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/></>,
    inspect: <><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/></>,
    datasets: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    history: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2M3 12H1"/></>,
    evaluation: <><path d="M3 3v18h18M7 14l4-4 3 3 6-7"/><path d="M16 6h4v4"/></>,
    report: <><path d="M6 3h9l4 4v14H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z"/><path d="M14 3v5h5M8 12h8M8 16h8"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1 1.3 1-1.4 2.4-1.6-.6a8 8 0 0 1-1.7 1l-.3 1.7h-2.8l-.3-1.7a8 8 0 0 1-1.7-1l-1.6.6L6 16.1l1.3-1a7 7 0 0 1 0-2.1L6 12l1.4-2.4 1.6.6a8 8 0 0 1 1.7-1l.3-1.7h2.8l.3 1.7a8 8 0 0 1 1.7 1l1.6-.6 1.4 2.4-1.3 1a7 7 0 0 1-.1 2Z"/></>,
    upload: <><path d="M12 16V4m-4 4 4-4 4 4"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></>,
    check: <><circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4"/></>,
    threshold: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/></>,
    cube: <><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z"/><path d="m3.5 8.5 8.5 5 8.5-5M12 13.5V21"/></>,
    timer: <><circle cx="12" cy="13" r="8"/><path d="M12 13V8m-3-6h6m-3 0v3m5 2 2-2"/></>,
    image: <><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m21 15-5-5L5 20"/></>,
    mask: <><path d="M12 3c5 0 9 4 9 9s-4 9-9 9-9-4-9-9 4-9 9-9Z"/><path d="M8 10h.01M16 10h.01M8 15c2 2 6 2 8 0"/></>,
    spark: <><path d="m12 2 1.8 7.2L21 12l-7.2 1.8L12 21l-1.8-7.2L3 12l7.2-2.8L12 2Z"/><path d="m19 2 .7 2.3L22 5l-2.3.7L19 8l-.7-2.3L16 5l2.3-.7L19 2Z"/></>,
    bell: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9ZM10 21h4"/></>,
    search: <><circle cx="10.8" cy="10.8" r="7"/><path d="m16 16 5 5"/></>,
    play: <path d="m8 5 12 7-12 7V5Z" fill="currentColor" stroke="none"/>,
    grid: <><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    minus: <path d="M5 12h14"/>,
    fullscreen: <><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    chevron: <path d="m6 9 6 6 6-6"/>,
    wifi: <><path d="M5 12.5a10 10 0 0 1 14 0M8 16a5.5 5.5 0 0 1 8 0M12 20h.01"/></>,
  };
  return <svg {...common}>{paths[name] ?? paths.image}</svg>;
}

const navGroups = [
  { title: "Monitor", links: [{ label: "Overview", icon: "overview" }, { label: "Inspect", icon: "inspect" }, { label: "Batch Inspection", icon: "inspect" }, { label: "Datasets", icon: "datasets" }, { label: "Inspection History", icon: "history" }] },
  { title: "Analytics", links: [{ label: "Model Evaluation", icon: "evaluation" }, { label: "Reports", icon: "report" }] },
  { title: "Platform", links: [{ label: "Settings", icon: "settings" }] },
];

function App() {
  const [categories, setCategories] = useState<string[]>([]);
  const [category, setCategory] = useState("");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const categoriesRequest = useRef(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [statusOpen, setStatusOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [view, setView] = useState<ViewMode>("comparison");
  const [zoom, setZoom] = useState(1);
  const [activeNav, setActiveNav] = useState("Overview");
  const [datasetInitialCategory, setDatasetInitialCategory] = useState<string | undefined>(undefined);
  const [historyInitialId, setHistoryInitialId] = useState<string | undefined>(undefined);
  const inputRef = useRef<HTMLInputElement>(null);
  const comparisonRef = useRef<HTMLDivElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error(`Health check returned ${response.status}`);
      const data = (await response.json()) as Health;
      setHealth(data);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
      setHealth(null);
    }
  }, []);

  const loadCategories = useCallback(async () => {
    if (categoriesRequest.current) return;
    categoriesRequest.current = true;
    try {
      const response = await fetch(`${API_BASE}/categories`, { signal: AbortSignal.timeout(7000) });
      if (!response.ok) throw new Error(`Categories request returned ${response.status}`);
      const data: unknown = await response.json();
      if (!Array.isArray(data) || !data.every((item) => typeof item === "string")) throw new Error("Invalid categories response");
      setCategories(data);
      setCategory((current) => current || data[0] || "");
      setError("");
    } catch {
      setError(`Could not load categories from ${API_BASE}. Start the FastAPI service and refresh this page.`);
    } finally {
      categoriesRequest.current = false;
    }
  }, []);

  useEffect(() => {
    void loadCategories();
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 15000);
    return () => {
      window.clearInterval(timer);
    };
  }, [loadCategories, refreshHealth]);

  useEffect(() => {
    if (apiOnline && categories.length === 0) void loadCategories();
  }, [apiOnline, categories.length, loadCategories]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const setSelectedFile = (selected: File | undefined) => {
    if (!selected) return;
    setError("");
    setPrediction(null);
    setView("comparison");
    setZoom(1);
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setFile(null);
      setPreviewUrl("");
      setError("Choose a PNG, JPG, JPEG, WEBP, or BMP image.");
      return;
    }
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => setSelectedFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    setSelectedFile(event.dataTransfer.files?.[0]);
  };

  const runInspection = async (event: FormEvent) => {
    event.preventDefault();
    if (!file || !category || loading) return;
    setError("");
    setNotice("");
    setPrediction(null);
    setLoading(true);
    const form = new FormData();
    form.append("category", category);
    form.append("image", file);
    try {
      const response = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const info = payload as { detail?: { message?: string } | string } | null;
        const message = typeof info?.detail === "string" ? info.detail : info?.detail?.message;
        throw new Error(message || `Inspection failed with HTTP ${response.status}.`);
      }
      const data = payload as Prediction;
      if (data.category !== category || typeof data.anomaly_map_visualization_base64 !== "string") {
        throw new Error("The API returned an incomplete prediction response.");
      }
      setPrediction(data);
      setApiOnline(true);
      setHealth((current) => current ? { ...current, model_ready: true, loaded_category: data.category, device: data.device, last_model_error: null } : current);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The inspection request failed.");
      void refreshHealth();
    } finally {
      setLoading(false);
    }
  };

  const removeFile = () => {
    setFile(null);
    setPreviewUrl("");
    setPrediction(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const heatmapUrl = useMemo(() => prediction ? `data:${prediction.anomaly_map_visualization_mime_type};base64,${prediction.anomaly_map_visualization_base64}` : "", [prediction]);
  const scoreLabel = prediction?.predicted_label === "defective" ? "Defect detected" : "Normal";
  const insight = prediction
    ? `The image-level anomaly score (${prediction.anomaly_score.toFixed(4)}) is ${prediction.anomaly_score >= prediction.threshold ? "at or above" : "below"} the configured threshold (${prediction.threshold.toFixed(4)}). WinCLIP classifies this ${prediction.category} image as ${prediction.predicted_label}. The score is an anomaly signal, not a probability.`
    : "Upload an image and run an inspection to see a concise explanation of the model output.";

  const handleFullscreen = () => {
    if (!comparisonRef.current) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void comparisonRef.current.requestFullscreen();
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><Icon name="logo" size={34} /></div><div><strong>DefectZero AI</strong><span>Zero-Shot Defect Detection</span></div></div>
        <button className="dataset-switch" type="button" aria-label="Open MVTec AD dataset explorer" onClick={() => { setActiveNav("Datasets"); setDatasetInitialCategory(undefined); setHistoryInitialId(undefined); setNotice(""); }}><span>MVTec AD</span><Icon name="chevron" size={17} /></button>
        <GlobalSearch categories={categories} onNavigate={(page, targetCategory, inspectionId) => { setActiveNav(page); setNotice(""); setDatasetInitialCategory(targetCategory); setHistoryInitialId(inspectionId); }} />
        <nav aria-label="Main navigation" className="navigation">
          {navGroups.map((group) => <section className="nav-group" key={group.title}><h2>{group.title}</h2>{group.links.map((link) => <button type="button" key={link.label} className={`nav-link ${activeNav === link.label ? "active" : ""}`} onClick={() => { setActiveNav(link.label); setHistoryInitialId(undefined); setNotice(""); }}><Icon name={link.icon} size={19} /><span>{link.label}</span></button>)}</section>)}
        </nav>
        <div className="sidebar-profile"><div className="avatar">VM</div><div className="profile-copy"><strong>Vanshika Malik</strong><span>DefectZero Operator</span></div><Icon name="chevron" size={15} /></div>
      </aside>

      <main className="main-area">
        <header className="topbar"><div className="workspace-icon"><Icon name={activeNav === "Overview" ? "overview" : activeNav === "Model Evaluation" ? "evaluation" : activeNav === "Datasets" ? "datasets" : activeNav === "Reports" ? "report" : activeNav === "Settings" ? "settings" : activeNav === "Inspection History" ? "history" : "inspect"} size={28} /></div><div className="page-heading"><h1>{activeNav === "Overview" ? "Overview" : activeNav === "Inspection History" ? "Inspection History" : activeNav === "Model Evaluation" ? "Model Evaluation" : activeNav === "Batch Inspection" ? "Batch Inspection" : activeNav === "Datasets" ? "Dataset Explorer" : activeNav === "Reports" ? "Reports" : activeNav === "Settings" ? "Settings" : "Inspection Workspace"}</h1><p>{activeNav === "Overview" ? "Monitor recent zero-shot inspections and saved evaluation results." : activeNav === "Inspection History" ? "Review saved inspection results and download reports." : activeNav === "Model Evaluation" ? "Explore saved zero-shot evaluation metrics across MVTec AD." : activeNav === "Batch Inspection" ? "Inspect multiple images sequentially and review each result." : activeNav === "Datasets" ? "Browse verified MVTec AD categories, counts, and representative samples." : activeNav === "Reports" ? "Open saved inspection reports and review evaluation outputs." : activeNav === "Settings" ? "Review service status, inference settings, and input limits." : "Upload an image, select the MVTec AD category, and analyze results with visual comparison."}</p></div><div className={`connection-pill ${apiOnline === true ? "online" : apiOnline === false ? "offline" : "connecting"}`}><span className="status-dot" />{apiOnline === true ? "API Connected" : apiOnline === false ? "API Offline" : "Connecting..."}</div><button className="icon-button notification" type="button" aria-label="Service status details" aria-expanded={statusOpen} aria-controls="service-status-popover" title="Service status details" onClick={() => setStatusOpen((open) => !open)}><Icon name="bell" size={22} /><span className="notification-dot" /></button><div className="avatar top-avatar" aria-label="Vanshika Malik">VM</div></header>

        {statusOpen && <section className="service-status-popover" id="service-status-popover" aria-label="Service status details"><div className="service-status-heading"><strong>Service status</strong><button type="button" onClick={() => setStatusOpen(false)} aria-label="Close service status">×</button></div><div className="service-status-row"><span>FastAPI</span><strong className={apiOnline ? "online" : apiOnline === false ? "offline" : "pending"}>{apiOnline ? "Connected" : apiOnline === false ? "Offline" : "Checking"}</strong></div><div className="service-status-row"><span>WinCLIP model</span><strong>{health?.model_ready ? `Ready · ${health.device ?? "device unknown"}` : "Loads on first inspection"}</strong></div>{health?.last_model_error && <p className="service-status-error">{health.last_model_error}</p>}</section>}

        {notice && <div className="notice-banner" role="status">{notice}<button type="button" onClick={() => setNotice("")} aria-label="Dismiss"><Icon name="close" size={16} /></button></div>}
        {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" onClick={() => setError("")} aria-label="Dismiss error"><Icon name="close" size={16} /></button></div>}

        {activeNav === "Overview" ? <Overview onNavigate={(page, inspectionId) => { setActiveNav(page); setHistoryInitialId(inspectionId); setNotice(""); }} /> : activeNav === "Inspection History" ? <InspectionHistory initialInspectionId={historyInitialId} /> : activeNav === "Model Evaluation" ? <ModelEvaluation /> : activeNav === "Batch Inspection" ? <BatchInspection categories={categories} /> : activeNav === "Datasets" ? <DatasetExplorer initialCategory={datasetInitialCategory} /> : activeNav === "Reports" ? <Reports /> : activeNav === "Settings" ? <Settings /> : <div className="workspace-grid">
          <section className="panel setup-panel">
            <div className="panel-heading"><span className="round-icon"><Icon name="upload" size={21} /></span><h2>Inspection Setup</h2></div>
            <form onSubmit={runInspection}>
              <label className="field-label" htmlFor="category">Category</label>
              <select id="category" value={category} onChange={(event) => { setCategory(event.target.value); setPrediction(null); }} disabled={categories.length === 0 || loading}>
                {categories.length === 0 && <option value="">Waiting for API…</option>}
                {categories.map((item) => <option key={item} value={item}>{item.replaceAll("_", " ")}</option>)}
              </select>

              <div className="upload-label-row"><span className="field-label">Upload Image</span><span className="optional-label">PNG · JPG · JPEG · WEBP</span></div>
              <div className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop} onClick={() => inputRef.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} role="button" tabIndex={0} aria-label="Choose or drop an image">
                <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" onChange={onFileChange} hidden />
                {!file ? <><div className="upload-illustration"><Icon name="image" size={35} /></div><strong>Drag &amp; drop an image here</strong><span>or click to browse</span><small>Single-frame inspection image</small></> : <><div className="upload-illustration"><Icon name="upload" size={34} /></div><strong>Image ready</strong><span>Drop another image or click to replace</span><small>{file.type || "Image file"}</small></>}
              </div>
              {file && <div className="file-card"><img src={previewUrl} alt="Selected inspection preview" /><div className="file-details"><strong title={file.name}>{file.name}</strong><span>{file.type || "Image"} · {formatBytes(file.size)}</span></div><button type="button" className="icon-button remove-file" onClick={removeFile} aria-label="Remove selected image"><Icon name="close" size={19} /></button></div>}
              <button className="run-button" type="submit" disabled={!file || !category || loading || apiOnline === false}><span className={loading ? "spinner" : "run-icon"}>{loading ? "" : <Icon name="play" size={17} />}</span>{loading ? "Running Inspection…" : "Run Inspection"}</button>
              {loading && <p className="loading-note"><span className="loading-pulse" />WinCLIP is analyzing the image. First use may take longer while the category model loads.</p>}
            </form>
          </section>

          <section className="panel results-panel">
            <div className="panel-heading result-heading"><span className="round-icon"><Icon name="check" size={22} /></span><h2>Inspection Result</h2><div className="elapsed"><span>ϟ</span>{prediction ? `Processed in ${(prediction.inference_time_ms / 1000).toFixed(2)} s` : "Awaiting image"}</div></div>
            <div className={`result-banner ${prediction ? prediction.predicted_label : "empty"}`}>
              <span className="result-status-icon">{prediction ? (prediction.predicted_label === "normal" ? "✓" : "!") : <Icon name="activity" size={27} />}</span>
              <div><strong>{prediction ? scoreLabel : loading ? "Inspection in progress" : "No inspection yet"}</strong><span>{prediction ? (prediction.predicted_label === "normal" ? "No defect detected in the image." : "WinCLIP detected anomalous visual features.") : loading ? "The model is processing your upload." : "Select an image and run an inspection to see results."}</span></div>
            </div>
            <div className="metrics-grid">
              <Metric icon="activity" label="Anomaly Score" value={prediction ? prediction.anomaly_score.toFixed(4) : "—"} />
              <Metric icon="threshold" label="Threshold" value={prediction ? prediction.threshold.toFixed(4) : "—"} />
              <Metric icon="cube" label="Category" value={(prediction?.category.replaceAll("_", " ") ?? category) || "—"} />
              <Metric icon="timer" label="Inference Time" value={prediction ? `${(prediction.inference_time_ms / 1000).toFixed(2)} s` : "—"} />
            </div>

            <div className="comparison-section" ref={comparisonRef}>
              <div className={`comparison-grid mode-${view}`}>
                {view === "comparison" && <VisualPanel title="Normal Image" subtitle="Input" icon="image" className="original-visual">
                  {previewUrl ? <img className="visual-image" style={{ transform: `scale(${zoom})` }} src={previewUrl} alt="Uploaded inspection image" /> : <VisualPlaceholder icon="image" text="Upload an image to preview" />}
                </VisualPanel>}
                {view === "comparison" && <VisualPanel title="Ground Truth" subtitle={prediction?.ground_truth_available ? `Defect Mask · ${prediction.ground_truth_defect_type}` : "Defect Mask"} icon="mask" className="ground-truth-visual">
                  {prediction?.ground_truth_available && prediction.ground_truth_mask_base64 && prediction.ground_truth_mask_mime_type
                    ? <img className="visual-image ground-truth-image" src={`data:${prediction.ground_truth_mask_mime_type};base64,${prediction.ground_truth_mask_base64}`} alt={`Ground-truth defect mask for ${prediction.category} ${prediction.ground_truth_defect_type ?? "sample"}`} />
                    : <div className="unavailable-state"><span className="unavailable-icon"><Icon name="mask" size={25} /></span><strong>Ground truth unavailable</strong><span>No unique defective MVTec image match with a mask was found.</span></div>}
                </VisualPanel>}
                {view !== "original" && view !== "overlay" && <VisualPanel title="Heatmap" subtitle="Anomaly Localization" icon="spark" className="heatmap-visual">
                  {heatmapUrl ? <img className="visual-image" style={{ transform: `scale(${zoom})` }} src={heatmapUrl} alt="WinCLIP anomaly localization heatmap" /> : <VisualPlaceholder icon="spark" text="Heatmap appears after inspection" />}
                </VisualPanel>}
                {view === "overlay" && <VisualPanel title="Overlay" subtitle="Input + anomaly localization" icon="layers" className="overlay-visual">
                  <div className="overlay-canvas">{previewUrl ? <img className="visual-image" style={{ transform: `scale(${zoom})` }} src={previewUrl} alt="Uploaded image with heatmap overlay" /> : <VisualPlaceholder icon="image" text="Upload an image to preview" />}{heatmapUrl && <img className="overlay-heatmap" style={{ transform: `scale(${zoom})` }} src={heatmapUrl} alt="Anomaly heatmap overlay" />}</div>
                </VisualPanel>}
                {view === "original" && <VisualPanel title="Original Image" subtitle="Input image" icon="image" className="original-only">
                  {previewUrl ? <img className="visual-image" style={{ transform: `scale(${zoom})` }} src={previewUrl} alt="Uploaded inspection image" /> : <VisualPlaceholder icon="image" text="Upload an image to preview" />}
                </VisualPanel>}
                {(view === "comparison" || view === "overlay") && <div className="color-scale" aria-label="Heatmap scale from low to high"><span>High</span><div className="scale-bar" /><span>Low</span></div>}
              </div>
              <div className="controls-heading">View Controls</div>
              <div className="view-controls">
                <div className="view-tabs" role="group" aria-label="Comparison view">
                  <button className={view === "comparison" ? "selected" : ""} type="button" onClick={() => setView("comparison")}><Icon name="grid" size={17} /><span>Three-View Comparison</span></button>
                  <button className={view === "overlay" ? "selected" : ""} type="button" onClick={() => setView("overlay")}><Icon name="layers" size={17} /><span>Overlay</span></button>
                  <button className={view === "original" ? "selected" : ""} type="button" onClick={() => { setView("original"); setZoom(1); }}><Icon name="image" size={17} /><span>Original</span></button>
                </div>
                <div className="zoom-controls"><button type="button" onClick={() => setZoom((value) => Math.min(2.5, +(value + 0.15).toFixed(2)))} aria-label="Zoom in"><Icon name="plus" size={18} /><span>Zoom In</span></button><button type="button" onClick={() => setZoom((value) => Math.max(1, +(value - 0.15).toFixed(2)))} aria-label="Zoom out" disabled={zoom <= 1}><Icon name="minus" size={18} /><span>Zoom Out</span></button><button type="button" onClick={handleFullscreen}><Icon name="fullscreen" size={18} /><span>Fullscreen</span></button></div>
              </div>
            </div>

            <section className="insight-panel"><div className="insight-title"><Icon name="spark" size={22} /><h2>AI Insight</h2><span className="zero-shot-tag">ZERO-SHOT</span></div><p>{insight}</p></section>
            {apiOnline && health && <div className="model-footnote"><Icon name="wifi" size={15} /> API available · {health.model_ready ? `model ready on ${health.device ?? "device"}` : "category model loads on first inspection"} · pretrained weights · k-shot 0</div>}
          </section>
        </div>}
        <footer className="page-footer"><span>DefectZero AI</span><span>Zero-shot industrial inspection · Pretrained WinCLIP</span></footer>
      </main>
    </div>
  );
}

function Metric({ icon, label, value }: { icon: string; label: string; value: string }) {
  return <div className="metric-card"><span className="round-icon"><Icon name={icon} size={21} /></span><div><span>{label}</span><strong>{value}</strong></div></div>;
}

function VisualPanel({ title, subtitle, icon, className, children }: { title: string; subtitle: string; icon: string; className: string; children: ReactNode }) {
  return <section className={`visual-panel ${className}`}><div className="visual-header"><span className="visual-icon"><Icon name={icon} size={22} /></span><div><h3>{title}</h3><span>{subtitle}</span></div></div><div className="visual-frame">{children}</div></section>;
}

function VisualPlaceholder({ icon, text }: { icon: string; text: string }) {
  return <div className="visual-placeholder"><Icon name={icon} size={28} /><span>{text}</span></div>;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default App;
