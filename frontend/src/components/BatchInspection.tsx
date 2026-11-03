import { useEffect, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const MAX_FILES = 30;
const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp", "image/bmp"]);

type Item = { key: string; file: File; previewUrl: string; status: "queued" | "running" | "completed" | "failed" | "invalid"; error?: string; prediction?: Prediction };
type Prediction = { inspection_id: string; category: string; predicted_label: "normal" | "defective"; anomaly_score: number; threshold: number; inference_time_ms: number; anomaly_map_visualization_base64: string; anomaly_map_visualization_mime_type: string };

export default function BatchInspection({ categories }: { categories: string[] }) {
  const [category, setCategory] = useState(categories[0] ?? "");
  const [items, setItems] = useState<Item[]>([]);
  const [running, setRunning] = useState(false);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { if (!category && categories.length) setCategory(categories[0]); }, [categories, category]);

  const addFiles = (files: FileList | File[]) => {
    const incoming = Array.from(files);
    const available = Math.max(0, MAX_FILES - items.length);
    if (incoming.length > available) setError(`A batch can contain at most ${MAX_FILES} images.`);
    setItems((current) => {
      const additions = incoming.slice(0, available).map((file) => ({
        key: `${file.name}-${file.size}-${crypto.randomUUID()}`,
        file,
        previewUrl: URL.createObjectURL(file),
        status: ACCEPTED.has(file.type) ? "queued" as const : "invalid" as const,
        error: ACCEPTED.has(file.type) ? undefined : "Unsupported format. Use PNG, JPG, WEBP, or BMP.",
      }));
      return [...current, ...additions];
    });
    if (inputRef.current) inputRef.current.value = "";
  };

  const runBatch = async () => {
    if (running || !category) return;
    const queue = items.filter((item) => item.status === "queued");
    if (!queue.length) return;
    setRunning(true);
    setError("");
    for (const item of queue) {
      setItems((current) => current.map((candidate) => candidate.key === item.key ? { ...candidate, status: "running", error: undefined } : candidate));
      try {
        const form = new FormData();
        form.append("category", category);
        form.append("image", item.file);
        const response = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
        const payload = await response.json().catch(() => null) as (Prediction & { detail?: { message?: string } | string }) | null;
        if (!response.ok) {
          const detail = typeof payload?.detail === "string" ? payload.detail : payload?.detail?.message;
          throw new Error(detail || `Inspection failed with HTTP ${response.status}.`);
        }
        if (!payload?.inspection_id || !payload.anomaly_map_visualization_base64) throw new Error("The API returned an incomplete result.");
        setItems((current) => current.map((candidate) => candidate.key === item.key ? { ...candidate, status: "completed", prediction: payload } : candidate));
      } catch (caught) {
        setItems((current) => current.map((candidate) => candidate.key === item.key ? { ...candidate, status: "failed", error: caught instanceof Error ? caught.message : "Inspection request failed." } : candidate));
      }
    }
    setRunning(false);
  };

  const downloadIndividual = async (id: string) => {
    try {
      const response = await fetch(`${API_BASE}/inspections/${encodeURIComponent(id)}/report.pdf`);
      if (!response.ok) throw new Error(`Report download failed (HTTP ${response.status}).`);
      saveBlob(await response.blob(), `defectzero-inspection-${id}.pdf`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not download the inspection report.");
    }
  };

  const downloadSummary = async () => {
    const ids = items.flatMap((item) => item.prediction ? [item.prediction.inspection_id] : []);
    if (!ids.length || summaryBusy) return;
    setSummaryBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/inspections/batch-report.pdf`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ inspection_ids: ids }) });
      if (!response.ok) throw new Error(`Batch summary failed (HTTP ${response.status}).`);
      saveBlob(await response.blob(), "defectzero-batch-summary.pdf");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not download the batch summary.");
    } finally {
      setSummaryBusy(false);
    }
  };

  const removeItem = (key: string) => setItems((current) => {
    const removed = current.find((item) => item.key === key);
    if (removed) URL.revokeObjectURL(removed.previewUrl);
    return current.filter((item) => item.key !== key);
  });
  const clearItems = () => { items.forEach((item) => URL.revokeObjectURL(item.previewUrl)); setItems([]); setError(""); };
  const completeCount = items.filter((item) => item.status === "completed").length;
  const settledCount = items.filter((item) => ["completed", "failed", "invalid"].includes(item.status)).length;

  return <section className="batch-page">
    <div className="batch-heading"><div><span className="history-eyebrow">SEQUENTIAL ZERO-SHOT RUN</span><h2>Batch Inspection</h2><p>Inspect a set of images in order. Each successful image is saved to Inspection History.</p></div><div className="batch-heading-actions"><button className="history-secondary-button" type="button" onClick={clearItems} disabled={!items.length || running}>Clear batch</button><button className="run-button batch-run-button" type="button" onClick={() => void runBatch()} disabled={running || !category || !items.some((item) => item.status === "queued")}>{running ? "Processing batch…" : "Run batch"}</button></div></div>
    <div className="batch-setup-row"><label>Category<select value={category} onChange={(event) => setCategory(event.target.value)} disabled={running || !categories.length}>{categories.map((item) => <option value={item} key={item}>{item.replaceAll("_", " ")}</option>)}</select></label><button className="batch-upload-button" type="button" onClick={() => inputRef.current?.click()} disabled={running || items.length >= MAX_FILES}>Add images<input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" multiple hidden onChange={(event) => event.target.files && addFiles(event.target.files)} /></button><span className="batch-limit">Up to {MAX_FILES} images per batch</span></div>
    {items.length > 0 && <div className="batch-progress-area"><div className="batch-progress-label"><strong>{running ? "Inspection in progress" : "Batch progress"}</strong><span>{settledCount} / {items.length} processed · {completeCount} saved</span></div><div className="batch-progress-track"><span style={{ width: `${items.length ? settledCount / items.length * 100 : 0}%` }} /></div></div>}
    {error && <div className="history-error" role="alert">{error}</div>}
    {!items.length ? <div className="batch-empty"><strong>Select multiple images to begin</strong><span>Valid files are processed sequentially; unsupported files are marked without stopping the batch.</span><button className="batch-upload-button" type="button" onClick={() => inputRef.current?.click()}>Browse images<input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" multiple hidden onChange={(event) => event.target.files && addFiles(event.target.files)} /></button></div> : <div className="batch-results-grid">{items.map((item) => <article className="batch-result-card" key={item.key}>
      <div className="batch-result-heading"><span className={`batch-status ${item.status}`}>{statusLabel(item.status)}</span><button className="batch-remove" type="button" disabled={running} onClick={() => removeItem(item.key)} aria-label={`Remove ${item.file.name}`}>×</button></div>
      <div className={`batch-image-pair ${item.prediction ? "has-heatmap" : ""}`}><div className="batch-image-frame"><img src={item.previewUrl} alt={`Input ${item.file.name}`} /><span>Input</span></div>{item.prediction && <div className="batch-image-frame"><img src={`data:${item.prediction.anomaly_map_visualization_mime_type};base64,${item.prediction.anomaly_map_visualization_base64}`} alt={`Heatmap for ${item.file.name}`} /><span>Heatmap</span></div>}</div>
      <strong className="batch-filename" title={item.file.name}>{item.file.name}</strong>
      {item.prediction ? <div className="batch-result-metrics"><span className={`history-label ${item.prediction.predicted_label}`}>{item.prediction.predicted_label}</span><span>Score <b>{item.prediction.anomaly_score.toFixed(4)}</b></span><span>{(item.prediction.inference_time_ms / 1000).toFixed(2)} s</span></div> : <p className={`batch-item-message ${item.status}`}>{item.error ?? (item.status === "running" ? "Running WinCLIP inference…" : "Waiting in queue")}</p>}
      {item.prediction && <button className="history-report-button batch-report-button" type="button" onClick={() => void downloadIndividual(item.prediction!.inspection_id)}>Individual PDF report</button>}
    </article>)}</div>}
    {completeCount > 0 && <div className="batch-summary-action"><span>{completeCount} inspection(s) saved in history.</span><button className="history-report-button prominent" type="button" onClick={() => void downloadSummary()} disabled={summaryBusy}>{summaryBusy ? "Preparing summary…" : "Download batch summary PDF"}</button></div>}
  </section>;
}

function statusLabel(status: Item["status"]) { return ({ queued: "Queued", running: "Running", completed: "Saved", failed: "Failed", invalid: "Invalid file" })[status]; }
function saveBlob(blob: Blob, filename: string) { const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000); }
