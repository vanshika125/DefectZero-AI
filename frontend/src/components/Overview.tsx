import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchInspections } from "../api";
import type { InspectionRecord } from "../api";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const RECENT_LIMIT = 20;

type EvaluationItem = { category: string; image_auroc: number; pixel_auroc: number; image_count: number; normal_count: number; defective_count: number };
type EvaluationResponse = { results: EvaluationItem[] };
type DatasetSummary = { categories: unknown[]; category_count: number };

export default function Overview({ onNavigate }: { onNavigate: (page: string, inspectionId?: string) => void }) {
  const [records, setRecords] = useState<InspectionRecord[]>([]);
  const [categoryCount, setCategoryCount] = useState<number | null>(null);
  const [datasetCategoryCount, setDatasetCategoryCount] = useState<number | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const [evaluationError, setEvaluationError] = useState("");
  const [datasetError, setDatasetError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setHistoryError("");
    setEvaluationError("");
    setDatasetError("");

    const [historyResult, categoriesResult, evaluationResult, datasetResult] = await Promise.allSettled([
      fetchInspections(RECENT_LIMIT, 0),
      fetch(`${API_BASE}/categories`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json() as string[];
      }),
      fetch(`${API_BASE}/evaluation`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json() as EvaluationResponse;
      }),
      fetch(`${API_BASE}/dataset`).then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json() as DatasetSummary;
      }),
    ]);

    if (historyResult.status === "fulfilled") setRecords(historyResult.value);
    else { setRecords([]); setHistoryError(errorMessage(historyResult.reason)); }
    if (categoriesResult.status === "fulfilled") setCategoryCount(categoriesResult.value.length);
    else setCategoryCount(null);
    if (evaluationResult.status === "fulfilled" && Array.isArray(evaluationResult.value.results)) setEvaluation(evaluationResult.value.results.filter((item) => Number.isFinite(item.image_auroc) && Number.isFinite(item.pixel_auroc) && Number.isFinite(item.image_count)));
    else { setEvaluation([]); setEvaluationError(evaluationResult.status === "rejected" ? errorMessage(evaluationResult.reason) : "Evaluation data was malformed."); }
    if (datasetResult.status === "fulfilled") setDatasetCategoryCount(datasetResult.value.category_count);
    else { setDatasetCategoryCount(null); setDatasetError(errorMessage(datasetResult.reason)); }
    setLoading(false);
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const summary = useMemo(() => {
    const meanImageAuroc = evaluation.length ? evaluation.reduce((total, item) => total + item.image_auroc, 0) / evaluation.length : null;
    const meanPixelAuroc = evaluation.length ? evaluation.reduce((total, item) => total + item.pixel_auroc, 0) / evaluation.length : null;
    return {
      defective: records.filter((record) => record.predicted_label === "defective").length,
      normal: records.filter((record) => record.predicted_label === "normal").length,
      meanImageAuroc,
      meanPixelAuroc,
      evaluationImages: evaluation.reduce((total, item) => total + item.image_count, 0),
    };
  }, [records, evaluation]);

  return <section className="overview-page">
    <div className="overview-heading"><div><span className="history-eyebrow">SYSTEM OVERVIEW</span><h2>Overview</h2><p>At-a-glance status for zero-shot inspections, saved results, and the evaluation pilot.</p></div><button className="history-secondary-button" type="button" onClick={() => void refresh()} disabled={loading}>{loading ? "Refreshing…" : "Refresh dashboard"}</button></div>

    <section className="overview-welcome"><div><span className="overview-live-dot" /><div><strong>DefectZero AI · Zero-shot inspection</strong><span>Pretrained WinCLIP analyzes industrial images without training or fine-tuning in this project.</span></div></div><button type="button" className="overview-primary-action" onClick={() => onNavigate("Inspect")}>Start an inspection <span aria-hidden="true">→</span></button></section>

    <div className="overview-metrics-grid">
      <OverviewMetric label="MVTec categories" value={categoryCount == null ? "—" : `${categoryCount} / 15`} note="Supported by the running API" />
      <OverviewMetric label="Categories on disk" value={datasetCategoryCount == null ? "—" : `${datasetCategoryCount} / 15`} note={datasetError ? "Dataset summary unavailable" : "Verified dataset folders"} />
      <OverviewMetric label="Mean image AUROC" value={summary.meanImageAuroc == null ? "—" : summary.meanImageAuroc.toFixed(4)} note={evaluation.length ? `Saved pilot · ${evaluation.length} categories` : "No saved evaluation data"} />
      <OverviewMetric label="Mean pixel AUROC" value={summary.meanPixelAuroc == null ? "—" : summary.meanPixelAuroc.toFixed(4)} note={evaluation.length ? `${summary.evaluationImages} evaluation images` : "No saved evaluation data"} />
    </div>

    {evaluationError && <div className="overview-inline-note" role="status">Saved evaluation metrics are unavailable: {evaluationError}</div>}
    {datasetError && <div className="overview-inline-note" role="status">Dataset summary is unavailable: {datasetError}</div>}

    <div className="overview-content-grid">
      <section className="overview-panel overview-recent-panel"><div className="overview-panel-heading"><div><h3>Recent inspections</h3><p>Newest saved inspections returned by history.</p></div><button type="button" className="overview-text-action" onClick={() => onNavigate("Inspection History")}>View history <span aria-hidden="true">→</span></button></div>
        {historyError ? <div className="overview-empty"><strong>Inspection history unavailable</strong><span>{historyError}</span></div> : loading && records.length === 0 ? <div className="overview-empty">Loading recent inspections…</div> : records.length === 0 ? <div className="overview-empty"><strong>No saved inspections yet</strong><span>Run a single or batch inspection to create history records.</span><button type="button" className="overview-text-action" onClick={() => onNavigate("Inspect")}>Open inspection workspace <span aria-hidden="true">→</span></button></div> : <>
          <div className="overview-record-summary"><span><i className="overview-normal-dot" />{summary.normal} normal</span><span><i className="overview-defective-dot" />{summary.defective} defective</span><small>Latest {records.length} records</small></div>
          <div className="overview-record-list">{records.slice(0, 6).map((record) => <button type="button" className="overview-record-row" key={record.inspection_id} onClick={() => onNavigate("Inspection History", record.inspection_id)}><span className={`overview-record-status ${record.predicted_label}`} aria-label={record.predicted_label} /><span className="overview-record-file"><strong>{record.filename}</strong><small>{record.category.replaceAll("_", " ")} · {formatDate(record.timestamp_utc)}</small></span><span className={`history-label ${record.predicted_label}`}>{record.predicted_label}</span><strong className="overview-record-score">{record.anomaly_score.toFixed(4)}</strong></button>)}</div>
        </>}
      </section>

      <section className="overview-panel overview-actions-panel"><div className="overview-panel-heading"><div><h3>Workspaces</h3><p>Open a project area.</p></div></div><div className="overview-workspace-links">
        <WorkspaceLink title="Single inspection" detail="Upload an image and view its anomaly map." onClick={() => onNavigate("Inspect")} />
        <WorkspaceLink title="Batch inspection" detail="Run images sequentially and track each result." onClick={() => onNavigate("Batch Inspection")} />
        <WorkspaceLink title="Dataset explorer" detail="Browse categories, counts, samples, and masks." onClick={() => onNavigate("Datasets")} />
        <WorkspaceLink title="Model evaluation" detail="Review saved pilot metrics by category." onClick={() => onNavigate("Model Evaluation")} />
        <WorkspaceLink title="Reports" detail="Open saved inspection and evaluation reports." onClick={() => onNavigate("Reports")} />
      </div></section>
    </div>

    <section className="overview-pilot-note"><strong>Evaluation context</strong><span>The dashboard uses saved evaluation artifacts only. The project’s pilot results use limited samples per category, so they are exploratory and do not establish production performance.</span></section>
  </section>;
}

function OverviewMetric({ label, value, note }: { label: string; value: string; note: string }) { return <article className="overview-metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></article>; }
function WorkspaceLink({ title, detail, onClick }: { title: string; detail: string; onClick: () => void }) { return <button type="button" className="overview-workspace-link" onClick={onClick}><span><strong>{title}</strong><small>{detail}</small></span><b aria-hidden="true">→</b></button>; }
function errorMessage(reason: unknown) { return reason instanceof Error ? reason.message : "Request failed."; }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? "Date unavailable" : date.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }); }
