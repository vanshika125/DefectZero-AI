import { useEffect, useMemo, useState } from "react";
import { downloadInspectionReport, fetchInspections, inspectionReportUrl } from "../api";
import type { InspectionRecord } from "../api";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const PAGE_SIZE = 100;

type EvaluationRow = { category: string; image_auroc: number; pixel_auroc: number; image_count: number; normal_count: number; defective_count: number };

export default function Reports() {
  const [records, setRecords] = useState<InspectionRecord[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationRow[]>([]);
  const [loadingInspections, setLoadingInspections] = useState(true);
  const [loadingEvaluation, setLoadingEvaluation] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [inspectionError, setInspectionError] = useState("");
  const [evaluationError, setEvaluationError] = useState("");
  const [reportBusyId, setReportBusyId] = useState<string | null>(null);

  const loadInspections = async (append = false) => {
    setLoadingInspections(true);
    setInspectionError("");
    try {
      const page = await fetchInspections(PAGE_SIZE, append ? offset : 0);
      setRecords((current) => append ? [...current, ...page] : page);
      const nextOffset = (append ? offset : 0) + page.length;
      setOffset(nextOffset);
      setHasMore(page.length === PAGE_SIZE);
    } catch (caught) {
      setInspectionError(caught instanceof Error ? caught.message : "Could not load saved inspection reports.");
    } finally {
      setLoadingInspections(false);
    }
  };

  const loadEvaluation = async () => {
    setLoadingEvaluation(true);
    setEvaluationError("");
    try {
      const response = await fetch(`${API_BASE}/evaluation`);
      if (!response.ok) throw new Error(`Could not load evaluation reports (HTTP ${response.status}).`);
      const data = await response.json() as { results?: EvaluationRow[] };
      if (!Array.isArray(data.results)) throw new Error("Saved evaluation summary is malformed.");
      setEvaluation(data.results);
    } catch (caught) {
      setEvaluationError(caught instanceof Error ? caught.message : "Could not load evaluation reports.");
    } finally {
      setLoadingEvaluation(false);
    }
  };

  useEffect(() => { void loadInspections(); void loadEvaluation(); }, []);

  const average = useMemo(() => evaluation.length ? {
    image: evaluation.reduce((sum, row) => sum + row.image_auroc, 0) / evaluation.length,
    pixel: evaluation.reduce((sum, row) => sum + row.pixel_auroc, 0) / evaluation.length,
    samples: evaluation.reduce((sum, row) => sum + row.image_count, 0),
  } : null, [evaluation]);

  const download = async (inspectionId: string) => {
    setReportBusyId(inspectionId);
    setInspectionError("");
    try { await downloadInspectionReport(inspectionId); }
    catch (caught) { setInspectionError(caught instanceof Error ? caught.message : "Could not download the PDF report."); }
    finally { setReportBusyId(null); }
  };

  return <section className="reports-page">
    <div className="reports-heading"><div><span className="history-eyebrow">INSPECTION & EVALUATION OUTPUTS</span><h2>Reports</h2><p>Open saved inspection reports and review the existing WinCLIP evaluation results.</p></div><button className="history-secondary-button" type="button" onClick={() => { void loadInspections(); void loadEvaluation(); }} disabled={loadingInspections || loadingEvaluation}>Refresh</button></div>

    <section className="reports-panel">
      <div className="reports-panel-heading"><div><h3>Inspection reports</h3><p>Reports are generated on request from saved inspection records and assets.</p></div><span className="reports-count">{records.length} loaded</span></div>
      {inspectionError && <div className="history-error" role="alert">{inspectionError}</div>}
      {loadingInspections && records.length === 0 ? <div className="history-state">Loading inspection reports…</div> : records.length === 0 ? <div className="history-empty"><strong>No inspection reports available</strong><span>Run and save an inspection first; its PDF report will appear here.</span></div> : <>
        <div className="reports-table-wrap"><table className="reports-table"><thead><tr><th>Date</th><th>Image filename</th><th>Category</th><th>Prediction</th><th>Anomaly score</th><th>PDF report</th></tr></thead><tbody>{records.map((record) => <tr key={record.inspection_id}>
          <td>{formatDate(record.timestamp_utc)}</td><th scope="row" title={record.filename}>{record.filename}</th><td>{formatCategory(record.category)}</td><td><span className={`history-label ${record.predicted_label}`}>{record.predicted_label}</span></td><td>{record.anomaly_score.toFixed(4)}</td><td><div className="reports-actions"><a className="reports-open-link" href={inspectionReportUrl(record.inspection_id, true)} target="_blank" rel="noreferrer">Open</a><button className="history-report-button" type="button" onClick={() => void download(record.inspection_id)} disabled={reportBusyId === record.inspection_id}>{reportBusyId === record.inspection_id ? "Preparing…" : "Download"}</button></div></td>
        </tr>)}</tbody></table></div>
        {hasMore && <div className="history-load-more"><button className="history-secondary-button" type="button" onClick={() => void loadInspections(true)} disabled={loadingInspections}>{loadingInspections ? "Loading…" : "Load more reports"}</button></div>}
      </>}
    </section>

    <section className="reports-panel">
      <div className="reports-panel-heading"><div><h3>Evaluation reports</h3><p>Links and summary values from the saved evaluation outputs.</p></div></div>
      {evaluationError && <div className="history-error" role="alert">{evaluationError}</div>}
      {loadingEvaluation && evaluation.length === 0 ? <div className="history-state">Loading evaluation summary…</div> : !evaluation.length ? <div className="history-empty"><strong>No evaluation report available</strong><span>{evaluationError || "No saved evaluation categories were found."}</span></div> : <>
        <div className="reports-evaluation-summary"><div><span>Categories</span><strong>{evaluation.length} / 15</strong></div><div><span>Images in saved evaluation</span><strong>{average?.samples ?? 0}</strong></div><div><span>Mean image AUROC</span><strong>{average?.image.toFixed(4) ?? "—"}</strong></div><div><span>Mean pixel AUROC</span><strong>{average?.pixel.toFixed(4) ?? "—"}</strong></div></div>
        <div className="reports-evaluation-links"><a href={`${API_BASE}/evaluation/report`} target="_blank" rel="noreferrer">Open saved evaluation summary</a><a href={`${API_BASE}/evaluation/assets/image_auroc`} target="_blank" rel="noreferrer">Open image AUROC chart</a><a href={`${API_BASE}/evaluation/assets/pixel_auroc`} target="_blank" rel="noreferrer">Open pixel AUROC chart</a></div>
        <p className="reports-evaluation-note">The evaluation summary uses the saved 20-image-per-category pilot. Values are exploratory and should not be treated as full-benchmark performance.</p>
      </>}
    </section>
  </section>;
}

function formatCategory(value: string) { return value.replaceAll("_", " "); }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? value : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); }
