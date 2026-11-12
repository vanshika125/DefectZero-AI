import { useEffect, useMemo, useState } from "react";
import { downloadInspectionReport, fetchInspection, fetchInspections, inspectionAssetUrl } from "../api";
import type { InspectionAsset, InspectionRecord } from "../api";

const PAGE_SIZE = 50;

export default function InspectionHistory({ initialInspectionId }: { initialInspectionId?: string }) {
  const [records, setRecords] = useState<InspectionRecord[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<InspectionRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [reportBusyId, setReportBusyId] = useState<string | null>(null);
  const [error, setError] = useState("");

  const loadInitial = async () => {
    setLoading(true);
    setError("");
    try {
      const page = await fetchInspections(PAGE_SIZE, 0);
      setRecords(page);
      setOffset(page.length);
      setHasMore(page.length === PAGE_SIZE);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load inspection history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void loadInitial(); }, []);

  const categories = useMemo(
    () => [...new Set(records.map((record) => record.category))].sort((a, b) => a.localeCompare(b)),
    [records],
  );
  const filteredRecords = useMemo(() => records.filter((record) =>
    (categoryFilter === "all" || record.category === categoryFilter)
    && (labelFilter === "all" || record.predicted_label === labelFilter),
  ), [records, categoryFilter, labelFilter]);

  const loadMore = async () => {
    setLoading(true);
    setError("");
    try {
      const page = await fetchInspections(PAGE_SIZE, offset);
      setRecords((current) => [...current, ...page]);
      setOffset((current) => current + page.length);
      setHasMore(page.length === PAGE_SIZE);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load more inspections.");
    } finally {
      setLoading(false);
    }
  };

  const openInspection = async (inspectionId: string) => {
    setSelectedId(inspectionId);
    setSelected(null);
    setDetailLoading(true);
    setError("");
    try {
      setSelected(await fetchInspection(inspectionId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load inspection details.");
    } finally {
      setDetailLoading(false);
    }
  };

  const closeInspection = () => {
    setSelectedId(null);
    setSelected(null);
    setError("");
  };

  useEffect(() => { if (initialInspectionId) void openInspection(initialInspectionId); }, [initialInspectionId]);

  const downloadReport = async (inspectionId: string) => {
    setReportBusyId(inspectionId);
    setError("");
    try {
      await downloadInspectionReport(inspectionId);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not download this report.");
    } finally {
      setReportBusyId(null);
    }
  };

  return (
    <section className="history-page">
      <div className="history-page-heading">
        <div>
          <span className="history-eyebrow">INSPECTION RECORDS</span>
          <h2>{selected ? "Inspection Details" : "Inspection History"}</h2>
          <p>{selected ? "Saved result, source image, and inspection visualizations." : "Browse completed inspections and download their reports."}</p>
        </div>
        <div className="history-heading-actions">
          {selected && <button className="history-secondary-button" type="button" onClick={closeInspection}>Back to history</button>}
          {!selected && <button className="history-secondary-button" type="button" onClick={() => void loadInitial()} disabled={loading}>Refresh</button>}
        </div>
      </div>

      {error && <div className="history-error" role="alert">{error}</div>}

      {selectedId ? (
        detailLoading ? <div className="history-state">Loading inspection details…</div>
          : selected ? <InspectionDetail record={selected} reportBusy={reportBusyId === selected.inspection_id} onDownload={() => void downloadReport(selected.inspection_id)} />
            : <div className="history-state">Inspection details are unavailable.</div>
      ) : (
        <>
          <div className="history-filter-bar">
            <label>Category
              <select value={categoryFilter} onChange={(event) => setCategoryFilter(event.target.value)}>
                <option value="all">All categories</option>
                {categories.map((item) => <option value={item} key={item}>{formatCategory(item)}</option>)}
              </select>
            </label>
            <label>Prediction
              <select value={labelFilter} onChange={(event) => setLabelFilter(event.target.value)}>
                <option value="all">All predictions</option>
                <option value="normal">Normal</option>
                <option value="defective">Defective</option>
              </select>
            </label>
            <span className="history-count">{filteredRecords.length} shown · {records.length} loaded</span>
          </div>

          {loading && records.length === 0 ? <div className="history-state">Loading saved inspections…</div>
            : filteredRecords.length === 0 ? <div className="history-empty"><strong>No inspections found</strong><span>{records.length ? "Try changing the filters." : "Completed inspections will appear here."}</span></div>
              : <div className="history-table-shell">
                <div className="history-table-header"><span>Date</span><span>Filename</span><span>Category</span><span>Prediction</span><span>Anomaly score</span><span>Inference</span><span>Report</span></div>
                {filteredRecords.map((record) => <div className="history-record-row" key={record.inspection_id}>
                  <button className="history-record-main" type="button" onClick={() => void openInspection(record.inspection_id)}>
                    <span className="history-date" data-label="Date">{formatDate(record.timestamp_utc)}</span>
                    <strong className="history-filename" data-label="Filename" title={record.filename}>{record.filename}</strong>
                    <span className="history-category" data-label="Category">{formatCategory(record.category)}</span>
                    <span className={`history-label ${record.predicted_label}`} data-label="Prediction">{record.predicted_label}</span>
                    <strong className="history-score" data-label="Anomaly score">{record.anomaly_score.toFixed(4)}</strong>
                    <span className="history-time" data-label="Inference">{(record.inference_time_ms / 1000).toFixed(2)} s</span>
                  </button>
                  <button className="history-report-button" type="button" onClick={() => void downloadReport(record.inspection_id)} disabled={reportBusyId === record.inspection_id} aria-label={`Download report for ${record.filename}`}>
                    {reportBusyId === record.inspection_id ? "Preparing…" : "Download PDF"}
                  </button>
                </div>)}
              </div>}
          {hasMore && <div className="history-load-more"><button className="history-secondary-button" type="button" onClick={() => void loadMore()} disabled={loading}>{loading ? "Loading…" : "Load more inspections"}</button></div>}
        </>
      )}
    </section>
  );
}

function InspectionDetail({ record, reportBusy, onDownload }: { record: InspectionRecord; reportBusy: boolean; onDownload: () => void }) {
  const assets: { key: InspectionAsset; title: string }[] = [
    { key: "image", title: "Input Image" },
    { key: "heatmap", title: "Anomaly Heatmap" },
    ...(record.ground_truth_mask_path ? [{ key: "ground_truth" as const, title: "Ground-Truth Mask" }] : []),
  ];

  return <div className="history-detail-layout">
    <section className="history-detail-panel">
      <div className="history-detail-title"><div><span className="history-eyebrow">SAVED INSPECTION</span><h3 title={record.filename}>{record.filename}</h3></div><button className="history-report-button prominent" type="button" onClick={onDownload} disabled={reportBusy}>{reportBusy ? "Preparing PDF…" : "Download PDF report"}</button></div>
      <div className="history-metadata-grid">
        <HistoryValue label="Category" value={formatCategory(record.category)} />
        <HistoryValue label="Prediction" value={record.predicted_label} prediction={record.predicted_label} />
        <HistoryValue label="Anomaly score" value={record.anomaly_score.toFixed(4)} />
        <HistoryValue label="Threshold" value={record.threshold.toFixed(4)} />
        <HistoryValue label="Inference time" value={`${(record.inference_time_ms / 1000).toFixed(2)} s`} />
        <HistoryValue label="Date" value={formatDate(record.timestamp_utc)} />
        <HistoryValue label="Inspection ID" value={record.inspection_id} wide />
      </div>
    </section>
    <div className={`history-visual-grid ${assets.length === 3 ? "three-assets" : "two-assets"}`}>
      {assets.map((asset) => <section className="history-asset-panel" key={asset.key}>
        <h3>{asset.title}</h3>
        <div className="history-asset-frame"><img src={inspectionAssetUrl(record.inspection_id, asset.key)} alt={`${asset.title} for ${record.filename}`} /></div>
      </section>)}
      {!record.ground_truth_mask_path && <section className="history-asset-panel history-mask-unavailable"><h3>Ground-Truth Mask</h3><div className="history-asset-frame"><span>No matching ground-truth mask was saved for this inspection.</span></div></section>}
    </div>
  </div>;
}

function HistoryValue({ label, value, prediction, wide = false }: { label: string; value: string; prediction?: "normal" | "defective"; wide?: boolean }) {
  return <div className={`history-value ${wide ? "wide" : ""}`}><span>{label}</span><strong className={prediction ? `history-label ${prediction}` : ""} title={value}>{value}</strong></div>;
}

function formatCategory(category: string) {
  return category.replaceAll("_", " ");
}

function formatDate(timestamp: string) {
  const date = new Date(timestamp);
  return Number.isNaN(date.valueOf()) ? timestamp : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
