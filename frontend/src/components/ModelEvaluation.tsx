import { useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

type EvaluationResult = {
  category: string;
  image_auroc: number;
  pixel_auroc: number;
  image_count: number;
  normal_count: number;
  defective_count: number;
  mean_inference_time_sec: number;
  std_inference_time_sec: number | null;
  min_inference_time_sec: number | null;
  max_inference_time_sec: number | null;
  total_evaluation_time_sec: number | null;
  class_distribution: Record<string, number>;
  model_name: string | null;
  inference_mode: string | null;
  timestamp_utc: string | null;
};

type EvaluationResponse = { results: EvaluationResult[]; source: string };

export default function ModelEvaluation() {
  const [results, setResults] = useState<EvaluationResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/evaluation`);
      if (!response.ok) throw new Error(`Could not load saved evaluation results (HTTP ${response.status}).`);
      const payload = await response.json() as EvaluationResponse;
      if (!Array.isArray(payload.results)) throw new Error("The saved evaluation response is malformed.");
      setResults(payload.results);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load saved evaluation results.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);

  const averages = useMemo(() => {
    const valid = results.filter((item) => Number.isFinite(item.image_auroc) && Number.isFinite(item.pixel_auroc));
    if (!valid.length) return { image: null, pixel: null, samples: 0 };
    return {
      image: valid.reduce((sum, item) => sum + item.image_auroc, 0) / valid.length,
      pixel: valid.reduce((sum, item) => sum + item.pixel_auroc, 0) / valid.length,
      samples: valid.reduce((sum, item) => sum + item.image_count, 0),
    };
  }, [results]);

  return <section className="evaluation-page">
    <div className="evaluation-heading">
      <div><span className="evaluation-eyebrow">ZERO-SHOT BENCHMARK</span><h2>Model Evaluation</h2><p>Saved WinCLIP pilot results across the MVTec AD categories.</p></div>
      <button className="history-secondary-button" type="button" onClick={() => void load()} disabled={loading}>{loading ? "Loading…" : "Refresh"}</button>
    </div>

    <div className="evaluation-notice"><strong>Pilot evaluation</strong><span>Each category uses the recorded 20-image sample (10 normal, 10 defective). Treat these results as exploratory; they do not establish production performance.</span></div>
    {error && <div className="history-error" role="alert">{error}</div>}
    {loading && results.length === 0 ? <div className="history-state">Loading saved metrics…</div> : results.length === 0 ? <div className="history-empty"><strong>No saved evaluation results</strong><span>{error || "Expected summary metrics were not found."}</span></div> : <>
      <div className="evaluation-summary-grid">
        <MetricCard label="Mean image AUROC" value={formatMetric(averages.image)} note="Unweighted across available categories" />
        <MetricCard label="Mean pixel AUROC" value={formatMetric(averages.pixel)} note="Unweighted across available categories" />
        <MetricCard label="Categories" value={`${results.length} / 15`} note="Categories with saved results" />
        <MetricCard label="Images evaluated" value={averages.samples.toLocaleString()} note="Total across saved categories" />
      </div>

      <section className="evaluation-panel">
        <div className="evaluation-panel-heading"><div><h3>Per-category AUROC</h3><p>Higher values indicate stronger ranking performance in this sample.</p></div><div className="evaluation-legend"><span><i className="legend-image" />Image-level</span><span><i className="legend-pixel" />Pixel-level</span></div></div>
        {results.map((item) => <div className="evaluation-chart-row" key={item.category}>
          <strong>{formatCategory(item.category)}</strong>
          <div className="evaluation-bars">
            <div className="evaluation-bar-track" title={`Image AUROC ${item.image_auroc.toFixed(4)}`}><span className="evaluation-bar image" style={{ width: `${clamp(item.image_auroc * 100)}%` }} /></div>
            <div className="evaluation-bar-track" title={`Pixel AUROC ${item.pixel_auroc.toFixed(4)}`}><span className="evaluation-bar pixel" style={{ width: `${clamp(item.pixel_auroc * 100)}%` }} /></div>
          </div>
          <span className="evaluation-chart-value">{item.image_auroc.toFixed(3)} <b>/</b> {item.pixel_auroc.toFixed(3)}</span>
        </div>)}
        <div className="evaluation-axis"><span>0.0</span><span>0.5</span><span>1.0</span></div>
      </section>

      <section className="evaluation-panel evaluation-table-panel">
        <div className="evaluation-panel-heading"><div><h3>Evaluation details</h3><p>Additional metrics are shown when present in the saved category JSON.</p></div><span className="evaluation-source">Existing evaluation artifacts</span></div>
        <div className="evaluation-table-wrap"><table className="evaluation-table"><thead><tr><th>Category</th><th>Image AUROC</th><th>Pixel AUROC</th><th>Images</th><th>Normal / Defective</th><th>Mean inference</th><th>Inference std</th></tr></thead><tbody>{results.map((item) => <tr key={item.category}><th scope="row">{formatCategory(item.category)}</th><td>{item.image_auroc.toFixed(4)}</td><td>{item.pixel_auroc.toFixed(4)}</td><td>{item.image_count}</td><td>{item.normal_count} / {item.defective_count}</td><td>{formatSeconds(item.mean_inference_time_sec)}</td><td>{item.std_inference_time_sec == null ? "—" : formatSeconds(item.std_inference_time_sec)}</td></tr>)}</tbody></table></div>
      </section>
      <p className="evaluation-footnote">Pretrained WinCLIP · zero-shot (k-shot = 0) · no training performed. Metrics and counts are read from <code>experiments/evaluation_results/</code>. Evaluation sample size and timing variability limit cross-category conclusions.</p>
    </>}
  </section>;
}

function MetricCard({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="evaluation-metric-card"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function formatMetric(value: number | null) { return value == null ? "—" : value.toFixed(4); }
function formatSeconds(value: number) { return `${value.toFixed(2)} s`; }
function formatCategory(value: string) { return value.replaceAll("_", " "); }
function clamp(value: number) { return Math.min(100, Math.max(0, value)); }
