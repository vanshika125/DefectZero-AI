import { useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

type CategoryStats = { category: string; training_count: number; normal_test_count: number; defective_test_count: number; test_count: number; ground_truth_mask_count: number; training_count_verified: boolean; normal_test_count_verified: boolean; defective_test_count_verified: boolean; ground_truth_mask_count_verified: boolean };
type DatasetSummary = { categories: CategoryStats[]; category_count: number };
type Sample = { sample_id: string; filename: string; group: "normal" | "defective"; defect_type: string; mask_available: boolean; image_url: string; mask_url: string | null };

export default function DatasetExplorer() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [error, setError] = useState("");
  const [samplesError, setSamplesError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/dataset`).then(async (response) => {
      if (!response.ok) throw new Error(`Dataset metadata request failed (HTTP ${response.status}).`);
      return await response.json() as DatasetSummary;
    }).then((data) => {
      if (!cancelled) {
        setSummary(data);
        setSelectedCategory(data.categories[0]?.category ?? "");
      }
    }).catch((caught: unknown) => {
      if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not load dataset metadata.");
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedCategory) return;
    let cancelled = false;
    setSamplesLoading(true);
    setSamplesError("");
    fetch(`${API_BASE}/dataset/${encodeURIComponent(selectedCategory)}/samples?limit_per_group=3`).then(async (response) => {
      if (!response.ok) throw new Error(`Sample request failed (HTTP ${response.status}).`);
      return await response.json() as { samples: Sample[] };
    }).then((data) => { if (!cancelled) setSamples(data.samples); }).catch((caught: unknown) => {
      if (!cancelled) { setSamples([]); setSamplesError(caught instanceof Error ? caught.message : "Could not load representative samples."); }
    }).finally(() => { if (!cancelled) setSamplesLoading(false); });
    return () => { cancelled = true; };
  }, [selectedCategory]);

  const selected = summary?.categories.find((item) => item.category === selectedCategory) ?? null;
  const totals = useMemo(() => {
    const rows = summary?.categories ?? [];
    return rows.reduce((acc, row) => ({
      train: acc.train + (row.training_count_verified ? row.training_count : 0),
      normal: acc.normal + (row.normal_test_count_verified ? row.normal_test_count : 0),
      defective: acc.defective + (row.defective_test_count_verified ? row.defective_test_count : 0),
      masks: acc.masks + (row.ground_truth_mask_count_verified ? row.ground_truth_mask_count : 0),
    }), { train: 0, normal: 0, defective: 0, masks: 0 });
  }, [summary]);

  return <section className="dataset-page">
    <div className="dataset-heading"><div><span className="history-eyebrow">MVTEC ANOMALY DETECTION</span><h2>Dataset Explorer</h2><p>Browse available categories, verified image counts, and representative test samples.</p></div><label className="dataset-category-filter">Filter category<select value={selectedCategory} onChange={(event) => setSelectedCategory(event.target.value)} disabled={!summary?.categories.length}><option value="">Choose a category</option>{summary?.categories.map((row) => <option key={row.category} value={row.category}>{formatCategory(row.category)}</option>)}</select></label></div>
    {error && <div className="history-error" role="alert">{error}</div>}
    {loading ? <div className="history-state">Loading dataset metadata…</div> : !summary?.categories.length ? <div className="history-empty"><strong>Dataset metadata unavailable</strong><span>{error || "No supported MVTec category folders were found."}</span></div> : <>
      <div className="dataset-overview-stats">
        <DatasetStat label="Categories present" value={`${summary.category_count} / 15`} />
        <DatasetStat label="Training images" value={totals.train.toLocaleString()} />
        <DatasetStat label="Normal test images" value={totals.normal.toLocaleString()} />
        <DatasetStat label="Defective test images" value={totals.defective.toLocaleString()} />
        <DatasetStat label="Ground-truth masks" value={totals.masks.toLocaleString()} />
      </div>
      <section className="dataset-panel">
        <div className="dataset-panel-heading"><div><h3>{selected ? formatCategory(selected.category) : "Category details"}</h3><p>Counts are read from the available local dataset files.</p></div><span className="dataset-availability">{selected ? "Dataset folder present" : "No category selected"}</span></div>
        {selected && <div className="dataset-count-grid">
          <DatasetCount label="Training · good" count={selected.training_count} verified={selected.training_count_verified} />
          <DatasetCount label="Test · good (normal)" count={selected.normal_test_count} verified={selected.normal_test_count_verified} />
          <DatasetCount label="Test · defect folders" count={selected.defective_test_count} verified={selected.defective_test_count_verified} />
          <DatasetCount label="Ground-truth masks" count={selected.ground_truth_mask_count} verified={selected.ground_truth_mask_count_verified} />
        </div>}
      </section>

      <section className="dataset-panel">
        <div className="dataset-panel-heading"><div><h3>Representative test samples</h3><p>Up to three normal and three defective test images for the selected category. Masks appear only when files exist.</p></div></div>
        {samplesError && <div className="history-error" role="alert">{samplesError}</div>}
        {samplesLoading ? <div className="history-state">Loading representative samples…</div> : samples.length === 0 ? <div className="dataset-sample-empty">No test samples are available for this category.</div> : <div className="dataset-sample-grid">{samples.map((sample) => <article className="dataset-sample-card" key={sample.sample_id}>
          <div className="dataset-sample-title"><strong title={sample.filename}>{sample.filename}</strong><span className={`history-label ${sample.group === "defective" ? "defective" : "normal"}`}>{sample.group}</span></div>
          <p>{sample.group === "normal" ? "good · normal" : formatCategory(sample.defect_type)}</p>
          <div className={`dataset-sample-images ${sample.mask_available ? "with-mask" : ""}`}><figure><img src={`${API_BASE}${sample.image_url}`} alt={`${sample.group} ${sample.filename}`} loading="lazy" /><figcaption>Test image</figcaption></figure>{sample.mask_available && sample.mask_url ? <figure><img src={`${API_BASE}${sample.mask_url}`} alt={`Ground-truth defect mask for ${sample.filename}`} loading="lazy" /><figcaption>Ground-truth mask</figcaption></figure> : <div className="dataset-mask-note">{sample.group === "normal" ? "Normal samples have no defect mask." : "Ground-truth mask unavailable."}</div>}</div>
        </article>)}</div>}
      </section>

      <section className="dataset-panel dataset-structure-panel"><div><h3>Dataset organization</h3><p>Training uses only normal examples. Test images are grouped into `good` and defect-type folders; masks are stored by defect type and matched by filename stem.</p></div><pre aria-label="MVTec AD folder structure"><code>{`dataset/\n  ${selectedCategory || "<category>"}/\n    train/good/\n    test/good/\n    test/<defect_type>/\n    ground_truth/<defect_type>/*_mask.png`}</code></pre></section>
    </>}
  </section>;
}

function DatasetStat({ label, value }: { label: string; value: string }) { return <div className="dataset-stat"><span>{label}</span><strong>{value}</strong></div>; }
function DatasetCount({ label, count, verified }: { label: string; count: number; verified: boolean }) { return <div className="dataset-count"><span>{label}</span><strong>{verified ? count.toLocaleString() : "Not available"}</strong></div>; }
function formatCategory(value: string) { return value.replaceAll("_", " "); }
