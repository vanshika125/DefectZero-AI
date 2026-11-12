import { useCallback, useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

type CategoryStats = { category: string; training_count: number; normal_test_count: number; defective_test_count: number; test_count: number; ground_truth_mask_count: number; training_count_verified: boolean; normal_test_count_verified: boolean; defective_test_count_verified: boolean; ground_truth_mask_count_verified: boolean };
type DatasetSummary = { categories: CategoryStats[]; category_count: number };
type Sample = { sample_id: string; filename: string; group: "normal" | "defective"; defect_type: string; mask_available: boolean; image_url: string; mask_url: string | null };

export default function DatasetExplorer({ initialCategory }: { initialCategory?: string }) {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [selectedCategory, setSelectedCategory] = useState(initialCategory ?? "");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [categoryQuery, setCategoryQuery] = useState("");
  const [sampleGroup, setSampleGroup] = useState<"all" | "normal" | "defective">("all");
  const [loading, setLoading] = useState(true);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [error, setError] = useState("");
  const [samplesError, setSamplesError] = useState("");

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/dataset`);
      if (!response.ok) throw new Error(`Dataset metadata request failed (HTTP ${response.status}).`);
      const data = await response.json() as DatasetSummary;
      setSummary(data);
      setSelectedCategory((current) => data.categories.find((item) => item.category === (initialCategory || current))?.category ?? data.categories[0]?.category ?? "");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load dataset metadata.");
    } finally {
      setLoading(false);
    }
  }, [initialCategory]);

  useEffect(() => {
    void loadSummary();
  }, [loadSummary]);

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
  const visibleCategories = (summary?.categories ?? []).filter((item) => item.category.replaceAll("_", " ").toLocaleLowerCase().includes(categoryQuery.trim().toLocaleLowerCase()));
  const visibleSamples = samples.filter((sample) => sampleGroup === "all" || sample.group === sampleGroup);
  const totals = useMemo(() => {
    const rows = summary?.categories ?? [];
    return rows.reduce((acc, row) => ({
      train: acc.train + (row.training_count_verified ? row.training_count : 0),
      normal: acc.normal + (row.normal_test_count_verified ? row.normal_test_count : 0),
      defective: acc.defective + (row.defective_test_count_verified ? row.defective_test_count : 0),
      masks: acc.masks + (row.ground_truth_mask_count_verified ? row.ground_truth_mask_count : 0),
      total: acc.total + (row.training_count_verified ? row.training_count : 0) + (row.normal_test_count_verified ? row.normal_test_count : 0) + (row.defective_test_count_verified ? row.defective_test_count : 0),
    }), { train: 0, normal: 0, defective: 0, masks: 0, total: 0 });
  }, [summary]);

  return <section className="dataset-page">
    <div className="dataset-heading"><div><span className="history-eyebrow">MVTEC ANOMALY DETECTION</span><h2>Dataset Explorer</h2><p>Browse verified local dataset counts, category coverage, samples, and available ground-truth masks.</p></div><div className="dataset-heading-actions"><label className="dataset-category-filter">Filter categories<input type="search" value={categoryQuery} onChange={(event) => setCategoryQuery(event.target.value)} placeholder="Search categories…" aria-label="Filter MVTec categories" /></label><label className="dataset-category-filter">Selected category<select value={selectedCategory} onChange={(event) => setSelectedCategory(event.target.value)} disabled={!summary?.categories.length}><option value="">Choose a category</option>{visibleCategories.map((row) => <option key={row.category} value={row.category}>{formatCategory(row.category)}</option>)}</select></label><button type="button" className="history-secondary-button" onClick={() => void loadSummary()} disabled={loading}>{loading ? "Refreshing…" : "Refresh data"}</button></div></div>
    {error && <div className="history-error" role="alert">{error}</div>}
    {loading ? <div className="history-state">Loading dataset metadata…</div> : !summary?.categories.length ? <div className="history-empty"><strong>Dataset metadata unavailable</strong><span>{error || "No supported MVTec category folders were found."}</span></div> : <>
      <div className="dataset-overview-stats">
        <DatasetStat label="Categories present" value={`${summary.category_count} / 15`} />
        <DatasetStat label="Total available images" value={totals.total.toLocaleString()} />
        <DatasetStat label="Training images" value={totals.train.toLocaleString()} />
        <DatasetStat label="Normal test images" value={totals.normal.toLocaleString()} />
        <DatasetStat label="Defective test images" value={totals.defective.toLocaleString()} />
        <DatasetStat label="Ground-truth masks" value={totals.masks.toLocaleString()} />
      </div>
      <p className="dataset-count-disclaimer">Totals include only counts whose dataset folders are present and verified. Refresh data to rescan the local dataset.</p>
      <section className="dataset-panel"><div className="dataset-panel-heading"><div><h3>Category coverage</h3><p>{visibleCategories.length} matching categories · select a category to inspect its sample data.</p></div><span className="dataset-availability">{summary.category_count} found</span></div>
        {visibleCategories.length ? <div className="dataset-category-grid">{visibleCategories.map((row) => <button type="button" key={row.category} className={`dataset-category-card ${selectedCategory === row.category ? "selected" : ""}`} onClick={() => setSelectedCategory(row.category)} aria-pressed={selectedCategory === row.category}><span>{formatCategory(row.category)}</span><strong>{row.normal_test_count_verified && row.defective_test_count_verified ? row.test_count.toLocaleString() : "—"}</strong><small>test images · {row.training_count_verified ? row.training_count.toLocaleString() : "—"} train</small></button>)}</div> : <div className="dataset-sample-empty">No categories match this filter.</div>}
      </section>
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
        <div className="dataset-panel-heading"><div><h3>Representative test samples</h3><p>Up to three normal and three defective test images for the selected category. Masks appear only when files exist.</p></div><div className="dataset-sample-filter" role="group" aria-label="Filter sample type">{(["all", "normal", "defective"] as const).map((group) => <button type="button" key={group} className={sampleGroup === group ? "active" : ""} aria-pressed={sampleGroup === group} onClick={() => setSampleGroup(group)}>{group === "all" ? "All" : group === "normal" ? "Normal" : "Defective"}</button>)}</div></div>
        {samplesError && <div className="history-error" role="alert">{samplesError}</div>}
        {samplesLoading ? <div className="history-state">Loading representative samples…</div> : visibleSamples.length === 0 ? <div className="dataset-sample-empty">No {sampleGroup === "all" ? "test" : sampleGroup} samples are available for this category.</div> : <div className="dataset-sample-grid">{visibleSamples.map((sample) => <article className="dataset-sample-card" key={sample.sample_id}>
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
