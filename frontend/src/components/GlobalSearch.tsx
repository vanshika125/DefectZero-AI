import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { fetchInspections } from "../api";
import type { InspectionRecord } from "../api";

const PAGES = ["Overview", "Inspect", "Batch Inspection", "Datasets", "Inspection History", "Model Evaluation", "Reports", "Settings"];

export default function GlobalSearch({ categories, onNavigate }: { categories: string[]; onNavigate: (page: string, category?: string, inspectionId?: string) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [records, setRecords] = useState<InspectionRecord[]>([]);
  const [recordsLoaded, setRecordsLoaded] = useState(false);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const recordsRequestStarted = useRef(false);
  const normalized = query.trim().toLocaleLowerCase();

  useEffect(() => {
    const onShortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
        inputRef.current?.focus();
      }
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onShortcut);
    return () => document.removeEventListener("keydown", onShortcut);
  }, []);

  useEffect(() => {
    if (!open || normalized.length < 2 || recordsLoaded || recordsRequestStarted.current) return;
    recordsRequestStarted.current = true;
    let cancelled = false;
    setRecordsLoading(true);
    fetchInspections(100, 0).then((rows) => { if (!cancelled) { setRecords(rows); setRecordsLoaded(true); setRecordsError(""); } }).catch((caught: unknown) => {
      if (!cancelled) { setRecordsError(caught instanceof Error ? caught.message : "Saved inspections are unavailable."); setRecordsLoaded(true); }
    }).finally(() => { if (!cancelled) setRecordsLoading(false); });
    return () => { cancelled = true; };
  }, [open, normalized, recordsLoaded]);

  const matchingPages = useMemo(() => normalized ? PAGES.filter((page) => page.toLocaleLowerCase().includes(normalized)) : [], [normalized]);
  const matchingCategories = useMemo(() => normalized ? categories.filter((category) => category.toLocaleLowerCase().replaceAll("_", " ").includes(normalized) || category.toLocaleLowerCase().includes(normalized)) : [], [categories, normalized]);
  const matchingRecords = useMemo(() => normalized ? records.filter((record) => `${record.filename} ${record.category.replaceAll("_", " ")} ${record.predicted_label}`.toLocaleLowerCase().includes(normalized)).slice(0, 5) : [], [normalized, records]);
  const hasResults = matchingPages.length + matchingCategories.length + matchingRecords.length > 0;

  const choose = (page: string, category?: string, inspectionId?: string) => {
    onNavigate(page, category, inspectionId);
    setQuery("");
    setOpen(false);
    inputRef.current?.blur();
  };

  return <div className="global-search">
    <div className="search-box"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="10.8" cy="10.8" r="7"/><path d="m16 16 5 5"/></svg><input ref={inputRef} value={query} aria-label="Search dashboard pages, categories, and inspections" aria-expanded={open && normalized.length > 0} aria-controls="global-search-results" role="combobox" aria-autocomplete="list" placeholder="Search..." onFocus={() => setOpen(true)} onBlur={() => window.setTimeout(() => setOpen(false), 120)} onChange={(event) => { setQuery(event.target.value); setOpen(true); }} onKeyDown={(event) => { if (event.key === "Escape") setOpen(false); if (event.key === "Enter") { if (matchingPages[0]) choose(matchingPages[0]); else if (matchingCategories[0]) choose("Datasets", matchingCategories[0]); else if (matchingRecords[0]) choose("Inspection History", undefined, matchingRecords[0].inspection_id); } }} /><kbd>Ctrl K</kbd></div>
    {open && normalized.length > 0 && <div className="global-search-results" id="global-search-results" role="listbox" aria-label="Search results">
      {!hasResults && !recordsLoading && <div className="search-no-results">No matching pages, categories, or recent inspections.</div>}
      {matchingPages.length > 0 && <SearchGroup title="Pages">{matchingPages.map((page) => <SearchOption key={`page-${page}`} label={page} detail="Dashboard section" onClick={() => choose(page)} />)}</SearchGroup>}
      {matchingCategories.length > 0 && <SearchGroup title="MVTec categories">{matchingCategories.map((category) => <SearchOption key={`category-${category}`} label={category.replaceAll("_", " ")} detail="Open in Dataset Explorer" onClick={() => choose("Datasets", category)} />)}</SearchGroup>}
      {matchingRecords.length > 0 && <SearchGroup title="Recent inspections">{matchingRecords.map((record) => <SearchOption key={record.inspection_id} label={record.filename} detail={`${record.category.replaceAll("_", " ")} · ${record.predicted_label} · ${record.anomaly_score.toFixed(4)}`} onClick={() => choose("Inspection History", undefined, record.inspection_id)} />)}</SearchGroup>}
      {recordsLoading && <div className="search-loading" role="status">Searching recent inspections…</div>}
      {recordsError && <div className="search-record-error" role="status">{recordsError} · Page and category search still works.</div>}
    </div>}
  </div>;
}

function SearchGroup({ title, children }: { title: string; children: ReactNode }) { return <section className="search-result-group"><h3>{title}</h3>{children}</section>; }
function SearchOption({ label, detail, onClick }: { label: string; detail: string; onClick: () => void }) { return <button className="search-result-option" type="button" role="option" onMouseDown={(event) => event.preventDefault()} onClick={onClick}><strong>{label}</strong><span>{detail}</span></button>; }
