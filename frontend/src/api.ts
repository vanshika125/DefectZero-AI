const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export type InspectionRecord = {
  inspection_id: string;
  category: string;
  filename: string;
  predicted_label: "normal" | "defective";
  anomaly_score: number;
  threshold: number;
  inference_time_ms: number;
  timestamp_utc: string;
  image_path: string;
  heatmap_path: string;
  ground_truth_mask_path: string | null;
};

export type InspectionAsset = "image" | "heatmap" | "ground_truth";

async function requireOk(response: Response) {
  if (response.ok) return response;
  const payload = await response.json().catch(() => null) as { detail?: { message?: string } | string } | null;
  const message = typeof payload?.detail === "string" ? payload.detail : payload?.detail?.message;
  throw new Error(message || `History request failed with HTTP ${response.status}.`);
}

export async function fetchInspections(limit: number, offset: number): Promise<InspectionRecord[]> {
  const response = await requireOk(await fetch(`${API_BASE}/inspections?limit=${limit}&offset=${offset}`));
  return await response.json() as InspectionRecord[];
}

export async function fetchInspection(inspectionId: string): Promise<InspectionRecord> {
  const response = await requireOk(await fetch(`${API_BASE}/inspections/${encodeURIComponent(inspectionId)}`));
  return await response.json() as InspectionRecord;
}

export function inspectionAssetUrl(inspectionId: string, asset: InspectionAsset): string {
  return `${API_BASE}/inspections/${encodeURIComponent(inspectionId)}/assets/${asset}`;
}

export function inspectionReportUrl(inspectionId: string, inline = false): string {
  return `${API_BASE}/inspections/${encodeURIComponent(inspectionId)}/report.pdf${inline ? "?inline=true" : ""}`;
}

export async function downloadInspectionReport(inspectionId: string): Promise<void> {
  const response = await requireOk(await fetch(inspectionReportUrl(inspectionId)));
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `defectzero-inspection-${inspectionId}.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
