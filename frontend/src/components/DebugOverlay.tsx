import type { ConnectionState } from "../types/ws";
import type { LandmarkCounts } from "../hooks/useMediapipeLandmarks";

type DebugOverlayProps = {
  visible: boolean;
  fps: number;
  connectionState: ConnectionState;
  mockMode: boolean;
  inferenceLatencyMs: number | null;
  modelVersion: string | null;
  vocabSize: number | null;
  confidence: number;
  landmarkCounts: LandmarkCounts;
  mediapipeStatus: string;
};

export function DebugOverlay({
  visible,
  fps,
  connectionState,
  mockMode,
  inferenceLatencyMs,
  modelVersion,
  vocabSize,
  confidence,
  landmarkCounts,
  mediapipeStatus,
}: DebugOverlayProps) {
  if (!visible) return null;

  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-50 max-w-xs rounded-lg border border-surface-border bg-surface/95 p-3 font-mono text-xs text-slate-200 shadow-xl backdrop-blur"
      aria-live="polite"
    >
      <p className="mb-2 font-sans text-[10px] uppercase tracking-wider text-slate-500">
        Debug
      </p>
      <dl className="space-y-1">
        <Row k="FPS" v={fps > 0 ? String(fps) : "—"} />
        <Row
          k="WS"
          v={`${connectionState}${mockMode ? " (mock)" : ""}`}
        />
        <Row
          k="Inference latency"
          v={
            inferenceLatencyMs !== null
              ? `${inferenceLatencyMs} ms`
              : "— (placeholder)"
          }
        />
        <Row k="Model" v={modelVersion ?? "—"} />
        <Row k="Vocab size" v={vocabSize !== null ? String(vocabSize) : "—"} />
        <Row k="Confidence" v={(confidence * 100).toFixed(1) + "%"} />
        <Row
          k="Landmarks"
          v={`H${landmarkCounts.hands} P${landmarkCounts.pose} F${landmarkCounts.face}`}
        />
        <Row k="MediaPipe" v={mediapipeStatus} />
      </dl>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-slate-500">{k}</dt>
      <dd className="text-right text-slate-100">{v}</dd>
    </div>
  );
}
