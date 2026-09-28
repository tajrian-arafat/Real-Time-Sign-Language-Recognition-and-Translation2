import type { RecognitionSnapshot } from "../hooks/useRecognitionWebSocket";
import type { ConnectionState } from "../types/ws";

type RecognitionPanelProps = {
  snapshot: RecognitionSnapshot;
  mockMode: boolean;
  threshold: number;
  connectionState: ConnectionState;
};

function connectionBadgeClass(state: ConnectionState): string {
  switch (state) {
    case "connected":
      return "bg-emerald-500/20 text-emerald-200";
    case "connecting":
      return "bg-sky-500/20 text-sky-200";
    case "error":
      return "bg-red-500/20 text-red-200";
    case "disconnected":
      return "bg-slate-500/20 text-slate-300";
    case "idle":
      return "bg-slate-600/20 text-slate-400";
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

function connectionLabel(state: ConnectionState): string {
  switch (state) {
    case "idle":
      return "WS idle";
    case "connecting":
      return "WS connecting";
    case "connected":
      return "WS connected";
    case "disconnected":
      return "WS disconnected";
    case "error":
      return "WS error";
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

export function RecognitionPanel({
  snapshot,
  mockMode,
  threshold,
  connectionState,
}: RecognitionPanelProps) {
  const hasPrediction = snapshot.lastUpdatedAt !== null;
  const displayWord = snapshot.currentWord ?? "—";
  const belowThreshold =
    hasPrediction &&
    snapshot.currentWord !== null &&
    snapshot.confidence < threshold;
  const confident =
    hasPrediction &&
    snapshot.currentWord !== null &&
    snapshot.confidence >= threshold;

  let wordClass = "text-slate-500";
  if (confident) {
    wordClass = "text-white";
  } else if (belowThreshold) {
    wordClass = "text-amber-200";
  }

  let confidenceClass = "text-slate-400";
  if (confident) {
    confidenceClass = "text-emerald-400";
  } else if (belowThreshold) {
    confidenceClass = "text-amber-300/90";
  }

  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4">
      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Recognition</h2>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {!mockMode && (
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-medium ${connectionBadgeClass(connectionState)}`}
            >
              {connectionLabel(connectionState)}
            </span>
          )}
          {mockMode && (
            <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-200">
              Mock WS
            </span>
          )}
        </div>
      </header>

      <div
        className={`rounded-lg bg-surface p-4 ${!hasPrediction ? "ring-1 ring-slate-700/80" : belowThreshold ? "ring-1 ring-amber-500/30" : ""}`}
      >
        <p className="text-xs uppercase tracking-wide text-slate-500">
          Current sign (English gloss)
        </p>
        <p className={`mt-1 text-3xl font-bold ${wordClass}`}>{displayWord}</p>
        <p className="mt-2 text-sm text-slate-400">
          {!hasPrediction ? (
            <>
              <span className="text-slate-500">No prediction yet</span>
              <span className="ml-2 text-slate-600">
                (confidence stays 0% until the server returns a gloss)
              </span>
            </>
          ) : (
            <>
              Confidence:{" "}
              <span className={confidenceClass}>
                {(snapshot.confidence * 100).toFixed(1)}%
              </span>
              {belowThreshold && (
                <span className="ml-2 text-amber-200/80">
                  Below {Math.round(threshold * 100)}% threshold — shown for
                  debugging; sentence commit still gated
                </span>
              )}
            </>
          )}
        </p>
      </div>

      {snapshot.topK.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm text-slate-400">
          {snapshot.topK.map((c) => (
            <li key={c.word} className="flex justify-between gap-4">
              <span>{c.word}</span>
              <span>{(c.confidence * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
