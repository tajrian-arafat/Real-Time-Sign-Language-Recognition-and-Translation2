import type { RecognitionSnapshot } from "../hooks/useRecognitionWebSocket";

type RecognitionPanelProps = {
  snapshot: RecognitionSnapshot;
  mockMode: boolean;
  threshold: number;
};

export function RecognitionPanel({
  snapshot,
  mockMode,
  threshold,
}: RecognitionPanelProps) {
  const confident =
    snapshot.confidence >= threshold && snapshot.currentWord !== null;

  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4">
      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Recognition</h2>
        {mockMode && (
          <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-200">
            Mock WS
          </span>
        )}
      </header>

      <div className="rounded-lg bg-surface p-4">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          Current sign (English gloss)
        </p>
        <p className="mt-1 text-3xl font-bold text-white">
          {confident ? snapshot.currentWord : "…"}
        </p>
        <p className="mt-2 text-sm text-slate-400">
          Confidence:{" "}
          <span className={confident ? "text-emerald-400" : "text-slate-300"}>
            {(snapshot.confidence * 100).toFixed(1)}%
          </span>
          {!confident && snapshot.currentWord && (
            <span className="ml-2 text-slate-500">
              (below {Math.round(threshold * 100)}% threshold)
            </span>
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
