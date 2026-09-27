type BanglaPanelProps = {
  wordGloss: string | null;
  wordByWordBangla: string;
  fluentSentence: string | null;
  translatePending: boolean;
  translateError?: string | null;
  mockMode: boolean;
  onTranslateSentence: () => void;
};

export function BanglaPanel({
  wordGloss,
  wordByWordBangla,
  fluentSentence,
  translatePending,
  translateError,
  mockMode,
  onTranslateSentence,
}: BanglaPanelProps) {
  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4 font-bangla">
      <h2 className="mb-3 text-lg font-semibold font-sans">Bangla output</h2>

      <div className="space-y-4">
        <div className="rounded-lg bg-surface p-4">
          <p className="font-sans text-xs uppercase tracking-wide text-slate-500">
            Instant (word-by-word gloss)
          </p>
          <p className="mt-2 text-2xl leading-relaxed text-white">
            {wordGloss ?? "—"}
          </p>
          {wordByWordBangla && (
            <p className="mt-2 font-sans text-xs text-slate-500">
              Accumulated gloss (not grammatical): {wordByWordBangla}
            </p>
          )}
        </div>

        <div className="rounded-lg border border-dashed border-surface-border bg-surface/50 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="font-sans text-xs uppercase tracking-wide text-slate-500">
              Fluent sentence (BanglaT5 — on demand)
            </p>
            <button
              type="button"
              disabled={translatePending}
              onClick={onTranslateSentence}
              className="rounded-lg border border-surface-border px-3 py-1.5 font-sans text-sm text-slate-200 hover:bg-surface-border/40 disabled:opacity-50"
            >
              {translatePending ? "Translating…" : "Translate sentence"}
            </button>
          </div>
          <p className="mt-2 min-h-[2.5rem] text-xl leading-relaxed text-slate-200">
            {fluentSentence ??
              (mockMode
                ? "Mock mode — set VITE_MOCK_WS=false and run the FastAPI backend for BanglaT5."
                : "Press Translate sentence for fluent Bangla (BanglaT5 via POST /api/translate/sentence).")}
          </p>
          {translateError && (
            <p className="mt-2 font-sans text-sm text-red-400">{translateError}</p>
          )}
        </div>
      </div>

      <p className="mt-3 font-sans text-xs text-slate-500">
        Instant gloss: POST /api/translate/word · Sentence bar: /api/translate/words
      </p>
    </section>
  );
}
