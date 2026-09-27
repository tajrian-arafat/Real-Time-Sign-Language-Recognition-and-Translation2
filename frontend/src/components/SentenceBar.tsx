type SentenceBarProps = {
  englishText: string;
  onRemoveLast: () => void;
  onClear: () => void;
  onPunctuation: (mark: "." | "?" | "!") => void;
};

export function SentenceBar({
  englishText,
  onRemoveLast,
  onClear,
  onPunctuation,
}: SentenceBarProps) {
  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Sentence buffer</h2>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onRemoveLast}
            className="rounded-lg border border-surface-border px-3 py-1.5 text-sm hover:bg-surface-border/30"
          >
            Remove last
          </button>
          <button
            type="button"
            onClick={onClear}
            className="rounded-lg border border-red-500/40 px-3 py-1.5 text-sm text-red-200 hover:bg-red-500/10"
          >
            Clear
          </button>
        </div>
      </header>

      <p className="min-h-[3rem] rounded-lg bg-surface p-3 text-base leading-relaxed text-slate-100">
        {englishText || (
          <span className="text-slate-500">
            Recognized English glosses accumulate here (isolated signs, not
            continuous ASL grammar).
          </span>
        )}
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-500">Manual punctuation:</span>
        {([".", "?", "!"] as const).map((mark) => (
          <button
            key={mark}
            type="button"
            onClick={() => onPunctuation(mark)}
            className="h-9 w-9 rounded-lg border border-surface-border text-lg hover:bg-surface-border/30"
            aria-label={`Append ${mark}`}
          >
            {mark}
          </button>
        ))}
        <span className="text-xs text-slate-500">
          ASL prosody is out of scope — user adds punctuation explicitly.
        </span>
      </div>
    </section>
  );
}
