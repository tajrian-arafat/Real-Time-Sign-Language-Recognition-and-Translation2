export function ScopeNotice() {
  return (
    <aside className="rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-100">
      <strong className="font-semibold">Scope:</strong> This app recognizes{" "}
      <em>isolated</em> ASL signs and accumulates English glosses — not full
      continuous ASL linguistic translation. Bangla is shown as instant
      word-level glosses plus an optional fluent sentence translation.
    </aside>
  );
}
