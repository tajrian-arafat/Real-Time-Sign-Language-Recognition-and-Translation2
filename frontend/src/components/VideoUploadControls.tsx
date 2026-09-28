import { useRef, useState } from "react";

type VideoUploadControlsProps = {
  disabled?: boolean;
};

type VideoSegment = {
  timestamp_ms: number;
  word: string;
  confidence: number;
};

type VideoRecognitionResponse = {
  segments: VideoSegment[];
  sentence_english: string;
  latency_ms: number;
  vocab_size: number;
};

export function VideoUploadControls({
  disabled = false,
}: VideoUploadControlsProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VideoRecognitionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onPick = () => inputRef.current?.click();

  const onChange = (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    ev.target.value = "";
    if (!file) {
      setSelectedName(null);
      setStatus(null);
      setResult(null);
      setError(null);
      return;
    }
    setSelectedName(file.name);
    setResult(null);
    setError(null);
    void upload(file);
  };

  const upload = async (file: File) => {
    setLoading(true);
    setStatus("Uploading and extracting landmarks…");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/recognize/video", {
        method: "POST",
        body,
      });
      const payload = (await response.json()) as
        | VideoRecognitionResponse
        | { detail?: string };
      if (!response.ok) {
        const detail =
          typeof payload === "object" && payload && "detail" in payload
            ? String(payload.detail)
            : `HTTP ${response.status}`;
        throw new Error(detail);
      }
      const data = payload as VideoRecognitionResponse;
      setResult(data);
      const top = data.segments[0];
      setStatus(
        top
          ? `Top segment: ${top.word} (${(top.confidence * 100).toFixed(1)}%)`
          : "No segments above display threshold.",
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4">
      <h2 className="mb-2 text-lg font-semibold">Uploaded video</h2>
      <p className="mb-3 text-sm text-slate-400">
        Server-side MediaPipe Tasks → ONNX classifier (same model as live mode).
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        disabled={disabled || loading}
        onChange={onChange}
      />
      <button
        type="button"
        disabled={disabled || loading}
        onClick={onPick}
        className="rounded-lg border border-surface-border px-4 py-2 text-sm hover:bg-surface-border/30 disabled:opacity-50"
      >
        {loading ? "Processing…" : "Choose video file"}
      </button>
      {selectedName && (
        <p className="mt-2 text-sm text-slate-300">Selected: {selectedName}</p>
      )}
      {status && <p className="mt-1 text-xs text-emerald-200/90">{status}</p>}
      {error && <p className="mt-1 text-xs text-red-300">{error}</p>}
      {result && result.segments.length > 0 && (
        <ul className="mt-3 max-h-40 space-y-1 overflow-y-auto text-xs text-slate-300">
          {result.segments.map((seg) => (
            <li key={`${seg.timestamp_ms}-${seg.word}`}>
              {Math.round(seg.timestamp_ms / 1000)}s — {seg.word} (
              {(seg.confidence * 100).toFixed(1)}%)
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
