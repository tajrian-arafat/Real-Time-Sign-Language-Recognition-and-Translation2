import { useRef, useState } from "react";

type VideoUploadControlsProps = {
  disabled?: boolean;
};

export function VideoUploadControls({
  disabled = false,
}: VideoUploadControlsProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const onPick = () => inputRef.current?.click();

  const onChange = (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    if (!file) {
      setSelectedName(null);
      setStatus(null);
      return;
    }
    setSelectedName(file.name);
    setStatus(
      "Stub: upload will POST to /api/video when the backend job queue (Agent 6) is ready.",
    );
    ev.target.value = "";
  };

  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4">
      <h2 className="mb-2 text-lg font-semibold">Uploaded video</h2>
      <p className="mb-3 text-sm text-slate-400">
        Alternative to live webcam — server-side MediaPipe Tasks processing
        (not wired yet).
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        disabled={disabled}
        onChange={onChange}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={onPick}
        className="rounded-lg border border-surface-border px-4 py-2 text-sm hover:bg-surface-border/30 disabled:opacity-50"
      >
        Choose video file
      </button>
      {selectedName && (
        <p className="mt-2 text-sm text-slate-300">Selected: {selectedName}</p>
      )}
      {status && <p className="mt-1 text-xs text-amber-200/90">{status}</p>}
    </section>
  );
}
