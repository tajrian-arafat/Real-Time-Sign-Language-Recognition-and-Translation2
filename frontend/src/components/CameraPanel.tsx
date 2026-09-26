import type { RefObject } from "react";
import type { WebcamStatus } from "../hooks/useWebcam";
import type { MediapipeStatus } from "../hooks/useMediapipeLandmarks";

type CameraPanelProps = {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  webcamStatus: WebcamStatus;
  webcamError: string | null;
  mediapipeStatus: MediapipeStatus;
  mediapipeError: string | null;
  cameraOn: boolean;
  onToggleCamera: () => void;
};

function statusLabel(
  webcam: WebcamStatus,
  mp: MediapipeStatus,
): string {
  if (webcam === "error") return "Camera error";
  if (webcam === "requesting") return "Starting camera…";
  if (webcam !== "active") return "Camera off";
  if (mp === "loading") return "Loading MediaPipe Tasks…";
  if (mp === "error") return "Landmarks unavailable";
  if (mp === "ready") return "Live + landmark overlay";
  return "Camera active";
}

export function CameraPanel({
  videoRef,
  canvasRef,
  webcamStatus,
  webcamError,
  mediapipeStatus,
  mediapipeError,
  cameraOn,
  onToggleCamera,
}: CameraPanelProps) {
  return (
    <section className="rounded-xl border border-surface-border bg-surface-raised p-4 shadow-lg">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold text-white">Camera</h2>
          <p className="text-sm text-slate-400">
            {statusLabel(webcamStatus, mediapipeStatus)}
          </p>
        </div>
        <button
          type="button"
          onClick={onToggleCamera}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-muted"
        >
          {cameraOn ? "Stop camera" : "Start camera"}
        </button>
      </header>

      <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black">
        <video
          ref={videoRef}
          className="h-full w-full object-cover mirror"
          playsInline
          muted
          aria-label="Webcam preview"
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full object-cover mirror"
          aria-hidden
        />
        {!cameraOn && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/80 text-sm text-slate-300">
            Enable the camera to begin live recognition.
          </div>
        )}
      </div>

      {(webcamError || mediapipeError) && (
        <p className="mt-2 text-sm text-amber-400" role="alert">
          {webcamError ?? mediapipeError}
        </p>
      )}

      <style>{`.mirror { transform: scaleX(-1); }`}</style>
    </section>
  );
}
