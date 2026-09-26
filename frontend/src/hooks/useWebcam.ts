import { useCallback, useEffect, useRef, useState } from "react";

export type WebcamStatus =
  | "idle"
  | "requesting"
  | "active"
  | "error"
  | "stopped";

export function useWebcam(enabled: boolean) {
  const videoRef = useRef<HTMLVideoElement>(null!);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("stopped");
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus("active");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Camera access failed";
      setError(message);
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      void start();
    } else {
      stop();
      setStatus("idle");
    }
    return () => stop();
  }, [enabled, start, stop]);

  return { videoRef, status, error, start, stop };
}
