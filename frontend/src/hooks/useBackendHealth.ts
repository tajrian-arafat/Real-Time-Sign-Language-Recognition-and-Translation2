import { useEffect, useState } from "react";

export type BackendHealthSnapshot = {
  loading: boolean;
  isStub: boolean | null;
  modelLoaded: boolean | null;
  error: string | null;
};

const INITIAL: BackendHealthSnapshot = {
  loading: true,
  isStub: null,
  modelLoaded: null,
  error: null,
};

type HealthResponse = {
  is_stub?: boolean | null;
  model_loaded?: boolean | null;
};

export function useBackendHealth(): BackendHealthSnapshot {
  const [health, setHealth] = useState<BackendHealthSnapshot>(INITIAL);

  useEffect(() => {
    let cancelled = false;

    void fetch("/health")
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return (await response.json()) as HealthResponse;
      })
      .then((data) => {
        if (cancelled) return;
        setHealth({
          loading: false,
          isStub: data.is_stub ?? null,
          modelLoaded: data.model_loaded ?? null,
          error: null,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message =
          err instanceof Error ? err.message : "Health check failed";
        setHealth({
          loading: false,
          isStub: null,
          modelLoaded: null,
          error: message,
        });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return health;
}

export function inferenceBackendLabel(
  health: BackendHealthSnapshot,
): string {
  if (health.loading) {
    return "inference backend…";
  }
  if (health.error || health.isStub === null) {
    return "inference backend (unknown)";
  }
  return health.isStub ? "stub ONNX" : "real ONNX";
}
