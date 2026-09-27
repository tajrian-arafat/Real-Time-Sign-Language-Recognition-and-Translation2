import { useCallback, useRef } from "react";
import type { LandmarkFrame } from "../types/ws";

/** Rolling buffer of landmark frames for WebSocket windows (matches config sequence_length_T). */
export const LANDMARK_WINDOW_SIZE = 64;

export function useLandmarkWindowBuffer(maxFrames = LANDMARK_WINDOW_SIZE) {
  const bufferRef = useRef<LandmarkFrame[]>([]);

  const pushFrame = useCallback(
    (frame: LandmarkFrame) => {
      const next = [...bufferRef.current, frame];
      if (next.length > maxFrames) {
        next.splice(0, next.length - maxFrames);
      }
      bufferRef.current = next;
    },
    [maxFrames],
  );

  const getWindow = useCallback((): LandmarkFrame[] => {
    return [...bufferRef.current];
  }, []);

  const clear = useCallback(() => {
    bufferRef.current = [];
  }, []);

  return { pushFrame, getWindow, clear };
}
