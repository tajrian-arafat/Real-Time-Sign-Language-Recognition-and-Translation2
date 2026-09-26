import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";

export type MediapipeStatus =
  | "idle"
  | "loading"
  | "ready"
  | "error"
  | "unsupported";

export type LandmarkCounts = {
  hands: number;
  pose: number;
  face: number;
  total: number;
};

const WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const MODEL_ASSETS = {
  hand:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  pose:
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  face:
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
} as const;

export function useMediapipeLandmarks(
  videoRef: RefObject<HTMLVideoElement>,
  canvasRef: RefObject<HTMLCanvasElement>,
  active: boolean,
) {
  const [status, setStatus] = useState<MediapipeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState<LandmarkCounts>({
    hands: 0,
    pose: 0,
    face: 0,
    total: 0,
  });

  const handRef = useRef<HandLandmarker | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const faceRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);

  const drawStubOverlay = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      handLandmarks: { x: number; y: number }[][],
      poseLandmarks: { x: number; y: number }[] | undefined,
    ) => {
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(59, 130, 246, 0.85)";
      ctx.fillStyle = "rgba(59, 130, 246, 0.9)";
      ctx.lineWidth = 2;

      for (const hand of handLandmarks) {
        for (const pt of hand) {
          ctx.beginPath();
          ctx.arc(pt.x * width, pt.y * height, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (poseLandmarks) {
        ctx.strokeStyle = "rgba(34, 197, 94, 0.7)";
        for (const pt of poseLandmarks) {
          ctx.beginPath();
          ctx.arc(pt.x * width, pt.y * height, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      ctx.fillStyle = "rgba(15, 20, 25, 0.55)";
      ctx.fillRect(8, 8, 168, 22);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "12px system-ui";
      ctx.fillText("MediaPipe Tasks overlay", 14, 23);
    },
    [],
  );

  useEffect(() => {
    if (!active) {
      setStatus("idle");
      return;
    }

    let cancelled = false;

    async function init() {
      setStatus("loading");
      setError(null);
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
        const [hand, pose, face] = await Promise.all([
          HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: MODEL_ASSETS.hand },
            runningMode: "VIDEO",
            numHands: 2,
          }),
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: MODEL_ASSETS.pose },
            runningMode: "VIDEO",
          }),
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: MODEL_ASSETS.face },
            runningMode: "VIDEO",
          }),
        ]);
        if (cancelled) {
          hand.close();
          pose.close();
          face.close();
          return;
        }
        handRef.current = hand;
        poseRef.current = pose;
        faceRef.current = face;
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof Error ? err.message : "MediaPipe init failed";
        setError(message);
        setStatus("error");
      }
    }

    void init();

    return () => {
      cancelled = true;
      handRef.current?.close();
      poseRef.current?.close();
      faceRef.current?.close();
      handRef.current = null;
      poseRef.current = null;
      faceRef.current = null;
    };
  }, [active]);

  useEffect(() => {
    if (!active || status !== "ready") {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }

    let lastVideoTime = -1;

    const loop = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const hand = handRef.current;
      const pose = poseRef.current;
      const face = faceRef.current;

      if (video && canvas && hand && pose && face && video.readyState >= 2) {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          const timestamp = performance.now();
          const handResult = hand.detectForVideo(video, timestamp);
          const poseResult = pose.detectForVideo(video, timestamp);
          const faceResult = face.detectForVideo(video, timestamp);

          const handPts =
            handResult.landmarks?.map((lm) =>
              lm.map((p) => ({ x: p.x, y: p.y })),
            ) ?? [];
          const posePts = poseResult.landmarks?.[0]?.map((p) => ({
            x: p.x,
            y: p.y,
          }));
          const faceCount = faceResult.faceLandmarks?.[0]?.length ?? 0;

          const handCount = handPts.reduce((n, h) => n + h.length, 0);
          const poseCount = posePts?.length ?? 0;

          setCounts({
            hands: handCount,
            pose: poseCount,
            face: faceCount,
            total: handCount + poseCount + faceCount,
          });

          const ctx = canvas.getContext("2d");
          if (ctx) {
            drawStubOverlay(
              ctx,
              canvas.width,
              canvas.height,
              handPts,
              posePts,
            );
          }
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [active, status, videoRef, canvasRef, drawStubOverlay]);

  return { status, error, counts };
}
