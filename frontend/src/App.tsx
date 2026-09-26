import { useCallback, useEffect, useRef, useState } from "react";
import { BanglaPanel } from "./components/BanglaPanel";
import { CameraPanel } from "./components/CameraPanel";
import { DebugOverlay } from "./components/DebugOverlay";
import { RecognitionPanel } from "./components/RecognitionPanel";
import { ScopeNotice } from "./components/ScopeNotice";
import { SentenceBar } from "./components/SentenceBar";
import { VideoUploadControls } from "./components/VideoUploadControls";
import { INFERENCE, MOCK_BANGLA_GLOSS } from "./config";
import { useFps } from "./hooks/useFps";
import { useMediapipeLandmarks } from "./hooks/useMediapipeLandmarks";
import { useRecognitionWebSocket } from "./hooks/useRecognitionWebSocket";
import { useSentenceBuffer } from "./hooks/useSentenceBuffer";
import { useWebcam } from "./hooks/useWebcam";

export default function App() {
  const [cameraOn, setCameraOn] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [fluentBangla, setFluentBangla] = useState<string | null>(null);
  const [translatePending, setTranslatePending] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null!);

  const { videoRef, status: webcamStatus, error: webcamError } =
    useWebcam(cameraOn);

  const { status: mpStatus, error: mpError, counts: landmarkCounts } =
    useMediapipeLandmarks(videoRef, canvasRef, cameraOn);

  const sessionActive = cameraOn;
  const fps = useFps(cameraOn && webcamStatus === "active");

  const {
    mockMode,
    connectionState,
    snapshot,
    sendLandmarkWindow,
  } = useRecognitionWebSocket(sessionActive);

  const {
    englishText,
    tryCommitWord,
    appendPunctuation,
    removeLast,
    clear,
  } = useSentenceBuffer();

  useEffect(() => {
    if (
      !sessionActive ||
      !snapshot.currentWord ||
      snapshot.confidence < INFERENCE.confidenceThreshold
    ) {
      return;
    }
    tryCommitWord(
      snapshot.currentWord,
      snapshot.confidence,
      INFERENCE.debounceCooldownMs,
    );
  }, [sessionActive, snapshot, tryCommitWord]);

  useEffect(() => {
    if (!sessionActive || mockMode) return;
    sendLandmarkWindow({
      type: "landmark_window",
      frames: [],
    });
  }, [sessionActive, mockMode, sendLandmarkWindow, snapshot.lastUpdatedAt]);

  const instantGloss =
    snapshot.currentWord &&
    snapshot.confidence >= INFERENCE.confidenceThreshold
      ? (MOCK_BANGLA_GLOSS[snapshot.currentWord] ?? `[${snapshot.currentWord}]`)
      : null;

  const wordByWordBangla = englishText
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => MOCK_BANGLA_GLOSS[w.replace(/[.?!]$/, "")] ?? w)
    .join(" ");

  const onTranslateSentence = useCallback(() => {
    if (!englishText.trim()) return;
    setTranslatePending(true);
    setFluentBangla(null);
    window.setTimeout(() => {
      setFluentBangla(
        mockMode
          ? `[Mock fluent Bangla for: "${englishText}"] — BanglaT5 wiring pending Agent 5/6.`
          : null,
      );
      setTranslatePending(false);
    }, 600);
  }, [englishText, mockMode]);

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 py-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Real-Time ASL → Bangla
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Browser MediaPipe Tasks + WebSocket inference (mock until backend)
          </p>
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={debugOpen}
            onChange={(e) => setDebugOpen(e.target.checked)}
            className="rounded border-surface-border"
          />
          Debug overlay
        </label>
      </header>

      <ScopeNotice />

      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <div className="flex flex-col gap-6">
          <CameraPanel
            videoRef={videoRef}
            canvasRef={canvasRef}
            webcamStatus={webcamStatus}
            webcamError={webcamError}
            mediapipeStatus={mpStatus}
            mediapipeError={mpError}
            cameraOn={cameraOn}
            onToggleCamera={() => setCameraOn((v) => !v)}
          />
          <VideoUploadControls />
        </div>

        <div className="flex flex-col gap-6">
          <RecognitionPanel
            snapshot={snapshot}
            mockMode={mockMode}
            threshold={INFERENCE.confidenceThreshold}
          />
          <BanglaPanel
            wordGloss={instantGloss}
            wordByWordBangla={wordByWordBangla}
            fluentSentence={fluentBangla}
            translatePending={translatePending}
            onTranslateSentence={onTranslateSentence}
          />
        </div>
      </div>

      <SentenceBar
        englishText={englishText}
        onRemoveLast={removeLast}
        onClear={clear}
        onPunctuation={appendPunctuation}
      />

      <DebugOverlay
        visible={debugOpen}
        fps={fps}
        connectionState={connectionState}
        mockMode={mockMode}
        inferenceLatencyMs={snapshot.lastLatencyMs}
        modelVersion={snapshot.modelVersion}
        vocabSize={snapshot.vocabSize}
        confidence={snapshot.confidence}
        landmarkCounts={landmarkCounts}
        mediapipeStatus={mpStatus}
      />
    </div>
  );
}
