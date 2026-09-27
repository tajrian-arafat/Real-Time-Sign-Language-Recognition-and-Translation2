import { useCallback, useEffect, useRef, useState } from "react";
import { BanglaPanel } from "./components/BanglaPanel";
import { CameraPanel } from "./components/CameraPanel";
import { DebugOverlay } from "./components/DebugOverlay";
import { RecognitionPanel } from "./components/RecognitionPanel";
import { ScopeNotice } from "./components/ScopeNotice";
import { SentenceBar } from "./components/SentenceBar";
import { VideoUploadControls } from "./components/VideoUploadControls";
import { INFERENCE } from "./config";
import { useFps } from "./hooks/useFps";
import { useLandmarkWindowBuffer } from "./hooks/useLandmarkWindowBuffer";
import { useMediapipeLandmarks } from "./hooks/useMediapipeLandmarks";
import { useRecognitionWebSocket } from "./hooks/useRecognitionWebSocket";
import { useSentenceBuffer } from "./hooks/useSentenceBuffer";
import { useTranslationApi } from "./hooks/useTranslationApi";
import { useWebcam } from "./hooks/useWebcam";
import type { LandmarkFrame } from "./types/ws";

const MIN_FRAMES_BEFORE_SEND = 8;
const LANDMARK_SEND_INTERVAL_MS = 250;

export default function App() {
  const [cameraOn, setCameraOn] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [fluentBangla, setFluentBangla] = useState<string | null>(null);
  const [translatePending, setTranslatePending] = useState(false);
  const [instantGloss, setInstantGloss] = useState<string | null>(null);
  const [wordByWordBangla, setWordByWordBangla] = useState("");
  const [translateError, setTranslateError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null!);

  const { videoRef, status: webcamStatus, error: webcamError } =
    useWebcam(cameraOn);

  const { pushFrame, getWindow, clear: clearLandmarkBuffer } =
    useLandmarkWindowBuffer();

  const onLandmarkFrame = useCallback(
    (frame: LandmarkFrame) => {
      pushFrame(frame);
    },
    [pushFrame],
  );

  const { status: mpStatus, error: mpError, counts: landmarkCounts } =
    useMediapipeLandmarks(videoRef, canvasRef, cameraOn, onLandmarkFrame);

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

  const { translateWord, translateWords, translateSentence } =
    useTranslationApi();

  useEffect(() => {
    if (!sessionActive) {
      clearLandmarkBuffer();
    }
  }, [sessionActive, clearLandmarkBuffer]);

  useEffect(() => {
    if (
      !sessionActive ||
      mockMode ||
      connectionState !== "connected"
    ) {
      return;
    }
    const timer = window.setInterval(() => {
      const frames = getWindow();
      if (frames.length < MIN_FRAMES_BEFORE_SEND) {
        return;
      }
      sendLandmarkWindow({
        type: "landmark_window",
        frames,
      });
    }, LANDMARK_SEND_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [
    sessionActive,
    mockMode,
    connectionState,
    getWindow,
    sendLandmarkWindow,
  ]);

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
    const word = snapshot.currentWord;
    if (
      !word ||
      snapshot.confidence < INFERENCE.confidenceThreshold
    ) {
      setInstantGloss(null);
      return;
    }
    let cancelled = false;
    void translateWord(word)
      .then((result) => {
        if (!cancelled) {
          setInstantGloss(result.bangla ?? `[${word}]`);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setInstantGloss(`[${word}]`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [snapshot.currentWord, snapshot.confidence, translateWord]);

  useEffect(() => {
    const words = englishText.split(/\s+/).filter(Boolean);
    if (!words.length) {
      setWordByWordBangla("");
      return;
    }
    let cancelled = false;
    void translateWords(words)
      .then((result) => {
        if (!cancelled) {
          setWordByWordBangla(result.joined_bangla);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWordByWordBangla("");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [englishText, translateWords]);

  const onTranslateSentence = useCallback(async () => {
    if (!englishText.trim()) {
      return;
    }
    setTranslatePending(true);
    setTranslateError(null);
    setFluentBangla(null);
    try {
      const result = await translateSentence(englishText);
      setFluentBangla(result.bangla);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Sentence translation failed";
      setTranslateError(message);
    } finally {
      setTranslatePending(false);
    }
  }, [englishText, translateSentence]);

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 py-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Real-Time ASL → Bangla
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Browser MediaPipe Tasks → landmark WebSocket → stub ONNX + Bangla REST
            {mockMode ? " (mock mode)" : ""}
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
            translateError={translateError}
            mockMode={mockMode}
            onTranslateSentence={() => {
              void onTranslateSentence();
            }}
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
