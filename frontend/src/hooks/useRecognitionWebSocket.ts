import { useCallback, useEffect, useRef, useState } from "react";
import {
  MOCK_VOCAB,
  MOCK_WEBSOCKET,
  resolveWebSocketUrl,
} from "../config";
import type {
  ConnectionState,
  PredictionMessage,
  ServerMessage,
} from "../types/ws";

export type RecognitionSnapshot = {
  currentWord: string | null;
  confidence: number;
  topK: PredictionMessage["top_k"];
  modelVersion: string | null;
  vocabSize: number | null;
  lastLatencyMs: number | null;
  lastUpdatedAt: number | null;
};

const INITIAL_SNAPSHOT: RecognitionSnapshot = {
  currentWord: null,
  confidence: 0,
  topK: [],
  modelVersion: null,
  vocabSize: null,
  lastLatencyMs: null,
  lastUpdatedAt: null,
};

function randomMockPrediction(): PredictionMessage {
  const word =
    MOCK_VOCAB[Math.floor(Math.random() * MOCK_VOCAB.length)] ?? "hello";
  const confidence = 0.55 + Math.random() * 0.4;
  const others = MOCK_VOCAB.filter((w) => w !== word)
    .slice(0, 2)
    .map((w) => ({
      word: w,
      confidence: Math.max(0.1, confidence - 0.15 - Math.random() * 0.2),
    }));
  return {
    type: "prediction",
    word,
    confidence,
    top_k: [{ word, confidence }, ...others],
    model_version: "mock-v0",
    vocab_size: 250,
    latency_ms: 25 + Math.floor(Math.random() * 40),
  };
}

function applyPrediction(
  msg: PredictionMessage,
): RecognitionSnapshot {
  return {
    currentWord: msg.word,
    confidence: msg.confidence,
    topK: msg.top_k,
    modelVersion: msg.model_version ?? null,
    vocabSize: msg.vocab_size ?? null,
    lastLatencyMs: msg.latency_ms ?? null,
    lastUpdatedAt: Date.now(),
  };
}

export function useRecognitionWebSocket(active: boolean) {
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("idle");
  const [mockMode] = useState(MOCK_WEBSOCKET);
  const [snapshot, setSnapshot] =
    useState<RecognitionSnapshot>(INITIAL_SNAPSHOT);
  const wsRef = useRef<WebSocket | null>(null);
  const mockTimerRef = useRef<number | null>(null);

  const handleServerMessage = useCallback((raw: string) => {
    try {
      const parsed = JSON.parse(raw) as ServerMessage;
      if (parsed.type === "prediction") {
        setSnapshot(applyPrediction(parsed));
      }
    } catch {
      /* ignore malformed frames in scaffolding */
    }
  }, []);

  useEffect(() => {
    if (!active) {
      setConnectionState("idle");
      setSnapshot(INITIAL_SNAPSHOT);
      if (mockTimerRef.current !== null) {
        window.clearInterval(mockTimerRef.current);
        mockTimerRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
      return;
    }

    if (mockMode) {
      setConnectionState("connected");
      mockTimerRef.current = window.setInterval(() => {
        setSnapshot(applyPrediction(randomMockPrediction()));
      }, 2200);
      return () => {
        if (mockTimerRef.current !== null) {
          window.clearInterval(mockTimerRef.current);
          mockTimerRef.current = null;
        }
        setConnectionState("disconnected");
      };
    }

    setConnectionState("connecting");
    const url = resolveWebSocketUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnectionState("connected");
    ws.onclose = () => setConnectionState("disconnected");
    ws.onerror = () => setConnectionState("error");
    ws.onmessage = (ev) => handleServerMessage(String(ev.data));

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [active, mockMode, handleServerMessage]);

  const sendLandmarkWindow = useCallback(
    (payload: object) => {
      if (mockMode || wsRef.current?.readyState !== WebSocket.OPEN) {
        return;
      }
      wsRef.current.send(JSON.stringify(payload));
    },
    [mockMode],
  );

  return {
    mockMode,
    connectionState,
    snapshot,
    sendLandmarkWindow,
  };
}
