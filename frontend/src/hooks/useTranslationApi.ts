import { useCallback } from "react";
import { MOCK_BANGLA_GLOSS, MOCK_WEBSOCKET } from "../config";

export type InstantTranslateResult = {
  english: string;
  bangla: string | null;
  source: "dictionary" | "missing";
};

export type BatchTranslateResult = {
  english_words: string[];
  bangla_glosses: (string | null)[];
  joined_bangla: string;
};

export type SentenceTranslateResult = {
  english: string;
  normalized_english: string;
  bangla: string;
  latency_sec: number;
  model_id: string;
};

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function useTranslationApi() {
  const mockMode = MOCK_WEBSOCKET;

  const translateWord = useCallback(
    async (english: string): Promise<InstantTranslateResult> => {
      const trimmed = english.trim();
      if (!trimmed) {
        return { english: "", bangla: null, source: "missing" };
      }
      if (mockMode) {
        return {
          english: trimmed,
          bangla: MOCK_BANGLA_GLOSS[trimmed] ?? null,
          source: MOCK_BANGLA_GLOSS[trimmed] ? "dictionary" : "missing",
        };
      }
      return readJson(
        await fetch("/api/translate/word", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ english: trimmed }),
        }),
      );
    },
    [mockMode],
  );

  const translateWords = useCallback(
    async (words: string[]): Promise<BatchTranslateResult> => {
      const cleaned = words.map((w) => w.replace(/[.?!]$/, "").trim()).filter(Boolean);
      if (!cleaned.length) {
        return { english_words: [], bangla_glosses: [], joined_bangla: "" };
      }
      if (mockMode) {
        const glosses = cleaned.map((w) => MOCK_BANGLA_GLOSS[w] ?? null);
        return {
          english_words: cleaned,
          bangla_glosses: glosses,
          joined_bangla: glosses.filter(Boolean).join(" "),
        };
      }
      return readJson(
        await fetch("/api/translate/words", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ words: cleaned }),
        }),
      );
    },
    [mockMode],
  );

  const translateSentence = useCallback(
    async (english: string): Promise<SentenceTranslateResult> => {
      const trimmed = english.trim();
      if (!trimmed) {
        throw new Error("english must not be empty");
      }
      if (mockMode) {
        return {
          english: trimmed,
          normalized_english: trimmed,
          bangla: `[Mock fluent Bangla for: "${trimmed}"]`,
          latency_sec: 0,
          model_id: "mock",
        };
      }
      return readJson(
        await fetch("/api/translate/sentence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ english: trimmed }),
        }),
      );
    },
    [mockMode],
  );

  return { mockMode, translateWord, translateWords, translateSentence };
}
