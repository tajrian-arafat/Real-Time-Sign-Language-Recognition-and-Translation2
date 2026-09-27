import { useCallback, useMemo, useState } from "react";

export type SentenceToken = {
  id: string;
  word: string;
  confidence: number;
  timestamp: number;
};

function tokenId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function useSentenceBuffer() {
  const [tokens, setTokens] = useState<SentenceToken[]>([]);
  const [lastCommittedWord, setLastCommittedWord] = useState<string | null>(
    null,
  );
  const [lastCommitAt, setLastCommitAt] = useState(0);

  const englishText = useMemo(
    () => tokens.map((t) => t.word).join(" "),
    [tokens],
  );

  const tryCommitWord = useCallback(
    (word: string, confidence: number, cooldownMs: number) => {
      const now = Date.now();
      if (
        lastCommittedWord === word &&
        now - lastCommitAt < cooldownMs
      ) {
        return false;
      }
      setTokens((prev) => [
        ...prev,
        { id: tokenId(), word, confidence, timestamp: now },
      ]);
      setLastCommittedWord(word);
      setLastCommitAt(now);
      return true;
    },
    [lastCommitAt, lastCommittedWord],
  );

  const appendPunctuation = useCallback((mark: "." | "?" | "!") => {
    setTokens((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (!last) return prev;
      const updated: SentenceToken = {
        ...last,
        word: `${last.word}${mark}`,
      };
      return [...prev.slice(0, -1), updated];
    });
  }, []);

  const removeLast = useCallback(() => {
    setTokens((prev) => prev.slice(0, -1));
  }, []);

  const clear = useCallback(() => {
    setTokens([]);
    setLastCommittedWord(null);
    setLastCommitAt(0);
  }, []);

  return {
    tokens,
    englishText,
    tryCommitWord,
    appendPunctuation,
    removeLast,
    clear,
  };
}
